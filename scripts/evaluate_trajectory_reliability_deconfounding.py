#!/usr/bin/env python3
"""Test whether zero-scene ensemble disagreement predicts trajectory risk beyond motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_trajectory_reliability import (  # noqa: E402
    RISK_COVERAGES,
    SEEDS,
    SequenceWithImageSize,
    _correlation,
    binary_detection_metrics,
    evaluate_split,
    input_order_sha256,
    load_model,
    sha256_file,
)


BOOTSTRAP_SEED = 9124
BOOTSTRAP_REPETITIONS = 1000
RANDOM_RANKING_REPETITIONS = 100
PRIMARY_SCORE = "u_mean"


def _as_vector(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    """Partial Spearman via rank-transform, OLS residualization, then Pearson r."""
    x = _as_vector("x", x)
    y = _as_vector("y", y)
    control = _as_vector("control", control)
    if len(x) != len(y) or len(x) != len(control) or len(x) < 4:
        raise ValueError("partial Spearman inputs must have equal length >= 4")
    ranked = [rankdata(value, method="average") for value in (x, y, control)]
    design = np.column_stack((np.ones(len(control)), ranked[2]))
    residual_x = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    residual_y = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    scale_x = max(1.0, float(np.std(ranked[0])))
    scale_y = max(1.0, float(np.std(ranked[1])))
    if np.std(residual_x) <= 1e-12 * scale_x or np.std(residual_y) <= 1e-12 * scale_y:
        # A score entirely determined by the control has no residual ranking signal.
        return 0.0
    value = float(np.corrcoef(residual_x, residual_y)[0, 1])
    return 0.0 if abs(value) < 1e-14 else value


def fit_motion_adjustment(motion: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    """Fit log1p(U) ~ quadratic(log1p(observed motion)); deliberately has no ADE input."""
    motion = _as_vector("motion", motion)
    uncertainty = _as_vector("uncertainty", uncertainty)
    if len(motion) != len(uncertainty) or len(motion) < 3:
        raise ValueError("motion and uncertainty must have equal length >= 3")
    if np.any(motion < 0) or np.any(uncertainty < 0):
        raise ValueError("motion and uncertainty magnitudes must be non-negative")
    x = np.log1p(motion)
    y = np.log1p(uncertainty)
    b2, b1, b0 = np.polyfit(x, y, deg=2)
    return {
        "b0": float(b0),
        "b1": float(b1),
        "b2": float(b2),
        "fit_split": "validation",
        "target": "log1p(u_mean)",
        "predictor": "log1p(observed_motion_magnitude_pixel)",
        "degree": 2,
        "fit_sample_count": int(len(motion)),
        "future_gt_not_used": True,
        "trajectory_error_not_used": True,
    }


def apply_motion_adjustment(
    motion: np.ndarray, uncertainty: np.ndarray, coefficients: dict[str, Any]
) -> np.ndarray:
    """Apply frozen validation coefficients and return log-space residual disagreement."""
    motion = _as_vector("motion", motion)
    uncertainty = _as_vector("uncertainty", uncertainty)
    if len(motion) != len(uncertainty):
        raise ValueError("motion and uncertainty lengths differ")
    if np.any(motion < 0) or np.any(uncertainty < 0):
        raise ValueError("motion and uncertainty magnitudes must be non-negative")
    x = np.log1p(motion)
    predicted_log_u = coefficients["b0"] + coefficients["b1"] * x + coefficients["b2"] * x**2
    return np.log1p(uncertainty) - predicted_log_u


def _detection(score: np.ndarray, positive: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(positive, dtype=np.int64)
    if np.unique(labels).size < 2:
        return {
            "auroc": None,
            "auprc": None,
            "positive_count": int(labels.sum()),
            "sample_count": int(len(labels)),
        }
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "auprc": float(average_precision_score(labels, score)),
        "positive_count": int(labels.sum()),
        "sample_count": int(len(labels)),
    }


def score_metrics(
    score: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    high_ade: np.ndarray,
    high_fde: np.ndarray,
) -> dict[str, Any]:
    return {
        "spearman_vs_ade": _correlation(score, ade),
        "spearman_vs_fde": _correlation(score, fde),
        "high_ade": _detection(score, high_ade),
        "high_fde": _detection(score, high_fde),
    }


def assign_quantile_strata(values: np.ndarray, cutpoints: np.ndarray) -> np.ndarray:
    """Assign values to validation-defined quantile strata, preserving tied boundaries."""
    values = _as_vector("values", values)
    cutpoints = _as_vector("cutpoints", cutpoints)
    if len(cutpoints) < 3 or np.any(np.diff(cutpoints) < 0):
        raise ValueError("cutpoints must be a sorted quantile boundary array")
    return np.searchsorted(cutpoints[1:-1], values, side="right").astype(np.int64)


def select_stratified_indices(
    score: np.ndarray, strata: np.ndarray, coverage: float
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep the lowest score fraction independently in each fixed stratum."""
    score = _as_vector("score", score)
    strata = np.asarray(strata).reshape(-1)
    if len(score) != len(strata) or not 0 < coverage <= 1:
        raise ValueError("score/strata lengths must match and coverage must be in (0,1]")
    kept: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        count = len(members) if coverage == 1 else max(1, int(np.floor(len(members) * coverage)))
        order = np.argsort(score[members], kind="mergesort")
        selected = members[order[:count]]
        kept.append(selected)
        counts[str(int(stratum))] = int(len(selected))
    return np.concatenate(kept), counts


def _risk_stats(indices: np.ndarray, ade: np.ndarray, fde: np.ndarray) -> dict[str, float | int]:
    return {
        "retained_samples": int(len(indices)),
        "actual_coverage": float(len(indices) / len(ade)),
        "ade": float(np.mean(ade[indices])),
        "fde": float(np.mean(fde[indices])),
    }


def global_risk_curves(
    scores: dict[str, np.ndarray], ade: np.ndarray, fde: np.ndarray, seed: int = BOOTSTRAP_SEED
) -> dict[str, Any]:
    n = len(ade)
    curves: dict[str, list[dict[str, Any]]] = {}
    for name, score in scores.items():
        order = np.argsort(score, kind="mergesort")
        curves[name] = []
        for coverage in RISK_COVERAGES:
            count = n if coverage == 1 else max(1, int(np.floor(n * coverage)))
            row = {"nominal_coverage": coverage, **_risk_stats(order[:count], ade, fde)}
            curves[name].append(row)

    random_rng = np.random.default_rng(seed)
    random_values = {coverage: {"ade": [], "fde": []} for coverage in RISK_COVERAGES}
    for _ in range(RANDOM_RANKING_REPETITIONS):
        order = random_rng.permutation(n)
        for coverage in RISK_COVERAGES:
            count = n if coverage == 1 else max(1, int(np.floor(n * coverage)))
            indices = order[:count]
            random_values[coverage]["ade"].append(float(np.mean(ade[indices])))
            random_values[coverage]["fde"].append(float(np.mean(fde[indices])))
    random_curve = []
    for coverage in RISK_COVERAGES:
        count = n if coverage == 1 else max(1, int(np.floor(n * coverage)))
        random_curve.append({
            "nominal_coverage": coverage,
            "retained_samples": count,
            "actual_coverage": count / n,
            "ade_mean_sample_std": {
                "mean": float(np.mean(random_values[coverage]["ade"])),
                "sample_std": float(np.std(random_values[coverage]["ade"], ddof=1)),
            },
            "fde_mean_sample_std": {
                "mean": float(np.mean(random_values[coverage]["fde"])),
                "sample_std": float(np.std(random_values[coverage]["fde"], ddof=1)),
            },
        })
    oracle_order = np.argsort(ade, kind="mergesort")
    oracle_curve = []
    for coverage in RISK_COVERAGES:
        count = n if coverage == 1 else max(1, int(np.floor(n * coverage)))
        oracle_curve.append({
            "nominal_coverage": coverage,
            **_risk_stats(oracle_order[:count], ade, fde),
        })
    curves["random_global"] = random_curve
    curves["oracle_by_true_ade"] = oracle_curve
    return {
        "ranking_direction": "ascending score; retain lowest values first",
        "motion_only_direction": "low observed motion is treated as lower risk and retained first",
        "random_reference": {"repetitions": RANDOM_RANKING_REPETITIONS, "seed": seed},
        "oracle_reference": "ranks by true ADE; upper bound only, not deployable",
        "coverage_sample_rule": "floor(N*coverage), with all samples at 100% and at least one otherwise",
        "curves": curves,
        "monotonic_reduction": {
            name: {
                metric: bool(all(
                    later[metric] <= earlier[metric] + 1e-12
                    for earlier, later in zip(curve, curve[1:])
                ))
                for metric in ("ade", "fde")
            }
            for name, curve in curves.items()
            if name not in ("random_global",)
        },
    }


def stratified_risk_curve(
    score: np.ndarray, strata: np.ndarray, ade: np.ndarray, fde: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for coverage in RISK_COVERAGES:
        indices, per_stratum = select_stratified_indices(score, strata, coverage)
        rows.append({
            "nominal_coverage": coverage,
            **_risk_stats(indices, ade, fde),
            "retained_per_motion_decile": per_stratum,
        })
    return rows


def within_motion_permutation_test(
    raw_score: np.ndarray,
    adjusted_score: np.ndarray,
    strata: np.ndarray,
    ade: np.ndarray,
    fde: np.ndarray,
    repetitions: int = RANDOM_RANKING_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    null_values = {
        coverage: {"ade": [], "fde": []} for coverage in RISK_COVERAGES
    }
    for _ in range(repetitions):
        permuted = np.asarray(raw_score, dtype=np.float64).copy()
        for stratum in np.unique(strata):
            indices = np.flatnonzero(strata == stratum)
            permuted[indices] = rng.permutation(permuted[indices])
        curve = stratified_risk_curve(permuted, strata, ade, fde)
        for row in curve:
            bucket = null_values[row["nominal_coverage"]]
            bucket["ade"].append(row["ade"])
            bucket["fde"].append(row["fde"])

    observed = {
        "raw_u_mean": stratified_risk_curve(raw_score, strata, ade, fde),
        "adjusted_u_mean": stratified_risk_curve(adjusted_score, strata, ade, fde),
    }
    rows = []
    for position, coverage in enumerate(RISK_COVERAGES):
        null_ade = np.asarray(null_values[coverage]["ade"])
        null_fde = np.asarray(null_values[coverage]["fde"])
        row: dict[str, Any] = {
            "nominal_coverage": coverage,
            "randomized_within_motion_mean_sample_std": {
                "ade": {"mean": float(null_ade.mean()), "sample_std": float(null_ade.std(ddof=1))},
                "fde": {"mean": float(null_fde.mean()), "sample_std": float(null_fde.std(ddof=1))},
            },
        }
        for method, curve in observed.items():
            item = curve[position]
            row[method] = {"ade": item["ade"], "fde": item["fde"]}
            row[f"{method}_empirical_lower_tail_p"] = {
                metric: float((1 + np.count_nonzero(np.asarray(null_values[coverage][metric]) <= item[metric])) / (repetitions + 1))
                for metric in ("ade", "fde")
            }
        rows.append(row)
    return {
        "method": "permute raw u_mean independently within each validation-defined motion decile; fixed motion, ADE, and FDE",
        "repetitions": repetitions,
        "seed": seed,
        "lower_tail_p_interpretation": "small values mean observed stratified risk is lower than the within-motion random ranking",
        "rows": rows,
    }


def _cluster_members(cluster_ids: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    labels = np.asarray(cluster_ids).astype(str).reshape(-1)
    unique, inverse = np.unique(labels, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    return unique, members


def cluster_bootstrap_indices(cluster_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample clusters with replacement and include every member of each draw."""
    _, members = _cluster_members(cluster_ids)
    return _draw_from_cluster_members(members, rng)[0]


def _draw_from_cluster_members(
    members: list[np.ndarray], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    draws = rng.integers(0, len(members), size=len(members))
    indices = np.concatenate([members[index] for index in draws])
    return indices, draws


def _metric_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"lower_95": None, "median": None, "upper_95": None, "valid_replicates": 0}
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "valid_replicates": int(len(values)),
    }


def cluster_bootstrap_metrics(
    cluster_ids: np.ndarray,
    scores: dict[str, np.ndarray],
    ade: np.ndarray,
    high_ade: np.ndarray,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    cluster_ids = np.asarray(cluster_ids).astype(str).reshape(-1)
    unique, members = _cluster_members(cluster_ids)
    if len(cluster_ids) != len(ade) or len(cluster_ids) != len(high_ade):
        raise ValueError("cluster labels and metric arrays must be aligned")
    metrics = {
        name: {metric: [] for metric in ("spearman_vs_ade", "high_ade_auroc")}
        for name in scores
    }
    rng = np.random.default_rng(seed)
    for _ in range(repetitions):
        indices, _ = _draw_from_cluster_members(members, rng)
        labels = high_ade[indices]
        for name, score in scores.items():
            rho = spearmanr(score[indices], ade[indices]).statistic
            if np.isfinite(rho):
                metrics[name]["spearman_vs_ade"].append(float(rho))
            if np.unique(labels).size == 2:
                metrics[name]["high_ade_auroc"].append(
                    float(roc_auc_score(labels, score[indices]))
                )
    return {
        "cluster_level": "video_scene_id",
        "cluster_count": int(len(unique)),
        "sample_count": int(len(cluster_ids)),
        "repetitions": repetitions,
        "seed": seed,
        "method": "resample the same number of clusters with replacement; include all samples for every draw, including repeated clusters",
        "metrics": {
            name: {metric: _metric_ci(values) for metric, values in items.items()}
            for name, items in metrics.items()
        },
    }


def _metric_block(
    score_map: dict[str, np.ndarray],
    ade: np.ndarray,
    fde: np.ndarray,
    high_ade: np.ndarray,
    high_fde: np.ndarray,
) -> dict[str, Any]:
    return {
        name: score_metrics(score, ade, fde, high_ade, high_fde)
        for name, score in score_map.items()
    }


def _reliability_bins(
    val_score: np.ndarray,
    val_ade: np.ndarray,
    val_fde: np.ndarray,
    test_score: np.ndarray,
    test_ade: np.ndarray,
    test_fde: np.ndarray,
) -> dict[str, Any]:
    q33, q67 = np.quantile(val_score, [1 / 3, 2 / 3])
    output: dict[str, Any] = {
        "validation_cutpoints": {"q33": float(q33), "q67": float(q67)},
        "cutpoints_fit_on": "validation only",
        "bins": {},
    }
    for split, score, ade, fde in (
        ("validation", val_score, val_ade, val_fde),
        ("test", test_score, test_ade, test_fde),
    ):
        definitions = (
            ("low", score <= q33),
            ("medium", (score > q33) & (score <= q67)),
            ("high", score > q67),
        )
        rows = []
        for label, mask in definitions:
            rows.append({
                "bin": label,
                "sample_count": int(mask.sum()),
                "mean_adjusted_u": float(score[mask].mean()) if mask.any() else None,
                "mean_ade": float(ade[mask].mean()) if mask.any() else None,
                "median_ade": float(np.median(ade[mask])) if mask.any() else None,
                "mean_fde": float(fde[mask].mean()) if mask.any() else None,
                "median_fde": float(np.median(fde[mask])) if mask.any() else None,
            })
        output["bins"][split] = rows
    test_bins = output["bins"]["test"]
    output["test_low_to_high_error_monotonic"] = {
        metric: bool(
            all(item[f"mean_{metric}"] is not None for item in test_bins)
            and test_bins[0][f"mean_{metric}"] < test_bins[1][f"mean_{metric}"] < test_bins[2][f"mean_{metric}"]
        )
        for metric in ("ade", "fde")
    }
    return output


def _motion_strata(
    val_motion: np.ndarray,
    test_motion: np.ndarray,
    test_scores: dict[str, np.ndarray],
    test_ade: np.ndarray,
    test_fde: np.ndarray,
) -> dict[str, Any]:
    q33, q67 = np.quantile(val_motion, [1 / 3, 2 / 3])
    definitions = (
        ("slow", test_motion <= q33),
        ("medium", (test_motion > q33) & (test_motion <= q67)),
        ("fast", test_motion > q67),
    )
    tertiles = []
    for name, mask in definitions:
        tertiles.append({
            "stratum": name,
            "sample_count": int(mask.sum()),
            "observed_motion_mean_pixel": float(test_motion[mask].mean()) if mask.any() else None,
            "spearman": {
                score_name: {
                    "vs_ade": _correlation(score[mask], test_ade[mask]),
                    "vs_fde": _correlation(score[mask], test_fde[mask]),
                }
                for score_name, score in test_scores.items()
            },
        })
    quantiles = np.quantile(val_motion, np.linspace(0, 1, 11))
    test_deciles = assign_quantile_strata(test_motion, quantiles)
    deciles = []
    for index in range(10):
        mask = test_deciles == index
        deciles.append({
            "decile": index + 1,
            "validation_motion_lower_pixel": float(quantiles[index]),
            "validation_motion_upper_pixel": float(quantiles[index + 1]),
            "sample_count": int(mask.sum()),
            "observed_motion_mean_pixel": float(test_motion[mask].mean()) if mask.any() else None,
            "spearman": {
                score_name: {
                    "vs_ade": _correlation(score[mask], test_ade[mask]),
                    "vs_fde": _correlation(score[mask], test_fde[mask]),
                }
                for score_name, score in test_scores.items()
                if score_name in ("raw_u_mean", "adjusted_u_mean")
            },
        })
    return {
        "tertile_cutpoints_from_validation_pixel": {"q33": float(q33), "q67": float(q67)},
        "slow_medium_fast_test": tertiles,
        "decile_boundaries_from_validation_pixel": [float(value) for value in quantiles],
        "decile_boundary_assignment": "searchsorted on the nine interior validation quantiles; exact ties go to the higher decile",
        "test_motion_deciles": deciles,
    }


def _scale_diagnostics(
    image_size: np.ndarray,
    raw_u: np.ndarray,
    ade: np.ndarray,
) -> dict[str, Any]:
    width = image_size[:, 0].astype(np.float64)
    height = image_size[:, 1].astype(np.float64)
    return {
        "image_size_unique_width_height_pairs": np.unique(image_size, axis=0).tolist(),
        "width_unique_count": int(np.unique(width).size),
        "height_unique_count": int(np.unique(height).size),
        "width_vs_raw_u_spearman": _correlation(width, raw_u),
        "width_vs_ade_spearman": _correlation(width, ade),
        "height_vs_raw_u_spearman": _correlation(height, raw_u),
        "height_vs_ade_spearman": _correlation(height, ade),
        "interpretation": "Correlation is undefined when width/height are constant; report the observed resolution distribution rather than treating null as evidence of zero association.",
    }


def _previous_audit_reproduction(split: str, current: dict[str, Any]) -> dict[str, Any]:
    old_path = PROJECT_ROOT / "results/trajectory_reliability_audit" / f"{split}_metrics.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    current_report = current["report"]
    fields = {
        "ensemble_ade_pixel": (
            current_report["ensemble_mean_performance"]["ade_pixel"],
            old["ensemble_mean_performance"]["ade_pixel"],
        ),
        "ensemble_fde_pixel": (
            current_report["ensemble_mean_performance"]["fde_pixel"],
            old["ensemble_mean_performance"]["fde_pixel"],
        ),
        "u_mean_average_pixel": (
            current_report["score_mean_sample_std"][PRIMARY_SCORE]["mean"],
            old["score_mean_sample_std"][PRIMARY_SCORE]["mean"],
        ),
        "u_mean_motion_spearman": (
            current_report["uncertainty_motion_correlations"][PRIMARY_SCORE]["spearman_vs_observed_motion_magnitude"]["rho"],
            old["uncertainty_motion_correlations"][PRIMARY_SCORE]["spearman_vs_observed_motion_magnitude"]["rho"],
        ),
    }
    comparison = {
        name: {
            "current": float(now),
            "previous_audit": float(before),
            "absolute_difference": float(abs(now - before)),
        }
        for name, (now, before) in fields.items()
    }
    return {
        "matches_previous_audit_within_1e-5": all(item["absolute_difference"] <= 1e-5 for item in comparison.values()),
        "metrics": comparison,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_paths = [
        args.checkpoint_root / f"trajectory_transformer_zero_scene_15x15_seed{seed}.pt"
        for seed in SEEDS
    ]
    checkpoint_hashes = {
        str(seed): sha256_file(path) for seed, path in zip(SEEDS, checkpoint_paths)
    }
    val_path = args.data_root / "val.npz"
    sample = SequenceWithImageSize(val_path)[0]
    models = []
    checkpoint_meta: dict[str, Any] = {}
    for seed, path in zip(SEEDS, checkpoint_paths):
        model, payload = load_model(path, seed, sample["scene_feat"].numel(), device)
        models.append(model)
        checkpoint_meta[str(seed)] = {
            "path": str(path),
            "sha256_before": checkpoint_hashes[str(seed)],
            "scene_mode": payload.get("scene_mode", payload.get("args", {}).get("scene_mode")),
            "seed": seed,
            "model_eval": not model.training,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        }

    evaluated: dict[str, dict[str, Any]] = {}
    metadata: dict[str, dict[str, np.ndarray]] = {}
    for split, filename in (("validation", "val.npz"), ("test", "test.npz")):
        data_path = args.data_root / filename
        report, diagnostics = evaluate_split(split, data_path, models, device, args.batch_size)
        n = report["sample_count"]
        with np.load(data_path, allow_pickle=False) as raw:
            meta = {
                "scene_id": raw["scene_id"].astype(str),
                "target_id": raw["target_id"].astype(str),
                "image_size": raw["image_size"].astype(np.int64),
            }
        if any(len(value) != n for value in meta.values()):
            raise RuntimeError(f"{split}: sample metadata is not aligned with shared inference order")
        diagnostic_arrays = diagnostics["sample_arrays"]
        normalized_arrays = {
            "u_mean_normalized": diagnostics["scores_normalized"]["u_mean"],
            **diagnostics["sample_errors_normalized"],
        }
        if any(not np.isfinite(value).all() for value in normalized_arrays.values()):
            raise RuntimeError(f"{split}: normalized-coordinate diagnostic has non-finite values")
        evaluated[split] = {
            "report": report,
            "arrays": diagnostic_arrays,
            "normalized": normalized_arrays,
            "sample_order_sha256": input_order_sha256(data_path),
            "reproduction": _previous_audit_reproduction(split, diagnostics),
        }
        metadata[split] = meta

    if not all(evaluated[split]["reproduction"]["matches_previous_audit_within_1e-5"] for split in evaluated):
        raise RuntimeError("Reused frozen-checkpoint inference does not reproduce the previous reliability audit")

    val_arrays = evaluated["validation"]["arrays"]
    test_arrays = evaluated["test"]["arrays"]
    val_motion = val_arrays["observed_motion_magnitude"]
    test_motion = test_arrays["observed_motion_magnitude"]
    val_raw_u = val_arrays[PRIMARY_SCORE]
    test_raw_u = test_arrays[PRIMARY_SCORE]
    val_ade, val_fde = val_arrays["ade_pixel"], val_arrays["fde_pixel"]
    test_ade, test_fde = test_arrays["ade_pixel"], test_arrays["fde_pixel"]

    prior_thresholds = json.loads(
        (PROJECT_ROOT / "results/trajectory_reliability_audit/high_error_thresholds.json").read_text(encoding="utf-8")
    )
    thresholds = {
        "high_ade_pixel_threshold": float(prior_thresholds["high_ade_pixel_threshold"]),
        "high_fde_pixel_threshold": float(prior_thresholds["high_fde_pixel_threshold"]),
        "source": "previous validation-only 80th percentile thresholds; reused unchanged",
        "validation_quantile": 0.8,
        "test_percentile_used_for_primary_labels": False,
    }
    high_labels = {
        "validation_ade": val_ade >= thresholds["high_ade_pixel_threshold"],
        "validation_fde": val_fde >= thresholds["high_fde_pixel_threshold"],
        "test_ade": test_ade >= thresholds["high_ade_pixel_threshold"],
        "test_fde": test_fde >= thresholds["high_fde_pixel_threshold"],
    }

    coefficients = fit_motion_adjustment(val_motion, val_raw_u)
    val_adjusted = apply_motion_adjustment(val_motion, val_raw_u, coefficients)
    test_adjusted = apply_motion_adjustment(test_motion, test_raw_u, coefficients)
    scores_by_split = {
        "validation": {
            "motion_only": val_motion,
            "raw_u_mean": val_raw_u,
            "adjusted_u_mean": val_adjusted,
        },
        "test": {
            "motion_only": test_motion,
            "raw_u_mean": test_raw_u,
            "adjusted_u_mean": test_adjusted,
        },
    }
    errors_by_split = {
        "validation": (val_ade, val_fde),
        "test": (test_ade, test_fde),
    }
    split_metrics = {}
    for split in ("validation", "test"):
        ade, fde = errors_by_split[split]
        split_metrics[split] = _metric_block(
            scores_by_split[split],
            ade,
            fde,
            high_labels[f"{split}_ade"],
            high_labels[f"{split}_fde"],
        )

    partial = {}
    for split in ("validation", "test"):
        scores = scores_by_split[split]
        ade, fde = errors_by_split[split]
        partial[split] = {
            "u_mean_vs_ade_given_observed_motion": partial_spearman(
                scores["raw_u_mean"], ade, scores["motion_only"]
            ),
            "u_mean_vs_fde_given_observed_motion": partial_spearman(
                scores["raw_u_mean"], fde, scores["motion_only"]
            ),
            "method": "rank-transform each variable; regress ranked U and ranked error separately on intercept + ranked observed motion; Pearson correlation of residuals",
        }

    adjusted_bins = _reliability_bins(
        val_adjusted, val_ade, val_fde, test_adjusted, test_ade, test_fde
    )
    val_decile_cutpoints = np.quantile(val_motion, np.linspace(0, 1, 11))
    test_motion_deciles = assign_quantile_strata(test_motion, val_decile_cutpoints)
    motion_stratified = _motion_strata(
        val_motion,
        test_motion,
        {
            "raw_u_mean": test_raw_u,
            "adjusted_u_mean": test_adjusted,
            "motion_only": test_motion,
        },
        test_ade,
        test_fde,
    )

    global_risk = global_risk_curves(
        {"raw_u_mean": test_raw_u, "adjusted_u_mean": test_adjusted, "motion_only": test_motion},
        test_ade,
        test_fde,
    )
    stratified_curves = {
        "raw_u_mean_within_motion_deciles": stratified_risk_curve(
            test_raw_u, test_motion_deciles, test_ade, test_fde
        ),
        "adjusted_u_mean_within_motion_deciles": stratified_risk_curve(
            test_adjusted, test_motion_deciles, test_ade, test_fde
        ),
    }
    permutation = within_motion_permutation_test(
        test_raw_u, test_adjusted, test_motion_deciles, test_ade, test_fde
    )
    risk_comparison_rows = []
    curve_names = ("raw_u_mean", "adjusted_u_mean", "motion_only", "random_global", "oracle_by_true_ade")
    for index, coverage in enumerate(RISK_COVERAGES):
        row: dict[str, Any] = {"nominal_coverage": coverage}
        for name in curve_names:
            row[name] = global_risk["curves"][name][index]
        for name, curve in stratified_curves.items():
            row[name] = curve[index]
        row["within_motion_permutation_reference"] = permutation["rows"][index]
        risk_comparison_rows.append(row)

    cluster_ids = metadata["test"]["scene_id"]
    video_bootstrap = cluster_bootstrap_metrics(
        cluster_ids,
        {"raw_u_mean": test_raw_u, "adjusted_u_mean": test_adjusted, "motion_only": test_motion},
        test_ade,
        high_labels["test_ade"],
        repetitions=args.bootstrap_reps,
        seed=BOOTSTRAP_SEED,
    )
    track_ids = np.char.add(
        np.char.add(metadata["test"]["scene_id"], "::"), metadata["test"]["target_id"]
    )
    track_bootstrap = cluster_bootstrap_metrics(
        track_ids,
        {"raw_u_mean": test_raw_u, "adjusted_u_mean": test_adjusted},
        test_ade,
        high_labels["test_ade"],
        repetitions=args.bootstrap_reps,
        seed=BOOTSTRAP_SEED,
    )
    track_bootstrap["cluster_level"] = "(scene_id, target_id) pedestrian track"

    scale = {
        split: _scale_diagnostics(
            metadata[split]["image_size"],
            evaluated[split]["arrays"][PRIMARY_SCORE],
            evaluated[split]["arrays"]["ade_pixel"],
        )
        for split in ("validation", "test")
    }
    normalized_sanity = {}
    for split in ("validation", "test"):
        norm = evaluated[split]["normalized"]
        normalized_sanity[split] = {
            "spearman_u_mean_normalized_vs_ade_normalized": _correlation(
                norm["u_mean_normalized"], norm["ade_normalized"]
            ),
            "spearman_u_mean_normalized_vs_fde_normalized": _correlation(
                norm["u_mean_normalized"], norm["fde_normalized"]
            ),
            "mean_u_mean_normalized": float(norm["u_mean_normalized"].mean()),
            "mean_ade_normalized": float(norm["ade_normalized"].mean()),
            "mean_fde_normalized": float(norm["fde_normalized"].mean()),
        }

    for seed, path in zip(SEEDS, checkpoint_paths):
        after = sha256_file(path)
        checkpoint_meta[str(seed)]["sha256_after"] = after
        checkpoint_meta[str(seed)]["unchanged"] = after == checkpoint_hashes[str(seed)]
        if after != checkpoint_hashes[str(seed)]:
            raise RuntimeError(f"Checkpoint changed during read-only audit: {path}")

    test_decile_counts = [int((test_motion_deciles == i).sum()) for i in range(10)]
    data_alignment = {
        split: {
            "sample_count": evaluated[split]["report"]["sample_count"],
            "shared_dataloader_all_three_models": True,
            "shuffle": False,
            "scene_input_all_zeros": True,
            "sample_order_sha256": evaluated[split]["sample_order_sha256"],
            "historical_audit_reproduction": evaluated[split]["reproduction"],
        }
        for split in ("validation", "test")
    }
    decision_inputs = {
        "partial_spearman_ade_test": partial["test"]["u_mean_vs_ade_given_observed_motion"],
        "adjusted_rho_ade_test": split_metrics["test"]["adjusted_u_mean"]["spearman_vs_ade"]["rho"],
        "adjusted_high_ade_auroc_test": split_metrics["test"]["adjusted_u_mean"]["high_ade"]["auroc"],
        "motion_high_ade_auroc_test": split_metrics["test"]["motion_only"]["high_ade"]["auroc"],
        "adjusted_video_cluster_rho_ade_ci_lower": video_bootstrap["metrics"]["adjusted_u_mean"]["spearman_vs_ade"]["lower_95"],
        "adjusted_stratified_ade_reduction_100_to_20": (
            stratified_curves["adjusted_u_mean_within_motion_deciles"][0]["ade"]
            - stratified_curves["adjusted_u_mean_within_motion_deciles"][-1]["ade"]
        ),
        "adjusted_stratified_fde_reduction_100_to_20": (
            stratified_curves["adjusted_u_mean_within_motion_deciles"][0]["fde"]
            - stratified_curves["adjusted_u_mean_within_motion_deciles"][-1]["fde"]
        ),
        "adjusted_within_motion_permutation_ade_p_at_80": permutation["rows"][2]["adjusted_u_mean_empirical_lower_tail_p"]["ade"],
    }
    partial_test = decision_inputs["partial_spearman_ade_test"]
    adjusted_rho = decision_inputs["adjusted_rho_ade_test"]
    adjusted_auc = decision_inputs["adjusted_high_ade_auroc_test"]
    motion_auc = decision_inputs["motion_high_ade_auroc_test"]
    cluster_lower = decision_inputs["adjusted_video_cluster_rho_ade_ci_lower"]
    adjusted_perm_p = decision_inputs["adjusted_within_motion_permutation_ade_p_at_80"]
    positive_risk = (
        decision_inputs["adjusted_stratified_ade_reduction_100_to_20"] > 0
        and decision_inputs["adjusted_stratified_fde_reduction_100_to_20"] > 0
    )
    go_conditions = {
        "partial_spearman_ade_at_least_0_20": partial_test is not None and partial_test >= 0.20,
        "adjusted_spearman_ade_at_least_0_20": adjusted_rho is not None and adjusted_rho >= 0.20,
        "video_cluster_ci_lower_above_0_10": cluster_lower is not None and cluster_lower > 0.10,
        "adjusted_high_ade_auroc_at_least_0_65_or_plus_0_03_vs_motion": (
            adjusted_auc is not None and motion_auc is not None
            and (adjusted_auc >= 0.65 or adjusted_auc >= motion_auc + 0.03)
        ),
        "adjusted_stratified_ade_and_fde_improve_100_to_20": positive_risk,
        "adjusted_stratified_ade_beats_within_motion_random_at_80_p_le_0_05": (
            adjusted_perm_p is not None and adjusted_perm_p <= 0.05
        ),
    }
    stop_conditions = {
        "partial_spearman_ade_below_0_10": partial_test is not None and partial_test < 0.10,
        "adjusted_high_ade_auroc_within_0_01_of_motion": (
            adjusted_auc is not None and motion_auc is not None and adjusted_auc <= motion_auc + 0.01
        ),
        "no_adjusted_stratified_ade_or_fde_improvement": not positive_risk,
        "no_within_motion_permutation_evidence_p_above_0_10": (
            adjusted_perm_p is not None and adjusted_perm_p > 0.10
        ),
    }
    go_conditions_met = int(sum(go_conditions.values()))
    if go_conditions_met >= 4:
        verdict = "GO"
    elif all(stop_conditions.values()):
        verdict = "STOP"
    else:
        verdict = "WEAK"
    decision = {
        "decision": verdict,
        "inputs": decision_inputs,
        "go_conditions": go_conditions,
        "go_conditions_met": go_conditions_met,
        "go_conditions_total": len(go_conditions),
        "stop_conditions": stop_conditions,
        "rule": "GO requires a majority of the six evidence checks (at least four); STOP requires all four null-pattern checks; otherwise WEAK. This operationalizes the task's instruction to satisfy most GO criteria without tuning a model on test.",
        "next_stage_authorization": "No reliability-gated intention model was started. Only GO would support considering it in a later explicitly authorized task.",
    }

    for split, item in evaluated.items():
        item["report"]["new_deconfounding_metrics"] = split_metrics[split]
    outputs = {
        "protocol.json": {
            "scientific_question": "Does zero-scene ensemble u_mean retain forecast-error ranking signal after controlling observed motion magnitude?",
            "primary_score": PRIMARY_SCORE,
            "u_mean_definition": "per future step, mean of the three models' 2D pixel distances from the ensemble mean; averaged across 15 future steps",
            "motion_definition": "pixel displacement between first and last observed target centers across 15 observation frames; observation only",
            "no_training": True,
            "no_intention_or_scene_input": True,
            "no_future_gt_in_deployable_score_or_adjustment": True,
            "high_error_thresholds": thresholds,
            "motion_decile_test_counts": test_decile_counts,
            "data_alignment": data_alignment,
        },
        "partial_correlation.json": {
            "method": "rank-transform and linear residualization on ranked observed motion, then Pearson correlation",
            "split_results": partial,
        },
        "motion_baselines.json": {
            "high_error_thresholds": thresholds,
            "split_metrics": split_metrics,
            "primary_test_comparison": {
                name: split_metrics["test"][name]
                for name in ("motion_only", "raw_u_mean", "adjusted_u_mean")
            },
            "motion_only_ranking_direction": "ascending observed_motion_magnitude; low motion is retained as lower risk",
        },
        "adjustment_fit.json": coefficients,
        "adjusted_score.json": {
            "score_name": "adjusted_u_mean",
            "formula": "log1p(u_mean) - (b0 + b1*log1p(observed_motion) + b2*log1p(observed_motion)^2)",
            "fit": coefficients,
            "fit_uses_ADE_or_FDE": False,
            "frozen_coefficients_applied_to_test_without_refitting": True,
            "raw_and_adjusted_motion_correlation": {
                split: {
                    "raw_u_mean_vs_motion": _correlation(
                        scores_by_split[split]["raw_u_mean"], scores_by_split[split]["motion_only"]
                    ),
                    "adjusted_u_mean_vs_motion": _correlation(
                        scores_by_split[split]["adjusted_u_mean"], scores_by_split[split]["motion_only"]
                    ),
                }
                for split in ("validation", "test")
            },
            "split_metrics": split_metrics,
            "reliability_bins": adjusted_bins,
            "gt_future_displacement_diagnostic_only": {
                split: {
                    "raw_u_mean": _correlation(
                        scores_by_split[split]["raw_u_mean"], evaluated[split]["arrays"]["gt_future_displacement_magnitude"]
                    ),
                    "adjusted_u_mean": _correlation(
                        scores_by_split[split]["adjusted_u_mean"], evaluated[split]["arrays"]["gt_future_displacement_magnitude"]
                    ),
                    "note": "future_gt used only after score construction for post-hoc diagnosis",
                }
                for split in ("validation", "test")
            },
        },
        "risk_coverage_comparison.json": {
            "global": global_risk,
            "motion_stratified": {
                "strata_definition": "validation motion deciles, cutpoints frozen for test",
                "ranking_method": "within each motion decile independently retain the lowest score fraction, then pool retained samples",
                "curves": stratified_curves,
                "within_motion_permutation": permutation,
            },
            "comparison_rows": risk_comparison_rows,
        },
        "motion_stratified_metrics.json": motion_stratified,
        "cluster_bootstrap.json": {
            "primary_video_cluster_bootstrap": video_bootstrap,
            "secondary_track_cluster_bootstrap": track_bootstrap,
            "high_ade_threshold_pixel": thresholds["high_ade_pixel_threshold"],
        },
        "scale_diagnostics.json": {
            "pixel_scale_by_image_dimensions": scale,
            "normalized_coordinate_sanity_check": normalized_sanity,
            "normalized_u_definition": "mean across 15 steps of the three normalized-coordinate predictions' mean 2D Euclidean distance from their normalized ensemble mean",
        },
        "checkpoint_audit.json": checkpoint_meta,
        "decision.json": decision,
    }
    for name, value in outputs.items():
        write_json(args.output_root / name, value)
    for split, item in evaluated.items():
        write_json(args.output_root / f"{split}_inference_audit.json", item["report"])
    return {
        "decision": verdict,
        "partial_spearman_ade_test": partial_test,
        "adjusted_spearman_ade_test": adjusted_rho,
        "test_metrics": split_metrics["test"],
        "adjustment_coefficients": {key: coefficients[key] for key in ("b0", "b1", "b2")},
        "output_root": str(args.output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/trajectory_reliability_deconfounding")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPETITIONS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    print(json.dumps(run_audit(args), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
