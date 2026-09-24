#!/usr/bin/env python3
"""Feasibility audit: zero-scene ensemble disagreement vs trajectory error."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.trajectory_transformer import SceneTrajectoryTransformer


SEEDS = (42, 123, 2024)
SCORE_NAMES = ("u_mean", "u_endpoint", "u_pairwise", "u_endpoint_pairwise")
SCORE_LABELS = {
    "u_mean": "Mean trajectory spread",
    "u_endpoint": "Endpoint spread",
    "u_pairwise": "Mean pairwise disagreement",
    "u_endpoint_pairwise": "Endpoint pairwise disagreement",
}
RISK_COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)


class SequenceWithImageSize(Dataset):
    """Attach pixel scale to each sample so order and scale cannot drift apart."""

    def __init__(self, path: Path) -> None:
        self.dataset = JAADSequenceDataset(path)
        with np.load(path, allow_pickle=False) as raw:
            self.image_size = torch.from_numpy(raw["image_size"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        item["image_size"] = self.image_size[index]
        return item


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_order_sha256(path: Path) -> str:
    """Fingerprint identity and all per-sample inputs used by the diagnostic."""
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as raw:
        for key in ("scene_id", "target_id", "obs_end_frame", "target_obs", "target_abs_obs", "future_gt", "image_size"):
            array = np.ascontiguousarray(raw[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def load_model(checkpoint_path: Path, expected_seed: int, scene_dim: int, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    if payload.get("scene_mode", args.get("scene_mode")) != "zero":
        raise ValueError(f"Expected a zero-scene checkpoint, got {checkpoint_path}")
    if int(args.get("seed", -1)) != expected_seed:
        raise ValueError(f"Checkpoint seed mismatch for {checkpoint_path}")
    model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=scene_dim,
        d_model=int(args.get("d_model", 128)),
        nhead=4,
        num_layers=int(args.get("num_layers", 3)),
        pred_len=15,
        dropout=0.1,
        max_obs_len=15,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


def _safe_float(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": None, "p_value": None}
    result = spearmanr(x, y)
    return {"rho": _safe_float(result.statistic), "p_value": _safe_float(result.pvalue)}


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": None, "p_value": None}
    result = pearsonr(x, y)
    return {"r": _safe_float(result.statistic), "p_value": _safe_float(result.pvalue)}


def _mean_std(values: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(x)),
        "sample_std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
        "n": int(len(x)),
    }


def prediction_diagnostics(
    predictions_normalized: np.ndarray,
    future_normalized: np.ndarray,
    image_size: np.ndarray,
) -> dict[str, Any]:
    """Compute pixel errors and four permutation-invariant ensemble spreads.

    Args:
        predictions_normalized: [M,N,T,2] outputs from the same ordered samples.
        future_normalized: [N,T,2] ground truth in the same normalized coordinates.
        image_size: [N,2] (width,height) for each corresponding sample.
    """
    pred = np.asarray(predictions_normalized, dtype=np.float64)
    gt = np.asarray(future_normalized, dtype=np.float64)
    size = np.asarray(image_size, dtype=np.float64)
    if pred.ndim != 4 or pred.shape[0] != 3 or pred.shape[-1] != 2:
        raise ValueError(f"predictions must be [3,N,T,2], got {pred.shape}")
    if gt.shape != pred.shape[1:] or size.shape != (pred.shape[1], 2):
        raise ValueError(f"Incompatible gt/image_size shapes: pred={pred.shape}, gt={gt.shape}, size={size.shape}")

    scale = size[None, :, None, :]
    pred_px = pred * scale
    gt_px = gt * size[:, None, :]
    mean_pred_px = pred_px.mean(axis=0)
    point_error = np.linalg.norm(mean_pred_px - gt_px, axis=-1)
    ade = point_error.mean(axis=1)
    fde = point_error[:, -1]
    ensemble_norm = pred.mean(axis=0)
    point_error_norm = np.linalg.norm(ensemble_norm - gt, axis=-1)
    ade_norm = point_error_norm.mean(axis=1)
    fde_norm = point_error_norm[:, -1]

    centered = pred_px - mean_pred_px[None, ...]
    spread_by_horizon = np.linalg.norm(centered, axis=-1).mean(axis=0)
    u_mean = spread_by_horizon.mean(axis=1)
    u_endpoint = spread_by_horizon[:, -1]
    centered_norm = pred - ensemble_norm[None, ...]
    normalized_spread_by_horizon = np.linalg.norm(centered_norm, axis=-1).mean(axis=0)
    u_mean_normalized = normalized_spread_by_horizon.mean(axis=1)

    pairs = ((0, 1), (0, 2), (1, 2))
    pairwise_by_horizon = np.stack(
        [np.linalg.norm(pred_px[a] - pred_px[b], axis=-1) for a, b in pairs], axis=0
    ).mean(axis=0)
    u_pairwise = pairwise_by_horizon.mean(axis=1)
    u_endpoint_pairwise = pairwise_by_horizon[:, -1]

    individual = {}
    for model_index, seed in enumerate(SEEDS):
        errors = np.linalg.norm(pred_px[model_index] - gt_px, axis=-1)
        errors_norm = np.linalg.norm(pred[model_index] - gt, axis=-1)
        individual[str(seed)] = {
            "ade_pixel": float(errors.mean()),
            "fde_pixel": float(errors[:, -1].mean()),
            "ade_normalized": float(errors_norm.mean()),
            "fde_normalized": float(errors_norm[:, -1].mean()),
        }

    return {
        "individual_model_performance": individual,
        "ensemble_mean_performance": {
            "ade_pixel": float(ade.mean()),
            "fde_pixel": float(fde.mean()),
            "ade_normalized": float(ade_norm.mean()),
            "fde_normalized": float(fde_norm.mean()),
        },
        "sample_errors": {"ade_pixel": ade, "fde_pixel": fde},
        "sample_errors_normalized": {"ade_normalized": ade_norm, "fde_normalized": fde_norm},
        "scores": {
            "u_mean": u_mean,
            "u_endpoint": u_endpoint,
            "u_pairwise": u_pairwise,
            "u_endpoint_pairwise": u_endpoint_pairwise,
        },
        "scores_normalized": {"u_mean": u_mean_normalized},
        "spread_by_horizon": spread_by_horizon,
        "point_error_by_horizon": point_error,
        "ensemble_prediction_pixel": mean_pred_px,
        "ground_truth_pixel": gt_px,
    }


def evaluate_split(
    split: str,
    data_path: Path,
    models: list[torch.nn.Module],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = SequenceWithImageSize(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_batches: list[list[np.ndarray]] = [[], [], []]
    gt_batches, size_batches, abs_obs_batches = [], [], []
    processed = 0
    with torch.no_grad():
        for batch in loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            scene = torch.zeros_like(batch["scene_feat"].to(device))
            gt_batches.append(batch["future_gt"].numpy())
            size_batches.append(batch["image_size"].numpy())
            abs_obs_batches.append(batch["target_abs_obs"].numpy())
            batch_predictions = []
            for model in models:
                prediction = model(target, scene)
                batch_predictions.append(prediction.detach().cpu().numpy())
            for index, prediction in enumerate(batch_predictions):
                prediction_batches[index].append(prediction)
            processed += len(batch["future_gt"])

    predictions = np.stack(
        [np.concatenate(batches, axis=0) for batches in prediction_batches], axis=0
    )
    future_gt = np.concatenate(gt_batches, axis=0)
    image_size = np.concatenate(size_batches, axis=0)
    target_abs_obs = np.concatenate(abs_obs_batches, axis=0)
    if processed != len(dataset) or predictions.shape[1] != len(dataset):
        raise RuntimeError(f"{split}: aligned inference count mismatch")
    if predictions.shape[2:] != future_gt.shape[1:] or image_size.shape != (len(dataset), 2):
        raise RuntimeError(f"{split}: prediction/future/image_size shape mismatch")

    diagnostics = prediction_diagnostics(predictions, future_gt, image_size)
    center_pixels = target_abs_obs[:, :, :2] * image_size[:, None, :]
    observed_motion = np.linalg.norm(center_pixels[:, -1] - center_pixels[:, 0], axis=-1)
    gt_future_displacement = np.linalg.norm(future_gt[:, -1] * image_size, axis=-1)
    sample_arrays = {
        **diagnostics["sample_errors"],
        **diagnostics["scores"],
        "observed_motion_magnitude": observed_motion,
        "gt_future_displacement_magnitude": gt_future_displacement,
        "spread_by_horizon": diagnostics["spread_by_horizon"],
        "point_error_by_horizon": diagnostics["point_error_by_horizon"],
    }
    for name, values in sample_arrays.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f"{split}: non-finite values in {name}")

    order_hash = input_order_sha256(data_path)
    report = {
        "split": split,
        "sample_count": len(dataset),
        "coordinate_space": "pixel; normalized (x,y) multiplied by per-sample (width,height)",
        "scene_input": "torch.zeros_like(scene_feat) for all three checkpoints",
        "dataloader": {"shuffle": False, "batch_size": batch_size, "num_workers": 0},
        "sample_order_sha256": order_hash,
        "alignment": {
            "one_shared_dataloader_for_all_models": True,
            "same_batch_target_future_and_image_size": True,
            "identifiers_and_input_arrays_sha256": order_hash,
            "number_of_predictions_per_model": processed,
        },
        "individual_model_performance": diagnostics["individual_model_performance"],
        "ensemble_mean_performance": diagnostics["ensemble_mean_performance"],
        "score_mean_sample_std": {name: _mean_std(values) for name, values in diagnostics["scores"].items()},
        "uncertainty_error_correlations": {},
        "uncertainty_motion_correlations": {},
    }
    for name, score in diagnostics["scores"].items():
        report["uncertainty_error_correlations"][name] = {
            "spearman_vs_ensemble_ade": _correlation(score, diagnostics["sample_errors"]["ade_pixel"]),
            "spearman_vs_ensemble_fde": _correlation(score, diagnostics["sample_errors"]["fde_pixel"]),
            "pearson_vs_ensemble_ade": pearson_correlation(score, diagnostics["sample_errors"]["ade_pixel"]),
            "pearson_vs_ensemble_fde": pearson_correlation(score, diagnostics["sample_errors"]["fde_pixel"]),
        }
        report["uncertainty_motion_correlations"][name] = {
            "spearman_vs_observed_motion_magnitude": _correlation(score, observed_motion),
            "spearman_vs_gt_future_displacement_magnitude_diagnostic_only": _correlation(
                score, gt_future_displacement
            ),
        }
    diagnostics["report"] = report
    diagnostics["sample_arrays"] = sample_arrays
    return report, diagnostics


def binary_detection_metrics(score: np.ndarray, positive: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(positive, dtype=np.int64)
    if np.unique(y).size < 2:
        return {"auroc": None, "auprc": None, "positive_count": int(y.sum()), "sample_count": int(len(y))}
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "positive_count": int(y.sum()),
        "sample_count": int(len(y)),
    }


def add_high_error_diagnostics(
    validation_report: dict[str, Any],
    test_report: dict[str, Any],
    validation: dict[str, Any],
    test: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = {
        "high_ade_pixel_threshold": float(np.quantile(validation["sample_errors"]["ade_pixel"], 0.8)),
        "high_fde_pixel_threshold": float(np.quantile(validation["sample_errors"]["fde_pixel"], 0.8)),
        "quantile": 0.8,
        "method": "linear validation percentile; fixed pixel threshold for test primary metrics",
    }
    result_by_split = {}
    for split, report, arrays, primary_thresholds in (
        ("validation", validation_report, validation["sample_arrays"], True),
        ("test", test_report, test["sample_arrays"], False),
    ):
        metrics = {}
        for score_name, score in arrays.items():
            if score_name not in SCORE_NAMES:
                continue
            ade_threshold = thresholds["high_ade_pixel_threshold"]
            fde_threshold = thresholds["high_fde_pixel_threshold"]
            if primary_thresholds:
                ade_label = arrays["ade_pixel"] >= ade_threshold
                fde_label = arrays["fde_pixel"] >= fde_threshold
            else:
                ade_label = arrays["ade_pixel"] >= ade_threshold
                fde_label = arrays["fde_pixel"] >= fde_threshold
            item = {
                "high_ade_fixed_validation_threshold": binary_detection_metrics(score, ade_label),
                "high_fde_fixed_validation_threshold": binary_detection_metrics(score, fde_label),
            }
            if split == "test":
                item["secondary_test_top20_percent_ade"] = binary_detection_metrics(
                    score, arrays["ade_pixel"] >= np.quantile(arrays["ade_pixel"], 0.8)
                )
                item["secondary_test_top20_percent_fde"] = binary_detection_metrics(
                    score, arrays["fde_pixel"] >= np.quantile(arrays["fde_pixel"], 0.8)
                )
            metrics[score_name] = item
        report["high_error_detection"] = metrics
        result_by_split[split] = metrics
    return thresholds, result_by_split


def select_primary_score(validation_report: dict[str, Any], tolerance: float = 0.01) -> dict[str, Any]:
    correlations = validation_report["uncertainty_error_correlations"]
    high_metrics = validation_report["high_error_detection"]
    ade_rho = {name: correlations[name]["spearman_vs_ensemble_ade"]["rho"] for name in SCORE_NAMES}
    if any(value is None for value in ade_rho.values()):
        raise RuntimeError("Cannot select a primary score with undefined validation correlations")
    maximum_rho = max(float(value) for value in ade_rho.values())
    corr_contenders = [name for name in SCORE_NAMES if float(ade_rho[name]) >= maximum_rho - tolerance]
    ade_auc = {
        name: high_metrics[name]["high_ade_fixed_validation_threshold"]["auroc"]
        for name in SCORE_NAMES
    }
    max_auc = max(float(ade_auc[name]) for name in corr_contenders)
    auc_contenders = [
        name for name in corr_contenders if float(ade_auc[name]) >= max_auc - tolerance
    ]
    selected = "u_mean" if "u_mean" in auc_contenders else max(
        auc_contenders, key=lambda name: (float(ade_rho[name]), float(ade_auc[name]))
    )
    return {
        "selected_score": selected,
        "label": SCORE_LABELS[selected],
        "split_used": "validation only",
        "selection_rule": "maximize Spearman(U,ADE); scores within 0.01 rho tie; among them prefer higher validation High-ADE AUROC (within 0.01 tie), then prefer simpler U_mean",
        "rho_tie_tolerance": tolerance,
        "validation_spearman_ade": ade_rho,
        "validation_high_ade_auroc": ade_auc,
        "correlation_tied_candidates": corr_contenders,
        "high_ade_tied_candidates": auc_contenders,
    }


def _bin_summary(name: str, mask: np.ndarray, uncertainty: np.ndarray, ade: np.ndarray, fde: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"bin": name, "sample_count": 0, "mean_uncertainty": None, "mean_ade": None, "median_ade": None, "mean_fde": None, "median_fde": None}
    return {
        "bin": name,
        "sample_count": int(mask.sum()),
        "mean_uncertainty": float(uncertainty[mask].mean()),
        "mean_ade": float(ade[mask].mean()),
        "median_ade": float(np.median(ade[mask])),
        "mean_fde": float(fde[mask].mean()),
        "median_fde": float(np.median(fde[mask])),
    }


def build_reliability_bins(
    validation: dict[str, Any], test: dict[str, Any], selected_score: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    val_u = validation["sample_arrays"][selected_score]
    test_u = test["sample_arrays"][selected_score]
    val_ade, val_fde = validation["sample_arrays"]["ade_pixel"], validation["sample_arrays"]["fde_pixel"]
    test_ade, test_fde = test["sample_arrays"]["ade_pixel"], test["sample_arrays"]["fde_pixel"]
    q33, q67 = np.quantile(val_u, [1 / 3, 2 / 3])
    val_bins = [
        _bin_summary("low", val_u <= q33, val_u, val_ade, val_fde),
        _bin_summary("medium", (val_u > q33) & (val_u <= q67), val_u, val_ade, val_fde),
        _bin_summary("high", val_u > q67, val_u, val_ade, val_fde),
    ]
    test_bins = [
        _bin_summary("low", test_u <= q33, test_u, test_ade, test_fde),
        _bin_summary("medium", (test_u > q33) & (test_u <= q67), test_u, test_ade, test_fde),
        _bin_summary("high", test_u > q67, test_u, test_ade, test_fde),
    ]
    quantiles = np.quantile(val_u, np.linspace(0, 1, 11))
    interior = quantiles[1:-1]
    decile_rows = []
    for split, uncertainty, ade, fde in (
        ("validation", val_u, val_ade, val_fde),
        ("test", test_u, test_ade, test_fde),
    ):
        bin_id = np.searchsorted(interior, uncertainty, side="right")
        for index in range(10):
            mask = bin_id == index
            stats = _bin_summary(f"D{index + 1}", mask, uncertainty, ade, fde)
            decile_rows.append({
                "split": split,
                "bin": stats["bin"],
                "validation_uncertainty_lower": float(quantiles[index]),
                "validation_uncertainty_upper": float(quantiles[index + 1]),
                **{key: stats[key] for key in ("sample_count", "mean_uncertainty", "mean_ade", "median_ade", "mean_fde", "median_fde")},
            })
    result = {
        "selected_score": selected_score,
        "tertile_cutpoints_from_validation": {"q33": float(q33), "q67": float(q67)},
        "validation_bins": val_bins,
        "test_bins_using_fixed_validation_cutpoints": test_bins,
        "test_error_monotonic_low_to_high": {
            "ade": bool(test_bins[0]["mean_ade"] < test_bins[1]["mean_ade"] < test_bins[2]["mean_ade"]),
            "fde": bool(test_bins[0]["mean_fde"] < test_bins[1]["mean_fde"] < test_bins[2]["mean_fde"]),
        },
        "decile_cutpoints_from_validation": [float(value) for value in quantiles],
        "decile_boundary_note": "Duplicate quantile cutpoints are retained; searchsorted assigns tied values consistently, so some bins can be empty.",
    }
    return result, decile_rows


def risk_curve(uncertainty: np.ndarray, ade: np.ndarray, fde: np.ndarray) -> dict[str, Any]:
    n = len(uncertainty)
    order = np.argsort(uncertainty, kind="mergesort")
    oracle_order = np.argsort(ade, kind="mergesort")
    random_rng = np.random.default_rng(9124)
    random_results = {coverage: {"ade": [], "fde": []} for coverage in RISK_COVERAGES}
    for _ in range(100):
        random_order = random_rng.permutation(n)
        for coverage in RISK_COVERAGES:
            count = max(1, int(np.floor(n * coverage)))
            chosen = random_order[:count]
            random_results[coverage]["ade"].append(float(ade[chosen].mean()))
            random_results[coverage]["fde"].append(float(fde[chosen].mean()))
    rows = []
    for coverage in RISK_COVERAGES:
        count = max(1, int(np.floor(n * coverage)))
        chosen = order[:count]
        oracle = oracle_order[:count]
        row = {
            "nominal_coverage": float(coverage),
            "retained_samples": count,
            "actual_coverage": count / n,
            "selected": {"ade": float(ade[chosen].mean()), "fde": float(fde[chosen].mean())},
            "random_100_mean_sample_std": {},
            "oracle_by_true_ade": {"ade": float(ade[oracle].mean()), "fde": float(fde[oracle].mean())},
        }
        for metric in ("ade", "fde"):
            values = np.asarray(random_results[coverage][metric], dtype=np.float64)
            row["random_100_mean_sample_std"][metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)),
            }
        rows.append(row)
    selected_ade = [row["selected"]["ade"] for row in rows]
    selected_fde = [row["selected"]["fde"] for row in rows]
    return {
        "selected_score_order": "ascending uncertainty; lowest uncertainty retained first",
        "coverage_levels": list(RISK_COVERAGES),
        "sample_count_rule": "floor(N*nominal_coverage), at least one sample",
        "random_reference": {"ranking": "uniform random permutation", "seed": 9124, "repetitions": 100},
        "oracle_reference": {"ranking": "sort by true ensemble ADE ascending", "upper_bound_only": True},
        "rows": rows,
        "selected_curve_monotonic_improvement": {
            "ade": bool(all(later <= earlier + 1e-12 for earlier, later in zip(selected_ade, selected_ade[1:]))),
            "fde": bool(all(later <= earlier + 1e-12 for earlier, later in zip(selected_fde, selected_fde[1:]))),
        },
        "selected_ade_reduction_100_to_20": float(selected_ade[0] - selected_ade[-1]),
        "selected_fde_reduction_100_to_20": float(selected_fde[0] - selected_fde[-1]),
    }


def bootstrap_test_ci(
    uncertainty: np.ndarray,
    ade: np.ndarray,
    high_ade: np.ndarray,
    repetitions: int,
    seed: int = 9124,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(uncertainty)
    rho_values, auc_values = [], []
    for _ in range(repetitions):
        indices = rng.integers(0, n, size=n)
        rho = spearmanr(uncertainty[indices], ade[indices]).statistic
        if np.isfinite(rho):
            rho_values.append(float(rho))
        labels = high_ade[indices]
        if np.unique(labels).size == 2:
            auc_values.append(float(roc_auc_score(labels, uncertainty[indices])))
    def bounds(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"lower_95": None, "upper_95": None, "valid_replicates": 0}
        lower, upper = np.quantile(values, [0.025, 0.975])
        return {"lower_95": float(lower), "upper_95": float(upper), "valid_replicates": len(values)}
    return {
        "method": "paired nonparametric bootstrap by sample with replacement; percentile 95% CI",
        "seed": seed,
        "requested_repetitions": repetitions,
        "spearman_selected_score_vs_ensemble_ade": bounds(rho_values),
        "high_ade_auroc_fixed_validation_threshold": bounds(auc_values),
    }


def horizon_analysis(validation: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for split, diagnostic in (("validation", validation), ("test", test)):
        arrays = diagnostic["sample_arrays"]
        errors = arrays["point_error_by_horizon"]
        disagreements = arrays["spread_by_horizon"]
        output[split] = [
            {
                "horizon": step + 1,
                "mean_ensemble_point_error_pixel": float(errors[:, step].mean()),
                "mean_disagreement_pixel": float(disagreements[:, step].mean()),
                "spearman_disagreement_vs_point_error": _correlation(disagreements[:, step], errors[:, step]),
            }
            for step in range(errors.shape[1])
        ]
    return output


def motion_strata(
    validation: dict[str, Any], test: dict[str, Any], selected_score: str
) -> dict[str, Any]:
    val_motion = validation["sample_arrays"]["observed_motion_magnitude"]
    test_arrays = test["sample_arrays"]
    q33, q67 = np.quantile(val_motion, [1 / 3, 2 / 3])
    rows = []
    for name, mask in (
        ("slow", test_arrays["observed_motion_magnitude"] <= q33),
        ("medium", (test_arrays["observed_motion_magnitude"] > q33) & (test_arrays["observed_motion_magnitude"] <= q67)),
        ("fast", test_arrays["observed_motion_magnitude"] > q67),
    ):
        u = test_arrays[selected_score][mask]
        error = test_arrays["ade_pixel"][mask]
        rows.append({
            "stratum": name,
            "sample_count": int(mask.sum()),
            "observed_motion_mean_pixel": float(test_arrays["observed_motion_magnitude"][mask].mean()) if mask.any() else None,
            "spearman_selected_uncertainty_vs_ade": _correlation(u, error),
        })
    return {
        "observed_motion_definition": "pixel displacement between first and last of 15 observed target centers, from target_abs_obs[...,0:2]",
        "slow_medium_fast_cutpoints_from_validation_pixel": {"q33": float(q33), "q67": float(q67)},
        "test_strata": rows,
        "selected_score_global_test_spearman_vs_observed_motion": test["report"]["uncertainty_motion_correlations"][selected_score]["spearman_vs_observed_motion_magnitude"],
        "selected_score_global_test_spearman_vs_gt_future_endpoint_displacement_diagnostic_only": test["report"]["uncertainty_motion_correlations"][selected_score]["spearman_vs_gt_future_displacement_magnitude_diagnostic_only"],
        "gt_future_displacement_note": "Uses future ground truth only for post-hoc diagnostic; never a model input or selection feature.",
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/trajectory_reliability_audit")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = [
        args.checkpoint_root / f"trajectory_transformer_zero_scene_15x15_seed{seed}.pt"
        for seed in SEEDS
    ]
    checkpoint_hashes = {str(seed): sha256_file(path) for seed, path in zip(SEEDS, checkpoint_paths)}
    sample = SequenceWithImageSize(args.data_root / "val.npz")[0]
    models, checkpoint_meta = [], {}
    for seed, path in zip(SEEDS, checkpoint_paths):
        model, payload = load_model(path, seed, sample["scene_feat"].numel(), device)
        if not hasattr(model, "scene_encoder"):
            raise RuntimeError("Expected scene_encoder to remain in the zero-input checkpoint architecture")
        models.append(model)
        checkpoint_meta[str(seed)] = {
            "path": str(path),
            "sha256_before": checkpoint_hashes[str(seed)],
            "scene_mode": payload.get("scene_mode", payload.get("args", {}).get("scene_mode")),
            "seed": int(payload["args"]["seed"]),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "model_eval": not model.training,
            "scene_encoder_retained": True,
        }

    reports: dict[str, dict[str, Any]] = {}
    full_diagnostics: dict[str, dict[str, Any]] = {}
    for split in ("validation", "test"):
        filename = "val.npz" if split == "validation" else "test.npz"
        report, diagnostic = evaluate_split(
            split, args.data_root / filename, models, device, args.batch_size
        )
        reports[split] = report
        full_diagnostics[split] = diagnostic

        historical_key = "val" if split == "validation" else "test"
        historical_reproduction = {}
        for seed in SEEDS:
            historical_path = PROJECT_ROOT / f"results/trajectory_transformer_zero_scene_15x15_seed{seed}/metrics.json"
            historical = json.loads(historical_path.read_text(encoding="utf-8"))[historical_key]
            current = report["individual_model_performance"][str(seed)]
            errors = {
                "ade_pixel": current["ade_pixel"] - historical["trajectory_ade_pixel"],
                "fde_pixel": current["fde_pixel"] - historical["trajectory_fde_pixel"],
            }
            historical_reproduction[str(seed)] = {
                "historical_ade_pixel": historical["trajectory_ade_pixel"],
                "current_ade_pixel": current["ade_pixel"],
                "ade_pixel_difference": errors["ade_pixel"],
                "historical_fde_pixel": historical["trajectory_fde_pixel"],
                "current_fde_pixel": current["fde_pixel"],
                "fde_pixel_difference": errors["fde_pixel"],
                "reproduced_within_1e_4_pixel": abs(errors["ade_pixel"]) <= 1e-4 and abs(errors["fde_pixel"]) <= 1e-4,
            }
        report["historical_reproduction"] = historical_reproduction
        report["all_individual_pixel_metrics_reproduced"] = all(
            item["reproduced_within_1e_4_pixel"] for item in historical_reproduction.values()
        )

    validation_report = reports["validation"]
    test_report = reports["test"]
    thresholds, _ = add_high_error_diagnostics(
        validation_report, test_report, full_diagnostics["validation"], full_diagnostics["test"]
    )
    selection = select_primary_score(validation_report)
    selected_score = selection["selected_score"]
    val_test_bins, decile_rows = build_reliability_bins(
        full_diagnostics["validation"], full_diagnostics["test"], selected_score
    )
    selected_test_arrays = full_diagnostics["test"]["sample_arrays"]
    fixed_high_ade = selected_test_arrays["ade_pixel"] >= thresholds["high_ade_pixel_threshold"]
    risk = risk_curve(
        selected_test_arrays[selected_score],
        selected_test_arrays["ade_pixel"],
        selected_test_arrays["fde_pixel"],
    )
    bootstrap = bootstrap_test_ci(
        selected_test_arrays[selected_score],
        selected_test_arrays["ade_pixel"],
        fixed_high_ade,
        args.bootstrap_reps,
    )
    horizon = horizon_analysis(full_diagnostics["validation"], full_diagnostics["test"])
    motion = motion_strata(full_diagnostics["validation"], full_diagnostics["test"], selected_score)
    bins_monotonic = val_test_bins["test_error_monotonic_low_to_high"]

    test_selected = test_report["uncertainty_error_correlations"][selected_score]
    high_ade_auc = test_report["high_error_detection"][selected_score]["high_ade_fixed_validation_threshold"]["auroc"]
    low_high = val_test_bins["test_bins_using_fixed_validation_cutpoints"]
    high_ade_ratio = (
        low_high[2]["mean_ade"] / low_high[0]["mean_ade"]
        if low_high[0]["mean_ade"] and low_high[2]["mean_ade"] is not None
        else None
    )
    risk_monotonic_ade = risk["selected_curve_monotonic_improvement"]["ade"]
    if (
        test_selected["spearman_vs_ensemble_ade"]["rho"] is not None
        and test_selected["spearman_vs_ensemble_ade"]["rho"] >= 0.30
        and high_ade_auc is not None
        and high_ade_auc >= 0.65
        and high_ade_ratio is not None
        and high_ade_ratio >= 1.10
        and risk_monotonic_ade
    ):
        decision = "Promising"
    elif (
        test_selected["spearman_vs_ensemble_ade"]["rho"] is not None
        and test_selected["spearman_vs_ensemble_ade"]["rho"] < 0.15
        and high_ade_auc is not None
        and abs(high_ade_auc - 0.5) < 0.05
        and risk["selected_ade_reduction_100_to_20"] <= 0
    ):
        decision = "Stop"
    else:
        decision = "Weak/inconclusive"

    for seed, path in zip(SEEDS, checkpoint_paths):
        after = sha256_file(path)
        checkpoint_meta[str(seed)]["sha256_after"] = after
        checkpoint_meta[str(seed)]["unchanged_during_audit"] = after == checkpoint_hashes[str(seed)]
        if after != checkpoint_hashes[str(seed)]:
            raise RuntimeError(f"Checkpoint changed during read-only audit: {path}")

    for split in ("validation", "test"):
        write_json(args.output_root / f"{split}_metrics.json", reports[split])
    write_json(args.output_root / "selection.json", selection)
    write_json(args.output_root / "reliability_bins.json", val_test_bins)
    write_json(args.output_root / "risk_coverage.json", risk)
    write_json(args.output_root / "horizon_analysis.json", horizon)
    write_json(args.output_root / "bootstrap.json", bootstrap)
    write_json(args.output_root / "checkpoint_audit.json", checkpoint_meta)
    write_json(args.output_root / "high_error_thresholds.json", thresholds)
    write_json(args.output_root / "motion_confounds.json", motion)
    write_json(args.output_root / "decision.json", {
        "decision": decision,
        "primary_score": selected_score,
        "test_spearman_ade": test_selected["spearman_vs_ensemble_ade"]["rho"],
        "test_high_ade_auroc": high_ade_auc,
        "test_high_to_low_ade_ratio": high_ade_ratio,
        "test_ade_risk_curve_monotonic": risk_monotonic_ade,
        "test_bins_monotonic_ade_fde": bins_monotonic,
        "criteria_note": "Promising requires rho>=0.30, fixed-validation High-ADE AUROC>=0.65, high-bin mean ADE >=1.10x low-bin ADE, and non-increasing ADE risk as coverage declines. Stop requires rho<0.15, AUROC within 0.05 of 0.5, and no ADE risk reduction; otherwise weak/inconclusive.",
    })
    csv_path = args.output_root / "reliability_bins.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        columns = [
            "split", "bin", "validation_uncertainty_lower", "validation_uncertainty_upper",
            "sample_count", "mean_uncertainty", "mean_ade", "median_ade", "mean_fde", "median_fde",
        ]
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(decile_rows)

    print(json.dumps({
        "decision": decision,
        "primary_score": selected_score,
        "validation_ensemble": validation_report["ensemble_mean_performance"],
        "test_ensemble": test_report["ensemble_mean_performance"],
        "test_primary_spearman_ade": test_selected["spearman_vs_ensemble_ade"]["rho"],
        "test_primary_high_ade_auroc": high_ade_auc,
        "output_root": str(args.output_root),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
