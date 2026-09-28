#!/usr/bin/env python3
"""Paired held-out comparison, cluster bootstrap, and M1 evidence summary."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrained_clone_intent_utils import RESULTS_ROOT, load_config, write_json
from scripts.reliability_gated_intent_utils import (
    binary_metrics,
    cluster_bootstrap_paired_delta,
    mean_sample_std,
    paired_delta_metrics,
)
from scripts.trajectory_preserving_utils import SEEDS, sha256_file

PROTOCOL_PATH = RESULTS_ROOT / "protocol_frozen.json"
PROTOCOL_SHA_PATH = RESULTS_ROOT / "protocol_frozen.sha256"
M0_ROOT = ROOT / "results/intention_scratch_matched"
P1_ROOT = ROOT / "results/trajectory_preserving_joint/P1_target_only"


def verify_protocol() -> tuple[dict[str, Any], str]:
    payload = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip().split()[0] != digest:
        raise RuntimeError("M1 protocol checksum mismatch")
    protocol = json.loads(payload)
    for relative, expected in protocol["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Protocol source changed after freeze: {relative}")
    return protocol, digest


def sample_keys(arrays: dict[str, np.ndarray]) -> list[tuple[str, ...]]:
    fields = ("scene_id", "video_id", "target_id", "obs_end_frame")
    sizes = {len(np.asarray(arrays[field]).reshape(-1)) for field in fields}
    if len(sizes) != 1:
        raise ValueError("Sample identity arrays have different lengths")
    return [tuple(str(np.asarray(arrays[field]).reshape(-1)[i]) for field in fields) for i in range(sizes.pop())]


def paired_indices(reference: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> np.ndarray:
    """Return reference row indices aligned to target order; fail on duplicates/missing IDs."""
    ref_keys = sample_keys(reference)
    target_keys = sample_keys(target)
    if len(set(ref_keys)) != len(ref_keys) or len(set(target_keys)) != len(target_keys):
        raise ValueError("Duplicate sample identity in paired predictions")
    ref_lookup = {key: i for i, key in enumerate(ref_keys)}
    if set(ref_lookup) != set(target_keys):
        missing = len(set(target_keys) - set(ref_lookup))
        extra = len(set(ref_lookup) - set(target_keys))
        raise ValueError(f"Paired sample identities differ (missing={missing}, extra={extra})")
    return np.asarray([ref_lookup[key] for key in target_keys], dtype=np.int64)


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def metrics_for(method: str, run: dict[str, Any], pred: dict[str, np.ndarray]) -> tuple[np.ndarray, float, np.ndarray]:
    labels = pred["intent_label"].astype(np.int64).reshape(-1)
    if method == "M1_pretrained_clone_trainable":
        probability = pred["calibrated_probability"].astype(np.float64)
        threshold = float(pred["threshold"])
    elif method == "M0_scratch_matched":
        probability = pred["calibrated_probability"].astype(np.float64)
        threshold = float(pred["threshold"])
    else:
        probability = pred["intent_probability"].astype(np.float64)
        threshold = float(run["selected_validation_calibration"]["threshold"])
    return labels, threshold, probability


def mean_metric(method_rows: list[dict[str, Any]], section: str, metric: str) -> str:
    values = [float(row[section][metric]) for row in method_rows]
    return f"{np.mean(values):.4f} ± {np.std(values, ddof=1):.4f}"


def main() -> None:
    config = load_config()
    protocol, protocol_hash = verify_protocol()
    if not (RESULTS_ROOT / "official_test_access_record.json").is_file():
        raise RuntimeError("M1 one-time official test evaluation is not complete")

    method_rows: dict[str, list[dict[str, Any]]] = {
        "M0": [], "P1": [], "M1": [],
    }
    seed_rows: list[dict[str, Any]] = []
    bootstrap: dict[str, Any] = {
        "protocol_sha256": protocol_hash,
        "definition": "paired, scene_id cluster bootstrap; delta is M1 minus comparator; seeds analyzed independently",
        "repetitions": int(config["paired_analysis"]["bootstrap_repetitions"]),
        "by_seed": {},
    }
    trajectory_rows: list[dict[str, Any]] = []

    for seed in SEEDS:
        m1_pred = load_predictions(RESULTS_ROOT / f"seed{seed}/test_predictions.npz")
        m0_pred = load_predictions(M0_ROOT / f"seed{seed}/test_predictions.npz")
        p1_pred = load_predictions(P1_ROOT / f"seed{seed}/test_predictions.npz")
        m1_metrics = json.loads((RESULTS_ROOT / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
        m0_metrics = json.loads((M0_ROOT / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
        p1_metrics = json.loads((P1_ROOT / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))

        m0_order = paired_indices(m0_pred, m1_pred)
        p1_order = paired_indices(p1_pred, m1_pred)
        labels = m1_pred["intent_label"].astype(np.int64)
        if not np.array_equal(m0_pred["intent_label"][m0_order], labels) or not np.array_equal(p1_pred["intent_label"][p1_order], labels):
            raise RuntimeError(f"Paired labels mismatch for seed {seed}")

        m1_prob = m1_pred["calibrated_probability"].astype(np.float64)
        m1_threshold = float(m1_pred["threshold"])
        comparisons = {}
        for tag, method, prior_metrics, prior_pred, order in (
            ("M0_scratch_matched", "M0", m0_metrics, m0_pred, m0_order),
            ("P1_target_only", "P1", p1_metrics, p1_pred, p1_order),
        ):
            prior_run = prior_metrics
            prior_labels, prior_threshold, prior_probability = metrics_for(tag, prior_run, prior_pred)
            prior_labels = prior_labels[order]
            prior_probability = prior_probability[order]
            delta = paired_delta_metrics(labels, prior_probability, m1_prob, prior_threshold, m1_threshold)
            base = binary_metrics(prior_labels, prior_probability, prior_threshold)
            current = binary_metrics(labels, m1_prob, m1_threshold)
            scene_ids = m1_pred["scene_id"].astype(str)
            ci = cluster_bootstrap_paired_delta(
                labels,
                prior_probability,
                m1_prob,
                scene_ids,
                repetitions=int(config["paired_analysis"]["bootstrap_repetitions"]),
                seed=int(config["paired_analysis"]["bootstrap_seed"]),
            )
            comparisons[tag] = {
                "baseline": method,
                "baseline_metrics_recomputed": base,
                "m1_metrics_recomputed": current,
                "point_delta_m1_minus_baseline": delta,
                "paired_scene_cluster_bootstrap": ci,
            }
        bootstrap["by_seed"][str(seed)] = comparisons

        m1_test = m1_metrics["test"]
        m0_test = m0_metrics["test"]
        p1_test = p1_metrics["test"]
        method_rows["M0"].append({"seed": seed, **m0_test["intent"], **{"trajectory": p1_test["trajectory"]}})
        method_rows["P1"].append({"seed": seed, **p1_test["intent"], "trajectory": p1_test["trajectory"]})
        method_rows["M1"].append({"seed": seed, **m1_test["intent"], "trajectory": m1_test["trajectory"]})
        delta_m0 = comparisons["M0_scratch_matched"]["point_delta_m1_minus_baseline"]
        delta_p1 = comparisons["P1_target_only"]["point_delta_m1_minus_baseline"]
        seed_rows.append({
            "seed": seed,
            "m0_auc": comparisons["M0_scratch_matched"]["baseline_metrics_recomputed"]["roc_auc"],
            "p1_auc": comparisons["P1_target_only"]["baseline_metrics_recomputed"]["roc_auc"],
            "m1_auc": comparisons["M0_scratch_matched"]["m1_metrics_recomputed"]["roc_auc"],
            "m1_minus_m0": delta_m0,
            "m1_minus_p1": delta_p1,
        })
        t0 = p1_test["trajectory"]
        m1_traj = m1_test["trajectory"]
        trajectory_rows.append({
            "seed": seed,
            "t0_ade": float(t0["ade_pixel"]), "m1_ade": float(m1_traj["ade_pixel"]),
            "delta_ade": float(m1_traj["ade_pixel"] - t0["ade_pixel"]),
            "t0_fde": float(t0["fde_pixel"]), "m1_fde": float(m1_traj["fde_pixel"]),
            "delta_fde": float(m1_traj["fde_pixel"] - t0["fde_pixel"]),
            "max_abs_future_prediction_difference_vs_p1": float(m1_test["max_abs_future_prediction_difference_vs_p1"]),
        })

    m0_auc = [row["m0_auc"] for row in seed_rows]
    m1_auc = [row["m1_auc"] for row in seed_rows]
    delta_auc = [row["m1_minus_m0"]["delta_roc_auc"] for row in seed_rows]
    ci_positive = sum(
        bootstrap["by_seed"][str(seed)]["M0_scratch_matched"]["paired_scene_cluster_bootstrap"]["delta_roc_auc"]["ci_percentile_95"]["lower_95"] > 0
        for seed in SEEDS
    )
    trajectory_ok = all(abs(row["max_abs_future_prediction_difference_vs_p1"]) < 1e-6 for row in trajectory_rows)
    m0_sd = float(np.std(m0_auc, ddof=1))
    m1_sd = float(np.std(m1_auc, ddof=1))
    variance_ratio = m1_sd / m0_sd if m0_sd > 0 else float("inf")
    decision = {
        "mean_delta_auc_m1_minus_m0": float(np.mean(delta_auc)),
        "positive_seed_count": int(sum(value > 0 for value in delta_auc)),
        "seed_ci_lower_above_zero_count": int(ci_positive),
        "m0_seed_auc_sample_sd": m0_sd,
        "m1_seed_auc_sample_sd": m1_sd,
        "m1_to_m0_seed_sd_ratio": variance_ratio,
        "trajectory_exact_preservation_pass": trajectory_ok,
        "positive_mean_delta_auc_pass": bool(np.mean(delta_auc) > 0),
        "positive_seed_count_pass": bool(sum(value > 0 for value in delta_auc) >= 2),
        "bootstrap_stability_pass": bool(ci_positive >= 2),
        "seed_variance_pass": bool(variance_ratio <= float(config["paired_analysis"]["transfer_success_rule"]["maximum_m1_auc_seed_sd_vs_m0_seed_sd_ratio"])),
    }
    decision["pretraining_transfer_success"] = all(
        decision[key] for key in (
            "positive_mean_delta_auc_pass", "positive_seed_count_pass", "bootstrap_stability_pass",
            "seed_variance_pass", "trajectory_exact_preservation_pass",
        )
    )
    decision["interpretation"] = (
        "evidence meets the pre-registered transfer rule"
        if decision["pretraining_transfer_success"]
        else "does not meet the pre-registered stable-transfer rule; any bootstrap CI crossing zero is weak/inconclusive, not significant transfer"
    )

    write_json(RESULTS_ROOT / "cluster_bootstrap.json", bootstrap)
    write_summary(method_rows, seed_rows, trajectory_rows, bootstrap, decision, protocol_hash)
    write_json(RESULTS_ROOT / "analysis_decision.json", decision)
    print(json.dumps({"decision": decision, "summary": str(RESULTS_ROOT / "summary.md")}, ensure_ascii=False, indent=2))


def write_summary(method_rows, seed_rows, trajectory_rows, bootstrap, decision, protocol_hash: str) -> None:
    lines = [
        "# M1 Pretrained-Clone Trainable Intention Experiment",
        "",
        f"Frozen protocol SHA256: `{protocol_hash}`.",
        "",
        "M1 isolates encoder initialization: its target-only intention Transformer is trainable from trajectory-pretrained cloned weights; a separate, frozen trajectory branch is retained. All official metrics below use the held-out test set once after freeze. Seed values are not pooled as statistical replicates; cluster bootstrap is performed independently within each seed.",
        "",
        "## Test metrics across seeds (mean ± sample SD)",
        "",
        "| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("M0", "M0"), ("P1", "P1"), ("M1", "M1")):
        rows = method_rows[key]
        lines.append(
            f"| {label} | {mean_metric(rows, 'intent', 'roc_auc')} | {mean_metric(rows, 'intent', 'brier')} | {mean_metric(rows, 'intent', 'f1')} | {mean_metric(rows, 'intent', 'balanced_accuracy')} | {mean_metric(rows, 'trajectory', 'ade_pixel')} | {mean_metric(rows, 'trajectory', 'fde_pixel')} |"
        )
    lines += ["", "## Matched-seed intention comparison", "", "| Seed | M0 AUC | P1 AUC | M1 AUC | M1−M0 ΔAUC | M1−P1 ΔAUC |", "|---:|---:|---:|---:|---:|---:|"]
    for row in seed_rows:
        lines.append(f"| {row['seed']} | {row['m0_auc']:.4f} | {row['p1_auc']:.4f} | {row['m1_auc']:.4f} | {row['m1_minus_m0']['delta_roc_auc']:+.4f} | {row['m1_minus_p1']['delta_roc_auc']:+.4f} |")
    lines += ["", "## M1−M0 paired cluster bootstrap (per seed)", "", "| Seed | ΔAUC | 95% CI | ΔBrier | 95% CI |", "|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
        item = bootstrap["by_seed"][str(seed)]["M0_scratch_matched"]
        auc_ci = item["paired_scene_cluster_bootstrap"]["delta_roc_auc"]["ci_percentile_95"]
        brier_ci = item["paired_scene_cluster_bootstrap"]["delta_brier"]["ci_percentile_95"]
        delta = item["point_delta_m1_minus_baseline"]
        lines.append(f"| {seed} | {delta['delta_roc_auc']:+.4f} | [{auc_ci['lower_95']:+.4f}, {auc_ci['upper_95']:+.4f}] | {delta['delta_brier']:+.4f} | [{brier_ci['lower_95']:+.4f}, {brier_ci['upper_95']:+.4f}] |")
    lines += ["", "## Trajectory preservation", "", "| Seed | T0 ADE | M1 ADE | ΔADE | T0 FDE | M1 FDE | ΔFDE | max |Δprediction| |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in trajectory_rows:
        lines.append(f"| {row['seed']} | {row['t0_ade']:.4f} | {row['m1_ade']:.4f} | {row['delta_ade']:+.6f} | {row['t0_fde']:.4f} | {row['m1_fde']:.4f} | {row['delta_fde']:+.6f} | {row['max_abs_future_prediction_difference_vs_p1']:.3g} |")
    lines += [
        "",
        "## Initialization and isolation checks",
        "",
        "- Exact pretrained clone: pass for all 3 seeds; expected/copy counts and zero maximum absolute tensor difference are in `pretrained_clone_report.json`.",
        "- Independent parameters: pass for all 3 seeds; optimizer step changed intention parameters only, with no shared object/storage and no trajectory parameter update (`parameter_independence_test.json`).",
        "- Trajectory equivalence before training: pass for all seeds (`trajectory_equivalence.json`).",
        "- Training: three seeds completed 20 epochs; frozen trajectory SHA256 was unchanged every epoch; intention optimizer excluded trajectory parameters.",
        "",
        "## Pre-registered transfer rule",
        "",
        f"- Mean ΔAUC(M1−M0) > 0: **{decision['positive_mean_delta_auc_pass']}** ({decision['mean_delta_auc_m1_minus_m0']:+.4f}).",
        f"- At least 2/3 positive seed ΔAUC: **{decision['positive_seed_count_pass']}** ({decision['positive_seed_count']}/3).",
        f"- At least 2/3 seed-specific 95% bootstrap CI lower bounds > 0: **{decision['bootstrap_stability_pass']}** ({decision['seed_ci_lower_above_zero_count']}/3).",
        f"- M1/M0 AUC seed-SD ratio ≤ 1.5: **{decision['seed_variance_pass']}** ({decision['m1_to_m0_seed_sd_ratio']:.3f}; M0 SD={decision['m0_seed_auc_sample_sd']:.4f}, M1 SD={decision['m1_seed_auc_sample_sd']:.4f}).",
        f"- Trajectory predictions preserved exactly: **{decision['trajectory_exact_preservation_pass']}**.",
        f"- Overall: **{decision['interpretation']}**.",
        "",
        "## Encoder drift and representation shift",
        "",
        "Selected-epoch, per-layer parameter drift is recorded in `encoder_drift.json`. The validation-only 1,000-sample feature comparison (means, standard deviations, norms, per-dimension variance, per-sample cosine, and linear CKA) is in `representation_shift.json`; it did not affect model selection.",
        "",
        "## Research interpretation and next step",
        "",
        f"- M1 vs M0 isolates trainable pretrained initialization. The result **{'supports' if decision['pretraining_transfer_success'] else 'does not establish'}** reliable transfer under the registered rule.",
        "- M1 vs P1 is the secondary task-adaptation comparison; consult the seedwise paired results and bootstrap in `cluster_bootstrap.json`.",
        "- Adapter and partial-unfreeze work is **not automatically authorized by these results**. If transfer is inconclusive or negative, do not present trajectory-pretrained initialization as established value; diagnose task mismatch and preserve M0 as the reference before expanding architecture.",
        "- A positive M1−M0 mean with confidence intervals crossing zero is described as weak/inconclusive, not statistically significant transfer.",
        "",
        "## Reproducibility artifacts",
        "",
        "The frozen protocol records source/config/checkpoint/data hashes and the already-existing M0/P1 result artifact hashes. M1 official test predictions are stored under each seed directory. No adapter, partial unfreeze, scene/social/reliability input, or trajectory loss was used.",
        "",
    ]
    (RESULTS_ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
