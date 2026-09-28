#!/usr/bin/env python3
"""Finalize M1 report from already-computed metrics/bootstrap; never loads test.npz."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_pretrained_clone_intent as original_analysis
from scripts.pretrained_clone_intent_utils import RESULTS_ROOT, write_json
from scripts.trajectory_preserving_utils import SEEDS, sha256_file

ANALYSIS_ADDENDUM = RESULTS_ROOT / "analysis_report_addendum.json"
ANALYSIS_ADDENDUM_SHA = RESULTS_ROOT / "analysis_report_addendum.sha256"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol, protocol_hash = original_analysis.verify_protocol()
    bootstrap_path = RESULTS_ROOT / "cluster_bootstrap.json"
    bootstrap = read_json(bootstrap_path)
    if bootstrap.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("Existing bootstrap does not match the frozen protocol")
    access = read_json(RESULTS_ROOT / "official_test_access_record.json")
    if access.get("official_test_evaluation_count") != 1:
        raise RuntimeError("Exactly one completed official evaluation is required")
    if bootstrap.get("repetitions") != protocol["paired_analysis_protocol"]["bootstrap_repetitions"]:
        raise RuntimeError("Bootstrap replicate count differs from frozen protocol")

    method_rows = {"M0": [], "P1": [], "M1": []}
    seed_rows = []
    trajectory_rows = []
    for seed in SEEDS:
        m0 = read_json(ROOT / f"results/intention_scratch_matched/seed{seed}/metrics.json")
        p1 = read_json(ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/metrics.json")
        m1 = read_json(RESULTS_ROOT / f"seed{seed}/metrics.json")
        m0_test, p1_test, m1_test = m0["test"], p1["test"], m1["test"]
        method_rows["M0"].append({"seed": seed, "intent": m0_test["intent"], "trajectory": p1_test["trajectory"]})
        method_rows["P1"].append({"seed": seed, "intent": p1_test["intent"], "trajectory": p1_test["trajectory"]})
        method_rows["M1"].append({"seed": seed, "intent": m1_test["intent"], "trajectory": m1_test["trajectory"]})

        primary = bootstrap["by_seed"][str(seed)]["M0_scratch_matched"]
        secondary = bootstrap["by_seed"][str(seed)]["P1_target_only"]
        seed_rows.append({
            "seed": seed,
            "m0_auc": primary["baseline_metrics_recomputed"]["roc_auc"],
            "p1_auc": secondary["baseline_metrics_recomputed"]["roc_auc"],
            "m1_auc": primary["m1_metrics_recomputed"]["roc_auc"],
            "m1_minus_m0": primary["point_delta_m1_minus_baseline"],
            "m1_minus_p1": secondary["point_delta_m1_minus_baseline"],
        })
        t0 = p1_test["trajectory"]
        trajectory = m1_test["trajectory"]
        trajectory_rows.append({
            "seed": seed,
            "t0_ade": float(t0["ade_pixel"]), "m1_ade": float(trajectory["ade_pixel"]),
            "delta_ade": float(trajectory["ade_pixel"] - t0["ade_pixel"]),
            "t0_fde": float(t0["fde_pixel"]), "m1_fde": float(trajectory["fde_pixel"]),
            "delta_fde": float(trajectory["fde_pixel"] - t0["fde_pixel"]),
            "max_abs_future_prediction_difference_vs_p1": float(m1_test["max_abs_future_prediction_difference_vs_p1"]),
        })

    m0_auc = [row["m0_auc"] for row in seed_rows]
    m1_auc = [row["m1_auc"] for row in seed_rows]
    deltas = [row["m1_minus_m0"]["delta_roc_auc"] for row in seed_rows]
    auc_cis = [bootstrap["by_seed"][str(seed)]["M0_scratch_matched"]["paired_scene_cluster_bootstrap"]["delta_roc_auc"]["ci_percentile_95"] for seed in SEEDS]
    m0_sd = float(np.std(m0_auc, ddof=1))
    m1_sd = float(np.std(m1_auc, ddof=1))
    variance_ratio = m1_sd / m0_sd if m0_sd > 0 else float("inf")
    ci_positive = sum(ci["lower_95"] > 0 for ci in auc_cis)
    trajectory_ok = all(row["max_abs_future_prediction_difference_vs_p1"] < 1e-6 for row in trajectory_rows)
    config_rule = protocol["paired_analysis_protocol"]["transfer_success_rule"]
    decision = {
        "mean_delta_auc_m1_minus_m0": float(np.mean(deltas)),
        "positive_seed_count": int(sum(delta > 0 for delta in deltas)),
        "seed_ci_lower_above_zero_count": int(ci_positive),
        "m0_seed_auc_sample_sd": m0_sd,
        "m1_seed_auc_sample_sd": m1_sd,
        "m1_to_m0_seed_sd_ratio": variance_ratio,
        "trajectory_exact_preservation_pass": trajectory_ok,
        "positive_mean_delta_auc_pass": bool(np.mean(deltas) > 0),
        "positive_seed_count_pass": bool(sum(delta > 0 for delta in deltas) >= int(config_rule["minimum_positive_seed_delta_auc_count"])),
        "bootstrap_stability_pass": bool(ci_positive >= int(config_rule["minimum_seed_auc_ci_stable_positive_count"])),
        "seed_variance_pass": bool(variance_ratio <= float(config_rule["maximum_m1_auc_seed_sd_vs_m0_seed_sd_ratio"])),
    }
    decision["pretraining_transfer_success"] = all(decision[key] for key in (
        "positive_mean_delta_auc_pass", "positive_seed_count_pass", "bootstrap_stability_pass",
        "seed_variance_pass", "trajectory_exact_preservation_pass",
    ))
    decision["interpretation"] = (
        "evidence meets the pre-registered transfer rule" if decision["pretraining_transfer_success"]
        else "does not meet the pre-registered stable-transfer rule; bootstrap intervals crossing zero are weak/inconclusive, not significant transfer"
    )

    original_analysis.write_summary(method_rows, seed_rows, trajectory_rows, bootstrap, decision, protocol_hash)
    summary_path = RESULTS_ROOT / "summary.md"
    with summary_path.open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## Execution and reporting notes\n\n"
            "The held-out NPZ contains `scene_id`, `target_id`, and `obs_end_frame` but no standalone `video_id`. "
            "After the frozen evaluator stopped at field validation (before inference or test-metric computation), "
            "video IDs were attached by an exact one-to-one join to the hash-frozen P1 outputs; the original "
            "protocol remains unchanged and this metadata-only correction is documented in `protocol_frozen_addendum.json`. "
            "A report-shape error occurred after all bootstrap results had been written; this summary was rebuilt from "
            "those existing bootstrap and metrics files without reopening the test archive or recomputing the bootstrap.\n"
        )
    write_json(RESULTS_ROOT / "analysis_decision.json", decision)

    if ANALYSIS_ADDENDUM.exists() or ANALYSIS_ADDENDUM_SHA.exists():
        raise RuntimeError("Report addendum already exists")
    addendum = {
        "addendum_id": "M1-report-aggregation-fix-v1",
        "parent_protocol_sha256": protocol_hash,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "The frozen analyzer expected nested test metric dictionaries although the bootstrap output had already been computed and saved.",
        "test_archive_reopened": False,
        "bootstrap_recomputed": False,
        "model_or_checkpoint_changed": False,
        "decision_rule_changed": False,
        "existing_bootstrap_sha256": sha256_file(bootstrap_path),
        "recovery_report_script_sha256": sha256_file(Path(__file__)),
        "parent_analyzer_sha256": sha256_file(ROOT / "scripts/analyze_pretrained_clone_intent.py"),
        "summary_sha256": sha256_file(summary_path),
    }
    write_json(ANALYSIS_ADDENDUM, addendum)
    digest = hashlib.sha256(ANALYSIS_ADDENDUM.read_bytes()).hexdigest()
    ANALYSIS_ADDENDUM_SHA.write_text(f"{digest}  analysis_report_addendum.json\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "summary": str(summary_path), "bootstrap_recomputed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
