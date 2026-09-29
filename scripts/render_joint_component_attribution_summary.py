#!/usr/bin/env python3
"""Render the frozen component study summary with the intended matched M0 context.

This is post-freeze reporting only: it reads saved predictions/metrics, does
not load the test archive, run inference, or recompute component bootstraps.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_joint_component_attribution as study


def main() -> None:
    protocol_raw = study.PROTOCOL_PATH.read_bytes()
    protocol_sha = hashlib.sha256(protocol_raw).hexdigest()
    declared_sha = study.PROTOCOL_SHA_PATH.read_text(encoding="utf-8").split()[0]
    if protocol_sha != declared_sha:
        raise RuntimeError("Frozen protocol SHA256 mismatch")
    protocol = json.loads(protocol_raw)
    if protocol.get("status") != "frozen_before_first_access_to_test_archive":
        raise RuntimeError("The component study protocol is not frozen")
    access = json.loads(study.TEST_ACCESS_PATH.read_text(encoding="utf-8"))
    if access.get("status") != "all_15_official_test_evaluations_complete":
        raise RuntimeError("Official test evaluation is not complete")

    effects_path = study.RESULTS / "component_effects.json"
    effects = json.loads(effects_path.read_text(encoding="utf-8"))
    effects.setdefault("per_seed", {})
    rows: list[dict[str, float]] = []
    m0_sources: dict[str, str] = {}
    for seed in study.SEEDS:
        prediction_path = ROOT / f"results/intention_scratch_matched/seed{seed}/test_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not {"intent_label", "raw_probability"}.issubset(archive.files):
                raise RuntimeError(f"Matched M0 raw prediction fields are absent: {prediction_path}")
            labels = archive["intent_label"].astype(np.int64)
            probabilities = archive["raw_probability"].astype(np.float64)
        metrics = study.binary_test_metrics(labels, probabilities)
        row = {
            "roc_auc": metrics["roc_auc"],
            "brier": metrics["brier"],
            "f1": metrics["f1"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "accuracy": metrics["accuracy"],
        }
        rows.append(row)
        effects["per_seed"].setdefault(str(seed), {})["M0_target_only"] = {
            **row,
            "source": str(prediction_path.relative_to(ROOT)),
            "probability_field": "raw_probability",
            "threshold": 0.5,
        }
        m0_sources[str(seed)] = study.sha256_file(prediction_path)

    m0_summary = {}
    for metric in ("roc_auc", "brier", "f1", "balanced_accuracy", "accuracy"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        m0_summary[metric] = {
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "n": len(values),
        }
    effects["method_summary"]["M0_target_only"] = m0_summary
    effects["m0_context_source"] = "intention_scratch_matched raw_probability, threshold=0.5; historical matched control"
    study.write_json(effects_path, effects)

    bootstrap = json.loads((study.RESULTS / "cluster_bootstrap.json").read_text(encoding="utf-8"))
    summary = study.make_summary_markdown(effects, bootstrap, effects["per_seed"], protocol)
    summary_lines = summary.splitlines()
    for index, line in enumerate(summary_lines):
        if line.startswith("9. `M0→J0` AUC gap"):
            gap = effects["auc_gap_j0_minus_m0"]
            largest = effects["max_single_component_mean_contribution"]
            summary_lines[index] = (
                f"9. `M0→J0` AUC gap 为 `{gap:.4f}`；No Scene 的 Full−Ablation 点估计为 `{largest:+.4f}`，"
                "几乎与该 gap 同量级。但 3/3 seed 的 AUC bootstrap CI 均跨 0，故目前不能据此确认 scene 单独解释了全部差距。"
            )
            break
    (study.RESULTS / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    amendment = {
        "status": "post_freeze_read_only_report_render",
        "protocol_sha256": protocol_sha,
        "reason": "The generic historical target-only resolver can match a different intent_target_only_15x15 experiment; the requested M0 is the matched intention_scratch_matched control.",
        "m0_source": "results/intention_scratch_matched/seed{seed}/test_predictions.npz raw_probability",
        "m0_prediction_sha256_by_seed": m0_sources,
        "m0_recomputed_metrics": m0_summary,
        "reporting_only": True,
        "renderer_opened_test_archive": False,
        "model_inference_repeated": False,
        "component_bootstrap_values_changed": False,
        "bootstrap_computation": "already completed from frozen predictions after protocol freeze; this renderer reuses cluster_bootstrap.json",
        "A0_A1_A5_test_metrics_and_bootstrap_values_changed": False,
        "interpretation_note": "No Scene point estimate nearly equals the M0-to-J0 AUC gap, but all three paired cluster-bootstrap AUC CIs cross zero; the report therefore does not claim a confirmed single-component explanation.",
    }
    amendment["renderer_sha256"] = study.sha256_file(Path(__file__))
    study.write_json(study.RESULTS / "post_freeze_report_amendment.json", amendment)
    print(json.dumps({"status": "rendered", "protocol_sha256": protocol_sha, "m0_auc_mean": m0_summary["roc_auc"]["mean"], "summary": "results/joint_component_attribution/summary.md"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
