#!/usr/bin/env python3
"""Render compact, reproducible markdown from the frozen feature experiment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reliability_feature_intent_utils import load_json

OUTPUT_ROOT = PROJECT_ROOT / "results/reliability_feature_intent_15x15"
SEEDS = (42, 123, 2024)


def fmt(values: dict[str, Any], *, digits: int = 4) -> str:
    return f"{values['mean']:.{digits}f} ± {values['sample_std']:.{digits}f}"


def ci_text(record: dict[str, Any]) -> str:
    ci = record["delta_roc_auc"]["ci_percentile_95"]
    ci_b = record["delta_brier"]["ci_percentile_95"]
    return f"AUC [{ci['lower_95']:.4f}, {ci['upper_95']:.4f}]; Brier [{ci_b['lower_95']:.4f}, {ci_b['upper_95']:.4f}]"


def main() -> None:
    protocol = load_json(OUTPUT_ROOT / "protocol_frozen.json")
    metrics = load_json(OUTPUT_ROOT / "test_metrics.json")
    paired = load_json(OUTPUT_ROOT / "paired_delta.json")
    bootstrap = load_json(OUTPUT_ROOT / "cluster_bootstrap.json")
    strata = load_json(OUTPUT_ROOT / "reliability_strata.json")
    distribution = load_json(OUTPUT_ROOT / "train_distribution.json")
    importance = load_json(OUTPUT_ROOT / "feature_importance.json")
    access = load_json(OUTPUT_ROOT / "test_access_record.json")

    lines = [
        "# Reliability as an intention feature: frozen JAAD experiment",
        "",
        f"- Protocol SHA-256: `{metrics['protocol_sha256']}`",
        f"- Labeled test cache SHA-256: `{metrics['test_cache_sha256']}`",
        f"- Samples: {protocol['data']['train_oof_sample_count']:,} train OOF; {protocol['data']['validation_sample_count']:,} validation; {access['sample_count']:,} test.",
        "- Trajectory predictors were frozen; no future ground truth, ADE/FDE, scene, or social inputs were used.",
        "- Normalization was fit only on official train OOF; validation-fitted temperatures and thresholds were frozen before test labels were decoded.",
        "- Test labels were used only for this single frozen evaluation, not training, selection, or normalization.",
        "",
        "## Test performance (mean ± sample SD across 3 seeds)",
        "",
        "| Model | ROC-AUC | Brier ↓ | ECE ↓ | Balanced accuracy | F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    display = ("A", "B", "C", "D", "E", "D_no_future")
    for variant in display:
        aggregate = metrics["models"][variant]["mean_sample_std"]
        lines.append(
            f"| {variant}: {metrics['models'][variant]['variant_name']} | {fmt(aggregate['roc_auc'])} | {fmt(aggregate['brier'])} | {fmt(aggregate['ece_15_equal_width'])} | {fmt(aggregate['balanced_accuracy'])} | {fmt(aggregate['f1_positive'])} |"
        )

    d_auc = metrics["models"]["D"]["mean_sample_std"]["roc_auc"]["mean"]
    b_auc = metrics["models"]["B"]["mean_sample_std"]["roc_auc"]["mean"]
    c_auc = metrics["models"]["C"]["mean_sample_std"]["roc_auc"]["mean"]
    e_auc = metrics["models"]["E"]["mean_sample_std"]["roc_auc"]["mean"]
    d_brier = metrics["models"]["D"]["mean_sample_std"]["brier"]["mean"]
    c_brier = metrics["models"]["C"]["mean_sample_std"]["brier"]["mean"]
    e_brier = metrics["models"]["E"]["mean_sample_std"]["brier"]["mean"]
    lines.extend(
        [
            "",
            "## Main finding",
            "",
            f"The primary adjusted-reliability model D did not improve discrimination over future-only B (ΔAUC {d_auc - b_auc:+.4f}); its Brier score was {d_brier - metrics['models']['B']['mean_sample_std']['brier']['mean']:+.4f} lower, so discrimination and calibration move in different directions. Motion adjustment did not improve AUC over raw reliability C (ΔAUC {d_auc - c_auc:+.4f}), although D's Brier was {d_brier - c_brier:+.4f} lower. Adding motion in E changed AUC by {e_auc - d_auc:+.4f} and Brier by {e_brier - d_brier:+.4f}; this is not a consistent gain. Overall, this run does not support the hypothesis that these reliability features improve intention recognition under the frozen setup.",
            "",
        ]
    )

    lines.extend(
        [
            "",
            "## Paired primary comparisons",
            "",
            "Deltas are candidate minus baseline; positive ΔAUC and negative ΔBrier favor the candidate. Bootstrap resamples `scene_id` clusters jointly (2,000 draws per seed).",
            "",
            "| Comparison | Mean ΔAUC | Mean ΔBrier | Seed 42 paired 95% CI | Seed 123 paired 95% CI | Seed 2024 paired 95% CI |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for key in ("D-B", "D-A", "D-C", "E-D"):
        delta = paired["comparisons"][key]["mean_sample_std"]
        boot = bootstrap["comparisons"][key]["per_seed"]
        lines.append(
            f"| {key} | {fmt(delta['delta_roc_auc'])} | {fmt(delta['delta_brier'])} | {ci_text(boot['42'])} | {ci_text(boot['123'])} | {ci_text(boot['2024'])} |"
        )

    lines.extend(["", "## Reliability strata", "", "Cutpoints were fixed at train-OOF adjusted-u q33/q67. Lower adjusted-u means comparatively higher reliability; the highest adjusted-u tertile is the low-reliability/high-uncertainty group.", "", "| Stratum | n | B AUC | D AUC | ΔAUC D-B (per seed) | B Brier | D Brier |", "|---|---:|---:|---:|---|---:|---:|"])
    for name in ("high_reliability", "medium_reliability", "low_reliability"):
        row = strata["strata"][name]
        b, d = row["models"]["B"]["mean_sample_std"], row["models"]["D"]["mean_sample_std"]
        values = row["deltas"]["D-B"]
        delta_auc = [values[str(seed)]["delta_roc_auc"] for seed in SEEDS]
        delta_display = ", ".join("NA" if value is None else f"{value:+.4f}" for value in delta_auc)
        lines.append(
            f"| {name} | {row['sample_count']:,} | {fmt(b['roc_auc']) if 'roc_auc' in b else 'NA'} | {fmt(d['roc_auc']) if 'roc_auc' in d else 'NA'} | {delta_display} | {fmt(b['brier'])} | {fmt(d['brier'])} |"
        )
    high_rel_n = strata["strata"]["high_reliability"]["sample_count"]
    lines.append(
        f"\nApplying the fixed train-OOF cutpoints yields {high_rel_n:,}/{access['sample_count']:,} test samples in high_reliability; this imbalance is reported as observed and was not corrected with test-derived quantiles."
    )

    lines.extend(["", "## Feature distributions", "", "| Split | Feature | Mean | SD | Median | q05–q95 |", "|---|---|---:|---:|---:|---:|"])
    for split, split_name in (("train_oof", "Train OOF"), ("val", "Validation"), ("test_unlabeled", "Test (features only)")):
        for feature, label in (("raw_u", "raw_u"), ("adjusted_u", "adjusted_u"), ("motion", "motion")):
            stat = distribution[split]["features"][feature]
            lines.append(
                f"| {split_name} | {label} | {stat['mean']:.4f} | {stat['std']:.4f} | {stat['median']:.4f} | {stat['q05']:.4f}–{stat['q95']:.4f} |"
            )

    lines.extend(["", "## Reliability-branch diagnostic", "", "The following are first-layer weight norms, not causal feature attributions and not computed from test labels.", "", "| Model | Seed | Input-column L2 norms |", "|---|---:|---|"])
    for variant, seeds in importance["models"].items():
        for seed, values in seeds.items():
            values_text = ", ".join(f"{key}={value:.4f}" for key, value in values["input_column_l2_norm"].items())
            lines.append(f"| {variant} | {seed} | {values_text} |")

    lines.extend(
        [
            "",
            "## Protocol and leakage audit",
            "",
            f"- Optimizer/config: AdamW, lr={protocol['training']['learning_rate']}, batch={protocol['training']['batch_size']}, epochs={protocol['training']['epochs_max']}, balanced BCE unchanged from the previous frozen run.",
            f"- Training seeds: {', '.join(map(str, protocol['training']['seeds']))}; normalization SHA: `{protocol['feature_normalization']['sha256']}`.",
            f"- Matched B/C/D/E observed/future initializations were checked: {len(protocol['matched_initialization']['common_observed_and_future_branch_initialization'])} seeds passed.",
            f"- Test access record: labels read after protocol freeze = `{access['labels_read_after_protocol_freeze']}`; used for training/selection/normalization = `{access['labels_used_for_training_selection_or_normalization']}`.",
            "- D_raw is the same trained model as C; D_adjusted is the same trained model as D. D_no_future is the separate observed + adjusted-reliability ablation.",
            "",
        ]
    )
    path = OUTPUT_ROOT / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
