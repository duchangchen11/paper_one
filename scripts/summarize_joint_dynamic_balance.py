#!/usr/bin/env python3
"""Create the paired DGB-20 vs fixed-100 report from frozen run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024)
METRIC_MAP = {
    "AUC": "intent_auc",
    "Brier": "intent_brier",
    "F1": "intent_f1",
    "BAcc": "intent_balanced_accuracy",
    "ADE": "trajectory_ade_pixel",
    "FDE": "trajectory_fde_pixel",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> str:
    data = np.asarray(values, dtype=np.float64)
    return f"{data.mean():.4f} ± {data.std(ddof=1):.4f}"


def metric_value(metrics: dict, name: str) -> float:
    return float(metrics[METRIC_MAP[name]])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/joint_dynamic_balance")
    parser.add_argument("--fixed-root", type=Path, default=ROOT / "results/joint_loss_balance/lambda100")
    parser.add_argument("--output", type=Path, default=ROOT / "results/joint_dynamic_balance/summary.md")
    args = parser.parse_args()

    fixed_runs = {}
    dgb_runs = {}
    dgb_histories = {}
    fixed_histories = {}
    for seed in SEEDS:
        fixed_dir = args.fixed_root / f"seed{seed}"
        dgb_dir = args.results_root / "dgb20" / f"seed{seed}"
        fixed_runs[seed] = read_json(fixed_dir / "metrics.json")["test"]
        dgb_runs[seed] = read_json(dgb_dir / "metrics.json")["test"]
        fixed_histories[seed] = read_json(fixed_dir / "gradient_history.json")
        dgb_histories[seed] = read_json(dgb_dir / "gradient_history.json")

    validation_lines = []
    for seed in SEEDS:
        metrics = read_json(args.results_root / "dgb20" / f"seed{seed}" / "metrics.json")
        selected = next(row for row in metrics["history"] if row["epoch"] == metrics["best_epoch"])
        val = selected["val"]
        validation_lines.append(
            f"| {seed} | {metrics['best_epoch']} | {val['intent_auc']:.4f} | {val['intent_brier']:.4f} | "
            f"{val['intent_f1']:.4f} | {val['intent_balanced_accuracy']:.4f} | "
            f"{val['trajectory_ade_pixel']:.2f} | {val['trajectory_fde_pixel']:.2f} |"
        )

    dgb_lambda_values = [
        float(value)
        for seed in SEEDS
        for epoch in dgb_histories[seed]["epochs"]
        for value in epoch["lambda_per_training_batch"]
    ]
    dgb_weighted_ratios = [
        float(sample["weighted_gradient_ratio"])
        for seed in SEEDS
        for epoch in dgb_histories[seed]["epochs"]
        for sample in epoch["gradient_samples"]
    ]
    fixed_weighted_ratios = [
        float(epoch["intent_over_weighted_trajectory_gradient_ratio"])
        for seed in SEEDS
        for epoch in fixed_histories[seed]["epochs"]
    ]
    dgb_updates = [
        record
        for seed in SEEDS
        for epoch in dgb_histories[seed]["epochs"]
        for record in epoch["update_records"]
        if record.get("controller_update")
    ]
    lower_hits = sum(bool(record.get("hit_lambda_min")) for record in dgb_updates)
    upper_hits = sum(bool(record.get("hit_lambda_max")) for record in dgb_updates)
    total_updates = len(dgb_updates)

    lines = [
        "# Dynamic Gradient Balance (DGB-20) results",
        "",
        "## Frozen protocol and validation-only review",
        "",
        "DGB-20 used initial λ=100, target weighted shared-gradient ratio ρ=20, log-space EMA β=0.9, λ∈[10,300], one fixed warm-up epoch, and updates every 10 training batches. Training settings, architecture, splits, optimizer, scheduler, and checkpoint selection were held at the `ad05ef5` baseline. No reliability/uncertainty weighting, adapters, PCGrad, or GradNorm were used.",
        "",
        "The three training runs completed with the test split withheld. The following validation metrics at each seed's selected checkpoint were reviewed before protocol freeze:",
        "",
        "| Seed | Best epoch | Val AUC | Val Brier | Val F1 | Val BAcc | Val ADE (px) | Val FDE (px) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *validation_lines,
        "",
        f"Frozen protocol SHA256: `{(args.results_root / 'protocol_frozen.sha256').read_text(encoding='utf-8').strip() if (args.results_root / 'protocol_frozen.sha256').exists() else 'not found'}`.",
        "",
        "## Test-set summary (mean ± sample SD across three seeds)",
        "",
        "| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) | Mean weighted gradient ratio | Mean λ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    fixed_metric_values = {
        name: [metric_value(fixed_runs[seed], name) for seed in SEEDS] for name in METRIC_MAP
    }
    dgb_metric_values = {
        name: [metric_value(dgb_runs[seed], name) for seed in SEEDS] for name in METRIC_MAP
    }
    fixed_ratio_by_seed = [
        np.mean(
            [epoch["intent_over_weighted_trajectory_gradient_ratio"] for epoch in fixed_histories[seed]["epochs"]]
        )
        for seed in SEEDS
    ]
    dgb_mean_lambda = float(np.mean(dgb_lambda_values))
    lines.append(
        f"| Fixed λ=100 | {mean_sd(fixed_metric_values['AUC'])} | {mean_sd(fixed_metric_values['Brier'])} | "
        f"{mean_sd(fixed_metric_values['F1'])} | {mean_sd(fixed_metric_values['BAcc'])} | "
        f"{mean_sd(fixed_metric_values['ADE'])} | {mean_sd(fixed_metric_values['FDE'])} | "
        f"{mean_sd(fixed_ratio_by_seed)} | 100.0000 ± 0.0000 |"
    )
    lines.append(
        f"| DGB-20 | {mean_sd(dgb_metric_values['AUC'])} | {mean_sd(dgb_metric_values['Brier'])} | "
        f"{mean_sd(dgb_metric_values['F1'])} | {mean_sd(dgb_metric_values['BAcc'])} | "
        f"{mean_sd(dgb_metric_values['ADE'])} | {mean_sd(dgb_metric_values['FDE'])} | "
        f"{np.mean(dgb_weighted_ratios):.4f} ± {np.std(dgb_weighted_ratios, ddof=1):.4f} | "
        f"{dgb_mean_lambda:.4f} ± {np.std(dgb_lambda_values, ddof=1):.4f} |"
    )

    lines.extend(
        [
            "",
            "## Per-seed test results and paired deltas (DGB − Fixed100)",
            "",
            "| Seed | Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) | Best epoch |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    paired = {name: [] for name in ("AUC", "Brier", "ADE", "FDE")}
    for seed in SEEDS:
        dgb_metrics_file = read_json(args.results_root / "dgb20" / f"seed{seed}" / "metrics.json")
        for method, metrics, best_epoch in (
            ("Fixed100", fixed_runs[seed], read_json(args.fixed_root / f"seed{seed}" / "metrics.json")["best_epoch"]),
            ("DGB-20", dgb_runs[seed], dgb_metrics_file["best_epoch"]),
        ):
            lines.append(
                f"| {seed} | {method} | {metric_value(metrics, 'AUC'):.4f} | {metric_value(metrics, 'Brier'):.4f} | "
                f"{metric_value(metrics, 'F1'):.4f} | {metric_value(metrics, 'BAcc'):.4f} | "
                f"{metric_value(metrics, 'ADE'):.2f} | {metric_value(metrics, 'FDE'):.2f} | {best_epoch} |"
            )
        for name in paired:
            paired[name].append(metric_value(dgb_runs[seed], name) - metric_value(fixed_runs[seed], name))
    lines.append(
        f"| Δ mean | DGB−Fixed100 | {np.mean(paired['AUC']):+.4f} AUC | {np.mean(paired['Brier']):+.4f} Brier | — | — | {np.mean(paired['ADE']):+.2f} ADE | {np.mean(paired['FDE']):+.2f} FDE | — |"
    )
    lines.extend(
        [
            "",
            "Paired differences are descriptive only; three seeds do not support a seed-level significance claim.",
            "",
            "## Controller and gradient diagnostics",
            "",
            f"- λ over all training batches: mean **{np.mean(dgb_lambda_values):.3f}**, std **{np.std(dgb_lambda_values, ddof=1):.3f}**, min **{np.min(dgb_lambda_values):.3f}**, max **{np.max(dgb_lambda_values):.3f}**.",
            f"- Applied controller updates: **{total_updates}**; λ_min hits **{lower_hits}/{total_updates} ({(100*lower_hits/total_updates if total_updates else 0):.2f}%)**; λ_max hits **{upper_hits}/{total_updates} ({(100*upper_hits/total_updates if total_updates else 0):.2f}%)**.",
            f"- DGB weighted gradient ratio at online measurement batches: mean **{np.mean(dgb_weighted_ratios):.3f}**, sample SD **{np.std(dgb_weighted_ratios, ddof=1):.3f}**, median **{np.median(dgb_weighted_ratios):.3f}**, range **[{np.min(dgb_weighted_ratios):.3f}, {np.max(dgb_weighted_ratios):.3f}]**.",
            f"- Fixed100 diagnostic ratio: mean **{np.mean(fixed_weighted_ratios):.3f}**, sample SD **{np.std(fixed_weighted_ratios, ddof=1):.3f}** across its fixed post-epoch audit batches.",
            "- Sampling caveat: DGB ratios are measured online every 10th training batch, while the pre-existing Fixed100 values are measured once per epoch on a fixed balanced diagnostic subset. Their spread is descriptive, not a strictly paired estimator.",
            "",
            "## Interpretation",
            "",
        ]
    )
    dgb_ade = float(np.mean(dgb_metric_values["ADE"]))
    dgb_fde = float(np.mean(dgb_metric_values["FDE"]))
    fixed_ade = float(np.mean(fixed_metric_values["ADE"]))
    fixed_fde = float(np.mean(fixed_metric_values["FDE"]))
    dgb_auc = float(np.mean(dgb_metric_values["AUC"]))
    fixed_auc = float(np.mean(fixed_metric_values["AUC"]))
    dgb_brier = float(np.mean(dgb_metric_values["Brier"]))
    fixed_brier = float(np.mean(fixed_metric_values["Brier"]))
    nearly_matches_fixed_ade = dgb_ade <= fixed_ade + 0.5
    no_material_fde_drop = dgb_fde <= fixed_fde * 1.05
    no_material_auc_drop = dgb_auc >= fixed_auc - 0.01
    no_material_brier_drop = dgb_brier <= fixed_brier + 0.01
    improves_or_matches = all(
        (nearly_matches_fixed_ade, no_material_fde_drop, no_material_auc_drop, no_material_brier_drop)
    )
    lines.extend(
        [
            f"- DGB-20 {'meets' if improves_or_matches else 'does not meet'} the predeclared practical comparison against Fixed100 under the simple descriptive checks (ADE within 0.5 px, FDE within 5%, AUC within 0.01, Brier within 0.01): ADE {dgb_ade:.2f} vs {fixed_ade:.2f} px; FDE {dgb_fde:.2f} vs {fixed_fde:.2f} px; AUC {dgb_auc:.4f} vs {fixed_auc:.4f}; Brier {dgb_brier:.4f} vs {fixed_brier:.4f}.",
            f"- Against the trajectory-only reference (ADE/FDE 11.06/19.59 px), DGB-20 {'does not restore' if dgb_ade > 11.06 or dgb_fde > 19.59 else 'reaches'} that reference. Remaining mean gaps: ADE **{dgb_ade - 11.06:+.2f} px**, FDE **{dgb_fde - 19.59:+.2f} px**.",
            f"- λ {'changed during training' if np.std(dgb_lambda_values) > 0 else 'did not change during training'}; boundary saturation {'is present' if lower_hits + upper_hits > 0 else 'was not observed'}.",
            f"- Recommendation: {'the results justify a carefully pre-registered follow-up that may include reliability-aware weighting, but they do not establish its benefit' if improves_or_matches else 'do not proceed to reliability-aware weighting yet; first have the next-stage research decision assess why DGB-20 did not outperform the fixed λ=100 control'}.",
            "- The test set is descriptive and was not used for DGB hyperparameter or checkpoint selection. No significance claim is made from three seeds.",
            "",
            "## Artifacts",
            "",
            "- Validation-only metrics and checkpoints: `results/joint_dynamic_balance/dgb20/seed{42,123,2024}/`.",
            "- Online gradient/lambda traces: each seed's `gradient_history.json`.",
            "- Frozen protocol: `results/joint_dynamic_balance/protocol_frozen.json` and `.sha256`.",
            "- Per-sample official-test predictions: each seed's `test_predictions.npz` (scene/video ID, target ID/frame, intention label/probability, future prediction/ground truth, image size, per-sample ADE/FDE).",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
