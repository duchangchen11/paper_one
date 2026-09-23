#!/usr/bin/env python3
"""Summarize frozen-backbone social residual experiments and diagnostics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODES = ("none", "always", "uncertainty")
SEEDS = (42, 123, 2024)
TEST_FIELDS = (
    "intent_auc",
    "intent_balanced_accuracy",
    "intent_f1",
    "intent_brier",
    "trajectory_ade_pixel",
    "trajectory_fde_pixel",
    "gate_mean",
    "gate_std",
    "entropy_mean",
    "entropy_std",
)
FIELD_LABELS = {
    "intent_auc": "AUC",
    "intent_balanced_accuracy": "BAcc",
    "intent_f1": "F1",
    "intent_brier": "Brier",
    "trajectory_ade_pixel": "ADE px",
    "trajectory_fde_pixel": "FDE px",
    "gate_mean": "Gate mean",
    "gate_std": "Gate std",
    "entropy_mean": "Entropy mean",
    "entropy_std": "Entropy std",
}


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_metrics(results_root: Path, stage: str) -> dict[str, dict[int, dict[str, Any]]]:
    found: dict[str, dict[int, dict[str, Any]]] = {mode: {} for mode in MODES}
    for mode in MODES:
        for seed in SEEDS:
            data = read_json(results_root / f"residual_social_stage{stage}_{mode}_seed{seed}" / "metrics.json")
            if data is not None:
                found[mode][seed] = data
    return found


def compact_runs(raw_runs: dict[str, dict[int, dict[str, Any]]]) -> dict[str, dict[str, dict[str, Any]]]:
    compact: dict[str, dict[str, dict[str, Any]]] = {mode: {} for mode in MODES}
    for mode, runs in raw_runs.items():
        for seed, run in runs.items():
            compact[mode][str(seed)] = {
                "stage": run.get("stage"),
                "seed": seed,
                "gate_mode": run.get("gate_mode"),
                "trajectory_residual_enabled": run.get("trajectory_residual_enabled"),
                "trajectory_backbone_frozen": run.get("trajectory_backbone_frozen"),
                "checkpoint": run.get("checkpoint"),
                "checkpoint_selected": run.get("checkpoint_selected"),
                "best_epoch": run.get("best_epoch"),
                "best_validation_intent_auc": run.get("best_validation_intent_auc"),
                "val_trajectory_ade_constraint_pixel": run.get("val_trajectory_ade_constraint_pixel"),
                "test": run.get("test", {}),
            }
    return compact


def multiseed_aggregates(
    compact: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    output = {}
    for mode, runs in compact.items():
        rows = [runs[str(seed)] for seed in SEEDS if str(seed) in runs]
        per_metric = {}
        for field in TEST_FIELDS:
            values = [row["test"].get(field) for row in rows]
            values = [value for value in values if value is not None]
            per_metric[field] = (
                {
                    "mean": float(statistics.mean(values)),
                    "std_sample": float(statistics.stdev(values)) if len(values) > 1 else None,
                    "n": len(values),
                }
                if values
                else None
            )
        output[mode] = {"completed_seeds": sorted(int(seed) for seed in runs), "metrics": per_metric}
    return output


def mean_sample_std(rows: list[dict[str, Any]], field: str) -> dict[str, float] | None:
    values = [row["test"][field] for row in rows if row.get("test", {}).get(field) is not None]
    if not values:
        return None
    return {
        "mean": float(statistics.mean(values)),
        "std_sample": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def fmt(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def build_markdown(payload: dict[str, Any]) -> str:
    baseline = payload["frozen_transformer_baseline"]
    stage_a = payload["stage_a"]
    stage_b = payload["stage_b"]
    lines = [
        "# Frozen Transformer + uncertainty-guided social residual experiments",
        "",
        "All trajectory distances are in image pixels. Three-seed summaries use mean ± sample standard deviation (ddof=1). `crossing=-1` was not used as training supervision.",
        "",
        "## 1. Frozen Transformer baseline",
        "",
    ]
    if baseline is None:
        lines.append("Seed123 reproduction was not run or its result file is missing.")
    else:
        delta = baseline.get("difference_from_reference", {})
        lines.extend(
            [
                f"Seed123 test ADE/FDE: **{baseline['trajectory_ade_pixel']:.6f} / {baseline['trajectory_fde_pixel']:.6f} px**.",
                f"Difference from stored baseline: ADE {delta.get('ade_pixel', float('nan')):+.6f} px; FDE {delta.get('fde_pixel', float('nan')):+.6f} px. Within 0.01 px: **{baseline.get('within_0_01_pixel_tolerance')}**.",
            ]
        )

    lines.extend(
        [
            "",
            "## 2. Stage A: intent-only social residual",
            "",
            "| Seed | Gate mode | Residual | AUC | BAcc | F1 | Brier | ADE px | FDE px | Gate mean±std | Entropy mean±std |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        for seed in SEEDS:
            run = stage_a[mode].get(str(seed))
            if run is None:
                continue
            test = run["test"]
            lines.append(
                f"| {seed} | {mode} | off | {fmt(test.get('intent_auc'))} | {fmt(test.get('intent_balanced_accuracy'))} | {fmt(test.get('intent_f1'))} | {fmt(test.get('intent_brier'))} | {fmt(test.get('trajectory_ade_pixel'), 3)} | {fmt(test.get('trajectory_fde_pixel'), 3)} | {fmt(test.get('gate_mean'))}±{fmt(test.get('gate_std'))} | {fmt(test.get('entropy_mean'))}±{fmt(test.get('entropy_std'))} |"
            )
    if not any(stage_a[mode] for mode in MODES):
        lines.append("No completed Stage A runs found.")

    lines.extend(["", "### Stage A multi-seed summary", ""])
    full_multiseed = all(len(stage_a[mode]) == len(SEEDS) for mode in MODES)
    if full_multiseed:
        lines.extend(
            [
                "| Gate mode | AUC | BAcc | F1 | Brier | ADE px | FDE px |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for mode in MODES:
            rows = [stage_a[mode][str(seed)] for seed in SEEDS]
            cells = [mode]
            for field in TEST_FIELDS[:6]:
                summary = mean_sample_std(rows, field)
                cells.append(f"{summary['mean']:.4f}±{summary['std_sample']:.4f}")
            lines.append("| " + " | ".join(cells) + " |")
    else:
        lines.append("Not run: the seed123 Stage A stop criterion fired (uncertainty AUC did not exceed no-social, and always-social had no AUC advantage). Multi-seed expansion was intentionally halted.")

    lines.extend(["", "## 3. Stage B: trajectory social residual", ""])
    any_stage_b = any(stage_b[mode] for mode in MODES)
    if not any_stage_b:
        lines.append("Not run: Stage A seed123 met the predefined stop condition. No Stage B training was started.")
    else:
        lines.extend(
            [
                "| Seed | Gate mode | AUC | ADE px | FDE px | ADE degradation | Within 5% |",
                "|---:|---|---:|---:|---:|---:|---|",
            ]
        )
        for mode in ("always", "uncertainty"):
            for seed in SEEDS:
                run = stage_b[mode].get(str(seed))
                if run is None:
                    continue
                test = run["test"]
                # Test degradation is reported relative to the seed-matched test baseline when present.
                reference = payload["seed_matched_test_baseline"].get(str(seed), {})
                ref_ade = reference.get("trajectory_ade_pixel")
                degradation = None if ref_ade is None else test["trajectory_ade_pixel"] / ref_ade - 1.0
                lines.append(
                    f"| {seed} | {mode} | {fmt(test.get('intent_auc'))} | {fmt(test.get('trajectory_ade_pixel'), 3)} | {fmt(test.get('trajectory_fde_pixel'), 3)} | {fmt(degradation * 100, 2) if degradation is not None else 'NA'}% | {test.get('trajectory_ade_pixel', float('inf')) <= (ref_ade * 1.05 if ref_ade else -1)} |"
                )

    lines.extend(["", "## 4. Uncertainty diagnostic", ""])
    diagnostic = payload.get("uncertainty_diagnostic")
    if diagnostic:
        lines.append(
            f"Seed {diagnostic['seed']}: ambiguous-vs-clean entropy AUROC {diagnostic['entropy_auroc_ambiguous_vs_clean']:.4f}; gate AUROC {diagnostic['gate_auroc_ambiguous_vs_clean']:.4f}. This is a diagnostic proxy on the ambiguous annotation subset, not uncertainty ground truth."
        )
        lines.append(
            f"Clean entropy {diagnostic['clean_entropy']['mean']:.4f}±{diagnostic['clean_entropy']['std']:.4f}; ambiguous entropy {diagnostic['ambiguous_entropy']['mean']:.4f}±{diagnostic['ambiguous_entropy']['std']:.4f}."
        )
        lines.append(
            f"Clean gate {diagnostic['clean_gate']['mean']:.4f}±{diagnostic['clean_gate']['std']:.4f}; ambiguous gate {diagnostic['ambiguous_gate']['mean']:.4f}±{diagnostic['ambiguous_gate']['std']:.4f}."
        )
    else:
        lines.append("No clean-vs-ambiguous diagnostic file found.")

    lines.extend(["", "## 5. Uncertainty-stratified analysis", ""])
    stratified = payload.get("uncertainty_stratified")
    if stratified:
        lines.extend(
            [
                "Strata use the seed-matched uncertainty model's clean-test entropy terciles; compare all three Stage A models on the same samples.",
                "",
                "| Stratum | N | Model | AUC | Brier | ADE px | FDE px |",
                "|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for name, group in stratified["strata"].items():
            for mode in MODES:
                metrics = group.get("models", {}).get(mode)
                if not metrics:
                    continue
                lines.append(
                    f"| {name} | {group['sample_count']} | {mode} | {fmt(metrics.get('auc'))} | {fmt(metrics.get('brier'))} | {fmt(metrics.get('ade_pixel'), 3)} | {fmt(metrics.get('fde_pixel'), 3)} |"
                )
        lines.extend(["", "Interpretation must be based on the stratum-level comparisons and sample counts; a single seed is exploratory."])
    else:
        lines.append("No uncertainty-stratified analysis file found.")

    lines.extend(
        [
            "",
            "## Current decision",
            "",
            payload["decision"],
            "",
            "The Transformer baseline remains frozen. No backbone fine-tuning or next-stage model work was started.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-md", type=Path, default=PROJECT_ROOT / "results/residual_social_joint_summary.md")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results/residual_social_joint_summary.json")
    parser.add_argument("--decision", default="")
    args = parser.parse_args()

    stage_a_raw = run_metrics(args.results_root, "A")
    stage_b_raw = run_metrics(args.results_root, "B")
    stage_a = compact_runs(stage_a_raw)
    stage_b = compact_runs(stage_b_raw)
    baseline_path = args.results_root / "residual_social_joint_validation/baseline_reproduction_seed123.json"
    baseline = read_json(baseline_path)
    diagnostic_path = args.results_root / "residual_social_analysis/clean_ambiguous_diagnostic_seed123.json"
    stratified_path = args.results_root / "residual_social_analysis/uncertainty_stratified_metrics.json"
    diagnostic = read_json(diagnostic_path)
    stratified = read_json(stratified_path)
    baseline_test_by_seed: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        path = args.results_root / f"trajectory_transformer_scene_15x15_seed{seed}" / "metrics.json"
        metrics = read_json(path)
        if metrics is not None:
            baseline_test_by_seed[str(seed)] = metrics["test"]

    if not args.decision:
        none_auc = stage_a["none"].get("123", {}).get("test", {}).get("intent_auc")
        always_auc = stage_a["always"].get("123", {}).get("test", {}).get("intent_auc")
        uncertainty_auc = stage_a["uncertainty"].get("123", {}).get("test", {}).get("intent_auc")
        if None not in (none_auc, always_auc, uncertainty_auc):
            args.decision = (
                f"Stage A seed123 test AUC: none={none_auc:.4f}, always={always_auc:.4f}, "
                f"uncertainty={uncertainty_auc:.4f}. The predefined stop condition fired: "
                "uncertainty did not exceed no-social and always-social did not outperform no-social. "
                "Therefore no multi-seed Stage A expansion or Stage B run was started."
            )
        else:
            args.decision = "Stage A comparison is incomplete; no automatic conclusion is made."

    payload = {
        "protocol": {
            "main_repository": "https://github.com/duchangchen11/paper_one",
            "stage_a_seeds_requested": list(SEEDS),
            "stage_a_runs_completed": {mode: sorted(runs) for mode, runs in stage_a.items()},
            "stage_b_runs_completed": {mode: sorted(runs) for mode, runs in stage_b.items()},
            "frozen_backbone": True,
            "ambiguous_training_supervision": False,
            "trajectory_unit": "pixel",
        },
        "frozen_transformer_baseline": baseline,
        "seed_matched_test_baseline": baseline_test_by_seed,
        "stage_a": stage_a,
        "stage_a_aggregates": multiseed_aggregates(stage_a),
        "stage_b": stage_b,
        "stage_b_aggregates": multiseed_aggregates(stage_b),
        "uncertainty_diagnostic": diagnostic,
        "uncertainty_stratified": stratified,
        "decision": args.decision,
    }
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
