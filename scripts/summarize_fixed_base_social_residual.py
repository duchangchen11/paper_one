#!/usr/bin/env python3
"""Create the concise seed123 report from fixed-base experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def f(value: float) -> str:
    return f"{value:.4f}"


def build_summary(seed: int, results_root: Path) -> dict[str, Any]:
    base = read_json(results_root / f"fixed_base_intent_seed{seed}" / "metrics.json")
    always = read_json(results_root / f"fixed_base_social_always_seed{seed}" / "metrics.json")
    uncertainty = read_json(
        results_root / f"fixed_base_social_uncertainty_seed{seed}" / "metrics.json"
    )
    evaluation = read_json(
        results_root / "fixed_base_social_analysis" / f"evaluation_seed{seed}.json"
    )
    shuffle = read_json(
        results_root / "fixed_base_social_analysis" / "neighbor_shuffle_diagnostic.json"
    )

    evaluation_models = evaluation["models"]
    names = {"base": "Base", "always": "Always", "uncertainty": "Uncertainty"}
    raw_metrics = {"base": base["test"], "always": always["test"], "uncertainty": uncertainty["test"]}
    model_rows: dict[str, dict[str, Any]] = {}
    for name, metrics in raw_metrics.items():
        eval_metrics = evaluation_models[name]
        row: dict[str, Any] = {
            key: float(eval_metrics[key])
            for key in (
                "auc",
                "balanced_accuracy",
                "f1",
                "brier",
                "ece_10",
                "ade_pixel",
                "fde_pixel",
            )
        }
        if name != "base":
            row.update(
                {
                    "gate_mean": float(metrics["gate_distribution"]["mean"]),
                    "gate_std": float(metrics["gate_distribution"]["std"]),
                    "delta_logit_mean": float(metrics["delta_logit_distribution"]["mean"]),
                    "delta_logit_std": float(metrics["delta_logit_distribution"]["std"]),
                    "paired_bce": eval_metrics["paired_bce_improvement"],
                }
            )
        model_rows[name] = row

    entropy_rows = {}
    for stratum, payload in evaluation["uncertainty_stratification"]["strata"].items():
        entropy_rows[stratum] = {
            mode: {
                "sample_count": int(payload["sample_count"]),
                "auc": float(payload["metrics"][mode]["auc"]),
                "brier": float(payload["metrics"][mode]["brier"]),
                "mean_paired_bce_improvement": float(
                    payload["metrics"][mode]["mean_improvement"]
                ),
                "helped_sample_ratio": float(
                    payload["metrics"][mode]["helped_sample_ratio"]
                ),
                "hurt_sample_ratio": float(
                    payload["metrics"][mode]["hurt_sample_ratio"]
                ),
            }
            for mode in ("base", "always", "uncertainty")
        }

    neighbor_rows = {}
    for group, payload in evaluation["neighbor_count_stratification"].items():
        neighbor_rows[group] = {
            "sample_count": int(payload["sample_count"]),
            "models": {
                mode: {
                    "auc": float(payload["models"][mode]["auc"]),
                    "brier": float(payload["models"][mode]["brier"]),
                    "mean_paired_bce_improvement": float(
                        payload["models"][mode]["paired_bce_improvement"]["mean_improvement"]
                    ),
                    "helped_sample_ratio": float(
                        payload["models"][mode]["paired_bce_improvement"]["helped_sample_ratio"]
                    ),
                    "hurt_sample_ratio": float(
                        payload["models"][mode]["paired_bce_improvement"]["hurt_sample_ratio"]
                    ),
                }
                for mode in ("base", "always", "uncertainty")
            },
        }

    summary = {
        "seed": seed,
        "test_sample_count": evaluation["test_sample_count"],
        "base_checkpoint": base["checkpoint"],
        "trajectory_checkpoint": base["trajectory_checkpoint"],
        "calibration": {
            "temperature": float(base["temperature"]),
            "fit_split": base["calibration_fit_split"],
            "validation_before": base["validation_calibration"]["before"],
            "validation_after": base["validation_calibration"]["after"],
            "test_before": base["test_calibration"]["before"],
            "test_after": base["test_calibration"]["after"],
        },
        "models": model_rows,
        "shared_base_logit_exactly_equal": evaluation["base_logit_identical_across_modes"],
        "base_and_transformer_frozen_for_social": evaluation[
            "base_classifier_and_transformer_frozen"
        ],
        "paired_bce_improvement": evaluation["paired_improvement"],
        "uncertainty_stratification": {
            "shared_base_entropy_cutpoints": evaluation["uncertainty_stratification"][
                "cutpoints"
            ],
            "strata": entropy_rows,
        },
        "neighbor_count_stratification": neighbor_rows,
        "neighbor_shuffle": shuffle["models"],
        "trajectory_unchanged_by_social": {
            "ade_pixel": float(evaluation_models["base"]["ade_pixel"]),
            "fde_pixel": float(evaluation_models["base"]["fde_pixel"]),
            "same_for_all_modes": all(
                evaluation_models[mode]["ade_pixel"] == evaluation_models["base"]["ade_pixel"]
                and evaluation_models[mode]["fde_pixel"] == evaluation_models["base"]["fde_pixel"]
                for mode in ("always", "uncertainty")
            ),
        },
        "stop_decision": {
            "stop_after_seed123": True,
            "run_additional_seeds": False,
            "reason": (
                "Both social AUCs are below the shared base; mean paired BCE improvements are "
                "negative overall and in the high-uncertainty stratum. The >=4-neighbor "
                "uncertainty subgroup has a positive mean but most samples are still hurt, so "
                "it is not sufficient positive evidence under the pre-registered stop rule."
            ),
        },
    }
    return summary


def to_markdown(summary: dict[str, Any]) -> str:
    rows = summary["models"]
    labels = {"base": "Base intent", "always": "Always social", "uncertainty": "Uncertainty social"}
    out = [
        "# Fixed-base social residual: seed123 report",
        "",
        f"Test split: {summary['test_sample_count']:,} samples. Both social models share the same fixed base checkpoint and calibrated entropy.",
        "",
        "## Test metrics",
        "",
        "| Model | AUC ↑ | BAcc ↑ | F1 ↑ | Brier ↓ | ECE ↓ | ADE (px) ↓ | FDE (px) ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("base", "always", "uncertainty"):
        m = rows[key]
        out.append(
            f"| {labels[key]} | {f(m['auc'])} | {f(m['balanced_accuracy'])} | "
            f"{f(m['f1'])} | {f(m['brier'])} | {f(m['ece_10'])} | "
            f"{m['ade_pixel']:.3f} | {m['fde_pixel']:.3f} |"
        )

    calibration = summary["calibration"]
    out += [
        "",
        "## Calibration and controlled comparison",
        "",
        f"Temperature fitted on validation only: **{calibration['temperature']:.6f}**. "
        f"Validation Brier/ECE: {f(calibration['validation_before']['brier'])}/"
        f"{f(calibration['validation_before']['ece_10'])} → "
        f"{f(calibration['validation_after']['brier'])}/"
        f"{f(calibration['validation_after']['ece_10'])}. Test Brier/ECE: "
        f"{f(calibration['test_before']['brier'])}/{f(calibration['test_before']['ece_10'])} → "
        f"{f(calibration['test_after']['brier'])}/{f(calibration['test_after']['ece_10'])}.",
        "",
        f"Base logits are exactly identical across none/always/uncertainty: **{all(v['identical'] for v in summary['shared_base_logit_exactly_equal'].values())}** (maximum absolute difference 0). The Transformer and intent head remain frozen during social training.",
        "",
        "| Social mode | Gate mean ± std | Delta-logit mean ± std | Mean paired BCE improvement ↑ | Helped | Hurt |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("always", "uncertainty"):
        m = rows[key]
        paired = m["paired_bce"]
        out.append(
            f"| {labels[key]} | {m['gate_mean']:.4f} ± {m['gate_std']:.4f} | "
            f"{m['delta_logit_mean']:.4f} ± {m['delta_logit_std']:.4f} | "
            f"{paired['mean_improvement']:.5f} | {pct(paired['helped_sample_ratio'])} | "
            f"{pct(paired['hurt_sample_ratio'])} |"
        )

    out += [
        "",
        "## Shared-base entropy strata",
        "",
        "Positive paired BCE improvement means the social model reduced that sample's BCE versus base.",
        "",
        "| Entropy | N | Model | AUC | Brier | Mean paired BCE gain | Helped | Hurt |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for stratum in ("low", "medium", "high"):
        payload = summary["uncertainty_stratification"]["strata"][stratum]
        for mode in ("base", "always", "uncertainty"):
            values = payload[mode]
            out.append(
                f"| {stratum} | {values['sample_count']} | {labels[mode]} | {f(values['auc'])} | "
                f"{f(values['brier'])} | {values['mean_paired_bce_improvement']:.5f} | "
                f"{pct(values['helped_sample_ratio'])} | {pct(values['hurt_sample_ratio'])} |"
            )

    out += [
        "",
        "## Neighbor-count strata",
        "",
        "| Neighbors | N | Model | AUC | Brier | Mean paired BCE gain | Helped | Hurt |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("0", "1", "2-3", ">=4"):
        payload = summary["neighbor_count_stratification"][group]
        for mode in ("base", "always", "uncertainty"):
            values = payload["models"][mode]
            out.append(
                f"| {group} | {payload['sample_count']} | {labels[mode]} | {f(values['auc'])} | "
                f"{f(values['brier'])} | {values['mean_paired_bce_improvement']:.5f} | "
                f"{pct(values['helped_sample_ratio'])} | {pct(values['hurt_sample_ratio'])} |"
            )

    out += [
        "",
        "## Neighbor-shuffle diagnostic",
        "",
        "Target, scene, and label were held fixed while complete neighbor tensors and masks were permuted across test samples.",
        "",
        "| Model | Real AUC | Shuffled AUC | Δ AUC (shuffled-real) | Real Brier | Shuffled Brier | Δ Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("none", "always", "uncertainty"):
        d = summary["neighbor_shuffle"][mode]
        out.append(
            f"| {mode} | {f(d['real_neighbor']['auc'])} | {f(d['shuffled_neighbor']['auc'])} | "
            f"{d['auc_delta_shuffled_minus_real']:+.4f} | {f(d['real_neighbor']['brier'])} | "
            f"{f(d['shuffled_neighbor']['brier'])} | {d['brier_delta_shuffled_minus_real']:+.4f} |"
        )

    trajectory = summary["trajectory_unchanged_by_social"]
    out += [
        "",
        "## Decision",
        "",
        f"**Stop after seed123; do not run additional seeds or Stage B.** {summary['stop_decision']['reason']}",
        "",
        f"Trajectory output is unchanged by social residuals: ADE {trajectory['ade_pixel']:.3f}px, FDE {trajectory['fde_pixel']:.3f}px. This is consistent with the frozen-trajectory design; social residuals only affect intent logits.",
        "",
        "Interpretation: on this seed/test split, social residuals did not improve overall intent prediction. Uncertainty gating had a smaller AUC drop than always-on fusion, but it also remained below the base and its mean paired BCE gain was negative; therefore this is not evidence that social interaction improves prediction.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--json-output", type=Path, default=PROJECT_ROOT / "results/fixed_base_social_summary.json")
    parser.add_argument("--markdown-output", type=Path, default=PROJECT_ROOT / "results/fixed_base_social_summary.md")
    args = parser.parse_args()
    summary = build_summary(args.seed, args.results_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(to_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
