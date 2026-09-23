#!/usr/bin/env python3
"""Summarize balanced, natural, and visibility-aware social protocols for seed123."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def condition_metrics(
    val_auc: float | None,
    test_metrics: dict[str, Any],
    paired: dict[str, Any],
    delta: dict[str, Any],
    shuffle: dict[str, Any] | None,
    **extra: Any,
) -> dict[str, Any]:
    normalized_pair = {
        "mean": float(paired.get("mean", paired.get("mean_improvement", 0.0))),
        "median": float(paired.get("median", paired.get("median_improvement", 0.0))),
        "helped_ratio": float(
            paired.get("helped_ratio", paired.get("helped_sample_ratio", 0.0))
        ),
        "hurt_ratio": float(
            paired.get("hurt_ratio", paired.get("hurt_sample_ratio", 0.0))
        ),
        "unchanged_ratio": float(
            paired.get("unchanged_ratio", paired.get("unchanged_sample_ratio", 0.0))
        ),
    }
    return {
        "validation_auc": val_auc,
        "test": {
            key: float(test_metrics[key])
            for key in ("auc", "balanced_accuracy", "f1", "brier", "ece_10", "ade_pixel", "fde_pixel")
            if key in test_metrics
        },
        "paired_bce_improvement": normalized_pair,
        "delta_logit": delta,
        "neighbor_shuffle": shuffle,
        **extra,
    }


def old_shuffle_payload(old_shuffle: dict[str, Any], mode: str) -> dict[str, Any]:
    d = old_shuffle["models"][mode]
    return {
        "real_auc": d["real_neighbor"]["auc"],
        "shuffled_auc": d["shuffled_neighbor"]["auc"],
        "delta_auc_shuffled_minus_real": d["auc_delta_shuffled_minus_real"],
        "real_brier": d["real_neighbor"]["brier"],
        "shuffled_brier": d["shuffled_neighbor"]["brier"],
        "delta_brier_shuffled_minus_real": d["brier_delta_shuffled_minus_real"],
    }


def new_shuffle_payload(new_shuffle: dict[str, Any], key: str) -> dict[str, Any]:
    d = new_shuffle["models"][key]
    return {
        "real_auc": d["real_neighbor"]["auc"],
        "shuffled_auc": d["shuffled_neighbor"]["auc"],
        "delta_auc_shuffled_minus_real": d["auc_delta_shuffled_minus_real"],
        "real_brier": d["real_neighbor"]["brier"],
        "shuffled_brier": d["shuffled_neighbor"]["brier"],
        "delta_brier_shuffled_minus_real": d["brier_delta_shuffled_minus_real"],
    }


def best_nonzero_epoch(metrics: dict[str, Any]) -> dict[str, Any] | None:
    later = [entry for entry in metrics["history"] if entry["epoch"] > 0]
    if not later:
        return None
    selected = max(later, key=lambda entry: entry["val"]["auc"])
    return {
        "epoch": int(selected["epoch"]),
        "validation_auc": float(selected["val"]["auc"]),
        "delta_logit_mean": float(selected["val"]["delta_logit_distribution"]["mean"]),
        "delta_logit_std": float(selected["val"]["delta_logit_distribution"]["std"]),
    }


def build_summary() -> dict[str, Any]:
    base = load_json(RESULTS / "fixed_base_intent_seed123/metrics.json")
    old_eval = load_json(RESULTS / "fixed_base_social_analysis/evaluation_seed123.json")
    old_shuffle = load_json(RESULTS / "fixed_base_social_analysis/neighbor_shuffle_diagnostic.json")
    repaired_eval = load_json(RESULTS / "social_protocol_repair_analysis/evaluation_seed123.json")

    base_eval = old_eval["models"]["base"]
    zero_pair = {
        "mean": 0.0,
        "median": 0.0,
        "helped_ratio": 0.0,
        "hurt_ratio": 0.0,
        "unchanged_ratio": 1.0,
    }
    zero_delta = {"mean": 0.0, "std": 0.0, "abs_mean": 0.0}
    conditions: dict[str, dict[str, Any]] = {
        "base": condition_metrics(
            base["validation_calibration"]["after"]["auc"],
            base_eval,
            zero_pair,
            zero_delta,
            None,
            selected_epoch=None,
            sampling="fixed base; no social training",
        )
    }

    for mode in ("always", "uncertainty"):
        run = load_json(RESULTS / f"fixed_base_social_{mode}_seed123/metrics.json")
        evaluated = old_eval["models"][mode]
        conditions[f"balanced_{mode}"] = condition_metrics(
            run["best_validation_auc"],
            evaluated,
            evaluated["paired_bce_improvement"],
            {
                "mean": run["test"]["delta_logit_distribution"]["mean"],
                "std": run["test"]["delta_logit_distribution"]["std"],
                "abs_mean": run["test"]["delta_logit_abs_mean"],
            },
            old_shuffle_payload(old_shuffle, mode),
            selected_epoch=int(run["best_epoch"]),
            sampling="balanced",
            epoch0_validation_auc=None,
        )

    run_names = {
        "natural_always": ("natural", "always"),
        "natural_uncertainty": ("natural", "uncertainty"),
        "visible_always": ("visible", "always"),
        "visible_uncertainty": ("visible", "uncertainty"),
    }
    for key, (protocol, mode) in run_names.items():
        directory = (
            f"fixed_base_social_{mode}_natural_seed123"
            if protocol == "natural"
            else f"fixed_base_social_visible_{mode}_seed123"
        )
        run = load_json(RESULTS / directory / "metrics.json")
        evaluated = repaired_eval["models"][key]
        conditions[key] = condition_metrics(
            run["best_validation_auc"],
            evaluated,
            evaluated["paired_bce_improvement"],
            evaluated["delta_logit"],
            new_shuffle_payload(repaired_eval["neighbor_shuffle"], key),
            selected_epoch=int(run["selected_epoch"]),
            epoch0_validation_auc=float(run["epoch0_validation_auc"]),
            best_nonzero_epoch=best_nonzero_epoch(run),
            sampling="natural",
            visibility_aware=protocol == "visible",
            social_improvement_detected=bool(run["social_improvement_detected"]),
        )

    old_count = old_eval["neighbor_count_stratification"]
    new_count = repaired_eval["neighbor_count_stratification"]
    neighbor_count = {}
    for group in ("0", "1", "2-3", ">=4"):
        neighbor_count[group] = {
            "sample_count": int(old_count[group]["sample_count"]),
            "models": {
                "base": old_count[group]["models"]["base"],
                "balanced_always": old_count[group]["models"]["always"],
                "balanced_uncertainty": old_count[group]["models"]["uncertainty"],
                "natural_always": new_count[group]["models"]["natural_always"],
                "natural_uncertainty": new_count[group]["models"]["natural_uncertainty"],
                "visible_always": new_count[group]["models"]["visible_always"],
                "visible_uncertainty": new_count[group]["models"]["visible_uncertainty"],
            },
        }

    visibility = repaired_eval["visibility_stratification"]
    base_auc = conditions["base"]["test"]["auc"]
    visible_uncertainty = conditions["visible_uncertainty"]
    visible_shuffle = visible_uncertainty["neighbor_shuffle"]
    stop_auc = visible_uncertainty["test"]["auc"] <= base_auc
    stop_paired = visible_uncertainty["paired_bce_improvement"]["mean"] <= 0
    # The report stores shuffled-minus-real, so >= 0 means real neighbors were not better.
    stop_shuffle = visible_shuffle["delta_auc_shuffled_minus_real"] >= 0
    stop = stop_auc and stop_paired and stop_shuffle
    return {
        "seed": 123,
        "base_checkpoint": base["checkpoint"],
        "test_sample_count": repaired_eval["test_sample_count"],
        "conditions": conditions,
        "fixed_base_logits_identical": repaired_eval[
            "fixed_base_logits_identical_across_all_runs"
        ],
        "neighbor_count_stratification": neighbor_count,
        "visibility_stratification": visibility,
        "visibility_observation": {
            "eligible_samples": visibility["eligible_sample_count"],
            "zero_neighbor_samples_excluded": visibility["excluded_zero_neighbor_count"],
            "mean_valid_neighbor_visible_ratio": visibility["mean_ratio"],
            "q1": visibility["tercile_cutpoints"]["q1"],
            "q2": visibility["tercile_cutpoints"]["q2"],
            "tercile_cutpoints_tied": visibility["cutpoints_tied_and_bins_collapsed"],
            "stratum_counts": {
                name: item["sample_count"]
                for name, item in visibility["strata"].items()
            },
        },
        "sampling_shift_diagnostic": {
            "balanced_selected_delta_mean": {
                mode: conditions[f"balanced_{mode}"]["delta_logit"]["mean"]
                for mode in ("always", "uncertainty")
            },
            "natural_selected_delta_mean": {
                mode: conditions[f"natural_{mode}"]["delta_logit"]["mean"]
                for mode in ("always", "uncertainty")
            },
            "natural_best_nonzero_epoch": {
                mode: conditions[f"natural_{mode}"]["best_nonzero_epoch"]
                for mode in ("always", "uncertainty")
            },
            "interpretation": (
                "The balanced selected models have negative mean residual logits. Natural runs select epoch 0, so the deployed residual is exactly zero; their best nonzero epochs have positive residual-logit means but lower validation AUC than epoch 0."
            ),
        },
        "stop_rule": {
            "visible_uncertainty_test_auc_le_base": stop_auc,
            "visible_uncertainty_paired_bce_mean_nonpositive": stop_paired,
            "real_neighbor_not_better_than_shuffled": stop_shuffle,
            "stop_social_gru_residual_route": stop,
            "run_additional_seeds": False,
            "conclusion": "current trajectory-based social representation does not provide reliable incremental intent information under this protocol." if stop else "Stop and report evidence; do not launch additional seeds without direction from the lead AI.",
        },
        "trajectory": {
            "ade_pixel": float(base_eval["ade_pixel"]),
            "fde_pixel": float(base_eval["fde_pixel"]),
            "all_social_conditions_preserve_trajectory": all(
                abs(conditions[key]["test"]["ade_pixel"] - base_eval["ade_pixel"]) < 1e-6
                and abs(conditions[key]["test"]["fde_pixel"] - base_eval["fde_pixel"]) < 1e-6
                for key in conditions if key != "base"
            ),
        },
    }


def fmt(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def to_markdown(summary: dict[str, Any]) -> str:
    order = (
        "base", "balanced_always", "balanced_uncertainty", "natural_always",
        "natural_uncertainty", "visible_always", "visible_uncertainty",
    )
    labels = {
        "base": "Base",
        "balanced_always": "旧 balanced · Always",
        "balanced_uncertainty": "旧 balanced · Uncertainty",
        "natural_always": "Natural · Always",
        "natural_uncertainty": "Natural · Uncertainty",
        "visible_always": "Natural+visibility · Always",
        "visible_uncertainty": "Natural+visibility · Uncertainty",
    }
    rows = summary["conditions"]
    out = [
        "# Social protocol repair — seed123",
        "",
        "本轮只排查 social residual 的采样先验偏移与邻居不可见帧处理；所有社交模型使用同一 fixed-base checkpoint，轨迹网络冻结。旧结果保留，新增结果使用独立目录。",
        "",
        "## 七组对照",
        "",
        "| Condition | Val AUC | Selected epoch | Test AUC | Brier | ECE | Paired BCE gain | Helped / hurt | Δlogit mean ± std | Shuffle ΔAUC* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in order:
        c = rows[key]
        test = c["test"]
        pair = c["paired_bce_improvement"]
        delta = c["delta_logit"]
        shuffle = c["neighbor_shuffle"]
        selected_epoch = c.get("selected_epoch")
        out.append(
            f"| {labels[key]} | {fmt(c['validation_auc'])} | "
            f"{'—' if selected_epoch is None else selected_epoch} | "
            f"{fmt(test['auc'])} | {fmt(test['brier'])} | {fmt(test['ece_10'])} | "
            f"{pair['mean']:+.5f} | {pct(pair['helped_ratio'])} / {pct(pair['hurt_ratio'])} | "
            f"{delta['mean']:+.4f} ± {delta['std']:.4f} | "
            f"{fmt(shuffle['delta_auc_shuffled_minus_real'] if shuffle else None)} |"
        )
    out += [
        "",
        "*Shuffle ΔAUC is shuffled minus real; negative values mean real neighbors scored higher. Base has no social input. Old balanced shuffle values come from the prior diagnostic; new natural/visibility values permute neighbor_obs, neighbor_mask, and neighbor_visible_mask together.",
        "",
        "## Epoch-0 selection and sampling diagnosis",
        "",
        f"Natural always/uncertainty epoch-0 validation AUC: {fmt(rows['natural_always']['epoch0_validation_auc'])}. Both selected epoch **0** (best AUC {fmt(rows['natural_always']['validation_auc'])}); visible always/uncertainty also both selected epoch **0**. The selected models therefore preserve the base logits exactly and produce zero paired BCE change.",
        "",
        f"Old balanced selected test residual-logit means: always {rows['balanced_always']['delta_logit']['mean']:+.4f}, uncertainty {rows['balanced_uncertainty']['delta_logit']['mean']:+.4f}. Natural selected means are exactly zero because epoch 0 was retained. The best nonzero natural epochs had positive residual means (always {summary['sampling_shift_diagnostic']['natural_best_nonzero_epoch']['always']['delta_logit_mean']:+.4f}, uncertainty {summary['sampling_shift_diagnostic']['natural_best_nonzero_epoch']['uncertainty']['delta_logit_mean']:+.4f}) but validation AUC below the base. This removes the selected model's negative residual offset by falling back to zero, not by finding a better social residual.",
        "",
        "## Visibility strata",
        "",
        f"Among {summary['visibility_observation']['eligible_samples']:,} test samples with at least one valid neighbor, mean valid-neighbor visible-frame ratio is {summary['visibility_observation']['mean_valid_neighbor_visible_ratio']:.6f}. Tercile cutpoints are q1={summary['visibility_observation']['q1']:.4f}, q2={summary['visibility_observation']['q2']:.4f}; they tie, so the middle bin is empty rather than splitting identical visibility values arbitrarily. {summary['visibility_observation']['zero_neighbor_samples_excluded']:,} zero-neighbor samples are excluded.",
        "",
        "| Visibility stratum | N | Base AUC / Brier | Visible Always AUC / Brier | Visible Uncertainty AUC / Brier | Paired BCE gains (A/U) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("low_visibility", "medium_visibility", "high_visibility"):
        item = summary["visibility_stratification"]["strata"][name]
        models = item["models"]
        def metric_cell(model: str) -> str:
            values = models[model]
            if values.get("sample_count", item["sample_count"]) == 0 or "auc" not in values:
                return "—"
            return f"{values['auc']:.4f} / {values['brier']:.4f}"
        def gain_cell(model: str) -> str:
            values = models[model]
            return "—" if values.get("sample_count", item["sample_count"]) == 0 or "auc" not in values else f"{values['paired_bce_improvement']['mean']:+.4f}"
        out.append(
            f"| {name.replace('_', ' ')} | {item['sample_count']} | {metric_cell('base')} | "
            f"{metric_cell('visible_always')} | {metric_cell('visible_uncertainty')} | "
            f"{gain_cell('visible_always')} / {gain_cell('visible_uncertainty')} |"
        )

    out += [
        "",
        "## Neighbor-count strata",
        "",
        "Cells show AUC / Brier. The ≥4-neighbor subgroup remains exploratory and was not used to tune a gate.",
        "",
        "| Neighbors | N | Base (AUC/Brier) | Balanced Always · Uncertainty | Natural Always · Uncertainty | Visibility Always · Uncertainty |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in ("0", "1", "2-3", ">=4"):
        item = summary["neighbor_count_stratification"][group]
        models = item["models"]
        def cell(key: str) -> str:
            m = models[key]
            return f"{m['auc']:.4f} / {m['brier']:.4f}"
        out.append(
            f"| {group} | {item['sample_count']} | {cell('base')} | "
            f"Always {cell('balanced_always')} · Unc {cell('balanced_uncertainty')} | "
            f"Always {cell('natural_always')} · Unc {cell('natural_uncertainty')} | "
            f"Always {cell('visible_always')} · Unc {cell('visible_uncertainty')} |"
        )

    decision = summary["stop_rule"]
    out += [
        "",
        "## Decision",
        "",
        f"Stop current GRU social residual route: **{decision['stop_social_gru_residual_route']}**. Natural+visibility uncertainty test AUC ≤ Base: {decision['visible_uncertainty_test_auc_le_base']}; paired BCE mean ≤ 0: {decision['visible_uncertainty_paired_bce_mean_nonpositive']}; real neighbor better than shuffled: {not decision['real_neighbor_not_better_than_shuffled']}.",
        "",
        f"> {decision['conclusion']}",
        "",
        f"Trajectory remains ADE {summary['trajectory']['ade_pixel']:.4f}px / FDE {summary['trajectory']['fde_pixel']:.4f}px, unchanged across all seven conditions. No extra seeds were run.",
        "",
        "Detailed metrics and checkpoint-selection histories are in the companion JSON.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=RESULTS / "social_protocol_repair_summary.json")
    parser.add_argument("--markdown-output", type=Path, default=RESULTS / "social_protocol_repair_summary.md")
    args = parser.parse_args()
    summary = build_summary()
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(to_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.markdown_output}")
    print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
