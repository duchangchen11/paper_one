#!/usr/bin/env python3
"""Render concise, validation-selected reports for trajectory reliability audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_trajectory_reliability import SCORE_LABELS, SCORE_NAMES


OUT_ROOT = PROJECT_ROOT / "results/trajectory_reliability_audit"
SEEDS = (42, 123, 2024)


def read_json(name: str) -> dict[str, Any]:
    return json.loads((OUT_ROOT / name).read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def corr_cell(value: dict[str, Any], digits: int = 3) -> str:
    rho = value.get("rho")
    p_value = value.get("p_value")
    if rho is None:
        return "NA"
    p = "<1e-300" if p_value == 0 else f"{p_value:.2g}"
    return f"{rho:.{digits}f} (p={p})"


def build_summary() -> tuple[dict[str, Any], str]:
    validation = read_json("validation_metrics.json")
    test = read_json("test_metrics.json")
    selection = read_json("selection.json")
    thresholds = read_json("high_error_thresholds.json")
    bins = read_json("reliability_bins.json")
    risk = read_json("risk_coverage.json")
    horizon = read_json("horizon_analysis.json")
    bootstrap = read_json("bootstrap.json")
    motion = read_json("motion_confounds.json")
    checkpoint_audit = read_json("checkpoint_audit.json")
    decision = read_json("decision.json")
    selected = selection["selected_score"]
    selected_val_corr = validation["uncertainty_error_correlations"][selected]
    selected_test_corr = test["uncertainty_error_correlations"][selected]
    selected_high = test["high_error_detection"][selected]

    validation_score_rows = []
    for name in SCORE_NAMES:
        correlations = validation["uncertainty_error_correlations"][name]
        high = validation["high_error_detection"][name]
        validation_score_rows.append({
            "score": name,
            "label": SCORE_LABELS[name],
            "spearman_ade": correlations["spearman_vs_ensemble_ade"],
            "spearman_fde": correlations["spearman_vs_ensemble_fde"],
            "high_ade_auroc": high["high_ade_fixed_validation_threshold"]["auroc"],
            "high_fde_auroc": high["high_fde_fixed_validation_threshold"]["auroc"],
            "high_ade_auprc": high["high_ade_fixed_validation_threshold"]["auprc"],
            "high_fde_auprc": high["high_fde_fixed_validation_threshold"]["auprc"],
        })

    ensemble = {
        split: {
            "individual_model_performance": report["individual_model_performance"],
            "ensemble_mean_performance": report["ensemble_mean_performance"],
            "historical_reproduction": report["historical_reproduction"],
            "all_individual_pixel_metrics_reproduced": report["all_individual_pixel_metrics_reproduced"],
        }
        for split, report in (("validation", validation), ("test", test))
    }
    output = {
        "purpose": "feasibility audit only; no uncertainty predictor or intention model was trained",
        "ensemble_performance": ensemble,
        "validation_score_selection": {
            **selection,
            "all_scores": validation_score_rows,
            "high_error_thresholds_pixel": thresholds,
        },
        "test_reliability": {
            "primary_score": selected,
            "spearman_ade": selected_test_corr["spearman_vs_ensemble_ade"],
            "spearman_fde": selected_test_corr["spearman_vs_ensemble_fde"],
            "pearson_ade_secondary": selected_test_corr["pearson_vs_ensemble_ade"],
            "pearson_fde_secondary": selected_test_corr["pearson_vs_ensemble_fde"],
            "spearman_ade_bootstrap_95_ci": bootstrap["spearman_selected_score_vs_ensemble_ade"],
            "high_ade_fixed_validation_threshold": selected_high["high_ade_fixed_validation_threshold"],
            "high_ade_auroc_bootstrap_95_ci": bootstrap["high_ade_auroc_fixed_validation_threshold"],
            "high_fde_fixed_validation_threshold": selected_high["high_fde_fixed_validation_threshold"],
            "secondary_test_top20_percent_ade": selected_high["secondary_test_top20_percent_ade"],
            "secondary_test_top20_percent_fde": selected_high["secondary_test_top20_percent_fde"],
        },
        "reliability_bins": bins,
        "risk_coverage": risk,
        "motion_confounds": motion,
        "horizon_analysis": horizon,
        "checkpoint_audit": checkpoint_audit,
        "final_decision": decision,
    }

    model_rows = []
    for seed in SEEDS:
        val_item = validation["individual_model_performance"][str(seed)]
        test_item = test["individual_model_performance"][str(seed)]
        model_rows.append(
            f"| seed {seed} | {val_item['ade_pixel']:.3f}/{val_item['fde_pixel']:.3f} | "
            f"{test_item['ade_pixel']:.3f}/{test_item['fde_pixel']:.3f} |"
        )
    val_ens = validation["ensemble_mean_performance"]
    test_ens = test["ensemble_mean_performance"]
    model_rows.append(
        f"| Ensemble mean | {val_ens['ade_pixel']:.3f}/{val_ens['fde_pixel']:.3f} | "
        f"{test_ens['ade_pixel']:.3f}/{test_ens['fde_pixel']:.3f} |"
    )

    selection_rows = []
    for item in validation_score_rows:
        selection_rows.append(
            f"| {item['score']} | {corr_cell(item['spearman_ade'])} | {corr_cell(item['spearman_fde'])} | "
            f"{fmt(item['high_ade_auroc'], 3)} | {fmt(item['high_fde_auroc'], 3)} |"
        )

    test_bins = bins["test_bins_using_fixed_validation_cutpoints"]
    bin_rows = [
        f"| {item['bin']} | {item['sample_count']} | {fmt(item['mean_uncertainty'])} | "
        f"{fmt(item['mean_ade'])} | {fmt(item['median_ade'])} | {fmt(item['mean_fde'])} | {fmt(item['median_fde'])} |"
        for item in test_bins
    ]
    risk_rows = []
    for row in risk["rows"]:
        random_ade = row["random_100_mean_sample_std"]["ade"]
        random_fde = row["random_100_mean_sample_std"]["fde"]
        risk_rows.append(
            f"| {row['nominal_coverage']:.0%} | {row['selected']['ade']:.3f}/{row['selected']['fde']:.3f} | "
            f"{random_ade['mean']:.3f}±{random_ade['sample_std']:.3f} / "
            f"{random_fde['mean']:.3f}±{random_fde['sample_std']:.3f} | "
            f"{row['oracle_by_true_ade']['ade']:.3f}/{row['oracle_by_true_ade']['fde']:.3f} |"
        )
    horizon_rows = [
        f"| {row['horizon']} | {row['mean_ensemble_point_error_pixel']:.3f} | "
        f"{row['mean_disagreement_pixel']:.3f} | {corr_cell(row['spearman_disagreement_vs_point_error'])} |"
        for row in horizon["test"]
    ]
    motion_rows = [
        f"| {row['stratum']} | {row['sample_count']} | {fmt(row['observed_motion_mean_pixel'])} | "
        f"{corr_cell(row['spearman_selected_uncertainty_vs_ade'])} |"
        for row in motion["test_strata"]
    ]

    rho_ci = bootstrap["spearman_selected_score_vs_ensemble_ade"]
    auc_ci = bootstrap["high_ade_auroc_fixed_validation_threshold"]
    high_ade = selected_high["high_ade_fixed_validation_threshold"]
    high_fde = selected_high["high_fde_fixed_validation_threshold"]
    md = [
        "# Zero-scene trajectory ensemble reliability audit",
        "",
        "Feasibility audit only. No trajectory was retrained; no uncertainty predictor, intention classifier, social input, or scene input was used. All errors and disagreement scores are in pixels.",
        "",
        "## Ensemble performance",
        "",
        "| Model | Validation ADE/FDE px | Test ADE/FDE px |",
        "|---|---:|---:|",
        *model_rows,
        "",
        f"All three individual pixel ADE/FDE values reproduced their saved experiment metrics within 1e-4 px: "
        f"validation `{validation['all_individual_pixel_metrics_reproduced']}`, test `{test['all_individual_pixel_metrics_reproduced']}`.",
        "",
        "## Validation score selection",
        "",
        "High-error thresholds are the validation 80th percentiles and are frozen in pixel units before test evaluation.",
        f"ADE threshold = `{thresholds['high_ade_pixel_threshold']:.4f} px`; FDE threshold = `{thresholds['high_fde_pixel_threshold']:.4f} px`.",
        "",
        "| Score | Spearman vs ADE (rho, p) | Spearman vs FDE (rho, p) | High-ADE AUROC | High-FDE AUROC |",
        "|---|---:|---:|---:|---:|",
        *selection_rows,
        "",
        f"Primary score selected using validation only: **{selected} — {SCORE_LABELS[selected]}**. {selection['selection_rule']}.",
        "",
        "## Test reliability of selected score",
        "",
        f"- Spearman vs ADE: {corr_cell(selected_test_corr['spearman_vs_ensemble_ade'])}; bootstrap 95% CI `[{rho_ci['lower_95']:.3f}, {rho_ci['upper_95']:.3f}]` ({rho_ci['valid_replicates']}/{bootstrap['requested_repetitions']} valid replicates).",
        f"- Spearman vs FDE: {corr_cell(selected_test_corr['spearman_vs_ensemble_fde'])}.",
        f"- High-ADE (fixed validation threshold) AUROC/AUPRC: {fmt(high_ade['auroc'])}/{fmt(high_ade['auprc'])}; AUROC bootstrap 95% CI `[{auc_ci['lower_95']:.3f}, {auc_ci['upper_95']:.3f}]`.",
        f"- High-FDE (fixed validation threshold) AUROC/AUPRC: {fmt(high_fde['auroc'])}/{fmt(high_fde['auprc'])}.",
        f"- Secondary test-top20% diagnostic AUROC (not used for selection): ADE `{fmt(selected_high['secondary_test_top20_percent_ade']['auroc'])}`, FDE `{fmt(selected_high['secondary_test_top20_percent_fde']['auroc'])}`.",
        "",
        "## Reliability bins",
        "",
        f"Low/medium/high cutpoints were validation q33=`{bins['tertile_cutpoints_from_validation']['q33']:.4f}` and q67=`{bins['tertile_cutpoints_from_validation']['q67']:.4f}`; applied unchanged to test.",
        "",
        "| Test bin | N | Mean U | Mean ADE | Median ADE | Mean FDE | Median FDE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *bin_rows,
        f"- Low→medium→high error monotonicity: ADE `{bins['test_error_monotonic_low_to_high']['ade']}`, FDE `{bins['test_error_monotonic_low_to_high']['fde']}`.",
        "- Validation-derived uncertainty decile boundaries and validation/test bin statistics are in `reliability_bins.csv`.",
        "",
        "## Risk-coverage",
        "",
        "Keep the lowest-uncertainty fraction. Random reference is mean±sample-SD over 100 rankings (seed 9124); oracle ranks by true ADE and is not deployable.",
        "",
        "| Coverage | Selected ADE/FDE | Random ADE / FDE | Oracle-ADE-ranked ADE/FDE |",
        "|---:|---:|---:|---:|",
        *risk_rows,
        f"- Selected ADE curve non-increasing as coverage drops: `{risk['selected_curve_monotonic_improvement']['ade']}`; FDE: `{risk['selected_curve_monotonic_improvement']['fde']}`.",
        f"- Selected 100%→20% reduction: ADE `{risk['selected_ade_reduction_100_to_20']:.3f}px`, FDE `{risk['selected_fde_reduction_100_to_20']:.3f}px`.",
        "",
        "## Motion-magnitude confound",
        "",
        f"Observed displacement is the pixel distance between first/last observed target centers. Selected-score test Spearman vs observed displacement: {corr_cell(motion['selected_score_global_test_spearman_vs_observed_motion'])}; vs GT future endpoint displacement (post-hoc diagnostic only): {corr_cell(motion['selected_score_global_test_spearman_vs_gt_future_endpoint_displacement_diagnostic_only'])}.",
        f"Slow/medium/fast strata use validation motion q33=`{motion['slow_medium_fast_cutpoints_from_validation_pixel']['q33']:.3f}px`, q67=`{motion['slow_medium_fast_cutpoints_from_validation_pixel']['q67']:.3f}px`.",
        "",
        "| Test motion stratum | N | Mean motion px | Selected U vs ADE Spearman |",
        "|---|---:|---:|---:|",
        *motion_rows,
        "",
        "## Horizon analysis",
        "",
        "| t | Mean point error px | Mean disagreement px | Spearman(disagreement, point error) |",
        "|---:|---:|---:|---:|",
        *horizon_rows,
        "",
        f"## Final decision: {decision['decision']}",
        "",
        decision["criteria_note"],
        "",
        "This is a feasibility audit only; it does not establish that trajectory reliability should gate crossing-intention evidence.",
    ]
    return output, "\n".join(md) + "\n"


def main() -> None:
    result, markdown = build_summary()
    (OUT_ROOT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUT_ROOT / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "primary_score": result["validation_score_selection"]["selected_score"],
        "decision": result["final_decision"]["decision"],
        "summary": str(OUT_ROOT / "summary.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
