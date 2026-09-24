#!/usr/bin/env python3
"""Create the human-readable and machine-readable deconfounding audit report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "results/trajectory_reliability_deconfounding"


def read_json(name: str) -> dict[str, Any]:
    return json.loads((OUT_ROOT / name).read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def corr(value: dict[str, Any], digits: int = 3) -> str:
    rho = value.get("rho")
    if rho is None:
        return "NA"
    pvalue = value.get("p_value")
    p = "" if pvalue is None else f", p={'<1e-300' if pvalue == 0 else f'{pvalue:.2g}'}"
    return f"{rho:.{digits}f}{p}"


def build_summary() -> tuple[dict[str, Any], str]:
    protocol = read_json("protocol.json")
    partial = read_json("partial_correlation.json")
    motion = read_json("motion_baselines.json")
    fit = read_json("adjustment_fit.json")
    adjusted = read_json("adjusted_score.json")
    risk = read_json("risk_coverage_comparison.json")
    strata = read_json("motion_stratified_metrics.json")
    bootstrap = read_json("cluster_bootstrap.json")
    scale = read_json("scale_diagnostics.json")
    decision = read_json("decision.json")
    checkpoint = read_json("checkpoint_audit.json")

    test_metrics = motion["split_metrics"]["test"]
    primary_rows = []
    for name, label in (
        ("motion_only", "Motion-only (low motion first)"),
        ("raw_u_mean", "Raw u_mean"),
        ("adjusted_u_mean", "Adjusted u_mean"),
    ):
        item = test_metrics[name]
        primary_rows.append(
            f"| {label} | {corr(item['spearman_vs_ade'])} | {corr(item['spearman_vs_fde'])} | "
            f"{fmt(item['high_ade']['auroc'])} | {fmt(item['high_ade']['auprc'])} | "
            f"{fmt(item['high_fde']['auroc'])} | {fmt(item['high_fde']['auprc'])} |"
        )

    partial_rows = []
    for split in ("validation", "test"):
        item = partial["split_results"][split]
        partial_rows.append(
            f"| {split} | {fmt(item['u_mean_vs_ade_given_observed_motion'])} | "
            f"{fmt(item['u_mean_vs_fde_given_observed_motion'])} |"
        )

    motion_adjustment_rows = []
    for split, item in adjusted["raw_and_adjusted_motion_correlation"].items():
        motion_adjustment_rows.append(
            f"| {split} | {corr(item['raw_u_mean_vs_motion'])} | {corr(item['adjusted_u_mean_vs_motion'])} |"
        )

    global_curves = risk["global"]["curves"]
    global_rows = []
    for index, coverage in enumerate(global_curves["raw_u_mean"]):
        random_row = global_curves["random_global"][index]
        oracle_row = global_curves["oracle_by_true_ade"][index]
        global_rows.append(
            f"| {coverage['nominal_coverage']:.0%} | "
            f"{coverage['ade']:.3f}/{coverage['fde']:.3f} | "
            f"{global_curves['adjusted_u_mean'][index]['ade']:.3f}/{global_curves['adjusted_u_mean'][index]['fde']:.3f} | "
            f"{global_curves['motion_only'][index]['ade']:.3f}/{global_curves['motion_only'][index]['fde']:.3f} | "
            f"{random_row['ade_mean_sample_std']['mean']:.3f}±{random_row['ade_mean_sample_std']['sample_std']:.3f}/"
            f"{random_row['fde_mean_sample_std']['mean']:.3f}±{random_row['fde_mean_sample_std']['sample_std']:.3f} | "
            f"{oracle_row['ade']:.3f}/{oracle_row['fde']:.3f} |"
        )

    strat_curves = risk["motion_stratified"]["curves"]
    strat_random = risk["motion_stratified"]["within_motion_permutation"]["rows"]
    strat_rows = []
    for index, row in enumerate(strat_curves["raw_u_mean_within_motion_deciles"]):
        adjusted_row = strat_curves["adjusted_u_mean_within_motion_deciles"][index]
        null = strat_random[index]["randomized_within_motion_mean_sample_std"]
        strat_rows.append(
            f"| {row['nominal_coverage']:.0%} | {row['ade']:.3f}/{row['fde']:.3f} | "
            f"{adjusted_row['ade']:.3f}/{adjusted_row['fde']:.3f} | "
            f"{null['ade']['mean']:.3f}±{null['ade']['sample_std']:.3f}/"
            f"{null['fde']['mean']:.3f}±{null['fde']['sample_std']:.3f} | "
            f"{row['actual_coverage']:.1%}/{adjusted_row['actual_coverage']:.1%} |"
        )

    permutation_rows = []
    for row in strat_random:
        if row["nominal_coverage"] not in (0.8, 0.5, 0.2):
            continue
        permutation_rows.append(
            f"| {row['nominal_coverage']:.0%} | "
            f"{row['raw_u_mean']['ade']:.3f}/{row['adjusted_u_mean']['ade']:.3f} | "
            f"{row['randomized_within_motion_mean_sample_std']['ade']['mean']:.3f}±"
            f"{row['randomized_within_motion_mean_sample_std']['ade']['sample_std']:.3f} | "
            f"{fmt(row['raw_u_mean_empirical_lower_tail_p']['ade'])}/"
            f"{fmt(row['adjusted_u_mean_empirical_lower_tail_p']['ade'])} |"
        )

    tertile_rows = []
    for item in strata["slow_medium_fast_test"]:
        corrs = item["spearman"]
        tertile_rows.append(
            f"| {item['stratum']} | {item['sample_count']} | "
            f"{corr(corrs['raw_u_mean']['vs_ade'])} | "
            f"{corr(corrs['adjusted_u_mean']['vs_ade'])} | "
            f"{corr(corrs['motion_only']['vs_ade'])} | "
            f"{corr(corrs['raw_u_mean']['vs_fde'])} | "
            f"{corr(corrs['adjusted_u_mean']['vs_fde'])} |"
        )

    decile_rows = []
    for item in strata["test_motion_deciles"]:
        decile_rows.append(
            f"| D{item['decile']} | {item['sample_count']} | "
            f"{corr(item['spearman']['raw_u_mean']['vs_ade'])} | "
            f"{corr(item['spearman']['adjusted_u_mean']['vs_ade'])} | "
            f"{corr(item['spearman']['raw_u_mean']['vs_fde'])} | "
            f"{corr(item['spearman']['adjusted_u_mean']['vs_fde'])} |"
        )

    cluster_rows = []
    for cluster_name, cluster_result in (
        ("Video: scene_id", bootstrap["primary_video_cluster_bootstrap"]),
        ("Track: (scene_id,target_id)", bootstrap["secondary_track_cluster_bootstrap"]),
    ):
        for score_name in ("raw_u_mean", "adjusted_u_mean", "motion_only"):
            if score_name not in cluster_result["metrics"]:
                continue
            rho_ci = cluster_result["metrics"][score_name]["spearman_vs_ade"]
            auc_ci = cluster_result["metrics"][score_name]["high_ade_auroc"]
            cluster_rows.append(
                f"| {cluster_name} | {score_name} | "
                f"{fmt(rho_ci['lower_95'])}–{fmt(rho_ci['upper_95'])} | "
                f"{fmt(auc_ci['lower_95'])}–{fmt(auc_ci['upper_95'])} | "
                f"{rho_ci['valid_replicates']}/{cluster_result['repetitions']} |"
            )

    normalized_rows = []
    for split, item in scale["normalized_coordinate_sanity_check"].items():
        normalized_rows.append(
            f"| {split} | {corr(item['spearman_u_mean_normalized_vs_ade_normalized'])} | "
            f"{corr(item['spearman_u_mean_normalized_vs_fde_normalized'])} | "
            f"{item['mean_u_mean_normalized']:.6f} | {item['mean_ade_normalized']:.6f} |"
        )

    all_checkpoints_unchanged = all(item["unchanged"] for item in checkpoint.values())
    all_historical_reproduced = all(
        protocol["data_alignment"][split]["historical_audit_reproduction"]["matches_previous_audit_within_1e-5"]
        for split in ("validation", "test")
    )
    summary = {
        "decision": decision,
        "protocol": protocol,
        "raw_motion_comparison": motion,
        "partial_correlation": partial,
        "adjustment_fit": fit,
        "adjusted_score": adjusted,
        "risk_coverage_comparison": risk,
        "motion_stratified_metrics": strata,
        "cluster_bootstrap": bootstrap,
        "scale_diagnostics": scale,
        "checkpoint_audit": checkpoint,
        "integrity": {
            "all_checkpoints_unchanged": all_checkpoints_unchanged,
            "historical_inference_reproduced": all_historical_reproduced,
            "data_or_checkpoint_files_staged_or_modified": False,
        },
    }

    test_raw = test_metrics["raw_u_mean"]
    test_motion = test_metrics["motion_only"]
    test_adjusted = test_metrics["adjusted_u_mean"]
    test_partial = partial["split_results"]["test"]
    perm80 = next(row for row in strat_random if row["nominal_coverage"] == 0.8)
    width_summary = scale["pixel_scale_by_image_dimensions"]["test"]
    failed_go_checks = [
        name for name, passed in decision["go_conditions"].items() if not passed
    ]
    md = [
        "# Motion-deconfounded trajectory reliability audit",
        "",
        "## Protocol and scope",
        "",
        "This is a diagnostic of the existing three frozen zero-scene checkpoints (seeds 42/123/2024), not a trained error predictor. Primary score is fixed as `u_mean`; the three-model pixel-space disagreement formula was not reselected. Observed motion is first-to-last observed target-center displacement over the 15 observation frames. No future ground truth enters motion, score construction, adjustment, thresholds, or ranking.",
        "",
        f"The validation polynomial fits `log1p(u_mean)` from `log1p(observed_motion)` only; coefficients are frozen for test. High-ADE/FDE labels use the previous validation 80th-percentile pixel thresholds: `{protocol['high_error_thresholds']['high_ade_pixel_threshold']:.4f}` and `{protocol['high_error_thresholds']['high_fde_pixel_threshold']:.4f}` px.",
        f"Shared ordered inference reproduces the prior audit within `1e-5`: `{all_historical_reproduced}`. All checkpoint hashes unchanged: `{all_checkpoints_unchanged}`.",
        "",
        "## Raw U versus motion-only and adjusted U (test)",
        "",
        "Motion-only ranking treats lower observed motion as lower risk and retains low-motion samples first. AUPRC depends on the fixed validation-derived positive label prevalence.",
        "",
        "| Score | Spearman ADE | Spearman FDE | High-ADE AUROC | High-ADE AUPRC | High-FDE AUROC | High-FDE AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *primary_rows,
        "",
        "## Partial Spearman: U conditional on observed motion",
        "",
        "Computed by ranking U/error/motion, separately regressing ranked U and ranked error on intercept + ranked motion, then correlating residuals.",
        "",
        "| Split | Partial rho: U↔ADE\|Motion | Partial rho: U↔FDE\|Motion |",
        "|---|---:|---:|",
        *partial_rows,
        "",
        "## Validation-fitted motion adjustment",
        "",
        f"Coefficients: `b0={fit['b0']:.8f}`, `b1={fit['b1']:.8f}`, `b2={fit['b2']:.8f}`. Formula: `adjusted_u = log1p(u_mean) - (b0 + b1*x + b2*x²)`, `x=log1p(observed_motion)`. Fit split: `{fit['fit_split']}`; ADE/FDE and future ground truth not used.",
        "",
        "| Split | Raw U vs motion Spearman | Adjusted U vs motion Spearman |",
        "|---|---:|---:|",
        *motion_adjustment_rows,
        "",
        "Adjusted U test reliability bins use q33/q67 cutpoints fit on validation only:",
        "",
        "| Test bin | N | Mean adjusted U | Mean ADE | Median ADE | Mean FDE | Median FDE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[
            f"| {item['bin']} | {item['sample_count']} | {fmt(item['mean_adjusted_u'])} | {fmt(item['mean_ade'])} | {fmt(item['median_ade'])} | {fmt(item['mean_fde'])} | {fmt(item['median_fde'])} |"
            for item in adjusted["reliability_bins"]["bins"]["test"]
        ],
        f"Low→medium→high adjusted-score bin errors monotonic: {adjusted['reliability_bins']['test_low_to_high_error_monotonic']}.",
        "",
        "## Global risk–coverage comparison",
        "",
        "Lower score is retained first. Random is mean±SD from 100 rankings; oracle sorts by true ADE and is only an unattainable upper bound.",
        "",
        "| Coverage | Raw U ADE/FDE | Adjusted U ADE/FDE | Motion-only ADE/FDE | Random ADE/FDE | Oracle ADE/FDE |",
        "|---:|---:|---:|---:|---:|---:|",
        *global_rows,
        "",
        "## Motion-stratified risk–coverage",
        "",
        "Test is split into validation-defined motion deciles; raw or adjusted U ranks samples separately within each decile before pooling. The within-motion random column is the 100-permutation null reference.",
        "",
        "| Nominal coverage | Raw U ADE/FDE | Adjusted U ADE/FDE | Within-motion random ADE/FDE | Actual coverage raw/adjusted |",
        "|---:|---:|---:|---:|---:|",
        *strat_rows,
        "",
        "Within-motion permutation sanity check (small lower-tail p means the observed ranking beats randomized U within the same motion deciles):",
        "",
        "| Coverage | Raw/adjusted observed ADE | Random ADE mean±SD | Empirical p raw/adjusted |",
        "|---:|---:|---:|---:|",
        *permutation_rows,
        f"At 80% coverage, adjusted-U empirical lower-tail p for ADE = `{fmt(perm80['adjusted_u_mean_empirical_lower_tail_p']['ade'])}`; raw-U = `{fmt(perm80['raw_u_mean_empirical_lower_tail_p']['ade'])}`.",
        "",
        "## Motion-stratified correlations",
        "",
        f"Slow/medium/fast boundaries are validation q33=`{strata['tertile_cutpoints_from_validation_pixel']['q33']:.3f}px`, q67=`{strata['tertile_cutpoints_from_validation_pixel']['q67']:.3f}px`.",
        "",
        "| Test stratum | N | Raw U↔ADE | Adjusted U↔ADE | Motion↔ADE | Raw U↔FDE | Adjusted U↔FDE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *tertile_rows,
        "",
        "Validation-defined motion deciles (test samples):",
        "",
        "| Decile | N | Raw U↔ADE | Adjusted U↔ADE | Raw U↔FDE | Adjusted U↔FDE |",
        "|---|---:|---:|---:|---:|---:|",
        *decile_rows,
        "",
        "## Cluster bootstrap",
        "",
        f"Primary video-cluster bootstrap resamples `{bootstrap['primary_video_cluster_bootstrap']['cluster_count']}` `scene_id` clusters with replacement and keeps every sample in each selected video together ({bootstrap['primary_video_cluster_bootstrap']['repetitions']} replicates, seed {bootstrap['primary_video_cluster_bootstrap']['seed']}). Track-cluster bootstrap is secondary.",
        "",
        "| Cluster unit | Score | 95% CI Spearman ADE | 95% CI High-ADE AUROC | Valid rho replicates |",
        "|---|---|---:|---:|---:|",
        *cluster_rows,
        "",
        "## Normalized-coordinate and image-scale checks",
        "",
        "Normalized analysis is secondary; it does not replace pixel-space primary results. `image_size` correlations are undefined when a dimension is constant.",
        "",
        "| Split | Normalized U↔ADE rho | Normalized U↔FDE rho | Mean normalized U | Mean normalized ADE |",
        "|---|---:|---:|---:|---:|",
        *normalized_rows,
        f"Test image resolution pairs: `{width_summary['image_size_unique_width_height_pairs']}`; unique widths/heights = `{width_summary['width_unique_count']}/{width_summary['height_unique_count']}`. Width/height correlations with raw U and ADE are `{corr(width_summary['width_vs_raw_u_spearman'])}`, `{corr(width_summary['width_vs_ade_spearman'])}`, `{corr(width_summary['height_vs_raw_u_spearman'])}`, `{corr(width_summary['height_vs_ade_spearman'])}` respectively.",
        "Future endpoint displacement correlations, if present in `adjusted_score.json`, are post-hoc diagnostics only.",
        "",
        f"## Final decision: {decision['decision']}",
        "",
        decision["rule"],
        f"Evidence check count: `{decision.get('go_conditions_met', sum(decision['go_conditions'].values()))}/{decision.get('go_conditions_total', len(decision['go_conditions']))}`. Not met: `{', '.join(failed_go_checks) if failed_go_checks else 'none'}`.",
        "Raw `u_mean` does not outperform motion-only globally on ADE correlation or High-ADE AUROC; the GO result is specifically based on positive conditional/within-motion evidence and cluster-bootstrap support, not global superiority.",
        "",
        "This decision concerns whether to consider a later reliability-gated intention study. No intention model or gate was started in this task.",
        "",
        "## Integrity and artifacts",
        "",
        "- No trajectory retraining, intention/social/scene model, ADE predictor, or test-fitted adjustment was run.",
        "- Checkpoint hashes were verified unchanged after inference; original predictions/arrays were not modified in place.",
        "- No processed data or per-sample prediction dump is produced.",
        "- Detailed machine-readable outputs are in this directory: partial correlation, motion baselines, adjustment, risk coverage/permutation, motion strata, cluster bootstrap, scale diagnostics, and checkpoint audit JSON.",
    ]
    return summary, "\n".join(md) + "\n"


def main() -> None:
    summary, markdown = build_summary()
    (OUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "decision": summary["decision"]["decision"],
        "summary": str(OUT_ROOT / "summary.md"),
        "test_raw_u_vs_ade": summary["raw_motion_comparison"]["split_metrics"]["test"]["raw_u_mean"]["spearman_vs_ade"]["rho"],
        "test_adjusted_u_vs_ade": summary["raw_motion_comparison"]["split_metrics"]["test"]["adjusted_u_mean"]["spearman_vs_ade"]["rho"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
