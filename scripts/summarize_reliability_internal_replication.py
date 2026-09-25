#!/usr/bin/env python3
"""Summarize the internal train-video holdout only after its one-time evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_reliability_internal_replication import RESULT_ROOT, SEEDS, read_json, write_json  # noqa: E402


def fmt(value: Any, digits: int = 3) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def pair(ade: Any, fde: Any) -> str:
    return f"{fmt(ade)}/{fmt(fde)}"


def rho(value: dict[str, Any] | float | None) -> str:
    if isinstance(value, dict):
        number = value.get("rho")
        pvalue = value.get("p_value")
        if number is None:
            return "NA"
        p = "" if pvalue is None else f", p={'<1e-300' if pvalue == 0 else f'{pvalue:.2g}'}"
        return f"{number:.3f}{p}"
    return fmt(value)


def build_summary() -> tuple[dict[str, Any], str]:
    access = read_json(RESULT_ROOT / "holdout_access_record.json")
    if access.get("status") != "completed" or access.get("holdout_evaluated_after_protocol_frozen") is not True:
        raise RuntimeError("Summary is blocked until the frozen holdout evaluation/recovery is completed")
    recovery_audit = read_json(RESULT_ROOT / "holdout_recovery_audit.json")
    if recovery_audit.get("status") != "completed" or recovery_audit.get("recovery_authorized") is not True:
        raise RuntimeError("Recovery audit is not recorded as completed and authorized")
    holdout = read_json(RESULT_ROOT / "internal_holdout_reliability.json")
    decision = read_json(RESULT_ROOT / "decision.json")
    if decision.get("decision_source_split") != "internal_holdout only":
        raise RuntimeError("Decision source is not recorded as internal holdout only")

    manifest = read_json(RESULT_ROOT / "split_manifest.json")
    protocol = read_json(RESULT_ROOT / "protocol_frozen.json")
    val = read_json(RESULT_ROOT / "internal_val_reliability.json")
    adjustment = read_json(RESULT_ROOT / "motion_adjustment.json")
    risk = read_json(RESULT_ROOT / "risk_coverage.json")
    cluster = read_json(RESULT_ROOT / "cluster_bootstrap.json")
    run_metrics = {str(seed): read_json(RESULT_ROOT / f"trajectory_seed{seed}/metrics.json") for seed in SEEDS}

    # This is deliberately the first read of the prior official-split audit, and this
    # script refuses to run until internal_holdout is evaluated and decision-frozen.
    prior = read_json(PROJECT_ROOT / "results/trajectory_reliability_deconfounding/summary.json")
    prior_test = {
        "partial_spearman_u_ade_given_motion": prior["partial_correlation"]["split_results"]["test"]["u_mean_vs_ade_given_observed_motion"],
        "partial_spearman_u_fde_given_motion": prior["partial_correlation"]["split_results"]["test"]["u_mean_vs_fde_given_observed_motion"],
        "raw_u_mean_vs_ade_rho": prior["raw_motion_comparison"]["split_metrics"]["test"]["raw_u_mean"]["spearman_vs_ade"]["rho"],
        "adjusted_u_mean_vs_ade_rho": prior["raw_motion_comparison"]["split_metrics"]["test"]["adjusted_u_mean"]["spearman_vs_ade"]["rho"],
        "motion_only_vs_ade_rho": prior["raw_motion_comparison"]["split_metrics"]["test"]["motion_only"]["spearman_vs_ade"]["rho"],
        "split_name": "old official test; post-hoc descriptive comparison only",
    }

    split_rows = []
    for name in ("internal_train", "internal_val", "internal_holdout"):
        item = manifest["splits"][name]
        split_rows.append(
            f"| {name} | {item['video_count']} | {item['sample_count']} | {item['unique_target_count']} |"
        )

    trajectory_rows = []
    for seed in SEEDS:
        metrics = run_metrics[str(seed)]
        v = metrics["best_internal_val"]
        h = metrics["internal_holdout"]
        trajectory_rows.append(
            f"| seed {seed} | {metrics['best_epoch']} | {pair(v['trajectory_ade_pixel'], v['trajectory_fde_pixel'])} | "
            f"{pair(h['trajectory_ade_pixel'], h['trajectory_fde_pixel'])} |"
        )
    val_ens = val["ensemble_mean_performance"]
    hold_ens = holdout["ensemble_mean_performance"]
    trajectory_rows.append(
        f"| Ensemble mean | — | {pair(val_ens['ade_pixel'], val_ens['fde_pixel'])} | "
        f"{pair(hold_ens['ade_pixel'], hold_ens['fde_pixel'])} |"
    )

    score_rows = []
    for name, label in (
        ("raw_u_mean", "Raw u_mean"),
        ("motion_only", "Motion-only"),
        ("adjusted_u_mean", "Adjusted u_mean"),
    ):
        vm = val["metrics"][name]
        hm = holdout["metrics"][name]
        score_rows.append(
            f"| {label} | {rho(vm['spearman_vs_ade'])} | {rho(vm['spearman_vs_fde'])} | "
            f"{rho(hm['spearman_vs_ade'])} | {rho(hm['spearman_vs_fde'])} | "
            f"{fmt(hm['high_ade']['auroc'])}/{fmt(hm['high_ade']['auprc'])} | "
            f"{fmt(hm['high_fde']['auroc'])}/{fmt(hm['high_fde']['auprc'])} |"
        )

    val_partial = val["partial_spearman_raw_u_given_motion"]
    hold_partial = holdout["partial_spearman_raw_u_given_motion"]
    tertile_rows = []
    for item in holdout["motion_stratified_metrics"]["slow_medium_fast"]:
        tertile_rows.append(
            f"| {item['stratum']} | {item['sample_count']} | {rho(item['raw_u_vs_ade'])} | "
            f"{rho(item['adjusted_u_vs_ade'])} | {rho(item['motion_vs_ade'])} | "
            f"{rho(item['raw_u_vs_fde'])} | {rho(item['adjusted_u_vs_fde'])} |"
        )
    decile_rows = []
    for item in holdout["motion_stratified_metrics"]["motion_deciles"]:
        decile_rows.append(
            f"| D{item['decile']} | {item['sample_count']} | {rho(item['raw_u_vs_ade'])} | "
            f"{rho(item['adjusted_u_vs_ade'])} |"
        )

    global_curves = risk["global"]["curves"]
    global_rows = []
    for index, item in enumerate(global_curves["raw_u_mean"]):
        random = global_curves["random_global"][index]
        oracle = global_curves["oracle_by_true_ade"][index]
        global_rows.append(
            f"| {item['nominal_coverage']:.0%} | {pair(item['ade'], item['fde'])} | "
            f"{pair(global_curves['adjusted_u_mean'][index]['ade'], global_curves['adjusted_u_mean'][index]['fde'])} | "
            f"{pair(global_curves['motion_only'][index]['ade'], global_curves['motion_only'][index]['fde'])} | "
            f"{random['ade_mean_sample_std']['mean']:.3f}±{random['ade_mean_sample_std']['sample_std']:.3f}/"
            f"{random['fde_mean_sample_std']['mean']:.3f}±{random['fde_mean_sample_std']['sample_std']:.3f} | "
            f"{pair(oracle['ade'], oracle['fde'])} |"
        )
    strat_curves = risk["motion_stratified"]["curves"]
    perm = risk["motion_stratified"]["within_motion_permutation"]
    strat_rows = []
    selected_perm_rows = []
    for index, item in enumerate(strat_curves["raw_u_mean"]):
        adj = strat_curves["adjusted_u_mean"][index]
        null = perm["rows"][index]
        strat_rows.append(
            f"| {item['nominal_coverage']:.0%} | {pair(item['ade'], item['fde'])} | "
            f"{pair(adj['ade'], adj['fde'])} | "
            f"{null['randomized_within_motion_mean_sample_std']['ade']['mean']:.3f}±{null['randomized_within_motion_mean_sample_std']['ade']['sample_std']:.3f}/"
            f"{null['randomized_within_motion_mean_sample_std']['fde']['mean']:.3f}±{null['randomized_within_motion_mean_sample_std']['fde']['sample_std']:.3f} |"
        )
        if item["nominal_coverage"] in (0.8, 0.5, 0.2):
            selected_perm_rows.append(
                f"| {item['nominal_coverage']:.0%} | {fmt(null['raw_u_mean_empirical_lower_tail_p']['ade'])}/"
                f"{fmt(null['adjusted_u_mean_empirical_lower_tail_p']['ade'])} | "
                f"{fmt(null['raw_u_mean_empirical_lower_tail_p']['fde'])}/"
                f"{fmt(null['adjusted_u_mean_empirical_lower_tail_p']['fde'])} |"
            )

    bootstrap_rows = []
    for unit_name, result in (
        ("Video scene_id", cluster["video_level_primary"]),
        ("Track scene_id,target_id", cluster["track_level_secondary"]),
    ):
        for score_name in ("raw_u_mean", "adjusted_u_mean", "motion_only"):
            if score_name not in result["metrics"]:
                continue
            rho_ci = result["metrics"][score_name]["spearman_vs_ade"]
            auc_ci = result["metrics"][score_name]["high_ade_auroc"]
            bootstrap_rows.append(
                f"| {unit_name} | {score_name} | {fmt(rho_ci['lower_95'])}–{fmt(rho_ci['upper_95'])} | "
                f"{fmt(auc_ci['lower_95'])}–{fmt(auc_ci['upper_95'])} | {rho_ci['valid_replicates']}/{result['repetitions']} |"
            )

    normalized = holdout["normalized_space_sanity"]
    old_comparison = {
        "previous_official_test": prior_test,
        "new_internal_holdout": {
            "partial_spearman_u_ade_given_motion": hold_partial["u_mean_vs_ade"],
            "partial_spearman_u_fde_given_motion": hold_partial["u_mean_vs_fde"],
            "adjusted_u_mean_vs_ade_rho": holdout["metrics"]["adjusted_u_mean"]["spearman_vs_ade"]["rho"],
            "raw_u_mean_vs_ade_rho": holdout["metrics"]["raw_u_mean"]["spearman_vs_ade"]["rho"],
            "motion_only_vs_ade_rho": holdout["metrics"]["motion_only"]["spearman_vs_ade"]["rho"],
            "split_name": "independent video-level holdout drawn from official train",
        },
        "use_restriction": "descriptive post-hoc comparison only; not used for training, checkpoint selection, adjustment, threshold, criteria, or decision",
    }
    summary = {
        "decision": decision,
        "split_manifest": manifest,
        "protocol_frozen": protocol,
        "trajectory_runs": run_metrics,
        "internal_val_reliability": val,
        "internal_holdout_reliability": holdout,
        "motion_adjustment": adjustment,
        "risk_coverage": risk,
        "cluster_bootstrap": cluster,
        "holdout_recovery_audit": recovery_audit,
        "previous_official_test_posthoc_comparison": old_comparison,
        "protocol_audit": {
            "official_val_test_used_for_training_or_model_selection": False,
            "official_val_test_used_for_adjustment_or_thresholds": False,
            "official_val_test_used_for_decision": False,
            "decision_was_determined_from_internal_holdout_only": True,
            "holdout_evaluated_after_protocol_frozen": access["holdout_evaluated_after_protocol_frozen"],
            "initial_holdout_attempt_count": 1,
            "recovery_attempt_count": len(recovery_audit["recovery_attempts"]),
            "total_holdout_computation_attempts": 1 + len(recovery_audit["recovery_attempts"]),
        },
    }

    split = manifest["splits"]
    train_ids = set(split["internal_train"]["scene_ids"])
    val_ids = set(split["internal_val"]["scene_ids"])
    hold_ids = set(split["internal_holdout"]["scene_ids"])
    overlaps = {
        "train_val": len(train_ids & val_ids),
        "train_holdout": len(train_ids & hold_ids),
        "val_holdout": len(val_ids & hold_ids),
    }
    criterion_rows = [
        f"| {name} | {passed} |"
        for name, passed in decision["conditions"].items()
    ]
    md = [
        "# Independent internal-video replication of motion-controlled trajectory reliability",
        "",
        "## Protocol and blind holdout",
        "",
        "Only `data/processed/jaad_sequences_scene_15x15/train.npz` was used to create the internal split, train models, select checkpoints, fit motion adjustment, set high-error thresholds, and make the decision. Official `val.npz` and `test.npz` were not opened by the experiment. The initial frozen PHASE D computation was interrupted; one explicitly authorized recovery re-executed the same frozen evaluation without changing models or protocol.",
        f"Manifest seed `{manifest['random_seed']}`; train.npz SHA256 `{manifest['train_npz_sha256']}`; canonical manifest SHA256 `{manifest['manifest_sha256']}`.",
        f"Frozen protocol SHA256 `{protocol['protocol_sha256']}`; recovery status `{recovery_audit['status']}`; authorized recovery attempts `{len(recovery_audit['recovery_attempts'])}`.",
        "Frozen checkpoint SHA256 values: " + "; ".join(f"seed {seed} `{value}`" for seed, value in protocol["checkpoint_sha256"].items()) + ".",
        "",
        "| Internal split | Videos | Samples | Unique target IDs |",
        "|---|---:|---:|---:|",
        *split_rows,
        f"Scene overlap counts: `{overlaps}`. All pairwise overlaps are zero: `{all(value == 0 for value in overlaps.values())}`.",
        f"`holdout_evaluated_after_protocol_frozen`: `{access['holdout_evaluated_after_protocol_frozen']}`; one-time access record status: `{access['status']}`.",
        f"Shared ordered inference sample metadata SHA256: `{holdout['sample_order']['sha256']}` for `{holdout['sample_order']['sample_count']}` rows (ordered scene_id, target_id, obs_end_frame recorded in holdout JSON).",
        "",
        "## Frozen trajectory ensemble",
        "",
        "All three models were trained from fresh random initialization with zero scene input. Checkpoint selection used only the lowest internal-val pixel ADE; no holdout was used for epoch selection.",
        "",
        "| Model | Best epoch | Internal-val ADE/FDE px | Holdout ADE/FDE px |",
        "|---|---:|---:|---:|",
        *trajectory_rows,
        "",
        "## Internal-validation reliability (descriptive; protocol fitting split)",
        "",
        "Primary score remained fixed as pixel `u_mean`. Motion is first-to-last observed target-center displacement from the 15 input frames. High-error thresholds and the quadratic motion adjustment were fit only on internal_val.",
        "",
        f"Internal-val fixed high-error thresholds: ADE `{protocol['high_error_thresholds']['high_ade_pixel_threshold']:.4f}px`; FDE `{protocol['high_error_thresholds']['high_fde_pixel_threshold']:.4f}px`.",
        f"Adjustment: `adjusted_u=log1p(u_mean)-(b0+b1*x+b2*x²)`, `x=log1p(motion)`; coefficients b0/b1/b2 = `{adjustment['b0']:.8f}`, `{adjustment['b1']:.8f}`, `{adjustment['b2']:.8f}`.",
        "",
        "| Score | Val rho ADE | Val rho FDE | Holdout rho ADE | Holdout rho FDE | Holdout High-ADE AUROC/AUPRC | Holdout High-FDE AUROC/AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *score_rows,
        f"Partial Spearman raw U↔ADE|Motion: internal-val `{rho(val_partial['u_mean_vs_ade'])}`, holdout `{rho(hold_partial['u_mean_vs_ade'])}`; U↔FDE|Motion: internal-val `{rho(val_partial['u_mean_vs_fde'])}`, holdout `{rho(hold_partial['u_mean_vs_fde'])}`.",
        "",
        "## Independent internal holdout: motion-controlled analyses",
        "",
        "### Slow / medium / fast",
        "",
        "| Stratum | N | Raw U↔ADE | Adjusted U↔ADE | Motion↔ADE | Raw U↔FDE | Adjusted U↔FDE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *tertile_rows,
        "",
        "### Internal-val motion deciles",
        "",
        "| Decile | N | Raw U↔ADE rho | Adjusted U↔ADE rho |",
        "|---|---:|---:|---:|",
        *decile_rows,
        "",
        "### Global risk–coverage (ADE/FDE px)",
        "",
        "Random is 100 random rankings (mean±SD). Oracle ranks by holdout ADE and is an upper bound only.",
        "",
        "| Coverage | Raw U | Adjusted U | Motion-only | Random | Oracle |",
        "|---:|---:|---:|---:|---:|---:|",
        *global_rows,
        "",
        "### Motion-stratified risk–coverage (ADE/FDE px)",
        "",
        "Within each internal-val motion decile, retain the lowest-score fraction and then pool. Random is the 500-permutation within-motion reference.",
        "",
        "| Coverage | Raw U | Adjusted U | Within-motion random mean±SD |",
        "|---:|---:|---:|---:|",
        *strat_rows,
        "",
        "### Within-motion permutation (lower-tail empirical p)",
        "",
        "| Coverage | ADE p raw/adjusted | FDE p raw/adjusted |",
        "|---:|---:|---:|",
        *selected_perm_rows,
        "The permutation minimum is `1/501`; interpret it as a finite-resolution reference, not extreme certainty.",
        "",
        "## Cluster bootstrap",
        "",
        f"Primary video bootstrap uses `{cluster['video_level_primary']['cluster_count']}` holdout videos and 2000 resamples; secondary track bootstrap uses `{cluster['track_level_secondary']['cluster_count']}` tracks and 1000 resamples. With only about 21 videos, video-cluster intervals are necessarily sensitive to the small number of clusters.",
        "",
        "| Cluster unit | Score | 95% CI Spearman ADE | 95% CI High-ADE AUROC | Valid rho replicates |",
        "|---|---|---:|---:|---:|",
        *bootstrap_rows,
        "",
        "## Normalized-space sanity check",
        "",
        f"Holdout normalized-coordinate Spearman: u_mean↔ADE `{rho(normalized['u_mean_normalized_vs_ade_normalized'])}`, u_mean↔FDE `{rho(normalized['u_mean_normalized_vs_fde_normalized'])}`. Pixel remains the primary unit.",
        "",
        "## Post-hoc comparison with prior official-test diagnostic",
        "",
        "This section was generated only after the internal holdout metrics and decision were frozen. It did not affect any training, checkpoint, adjustment, threshold, or decision step.",
        "",
        "| Diagnostic | Previous official test | New independent internal holdout |",
        "|---|---:|---:|",
        f"| Partial Spearman U↔ADE\|Motion | {fmt(prior_test['partial_spearman_u_ade_given_motion'])} | {rho(hold_partial['u_mean_vs_ade'])} |",
        f"| Partial Spearman U↔FDE\|Motion | {fmt(prior_test['partial_spearman_u_fde_given_motion'])} | {rho(hold_partial['u_mean_vs_fde'])} |",
        f"| Adjusted U↔ADE rho | {fmt(prior_test['adjusted_u_mean_vs_ade_rho'])} | {rho(holdout['metrics']['adjusted_u_mean']['spearman_vs_ade'])} |",
        f"| Raw U↔ADE rho | {fmt(prior_test['raw_u_mean_vs_ade_rho'])} | {rho(holdout['metrics']['raw_u_mean']['spearman_vs_ade'])} |",
        f"| Motion-only↔ADE rho | {fmt(prior_test['motion_only_vs_ade_rho'])} | {rho(holdout['metrics']['motion_only']['spearman_vs_ade'])} |",
        "",
        f"## Final decision: {decision['decision']}",
        "",
        decision["rule"],
        f"Frozen criteria passed: `{decision['conditions_passed']}/{decision['conditions_total']}`. Decision source: `{decision['decision_source_split']}`. Official validation/test were not used for the decision: `{not decision['official_validation_test_used_in_decision']}`.",
        "",
        "| Frozen replication condition | Passed |",
        "|---|---:|",
        *criterion_rows,
        "",
        "No intention classifier or reliability gate was trained. The experiment stops here as instructed.",
    ]
    return summary, "\n".join(md) + "\n"


def main() -> None:
    summary, markdown = build_summary()
    write_json(RESULT_ROOT / "summary.json", summary)
    (RESULT_ROOT / "summary.md").write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "decision": summary["decision"]["decision"],
        "summary": str(RESULT_ROOT / "summary.md"),
        "previous_official_comparison": "post-hoc only; internal holdout decision already frozen",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
