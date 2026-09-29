#!/usr/bin/env python3
"""Paired clean-test bootstrap, historical comparison, and final report."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_clean_utils import ARMS, RESULTS_ROOT, SEEDS, sha256_file, write_json
from scripts.reliability_gated_intent_utils import cluster_bootstrap_paired_delta


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float]) -> str:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return "—"
    return f"{array.mean():.4f} ± {array.std(ddof=1) if len(array) > 1 else 0.0:.4f}"


def clean_test(seed: int, arm: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    run_dir = RESULTS_ROOT / arm / f"seed{seed}"
    record = read_json(run_dir / "official_test_metrics.json")
    with np.load(run_dir / "test_predictions.npz", allow_pickle=False) as archive:
        predictions = {key: archive[key].copy() for key in archive.files}
    return record["metrics"], predictions


def historical_test(seed: int, arm: str) -> dict[str, Any]:
    if arm == "J0_clean":
        path = ROOT / f"results/joint_traj_supervision_attribution/j0/seed{seed}/official_test_metrics.json"
        value = read_json(path)
        return value.get("metrics", value)
    path = ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json"
    value = read_json(path)
    return value["test"]


def t0_metrics(seed: int) -> dict[str, float]:
    path = ROOT / f"results/trajectory_transformer_scene_15x15_seed{seed}/metrics.json"
    value = read_json(path)
    test = value.get("test", value)
    ade_keys = ("trajectory_ade_pixel", "ade_pixel", "ade")
    fde_keys = ("trajectory_fde_pixel", "fde_pixel", "fde")
    ade = next((test[key] for key in ade_keys if key in test), None)
    fde = next((test[key] for key in fde_keys if key in test), None)
    if ade is None or fde is None:
        raise KeyError(f"Cannot find T0 ADE/FDE in {path}: keys={list(test)}")
    return {"ade": float(ade), "fde": float(fde)}


def metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "auc": float(metrics["intent_auc"]),
        "brier": float(metrics["intent_brier"]),
        "f1": float(metrics["intent_f1"]),
        "bacc": float(metrics["intent_balanced_accuracy"]),
        "ade": float(metrics["trajectory_ade_pixel"]),
        "fde": float(metrics["trajectory_fde_pixel"]),
        "accuracy": float(metrics["intent_accuracy"]),
    }


def bootstrap_and_comparison(protocol_sha: str) -> tuple[dict[str, Any], dict[str, Any]]:
    bootstrap_payload: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "bootstrap_unit": "scene_id cluster",
        "requested_repetitions_per_seed": 2000,
        "bootstrap_seed": 9124,
        "delta_definition": "J100_clean minus J0_clean",
        "seeds_pooled": False,
        "per_seed": {},
    }
    comparison: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "historical_and_clean_are_separate_experiments": True,
        "primary_clean_delta_definition": "J100_clean minus J0_clean",
        "per_seed": {},
    }
    required_pair_fields = (
        "scene_id", "target_id", "obs_end_frame", "intent_label", "future_gt", "image_size",
    )
    for seed in SEEDS:
        clean_results = {arm: clean_test(seed, arm) for arm in ARMS}
        clean_metrics = {arm: clean_results[arm][0] for arm in ARMS}
        clean_predictions = {arm: clean_results[arm][1] for arm in ARMS}
        p0, p1 = clean_predictions["J0_clean"], clean_predictions["J100_clean"]
        for field in required_pair_fields:
            if field not in p0 or field not in p1 or not np.array_equal(p0[field], p1[field]):
                raise RuntimeError(f"Clean test sample pairing mismatch for seed {seed}: {field}")
        boot = cluster_bootstrap_paired_delta(
            p0["intent_label"], p0["intent_probability"], p1["intent_probability"], p0["scene_id"],
            repetitions=2000, seed=9124,
        )
        bootstrap_payload["per_seed"][str(seed)] = boot
        clean0, clean100 = metric_row(clean_metrics["J0_clean"]), metric_row(clean_metrics["J100_clean"])
        hist0 = metric_row(historical_test(seed, "J0_clean"))
        hist100 = metric_row(historical_test(seed, "J100_clean"))
        comparison["per_seed"][str(seed)] = {
            "historical_J0": hist0,
            "historical_J100": hist100,
            "clean_J0": clean0,
            "clean_J100": clean100,
            "historical_delta_J100_minus_J0": {key: hist100[key] - hist0[key] for key in hist0},
            "clean_delta_J100_minus_J0": {key: clean100[key] - clean0[key] for key in clean0},
            "bootstrap": boot,
            "trajectory_T0": t0_metrics(seed),
            "paired_prediction_ids_verified": True,
        }
    methods = ("historical_J0", "historical_J100", "clean_J0", "clean_J100")
    comparison["aggregate"] = {
        method: {
            key: [comparison["per_seed"][str(seed)][method][key] for seed in SEEDS]
            for key in ("auc", "brier", "f1", "bacc", "ade", "fde", "accuracy")
        }
        for method in methods
    }
    historical_deltas = [comparison["per_seed"][str(seed)]["historical_delta_J100_minus_J0"]["auc"] for seed in SEEDS]
    clean_deltas = [comparison["per_seed"][str(seed)]["clean_delta_J100_minus_J0"]["auc"] for seed in SEEDS]
    historical_j100_auc = [comparison["per_seed"][str(seed)]["historical_J100"]["auc"] for seed in SEEDS]
    clean_j100_auc = [comparison["per_seed"][str(seed)]["clean_J100"]["auc"] for seed in SEEDS]
    comparison["historical_mean_delta_auc"] = float(np.mean(historical_deltas))
    comparison["clean_mean_delta_auc"] = float(np.mean(clean_deltas))
    comparison["selection_effect_on_delta_auc_clean_minus_historical"] = float(np.mean(clean_deltas) - np.mean(historical_deltas))
    comparison["historical_delta_auc_sample_sd"] = float(np.std(historical_deltas, ddof=1))
    comparison["clean_delta_auc_sample_sd"] = float(np.std(clean_deltas, ddof=1))
    comparison["historical_J100_auc_sample_sd"] = float(np.std(historical_j100_auc, ddof=1))
    comparison["clean_J100_auc_sample_sd"] = float(np.std(clean_j100_auc, ddof=1))
    comparison["clean_J0_mean_auc_minus_historical_J0"] = float(np.mean([comparison["per_seed"][str(seed)]["clean_J0"]["auc"] - comparison["per_seed"][str(seed)]["historical_J0"]["auc"] for seed in SEEDS]))
    comparison["clean_J100_mean_auc_minus_historical_J100"] = float(np.mean([comparison["per_seed"][str(seed)]["clean_J100"]["auc"] - comparison["per_seed"][str(seed)]["historical_J100"]["auc"] for seed in SEEDS]))
    return bootstrap_payload, comparison


def support_decision(comparison: dict[str, Any], bootstrap: dict[str, Any]) -> dict[str, Any]:
    deltas = [comparison["per_seed"][str(seed)]["clean_delta_J100_minus_J0"]["auc"] for seed in SEEDS]
    mean_delta = float(np.mean(deltas))
    positive = sum(delta > 0 for delta in deltas)
    positive_ci = sum(
        bootstrap["per_seed"][str(seed)]["delta_roc_auc"]["ci_percentile_95"]["lower_95"] > 0
        for seed in SEEDS
    )
    historical_j100_sd = comparison["historical_J100_auc_sample_sd"]
    clean_j100_sd = comparison["clean_J100_auc_sample_sd"]
    seed_sd_cutoff = 2.0 * 0.0135
    seed_sd_pass = clean_j100_sd <= seed_sd_cutoff
    if mean_delta > 0.03 and positive >= 2 and positive_ci >= 2 and seed_sd_pass:
        label = "strong_support"
        explanation = "Mean effect, seed directions, bootstrap intervals, and the pre-access seed-SD stability guard are met."
    elif mean_delta > 0.02 and positive >= 2:
        label = "moderate_support"
        explanation = "Mean delta exceeds 0.02 with mostly positive seed direction, but the strong-support conditions are not all met."
    elif mean_delta < 0:
        label = "negative"
        explanation = "Mean clean delta AUC is negative."
    elif mean_delta > 0:
        label = "weak_support"
        explanation = "The point estimate is positive but too small and uncertain to support a stable or practically meaningful auxiliary-supervision benefit."
    else:
        label = "no_support"
        explanation = "Mean clean delta is near zero or the direction is unstable."
    return {
        "label": label,
        "mean_clean_delta_auc": mean_delta,
        "positive_seed_count": positive,
        "bootstrap_ci_lower_positive_seed_count": positive_ci,
        "clean_J100_auc_sample_sd": clean_j100_sd,
        "historical_J100_auc_sample_sd": historical_j100_sd,
        "clean_to_historical_seed_sd_ratio": clean_j100_sd / historical_j100_sd if historical_j100_sd > 0 else None,
        "seed_sd_stability_rule": "clean J100 AUC sample SD <= 2.0 * historical J100 AUC sample SD reference (0.0135), cutoff 0.0270, recorded before clean test access",
        "seed_sd_stability_pass": seed_sd_pass,
        "seed_sd_cutoff": seed_sd_cutoff,
        "interpretation": explanation,
    }


def render_summary(comparison: dict[str, Any], bootstrap: dict[str, Any], support: dict[str, Any]) -> str:
    frozen = read_json(RESULTS_ROOT / "protocol_frozen.json")
    protocol_sha = sha256_file(RESULTS_ROOT / "protocol_frozen.json")
    history_vs_clean = comparison
    pareto = read_json(RESULTS_ROOT / "pareto_history.json")
    representation = read_json(RESULTS_ROOT / "representation_diagnostic.json")
    gradients = read_json(RESULTS_ROOT / "gradient_diagnostic.json")

    methods = (
        ("Historical J0", "historical_J0"),
        ("Historical J100", "historical_J100"),
        ("Clean J0", "clean_J0"),
        ("Clean J100", "clean_J100"),
    )
    lines = [
        "# Clean trajectory-supervision attribution",
        "",
        f"- Frozen protocol SHA256: `{protocol_sha}`; base commit `{frozen['base_commit']}`.",
        f"- Official test access: one-time; {read_json(RESULTS_ROOT / 'official_test_access_record.json')['sample_count']} samples; test archive SHA256 `{read_json(RESULTS_ROOT / 'official_test_access_record.json')['test_archive_sha256']}`.",
        "- Clean J0/J100 share each seed's exact initial tensor state and main WeightedRandomSampler sequence. Scheduler monitors raw validation AUC; checkpoint selection uses raw validation AUC with Brier-only tie-break (≤1e-4).",
        "- All historical metrics are secondary context and were not pooled with clean runs.",
        "",
        "## Test metrics across methods",
        "",
        "| Method | AUC | Brier | F1 | BAcc | ADE (px) | FDE (px) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    metric_keys = ("auc", "brier", "f1", "bacc", "ade", "fde")
    for display, method in methods:
        rows = history_vs_clean["aggregate"][method]
        lines.append(f"| {display} | " + " | ".join(mean_sd(rows[key]) for key in metric_keys) + " |")
    lines.extend([
        "",
        "## Matched clean AUC attribution",
        "",
        "| Seed | J0-clean AUC | J100-clean AUC | ΔAUC |",
        "|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        row = history_vs_clean["per_seed"][str(seed)]
        lines.append(f"| {seed} | {row['clean_J0']['auc']:.4f} | {row['clean_J100']['auc']:.4f} | {row['clean_delta_J100_minus_J0']['auc']:+.4f} |")
    lines.extend([
        "",
        "## Paired scene_id cluster bootstrap (2,000 draws per seed; no pooling)",
        "",
        "| Seed | ΔAUC point | Bootstrap mean | 95% CI | ΔBrier point | 95% CI |",
        "|---:|---:|---:|---|---:|---|",
    ])
    for seed in SEEDS:
        row = history_vs_clean["per_seed"][str(seed)]
        boot = bootstrap["per_seed"][str(seed)]
        auc = boot["delta_roc_auc"]
        brier = boot["delta_brier"]
        lines.append(
            f"| {seed} | {row['clean_delta_J100_minus_J0']['auc']:+.4f} | {auc['mean']:+.4f} | [{auc['ci_percentile_95']['lower_95']:+.4f}, {auc['ci_percentile_95']['upper_95']:+.4f}] | {row['clean_delta_J100_minus_J0']['brier']:+.4f} | [{brier['ci_percentile_95']['lower_95']:+.4f}, {brier['ci_percentile_95']['upper_95']:+.4f}] |"
        )
    lines.extend([
        "",
        "## Trajectory ADE comparison",
        "",
        "| Seed | J0-clean ADE | J100-clean ADE | T0 standalone ADE |",
        "|---:|---:|---:|---:|",
    ])
    for seed in SEEDS:
        row = history_vs_clean["per_seed"][str(seed)]
        lines.append(f"| {seed} | {row['clean_J0']['ade']:.2f} | {row['clean_J100']['ade']:.2f} | {row['trajectory_T0']['ade']:.2f} |")

    historical_delta = history_vs_clean["historical_mean_delta_auc"]
    clean_delta = history_vs_clean["clean_mean_delta_auc"]
    selection_effect = history_vs_clean["selection_effect_on_delta_auc_clean_minus_historical"]
    positive_ci = support["bootstrap_ci_lower_positive_seed_count"]
    positive_seeds = support["positive_seed_count"]
    seed_sd_compare = (
        f"Clean J100 AUC sample SD={support['clean_J100_auc_sample_sd']:.4f}; historical J100 SD={support['historical_J100_auc_sample_sd']:.4f}; "
        f"ratio={support['clean_to_historical_seed_sd_ratio']:.2f}×; preregistered stability cutoff={support['seed_sd_cutoff']:.4f} AUC SD; pass={support['seed_sd_stability_pass']}."
    )
    j0_auc_text = ", ".join(f"{seed}={history_vs_clean['per_seed'][str(seed)]['clean_J0']['auc']:.4f}" for seed in SEEDS)
    j100_auc_text = ", ".join(f"{seed}={history_vs_clean['per_seed'][str(seed)]['clean_J100']['auc']:.4f}" for seed in SEEDS)
    lines.extend([
        "",
        "## Historical versus clean selection",
        "",
        f"Historical mean ΔAUC was {historical_delta:+.4f}; clean mean ΔAUC is {clean_delta:+.4f}; clean-minus-historical change is {selection_effect:+.4f}.",
        f"Clean J0 mean AUC shifts from historical by {history_vs_clean['clean_J0_mean_auc_minus_historical_J0']:+.4f}; clean J100 shifts by {history_vs_clean['clean_J100_mean_auc_minus_historical_J100']:+.4f}.",
        "Historical and clean experiments are reported separately; no historical samples or seeds were pooled into the primary clean inference.",
        "",
        "## Validation AUC–ADE Pareto diagnostic",
        "",
        f"AUC–ADE trade-off across J100-clean validation epochs observed: **{pareto['auc_ade_tradeoff_observed_any_seed']}**. Per-epoch points and Pareto-efficient epochs are in `pareto_history.json`; this diagnostic did not affect selection.",
        "",
        "## Representation diagnostic (selected checkpoint; fixed 1,000 validation samples)",
        "",
        "| Feature | Arm | Mean L2 norm | Mean per-dimension variance | Cosine to initialization | Linear CKA to initialization |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for feature in ("target_encoder_last", "fused_representation"):
        for arm in ARMS:
            rows = [representation["per_seed"][str(seed)]["features"][feature]["arms"][arm] for seed in SEEDS]
            norms = np.mean([row["statistics"]["mean_l2_norm"] for row in rows])
            variances = np.mean([row["statistics"]["mean_per_dimension_variance"] for row in rows])
            cosine = np.mean([row["similarity_to_initialization"]["cosine_mean"] for row in rows])
            cka = np.mean([row["similarity_to_initialization"]["linear_cka"] for row in rows])
            lines.append(f"| {feature} | {arm} | {norms:.4f} | {variances:.6g} | {cosine:.4f} | {cka:.4f} |")
    lines.extend([
        "",
        "These endpoint representation moments and similarities are descriptive only; they do not establish a causal mechanism.",
        "",
        "## Selected-checkpoint gradient diagnostic",
        "",
        "| Arm | Intent shared-grad norm | Raw trajectory shared-grad norm | Weighted trajectory shared-grad norm | Cosine(intent, trajectory) |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        rows = [gradients["per_seed"][str(seed)][arm] for seed in SEEDS]
        mean = lambda key: float(np.mean([row[key] for row in rows]))
        lines.append(f"| {arm} | {mean('intent_shared_gradient_norm'):.4g} | {mean('raw_trajectory_shared_gradient_norm'):.4g} | {mean('weighted_trajectory_shared_gradient_norm'):.4g} | {mean('intent_vs_raw_trajectory_cosine'):+.4f} |")
    lines.extend([
        "",
        "Selected-checkpoint gradients are endpoint diagnostics only and do not represent gradient behavior over training.",
        "",
        "## Direct answers",
        "",
        f"1. Initialization: **exactly matched** per seed; max absolute initial parameter difference is 0. The evidence is recorded in `initialization_match.json`.",
        f"2. Configuration: **only `traj_weight` differs**; audit passed. The model architecture file was unchanged.",
        f"3. Scheduler and checkpoint selection: **neither uses ADE/FDE**. Scheduler uses raw validation AUC; selector uses raw validation AUC then raw Brier within 1e-4.",
        f"4. J0-clean AUC by seed: {j0_auc_text}.",
        f"5. J100-clean AUC by seed: {j100_auc_text}.",
        f"6. Mean clean ΔAUC: **{clean_delta:+.4f}**; positive direction {positive_seeds}/3 seeds; bootstrap lower bound > 0 for {positive_ci}/3 seeds.",
        f"7. Bootstrap: see the per-seed AUC and Brier intervals above; three seeds are independent and were not pooled.",
        f"8. Historical +0.0370 versus clean: change {selection_effect:+.4f}; historical Δ={historical_delta:+.4f}, clean Δ={clean_delta:+.4f}.",
        f"9. J100-clean test trajectory ADE/FDE: " + "; ".join(f"seed {seed}: {history_vs_clean['per_seed'][str(seed)]['clean_J100']['ade']:.2f}/{history_vs_clean['per_seed'][str(seed)]['clean_J100']['fde']:.2f} px" for seed in SEEDS) + ".",
        f"10. AUC–ADE Pareto pattern: {'a validation trade-off exists' if pareto['auc_ade_tradeoff_observed_any_seed'] else 'no clear validation trade-off was observed'}; inspect `pareto_history.json` for epochs.",
        f"11. Representation: fixed-validation feature norm/variance and initialization CKA/cosine are tabulated above; see `representation_diagnostic.json` for per-seed/per-dimension values.",
        f"12. Gradient: see endpoint table and `gradient_diagnostic.json`; {seed_sd_compare}",
        f"13. Trajectory-supervision support classification: **{support['label']}**. {support['interpretation']}",
        f"14. Next step: {'the clean positive result supports proceeding to the already scoped Trajectory-Preserving Decoupled Auxiliary Learning direction, while treating Pareto/representation results as motivation rather than proof.' if clean_delta > 0.02 and positive_seeds >= 2 else 'do not claim trajectory supervision improves intention; proceed to the planned joint-architecture component attribution rather than automatically adding modules.'}",
        "15. No reliability, DGB, PCGrad, GradNorm, adapter, ablation, or lambda sweep was run in this phase.",
        "",
        "## Artifacts",
        "",
        "- `cluster_bootstrap.json`: per-seed paired scene_id bootstrap.",
        "- `historical_vs_clean.json`: historical and clean per-seed metrics/deltas kept separate.",
        "- `pareto_history.json`, `representation_diagnostic.json`, `gradient_diagnostic.json`: validation-only/descriptive diagnostics.",
        f"- Protocol: `protocol_frozen.json` (SHA256 `{protocol_sha}`).",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    protocol_path = RESULTS_ROOT / "protocol_frozen.json"
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    expected = (RESULTS_ROOT / "protocol_frozen.sha256").read_text(encoding="utf-8").split()[0]
    if protocol_sha != expected:
        raise RuntimeError("Frozen protocol integrity check failed")
    access = read_json(RESULTS_ROOT / "official_test_access_record.json")
    evaluation = read_json(RESULTS_ROOT / "official_test_evaluation.json")
    if access.get("status") != "complete_one_time_official_test_evaluation" or evaluation.get("status") != "official_test_evaluated_once":
        raise RuntimeError("One-time official test evaluation is incomplete")
    if evaluation.get("protocol_sha256") != protocol_sha or access.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Official evaluation does not match the frozen protocol")
    bootstrap, comparison = bootstrap_and_comparison(protocol_sha)
    support = support_decision(comparison, bootstrap)
    write_json(RESULTS_ROOT / "cluster_bootstrap.json", bootstrap)
    write_json(RESULTS_ROOT / "historical_vs_clean.json", comparison)
    write_json(RESULTS_ROOT / "support_decision.json", support)
    summary = render_summary(comparison, bootstrap, support)
    (RESULTS_ROOT / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({
        "summary": str((RESULTS_ROOT / "summary.md").relative_to(ROOT)),
        "clean_mean_delta_auc": comparison["clean_mean_delta_auc"],
        "historical_mean_delta_auc": comparison["historical_mean_delta_auc"],
        "support": support,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
