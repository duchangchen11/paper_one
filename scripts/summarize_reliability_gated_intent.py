#!/usr/bin/env python3
"""Create a factual compact summary after the frozen one-pass test evaluation."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/reliability_gated_intent_15x15"
CACHE = RESULTS / "cache"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_reliability_gated_intent import build_gates, load_model, load_npz, predict, SEEDS, VARIANTS
from scripts.reliability_gated_intent_utils import binary_metrics


def load(name: str) -> dict[str, Any]:
    path = RESULTS / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(mean_std: dict[str, Any]) -> str:
    return f"{mean_std['mean']:.4f} ± {mean_std['sample_std']:.4f}"


def main() -> None:
    protocol = load("protocol_frozen.json")
    protocol_path = RESULTS / "protocol_frozen.json"
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    recorded_sha = (RESULTS / "protocol_frozen.sha256").read_text(encoding="utf-8").split()[0]
    if protocol_sha != recorded_sha:
        raise RuntimeError("Frozen protocol checksum mismatch while summarizing")
    test = load("test_metrics.json")
    paired = load("paired_deltas.json")
    bootstrap = load("cluster_bootstrap.json")
    strata = load("reliability_strata.json")
    gates = load("gate_diagnostics.json")
    shuffle = load("gate_shuffle.json")
    audit = load("feature_distribution_audit.json")
    crossfit_manifest = load("crossfit_manifest.json")
    crossfit_summary = load("trajectory_crossfit_summary.json")
    release = load("test_features_released_after_protocol.json")
    if test["protocol_sha256"] != protocol_sha or release["protocol_sha256"] != protocol_sha:
        raise RuntimeError("Test results do not reference the frozen protocol")

    # Recompute validation metrics from the already-frozen models; this never reads test labels.
    train_features = load_npz(CACHE / "train_oof_features.npz")
    val_features = load_npz(CACHE / "val_features.npz")
    transform = load("reliability_transform.json")
    unlabeled_test = load_npz(CACHE / "test_features_unlabeled.npz")
    fixed_gates = build_gates(train_features, val_features, unlabeled_test, transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation: dict[str, Any] = {"protocol_sha256": protocol_sha, "models": {}}
    val_labels = val_features["intent_label"].astype(np.int64)
    for variant in VARIANTS:
        validation["models"][variant] = {"seeds": {}}
        for seed in SEEDS:
            model, record = load_model(variant, seed, protocol, device)
            gate = None if variant == "A_observed_only" else (
                np.ones(len(val_labels), dtype=np.float32) if variant == "B_always_future" else
                fixed_gates["val"]["motion"] if variant == "C_motion_gate" else fixed_gates["val"]["reliability"]
            )
            outputs = predict(model, variant, val_features, gate, device)
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(outputs["final_logit"] / float(record["temperature"]), -60, 60)))
            validation["models"][variant]["seeds"][str(seed)] = {
                **binary_metrics(val_labels, probabilities, float(record["threshold"])),
                "temperature": float(record["temperature"]),
                "selected_epoch": int(record["selected_epoch"]),
            }
            del model
        validation["models"][variant]["mean_sample_std"] = {
            metric: {
                "mean": float(np.mean([validation["models"][variant]["seeds"][str(seed)][metric] for seed in SEEDS])),
                "sample_std": float(np.std([validation["models"][variant]["seeds"][str(seed)][metric] for seed in SEEDS], ddof=1)),
                "n": len(SEEDS),
            }
            for metric in ("roc_auc", "brier", "ece_15_equal_width", "balanced_accuracy", "f1_positive", "negative_recall_specificity")
        }
    (RESULTS / "validation_metrics.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    rows = []
    for variant, record in test["models"].items():
        aggregate = record["mean_sample_std"]
        rows.append({
            "variant": variant,
            "mean_sample_std": aggregate,
            "per_seed": record["seeds"],
        })
    result = {
        "protocol_sha256": test["protocol_sha256"],
        "protocol_integrity": {
            "test_labels_read_before_freeze": protocol["data"]["test_intent_labels_read_before_freeze"],
            "test_labels_released_after_freeze": release["test_intent_labels_read_after_protocol_freeze"],
            "one_unified_test_pass": True,
            "trajectory_predictors_fine_tuned_for_intention": protocol["trajectory_predictors_fine_tuned_for_intention"],
        },
        "official_test_sample_count": test["models"]["A_observed_only"]["seeds"]["42"]["sample_count"],
        "crossfit_manifest": crossfit_manifest,
        "trajectory_crossfit_summary": crossfit_summary,
        "validation_metrics": validation,
        "models": rows,
        "paired_deltas": paired["comparisons"],
        "video_cluster_bootstrap": bootstrap["comparisons"],
        "reliability_strata": strata["strata"],
        "gate_diagnostics": gates,
        "gate_shuffle": shuffle["results"],
        "feature_distribution_audit": audit.get("reliability_feature_distribution", {}),
        "test_run_report": load("test_run_report.json"),
        "test_labels_released_after_protocol_freeze": release["test_intent_labels_read_after_protocol_freeze"],
        "interpretation_guardrails": [
            "Report observed effects only; do not claim SOTA or superiority to unrelated literature.",
            "The 3-model deterministic ensemble disagreement is an epistemic-style proxy, not full aleatoric uncertainty.",
            "A near-zero effective gated residual does not support a mechanism-effectiveness claim.",
        ],
    }
    (RESULTS / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    metric_names = ("roc_auc", "brier", "ece_15_equal_width", "balanced_accuracy", "f1_positive", "negative_recall_specificity")
    lines = [
        "# Reliability-gated future trajectory intention experiment",
        "",
        f"Frozen protocol SHA256: `{test['protocol_sha256']}`",
        f"Official test samples: {result['official_test_sample_count']}",
        "",
        "## Protocol integrity",
        "",
        f"- Test labels were read before freeze: `{protocol['data']['test_intent_labels_read_before_freeze']}`; released after freeze: `{release['test_intent_labels_read_after_protocol_freeze']}`.",
        "- A/B/C/D were evaluated together in one fixed test pass; thresholds and scalar temperatures were fit on official val only.",
        "- Trajectory predictors were frozen for intention training; future_gt/ADE/FDE were not intention inputs, loss, gates, or selection criteria.",
        "",
        "## Crossfit trajectory feature construction",
        "",
        f"All {crossfit_manifest['train_sample_count']} official train samples have OOF predictions: `{sum(f['sample_count'] for f in crossfit_manifest['folds']) == crossfit_manifest['train_sample_count']}`.",
        "| Fold | Videos | Samples | Seed | Best epoch | Val ADE px | Held-out ADE px | Held-out FDE px |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    folds_by_index = {row["fold"]: row for row in crossfit_manifest["folds"]}
    for row in crossfit_summary["models"]:
        fold = folds_by_index[row["fold"]]
        lines.append(f"| {row['fold']} | {fold['video_count']} | {fold['sample_count']} | {row['seed']} | {row['best_epoch']} | {row['best_val_ade_pixel']:.3f} | {row['heldout_ade_pixel']:.3f} | {row['heldout_fde_pixel']:.3f} |")
    lines.extend([
        "",
        "## Feature distribution audit",
        "",
        "| Split | N | ADE px | FDE px | u_mean mean/std/median px | Motion mean/std/median px | Adjusted-U mean/std/median |",
        "|---|---:|---:|---:|---|---|---|",
    ])
    traj_audit = audit["trajectory_audit"]
    rel_audit = audit["reliability_feature_distribution"]
    for split_key, label in (("train_oof", "Train OOF"), ("val", "Val"), ("test", "Test")):
        t = traj_audit[split_key]
        r = rel_audit[split_key]
        def triple(row):
            return f"{row['mean']:.3f}/{row['std']:.3f}/{row['median']:.3f}"
        lines.append(f"| {label} | {t['sample_count']} | {t['ensemble_ade_pixel']:.3f} | {t['ensemble_fde_pixel']:.3f} | {triple(r['u_mean_pixel'])} | {triple(r['observed_motion_pixel'])} | {triple(r['adjusted_u'])} |")
    lines.extend([
        "",
        "Reliability polynomial: " + ", ".join(f"{key}={protocol['reliability_transform']['polynomial'][key]:.8g}" for key in ("b0", "b1", "b2")) + ".",
        "Motion and adjusted-U empirical CDFs were fit on official train OOF only; hashes are in `reliability_transform.json`.",
        "",
        "## Observed-only baseline",
        "",
        "## Always-future",
        "",
        "## Motion-gated future",
        "",
        "## Reliability-gated future",
        "",
    ])
    section_titles = {
        "A_observed_only": ("## Observed-only baseline", "A uses only the 15-frame observed target history; no future feature is passed to the model."),
        "B_always_future": ("## Always-future", "B adds the ensemble-mean predicted future trajectory with a fixed gate of 1."),
        "C_motion_gate": ("## Motion-gated future", "C uses the train-OOF motion empirical confidence as its fixed gate."),
        "D_reliability_gate": ("## Reliability-gated future", "D uses the train-OOF motion-adjusted disagreement confidence as its fixed gate."),
    }
    for variant in reversed(VARIANTS):
        heading, description = section_titles[variant]
        aggregate = test["models"][variant]["mean_sample_std"]
        performance = "; ".join(f"{name} {fmt(aggregate[metric])}" for name, metric in (("AUC", "roc_auc"), ("Brier", "brier"), ("BAcc", "balanced_accuracy"), ("F1", "f1_positive")))
        index = lines.index(heading)
        lines.insert(index + 2, f"{description} Test mean ± SD: {performance}.")
    lines.extend([
        "### Per-seed validation/test metrics",
        "",
        "AUC, Brier, balanced accuracy, and positive F1:",
        "",
        "| Model | Seed | Val AUC | Val Brier | Val BAcc | Val F1 | Test AUC | Test Brier | Test BAcc | Test F1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        for seed in SEEDS:
            vm = validation["models"][variant]["seeds"][str(seed)]
            tm = test["models"][variant]["seeds"][str(seed)]
            lines.append(f"| {variant} | {seed} | {vm['roc_auc']:.4f} | {vm['brier']:.4f} | {vm['balanced_accuracy']:.4f} | {vm['f1_positive']:.4f} | {tm['roc_auc']:.4f} | {tm['brier']:.4f} | {tm['balanced_accuracy']:.4f} | {tm['f1_positive']:.4f} |")
    lines.extend([
        "",
        "## Three-seed summary",
        "",
        "All entries are official-test mean ± sample SD (ddof=1) across seeds 42/123/2024.",
        "",
        "| Model | " + " | ".join(metric_names) + " |",
        "|---|" + "|".join(["---:"] * len(metric_names)) + "|",
    ])
    for variant, record in test["models"].items():
        aggregate = record["mean_sample_std"]
        lines.append("| " + variant + " | " + " | ".join(fmt(aggregate[name]) for name in metric_names) + " |")
    lines.extend(["", "## Paired deltas", "", "Each label is first model minus second (for example, D-A = D minus A); positive ΔAUC/ΔBAcc favors the first model, while negative ΔBrier favors the first model.", ""])
    for name, record in paired["comparisons"].items():
        values = record["mean_sample_std"]
        lines.append(f"- {name}: ΔAUC {fmt(values['delta_roc_auc'])}; ΔBrier {fmt(values['delta_brier'])}; ΔBAcc {fmt(values['delta_balanced_accuracy'])}.")
    lines.extend(["", "## Video-cluster bootstrap", "", "Paired 95% percentile CIs; each bootstrap draw resamples `scene_id` clusters jointly.", "", "| Comparison | Seed | ΔAUC 95% CI | ΔBrier 95% CI |", "|---|---:|---|---|"])
    for key, row in bootstrap["comparisons"].items():
        auc_ci = row["delta_roc_auc"]["ci_percentile_95"]
        brier_ci = row["delta_brier"]["ci_percentile_95"]
        lines.append(f"| {key.rsplit('_seed', 1)[0]} | {key.rsplit('_seed', 1)[1]} | [{auc_ci['lower_95']:.4f}, {auc_ci['upper_95']:.4f}] | [{brier_ci['lower_95']:.4f}, {brier_ci['upper_95']:.4f}] |")
    lines.extend(["", "## Reliability-stratified results", ""])
    for name, values in strata["strata"].items():
        lines.append(f"### {name}: N={values['sample_count']}")
        for variant in VARIANTS:
            aucs = [np.nan if values["models"][variant][str(seed)]["roc_auc"] is None else values["models"][variant][str(seed)]["roc_auc"] for seed in SEEDS]
            briers = [values["models"][variant][str(seed)]["brier"] for seed in SEEDS]
            lines.append(f"- {variant}: AUC {np.nanmean(aucs):.4f} ± {np.nanstd(aucs, ddof=1):.4f}; Brier {np.mean(briers):.4f} ± {np.std(briers, ddof=1):.4f}.")
        for comparison in ("B-A", "D-A", "D-B"):
            vals = values["deltas"][comparison]
            high = [vals[str(seed)]["delta_roc_auc"] for seed in SEEDS]
            lines.append(f"- {comparison} high/stratum AUC delta per seed: " + ", ".join("NA" if value is None else f"{value:+.4f}" for value in high) + ".")
    lines.extend([
        "",
        "## Gate diagnostics",
        "",
        "| Split | Gate | Mean | SD | q10 | q50 | q90 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for split in ("validation", "test"):
        for gate_name in ("motion_gate", "reliability_gate"):
            stats = gates[f"{gate_name}_{split}"]
            lines.append(f"| {split} | {gate_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['q10']:.4f} | {stats['q50']:.4f} | {stats['q90']:.4f} |")
    lines.extend([
        "",
        "D residual magnitude on test by seed (`|delta_logit|` mean; `|gate × delta_logit|` mean):",
    ])
    for seed in SEEDS:
        row = gates["test"]["residuals"]["D_reliability_gate"][str(seed)]
        lines.append(f"- Seed {seed}: {row['absolute_delta_logit']['mean_absolute']:.4f}; {row['absolute_gate_times_delta_logit']['mean_absolute']:.4f}.")
    lines.extend([
        "",
        "Crossing-label-specific gate statistics and low/medium/high uncertainty residuals are recorded in `gate_diagnostics.json`.",
        "",
        "## Gate shuffle",
        "",
        "Within train-defined motion deciles, 200 gate permutations per seed; diagnostic only, not used for selection.",
    ])
    for seed, row in shuffle["results"].items():
        lines.append(f"- Seed {seed}: true AUC={row['real_auc']:.4f}; shuffled AUC={fmt(row['shuffled_auc_mean_std'])}; ΔAUC={row['real_minus_shuffled_auc_mean']:+.4f}; true Brier={row['real_brier']:.4f}; shuffled Brier={fmt(row['shuffled_brier_mean_std'])}; ΔBrier={row['real_minus_shuffled_brier_mean']:+.4f}.")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- These are the results of the registered A/B/C/D comparison; they do not establish state-of-the-art performance or superiority over papers with different data/protocols.",
        "- Ensemble disagreement is a deterministic three-checkpoint spread proxy, not a complete estimate of aleatoric uncertainty.",
        "- D effective residual magnitudes are reported above; a near-zero contribution would not support a mechanism-effectiveness claim.",
        "- Gate-shuffle results are diagnostic only and were not used for model selection.",
        "- Future-shuffle diagnostic was not run.",
        "",
        "## Tests",
        "",
        f"Full pytest suite: {load('test_run_report.json')['passed']} passed, {load('test_run_report.json')['failed']} failed, {load('test_run_report.json')['skipped']} skipped.",
        "",
    ])
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULTS / 'summary.json'} and {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
