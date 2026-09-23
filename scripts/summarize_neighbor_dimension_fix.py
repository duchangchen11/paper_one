#!/usr/bin/env python3
"""Aggregate legacy and corrected scene-social ablations across fixed seeds."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "neighbor_dimension_fix"
SEEDS = (42, 123, 2024)
MODES = ("none", "always", "uncertainty")
METRICS = (
    "intent_auc",
    "intent_balanced_accuracy",
    "intent_f1",
    "intent_brier",
    "trajectory_ade_normalized",
    "trajectory_fde_normalized",
    "gate_mean",
    "entropy_mean",
)
DISPLAY = {
    "intent_auc": "AUC",
    "intent_balanced_accuracy": "Balanced accuracy",
    "intent_f1": "F1",
    "intent_brier": "Brier",
    "trajectory_ade_normalized": "ADE (normalized)",
    "trajectory_fde_normalized": "FDE (normalized)",
    "gate_mean": "Gate mean",
    "entropy_mean": "Entropy mean",
}


def read_test(mode: str, seed: int, corrected: bool) -> dict[str, float]:
    suffix = "_neighborfix" if corrected else ""
    path = ROOT / "results" / f"scene_{mode}{suffix}_seed{seed}" / "metrics.json"
    with path.open(encoding="utf-8") as stream:
        metrics = json.load(stream)["test"]
    missing = set(METRICS) - metrics.keys()
    if missing:
        raise ValueError(f"{path} is missing test metrics: {sorted(missing)}")
    return {key: float(metrics[key]) for key in METRICS}


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std_sample": statistics.stdev(values),
        "per_seed": dict(zip(SEEDS, values, strict=True)),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def summarize(values: dict[str, dict[int, dict[str, float]]]) -> dict:
    return {
        mode: {
            metric: aggregate([values[mode][seed][metric] for seed in SEEDS])
            for metric in METRICS
        }
        for mode in MODES
    }


def pairwise(corrected: dict, left: str, right: str, metric: str) -> dict:
    differences = {
        seed: corrected[left][seed][metric] - corrected[right][seed][metric]
        for seed in SEEDS
    }
    return {
        "left": left,
        "right": right,
        "metric": metric,
        "mean_difference": statistics.mean(differences.values()),
        "per_seed_difference": differences,
        "left_higher_seed_count": sum(value > 0 for value in differences.values()),
    }


def build_markdown(payload: dict) -> str:
    lines = [
        "# Neighbor-dimension correction: final ablation summary",
        "",
        "All entries are test-set mean ± sample standard deviation across seeds 42, 123, and 2024. The corrected runs use the same recorded training protocol and splits as the legacy runs; only the neighbor tensor axis interpretation in the two social models was corrected.",
        "",
        "Lower is better for Brier, ADE, and FDE; higher is better for AUC, balanced accuracy, and F1. Gate and entropy means are descriptive.",
        "",
        "## Corrected results",
        "",
        "| Gate mode | AUC | Balanced accuracy | F1 | Brier | ADE (normalized) | FDE (normalized) | Gate mean | Entropy mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = payload["corrected"][mode]
        cells = [mode]
        for metric in METRICS:
            stat = row[metric]
            cells.append(f"{fmt(stat['mean'])} ± {fmt(stat['std_sample'])}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Legacy results (axis bug; historical comparison only)",
            "",
            "| Gate mode | AUC | Balanced accuracy | F1 | Brier | ADE (normalized) | FDE (normalized) | Gate mean | Entropy mean |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        row = payload["legacy"][mode]
        cells = [mode]
        for metric in METRICS:
            stat = row[metric]
            cells.append(f"{fmt(stat['mean'])} ± {fmt(stat['std_sample'])}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Side-by-side: legacy vs neighborfix",
            "",
            "Each cell is legacy → corrected, with mean ± sample standard deviation.",
            "",
            "| Gate mode | AUC | Balanced accuracy | F1 | Brier | ADE (normalized) | FDE (normalized) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in MODES:
        cells = [mode]
        for metric in METRICS[:6]:
            old_stat = payload["legacy"][mode][metric]
            new_stat = payload["corrected"][mode][metric]
            old_value = f"{fmt(old_stat['mean'])} ± {fmt(old_stat['std_sample'])}"
            new_value = f"{fmt(new_stat['mean'])} ± {fmt(new_stat['std_sample'])}"
            cells.append(f"{old_value} → {new_value}")
        lines.append("| " + " | ".join(cells) + " |")

    auc_none = payload["corrected"]["none"]["intent_auc"]["mean"]
    auc_always = payload["corrected"]["always"]["intent_auc"]["mean"]
    auc_uncertainty = payload["corrected"]["uncertainty"]["intent_auc"]["mean"]
    uncertainty_vs_none = payload["pairwise"]["uncertainty_vs_none_auc"]
    uncertainty_vs_always = payload["pairwise"]["uncertainty_vs_always_auc"]
    always_vs_none = payload["pairwise"]["always_vs_none_auc"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Corrected mean AUC: no-social {auc_none:.4f}, always-social {auc_always:.4f}, uncertainty gate {auc_uncertainty:.4f}.",
            f"- Uncertainty gate versus no-social: mean AUC difference {uncertainty_vs_none['mean_difference']:+.4f}; it is higher in {uncertainty_vs_none['left_higher_seed_count']}/{len(SEEDS)} seeds. This is not a consistent per-seed improvement.",
            f"- Uncertainty gate versus always-social: mean AUC difference {uncertainty_vs_always['mean_difference']:+.4f}; it is higher in {uncertainty_vs_always['left_higher_seed_count']}/{len(SEEDS)} seeds.",
            f"- Always-social versus no-social: mean AUC difference {always_vs_none['mean_difference']:+.4f}; it is higher in {always_vs_none['left_higher_seed_count']}/{len(SEEDS)} seeds.",
            "- Do not claim that social interaction or uncertainty gating reliably improves intent prediction from this three-seed ablation alone. Report all controls and seed variability.",
            "- Treat trajectory ADE/FDE separately from intent metrics; the best intent AUC mode need not be the best trajectory mode.",
            "",
            "## Run artifacts",
            "",
            "Per-seed corrected `metrics.json` files are in `results/scene_{none,always,uncertainty}_neighborfix_seed{42,123,2024}/`. Checkpoints remain local and are not part of the repository.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    corrected_raw = {
        mode: {seed: read_test(mode, seed, corrected=True) for seed in SEEDS}
        for mode in MODES
    }
    legacy_raw = {
        mode: {seed: read_test(mode, seed, corrected=False) for seed in SEEDS}
        for mode in MODES
    }
    corrected = summarize(corrected_raw)
    legacy = summarize(legacy_raw)
    payload = {
        "protocol": {
            "seeds": list(SEEDS),
            "epochs": 6,
            "batch_size": 256,
            "hidden_dim": 128,
            "learning_rate": 0.001,
            "traj_weight": 1.0,
            "prior_weight": 0.5,
            "ambiguous_weight": 0.2,
            "data_root": "data/processed/jaad_sequences_scene_v4",
            "ambiguous_root": "data/processed/jaad_ambiguous_scene_v4",
        },
        "aggregation": "mean and sample standard deviation (ddof=1) across the three seeds",
        "corrected": corrected,
        "legacy": legacy,
        "pairwise": {
            "uncertainty_vs_none_auc": pairwise(corrected_raw, "uncertainty", "none", "intent_auc"),
            "uncertainty_vs_always_auc": pairwise(corrected_raw, "uncertainty", "always", "intent_auc"),
            "always_vs_none_auc": pairwise(corrected_raw, "always", "none", "intent_auc"),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "summary.md").write_text(build_markdown(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}")
    print(f"Wrote {OUTPUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
