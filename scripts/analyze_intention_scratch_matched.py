#!/usr/bin/env python3
"""Paired P1-vs-M0 held-out comparison, scene bootstrap, and final report."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reliability_gated_intent_utils import cluster_bootstrap_paired_delta
from scripts.trajectory_preserving_utils import SEEDS, sha256_file

RESULTS = ROOT / "results/intention_scratch_matched"
P1_ROOT = ROOT / "results/trajectory_preserving_joint/P1_target_only"
HISTORICAL_PATH = ROOT / "results/reliability_gated_intent_15x15/test_metrics.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_sd(values: list[float], digits: int = 4) -> str:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return f"{array.mean():.{digits}f} ± {std:.{digits}f}"


def metrics_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "auc": float(metrics["roc_auc"]),
        "brier": float(metrics["brier"]),
        "f1": float(metrics["f1"]),
        "bacc": float(metrics["balanced_accuracy"]),
        "accuracy": float(metrics["accuracy"]),
    }


def paired_bootstrap_p1_minus_m0(
    labels: np.ndarray,
    m0_probability: np.ndarray,
    p1_probability: np.ndarray,
    scene_ids: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """The shared helper returns b-a, so pass M0 first and P1 second."""
    return cluster_bootstrap_paired_delta(
        labels,
        m0_probability,
        p1_probability,
        scene_ids,
        repetitions=repetitions,
        seed=seed,
    )


def main() -> None:
    protocol_path = RESULTS / "protocol_frozen.json"
    protocol_bytes = protocol_path.read_bytes()
    protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    expected_protocol_sha = (RESULTS / "protocol_frozen.sha256").read_text(encoding="utf-8").split()[0]
    if protocol_sha != expected_protocol_sha:
        raise RuntimeError("Frozen protocol checksum mismatch; refusing comparison")
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if protocol.get("frozen") is not True or protocol.get("m0_test_accessed_before_freeze") is not False:
        raise RuntimeError("A valid frozen M0 protocol is required before test comparison")
    access = read_json(RESULTS / "test_access_record.json")
    if not access.get("test_archive_loaded") or access.get("evaluation_count") != 1:
        raise RuntimeError("Official M0 test evaluation record is incomplete")
    if access.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Test access record refers to a different frozen protocol")

    historical = read_json(HISTORICAL_PATH)["models"]["A_observed_only"]["seeds"]
    observed: dict[int, dict[str, float]] = {}
    p1_runs: dict[int, dict[str, float]] = {}
    m0_runs: dict[int, dict[str, float]] = {}
    matched_deltas: dict[int, dict[str, float]] = {}
    bootstrap_results: dict[str, Any] = {}
    representation = read_json(RESULTS / "representation_analysis.json")
    cluster_positive_seeds = 0

    for seed in SEEDS:
        observed_row = historical[str(seed)]
        observed[seed] = {
            "auc": float(observed_row["roc_auc"]),
            "brier": float(observed_row["brier"]),
            "f1": float(observed_row["f1_positive"]),
            "bacc": float(observed_row["balanced_accuracy"]),
        }
        p1_metrics = read_json(P1_ROOT / f"seed{seed}/metrics.json")
        m0_metrics = read_json(RESULTS / f"seed{seed}/metrics.json")
        p1_test = metrics_row(p1_metrics["test"]["intent"])
        m0_test = metrics_row(m0_metrics["test"]["intent"])
        p1_runs[seed], m0_runs[seed] = p1_test, m0_test
        matched_deltas[seed] = {
            name: p1_test[name] - m0_test[name]
            for name in ("auc", "brier", "f1", "bacc", "accuracy")
        }

        p1_prediction_path = P1_ROOT / f"seed{seed}/test_predictions.npz"
        m0_prediction_path = RESULTS / f"seed{seed}/test_predictions.npz"
        with np.load(p1_prediction_path, allow_pickle=False) as p1, np.load(m0_prediction_path, allow_pickle=False) as m0:
            same_order = all(
                np.array_equal(p1[key], m0[m0_key])
                for key, m0_key in (
                    ("scene_id", "scene_id"),
                    ("target_id", "target_id"),
                    ("obs_end_frame", "obs_end_frame"),
                    ("intent_label", "intent_label"),
                )
            )
            if not same_order:
                raise RuntimeError(f"P1/M0 per-sample order mismatch for seed {seed}")
            labels = m0["intent_label"].astype(np.int64)
            scenes = m0["scene_id"].astype(str)
            p1_probability = p1["intent_probability"].astype(np.float64)
            m0_probability = m0["calibrated_probability"].astype(np.float64)
            bootstrap = paired_bootstrap_p1_minus_m0(
                labels,
                m0_probability,
                p1_probability,
                scenes,
                repetitions=int(protocol["paired_analysis"]["bootstrap_repetitions"]),
                seed=int(protocol["paired_analysis"]["bootstrap_seed"]),
            )
        auc_ci = bootstrap["delta_roc_auc"]["ci_percentile_95"]
        brier_ci = bootstrap["delta_brier"]["ci_percentile_95"]
        if auc_ci["lower_95"] > 0:
            cluster_positive_seeds += 1
        bootstrap_results[str(seed)] = {
            "sample_count": int(len(labels)),
            "scene_cluster_count": int(len(np.unique(scenes))),
            "delta_definition": "P1 minus M0; positive AUC favors P1, negative Brier favors P1",
            "point_estimate": {
                "delta_auc": matched_deltas[seed]["auc"],
                "delta_brier": matched_deltas[seed]["brier"],
            },
            "bootstrap": bootstrap,
        }

    mean_delta_auc = float(np.mean([matched_deltas[seed]["auc"] for seed in SEEDS]))
    positive_seed_count = sum(matched_deltas[seed]["auc"] > 0 for seed in SEEDS)
    decision_rule = protocol["transfer_decision_rule"]
    transfer_supported = (
        mean_delta_auc > 0
        and positive_seed_count >= int(decision_rule["minimum_positive_seed_delta_auc_count"])
        and cluster_positive_seeds >= int(decision_rule["minimum_seed_auc_cluster_ci_lower_above_zero_count"])
    )
    bootstrap_payload = {
        "protocol_sha256": protocol_sha,
        "method": "paired scene_id cluster bootstrap, P1 minus M0",
        "repetitions_per_seed": int(protocol["paired_analysis"]["bootstrap_repetitions"]),
        "bootstrap_seed": int(protocol["paired_analysis"]["bootstrap_seed"]),
        "seeds_not_pooled": True,
        "per_seed": bootstrap_results,
        "mean_matched_seed_delta_auc": mean_delta_auc,
        "positive_delta_auc_seeds": positive_seed_count,
        "seeds_with_cluster_95ci_lower_auc_above_zero": cluster_positive_seeds,
        "predeclared_descriptive_transfer_rule": decision_rule,
        "predeclared_descriptive_transfer_rule_met": transfer_supported,
        "note": "P1's previously frozen test predictions are reused as the matched baseline; M0 was first evaluated only after its own protocol freeze.",
    }
    (RESULTS / "cluster_bootstrap.json").write_text(
        json.dumps(bootstrap_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def aggregate(runs: dict[int, dict[str, float]], metric: str) -> str:
        return mean_sd([runs[seed][metric] for seed in SEEDS], 4)

    seed_delta_auc_text = ", ".join(
        f"{seed}: {matched_deltas[seed]['auc']:+.4f}" for seed in SEEDS
    )

    lines = [
        "# Input-Matched Scratch Intention Transformer vs P1",
        "",
        f"M0 protocol `{protocol['protocol_id']}` SHA256: `{protocol_sha}`. P1 is reused from its separately frozen prior run. P1 test predictions have therefore been previously generated; the M0 test split was not opened until this protocol was frozen. No M0 test result was used for checkpoint selection or tuning.",
        "",
        "All summary values are mean ± sample SD across the three matched seeds. Higher AUC/F1/BAcc/accuracy is better; lower Brier is better. Brier/F1/BAcc/accuracy use each method's validation-fitted temperature and threshold. AUC is invariant to temperature scaling.",
        "",
        "## Input and architecture match",
        "",
        "The pre-training equivalence audit passed. Both methods receive `concat(target_obs, target_abs_obs)` with shape `[B,15,8]`, use the same Linear(8→128), learned `[1,15,128]` positional embedding, 3-layer/4-head/128-d Transformer encoder, last-token context, and identical LayerNorm→Linear(128,128)→GELU→Dropout(0.1)→Linear(128,1) intention head. No scene input is used.",
        "",
        "| Method | Input | Initialization/training | AUC | Brier | F1 | BAcc | Accuracy |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        f"| Historical observed-only | 15×4 | No trajectory pretraining; GRU baseline | {mean_sd([observed[s]['auc'] for s in SEEDS])} | {mean_sd([observed[s]['brier'] for s in SEEDS])} | {mean_sd([observed[s]['f1'] for s in SEEDS])} | {mean_sd([observed[s]['bacc'] for s in SEEDS])} | — |",
        f"| M0 Scratch Transformer | 15×8 | Random init; encoder and head trained | {aggregate(m0_runs, 'auc')} | {aggregate(m0_runs, 'brier')} | {aggregate(m0_runs, 'f1')} | {aggregate(m0_runs, 'bacc')} | {aggregate(m0_runs, 'accuracy')} |",
        f"| P1 Pretrained Frozen | 15×8 | Trajectory-pretrained encoder frozen; head trained | {aggregate(p1_runs, 'auc')} | {aggregate(p1_runs, 'brier')} | {aggregate(p1_runs, 'f1')} | {aggregate(p1_runs, 'bacc')} | {aggregate(p1_runs, 'accuracy')} |",
        "",
        "Historical observed-only is context only, not the primary control: it uses 15×4 and a different GRU architecture.",
        "",
        "## Matched-seed held-out comparison (P1 − M0)",
        "",
        "| Seed | M0 AUC | P1 AUC | ΔAUC | M0 Brier | P1 Brier | ΔBrier | M0 F1 | P1 F1 | ΔF1 | M0 BAcc | P1 BAcc | ΔBAcc | M0 Acc | P1 Acc | ΔAcc |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        lines.append(
            f"| {seed} | {m0_runs[seed]['auc']:.4f} | {p1_runs[seed]['auc']:.4f} | {matched_deltas[seed]['auc']:+.4f} | "
            f"{m0_runs[seed]['brier']:.4f} | {p1_runs[seed]['brier']:.4f} | {matched_deltas[seed]['brier']:+.4f} | "
            f"{m0_runs[seed]['f1']:.4f} | {p1_runs[seed]['f1']:.4f} | {matched_deltas[seed]['f1']:+.4f} | "
            f"{m0_runs[seed]['bacc']:.4f} | {p1_runs[seed]['bacc']:.4f} | {matched_deltas[seed]['bacc']:+.4f} | "
            f"{m0_runs[seed]['accuracy']:.4f} | {p1_runs[seed]['accuracy']:.4f} | {matched_deltas[seed]['accuracy']:+.4f} |"
        )
    lines.extend(
        [
            f"| Mean Δ | — | — | {mean_delta_auc:+.4f} | — | — | {np.mean([matched_deltas[s]['brier'] for s in SEEDS]):+.4f} | — | — | {np.mean([matched_deltas[s]['f1'] for s in SEEDS]):+.4f} | — | — | {np.mean([matched_deltas[s]['bacc'] for s in SEEDS]):+.4f} | — | — | {np.mean([matched_deltas[s]['accuracy'] for s in SEEDS]):+.4f} |",
            "",
            "## Per-seed scene-cluster bootstrap (2,000 replicates each; seeds not pooled)",
            "",
            "| Seed | ΔAUC point | ΔAUC 95% CI | ΔBrier point | ΔBrier 95% CI | Video/scene clusters |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for seed in SEEDS:
        item = bootstrap_results[str(seed)]
        auc_ci = item["bootstrap"]["delta_roc_auc"]["ci_percentile_95"]
        brier_ci = item["bootstrap"]["delta_brier"]["ci_percentile_95"]
        lines.append(
            f"| {seed} | {item['point_estimate']['delta_auc']:+.4f} | [{auc_ci['lower_95']:+.4f}, {auc_ci['upper_95']:+.4f}] | "
            f"{item['point_estimate']['delta_brier']:+.4f} | [{brier_ci['lower_95']:+.4f}, {brier_ci['upper_95']:+.4f}] | {item['scene_cluster_count']} |"
        )

    validation_gap_rows = []
    for seed in SEEDS:
        p1_validation_auc = float(
            read_json(P1_ROOT / f"seed{seed}/metrics.json")["selected_validation_raw_metrics"]["roc_auc"]
        )
        m0_validation_auc = float(
            read_json(RESULTS / f"seed{seed}/metrics.json")["selected_validation_raw_metrics"]["roc_auc"]
        )
        validation_gap_rows.append(
            (seed, m0_validation_auc, m0_runs[seed]["auc"], p1_validation_auc, p1_runs[seed]["auc"])
        )
    lines.extend(
        [
            "",
            "## Validation-to-test AUC gaps",
            "",
            "AUC checkpoint selection used validation only; the held-out difference is included as a stability diagnostic.",
            "",
            "| Seed | M0 val AUC | M0 test AUC | M0 test−val | P1 val AUC | P1 test AUC | P1 test−val |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            *[
                f"| {seed} | {m0_val:.4f} | {m0_test:.4f} | {m0_test-m0_val:+.4f} | {p1_val:.4f} | {p1_test:.4f} | {p1_test-p1_val:+.4f} |"
                for seed, m0_val, m0_test, p1_val, p1_test in validation_gap_rows
            ],
        ]
    )

    p1_linear_auc = [representation["seeds"][str(seed)]["pretrained_frozen_linear_separability"]["roc_auc"] for seed in SEEDS]
    m0_linear_auc = [representation["seeds"][str(seed)]["scratch_trained_linear_separability"]["roc_auc"] for seed in SEEDS]
    p1_linear_brier = [representation["seeds"][str(seed)]["pretrained_frozen_linear_separability"]["brier"] for seed in SEEDS]
    m0_linear_brier = [representation["seeds"][str(seed)]["scratch_trained_linear_separability"]["brier"] for seed in SEEDS]
    lines.extend(
        [
            "",
            "## Validation representation diagnostic",
            "",
            f"Representation moments use the same first {representation['actual_validation_samples']} validation samples (requested 1,000); logistic regression is fit on each representation's train features and evaluated on the full validation split. This diagnostic did not affect checkpoint selection.",
            "",
            "| Representation | Linear val AUC | Linear val Brier |",
            "|---|---:|---:|",
            f"| P1 pretrained frozen | {mean_sd(p1_linear_auc)} | {mean_sd(p1_linear_brier)} |",
            f"| M0 scratch trained | {mean_sd(m0_linear_auc)} | {mean_sd(m0_linear_brier)} |",
            "",
            "Feature mean/std, mean L2 norm, per-dimension variance, and each seed's linear diagnostic are in `representation_analysis.json`.",
            "",
            "## Conclusions and requested questions",
            "",
            "1. **Input match:** yes; both consume the exact P1 `[15,8]` history concatenation with the same stored arrays and no added normalization.",
            "2. **Architecture match:** yes; all P1 target encoder and intention head tensor shapes/names mapped exactly in `architecture_equivalence.json`; M0 has no scene encoder or trajectory decoder.",
            f"3. **M0 scores:** AUC {aggregate(m0_runs, 'auc')}, Brier {aggregate(m0_runs, 'brier')}, F1 {aggregate(m0_runs, 'f1')}, BAcc {aggregate(m0_runs, 'bacc')} (per-seed values above).",
            f"4. **P1 scores:** AUC {aggregate(p1_runs, 'auc')}, Brier {aggregate(p1_runs, 'brier')}, F1 {aggregate(p1_runs, 'f1')}, BAcc {aggregate(p1_runs, 'bacc')}.",
            f"5. **Matched ΔAUC (P1−M0):** {mean_sd([matched_deltas[s]['auc'] for s in SEEDS])}; seed-specific deltas are {seed_delta_auc_text}.",
            f"6. **Direction:** P1 has higher AUC in {positive_seed_count}/3 seeds; this is not sufficient by itself to claim transfer.",
            f"7. **Bootstrap:** see per-seed paired scene-cluster intervals above. {cluster_positive_seeds}/3 AUC interval lower bounds are above zero; the rule (positive mean, at least 2/3 positive seeds, and at least 2/3 positive lower bounds) {'is met' if transfer_supported else 'is not met'}.",
            f"8. **Transfer judgment:** {'evidence supports a stable positive trajectory-pretraining transfer under this protocol, subject to the caveats below.' if transfer_supported else 'the results do not establish stable positive trajectory-pretraining transfer under the predeclared descriptive rule.'}",
            "9. **Does the comparison rule out ‘just 8D + Transformer’?** This is the correct input/architecture control, so P1−M0 measures the practical difference between a frozen trajectory-pretrained target representation and a scratch-trained target Transformer. Because P1 freezes its encoder while M0 trains its encoder, it does not isolate initialization alone from the frozen-vs-trainable strategy.",
            f"10. **Linear separability:** P1 train-fitted logistic regression validation AUC/Brier is {mean_sd(p1_linear_auc)}/{mean_sd(p1_linear_brier)}; M0 is {mean_sd(m0_linear_auc)}/{mean_sd(m0_linear_brier)}. These values are descriptive only and do not replace the nonlinear intention-head comparison.",
            "11. **Next phase:** do not proceed automatically. If the bootstrap rule is met, a narrowly scoped task-specific adapter study has support; otherwise first treat transfer as inconclusive and retain P1/M0 as controls. Do not begin partial-unfreezing in this task.",
            "",
            "## Reproducibility and caveats",
            "",
            f"- Frozen protocol SHA256: `{protocol_sha}`; M0 test archive was read once after freeze for all three seeds.",
            "- P1's stored predictions and metrics are reused from the earlier frozen experiment and were not recomputed. Thus P1's test output was previously available; the paired bootstrap is descriptive on this existing holdout, not a fresh blinded replication.",
            "- Cluster bootstrap resamples `scene_id` videos separately within each seed, 2,000 draws per seed; the three seeds are never pooled as if they were samples.",
            "- M0 selected checkpoints, initialization reports, validation histories, per-sample test predictions, and protocol hashes are retained under `results/intention_scratch_matched/`.",
            "",
        ]
    )
    output = RESULTS / "summary.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(output), "cluster_bootstrap": str(RESULTS / "cluster_bootstrap.json"), "mean_delta_auc": mean_delta_auc, "positive_seed_count": positive_seed_count, "transfer_rule_met": transfer_supported}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
