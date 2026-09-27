#!/usr/bin/env python3
"""One-pass official test evaluation after verifying the frozen protocol hash."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.reliability_gated_intent import ObservedOnlyIntent, ReliabilityGatedIntent
from scripts.reliability_gated_intent_utils import (
    apply_motion_adjustment,
    array_sha256,
    binary_metrics,
    cluster_bootstrap_paired_delta,
    empirical_confidence,
    mean_sample_std,
    paired_delta_metrics,
    reliability_tertile,
    train_defined_deciles,
)

OUTPUT_ROOT = PROJECT_ROOT / "results/reliability_gated_intent_15x15"
CACHE_ROOT = OUTPUT_ROOT / "cache"
TEST_PATH = PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz"
SEEDS = (42, 123, 2024)
VARIANTS = ("A_observed_only", "B_always_future", "C_motion_gate", "D_reliability_gate")
BOOTSTRAP_COMPARISONS = (("A", "D"), ("B", "D"), ("C", "D"), ("A", "B"))
SHORT_TO_FULL = {"A": "A_observed_only", "B": "B_always_future", "C": "C_motion_gate", "D": "D_reliability_gate"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key].copy() for key in raw.files}


def verify_frozen_protocol() -> tuple[dict[str, Any], str]:
    path = OUTPUT_ROOT / "protocol_frozen.json"
    sidecar = OUTPUT_ROOT / "protocol_frozen.sha256"
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError("Official test evaluation refused: frozen protocol and SHA sidecar are required")
    actual = sha256_file(path)
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    if actual != recorded:
        raise RuntimeError("Frozen protocol SHA256 mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("test_selection") is not False or protocol.get("data", {}).get("test_intent_labels_read_before_freeze") is not False:
        raise RuntimeError("Protocol does not certify the test holdout boundary")
    if sha256_file(TEST_PATH) != protocol["data"]["test_npz_sha256"]:
        raise RuntimeError("Official test NPZ differs from the version registered before freeze")
    return protocol, actual


def load_test_labels_after_freeze(features: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # This is the first point in the workflow where the official test labels are read.
    with np.load(TEST_PATH, allow_pickle=False) as source:
        labels = source["intent_label"].astype(np.int64, copy=True)
        crossing_labels = source["crossing_label"].astype(np.int64, copy=True)
        scene_ids = source["scene_id"].astype(str)
        target_ids = source["target_id"].astype(str)
        frames = source["obs_end_frame"].astype(np.int64, copy=True)
    if not set(np.unique(labels).tolist()).issubset({0, 1}) or not set(np.unique(crossing_labels).tolist()).issubset({0, 1}):
        raise ValueError("Official test contains non-clean intent labels")
    if not (np.array_equal(scene_ids, features["scene_id"]) and np.array_equal(target_ids, features["target_id"]) and np.array_equal(frames, features["obs_end_frame"])):
        raise RuntimeError("Test feature cache sample order/identity differs from the official test split")
    return labels, crossing_labels, scene_ids


def build_gates(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], transform_payload: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    polynomial = transform_payload["polynomial"]
    adjusted = {
        "train": apply_motion_adjustment(train["observed_motion_pixel"], train["u_mean_pixel"], polynomial),
        "val": apply_motion_adjustment(val["observed_motion_pixel"], val["u_mean_pixel"], polynomial),
        "test": apply_motion_adjustment(test["observed_motion_pixel"], test["u_mean_pixel"], polynomial),
    }
    # Match the frozen transform's fit path exactly: log1p is evaluated on the
    # cached float32 score, then the empirical-reference helper promotes to float64.
    motion_reference = np.sort(np.log1p(train["observed_motion_pixel"]), kind="mergesort").astype(np.float64)
    reliability_reference = np.sort(adjusted["train"], kind="mergesort")
    if array_sha256(motion_reference) != transform_payload["motion_cdf"]["reference_sha256"]:
        raise RuntimeError("Train motion CDF reference hash differs from the frozen transform")
    if array_sha256(reliability_reference) != transform_payload["adjusted_u_cdf"]["reference_sha256"]:
        raise RuntimeError("Train adjusted-U CDF reference hash differs from the frozen transform")
    motion_cdf = {"reference": motion_reference}
    reliability_cdf = {"reference": reliability_reference}
    output: dict[str, dict[str, np.ndarray]] = {}
    for split, data in (("train", train), ("val", val), ("test", test)):
        output[split] = {
            "motion": empirical_confidence(np.log1p(data["observed_motion_pixel"]), motion_cdf),
            "reliability": empirical_confidence(adjusted[split], reliability_cdf),
            "adjusted_u": adjusted[split],
        }
    return output


def predict(model: torch.nn.Module, variant: str, data: dict[str, np.ndarray], gate: np.ndarray | None, device: torch.device, batch_size: int = 512) -> dict[str, np.ndarray]:
    target = torch.from_numpy(data["target_obs"].astype(np.float32))
    if variant == "A_observed_only":
        loader = DataLoader(TensorDataset(target), batch_size=batch_size, shuffle=False)
    else:
        if gate is None:
            raise ValueError("A deterministic gate is required")
        future = torch.from_numpy(data["future_pred_mean"].astype(np.float32))
        gates = torch.from_numpy(np.asarray(gate, dtype=np.float32))
        loader = DataLoader(TensorDataset(target, future, gates), batch_size=batch_size, shuffle=False)
    model.eval()
    base: list[np.ndarray] = []
    delta: list[np.ndarray] = []
    final: list[np.ndarray] = []
    gate_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            if variant == "A_observed_only":
                output = model(batch[0].to(device))
                gate_value = torch.zeros_like(output["final_logit"])
                delta_value = torch.zeros_like(output["final_logit"])
            else:
                output = model(batch[0].to(device), batch[1].to(device), batch[2].to(device))
                gate_value = batch[2]
                delta_value = output["delta_logit"]
                future_rows.append(batch[1].numpy())
            base.append(output["base_logit"].cpu().numpy())
            delta.append(delta_value.cpu().numpy())
            final.append(output["final_logit"].cpu().numpy())
            gate_rows.append(gate_value.detach().cpu().numpy())
    return {
        "base_logit": np.concatenate(base),
        "delta_logit": np.concatenate(delta),
        "final_logit": np.concatenate(final),
        "gate": np.concatenate(gate_rows),
        "future_pred_mean": np.concatenate(future_rows) if future_rows else data["future_pred_mean"],
    }


def load_model(variant: str, seed: int, protocol: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    record = protocol["selected_models"][f"{variant}_seed{seed}"]
    checkpoint = PROJECT_ROOT / record["checkpoint"]
    if sha256_file(checkpoint) != record["checkpoint_sha256"]:
        raise RuntimeError(f"Model checkpoint SHA mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("variant") != variant or int(payload.get("seed", -1)) != seed:
        raise RuntimeError(f"Model checkpoint metadata mismatch: {checkpoint}")
    if variant == "A_observed_only":
        model: torch.nn.Module = ObservedOnlyIntent().to(device)
    else:
        model = ReliabilityGatedIntent().to(device)
        model.freeze_base()
    model.load_state_dict(payload["model_state"], strict=True)
    if variant != "A_observed_only":
        model.freeze_base()
    return model.eval(), record


def probability(logits: np.ndarray, temperature: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits) / temperature, -60, 60)))


def gate_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q10, q50, q90 = np.quantile(array, [0.1, 0.5, 0.9])
    return {"mean": float(array.mean()), "std": float(array.std()), "q10": float(q10), "q50": float(q50), "q90": float(q90), "min": float(array.min()), "max": float(array.max())}


def magnitude_summary(values: np.ndarray) -> dict[str, float]:
    array = np.abs(np.asarray(values, dtype=np.float64))
    return {"mean_absolute": float(array.mean()), "median_absolute": float(np.median(array)), "q90_absolute": float(np.quantile(array, .9))}


def paired_gate_diagnostics(
    split: str,
    labels: np.ndarray,
    gates: dict[str, np.ndarray],
    predictions: dict[str, dict[int, dict[str, np.ndarray]]],
    tertiles: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"split": split, "gate_by_crossing_label": {}, "residuals": {}}
    for gate_name in ("motion", "reliability"):
        values = gates[gate_name]
        result["gate_by_crossing_label"][gate_name] = {
            str(label): gate_summary(values[labels == label]) if np.any(labels == label) else None
            for label in (0, 1)
        }
        result["gate_by_crossing_label"][gate_name]["overall"] = gate_summary(values)
    for variant in ("B_always_future", "C_motion_gate", "D_reliability_gate"):
        rows: dict[str, Any] = {}
        for seed, values in predictions[variant].items():
            gate_delta = values["gate"] * values["delta_logit"]
            rows[str(seed)] = {
                "absolute_delta_logit": magnitude_summary(values["delta_logit"]),
                "absolute_gate_times_delta_logit": magnitude_summary(gate_delta),
                "by_reliability_tertile": {
                    ("low_uncertainty", "medium_uncertainty", "high_uncertainty")[index]: {
                        "sample_count": int(np.sum(tertiles == index)),
                        "absolute_delta_logit": magnitude_summary(values["delta_logit"][tertiles == index]) if np.any(tertiles == index) else None,
                        "absolute_gate_times_delta_logit": magnitude_summary(gate_delta[tertiles == index]) if np.any(tertiles == index) else None,
                    }
                    for index in range(3)
                },
            }
        rows["mean_across_seeds"] = {
            "absolute_delta_logit": {key: mean_sample_std([rows[str(seed)]["absolute_delta_logit"][key] for seed in predictions[variant]]) for key in ("mean_absolute", "median_absolute", "q90_absolute")},
            "absolute_gate_times_delta_logit": {key: mean_sample_std([rows[str(seed)]["absolute_gate_times_delta_logit"][key] for seed in predictions[variant]]) for key in ("mean_absolute", "median_absolute", "q90_absolute")},
        }
        result["residuals"][variant] = rows
    return result


def main() -> None:
    protocol, protocol_sha = verify_frozen_protocol()
    test = load_npz(CACHE_ROOT / "test_features_unlabeled.npz")
    train = load_npz(CACHE_ROOT / "train_oof_features.npz")
    val = load_npz(CACHE_ROOT / "val_features.npz")
    labels, crossing_labels, scene_ids = load_test_labels_after_freeze(test)
    if len(labels) != len(test["sample_index"]):
        raise RuntimeError("test sample count mismatch")
    transform_payload = json.loads((OUTPUT_ROOT / "reliability_transform.json").read_text(encoding="utf-8"))
    if sha256_file(OUTPUT_ROOT / "reliability_transform.json") != protocol["reliability_transform_sha256"]:
        raise RuntimeError("Frozen reliability transform SHA mismatch")
    gates = build_gates(train, val, test, transform_payload)
    cutpoints = np.asarray(protocol["reliability_tertile_cutpoints_train_oof_adjusted_u"], dtype=np.float64)
    test_tertiles = reliability_tertile(gates["test"]["adjusted_u"], cutpoints)
    val_tertiles = reliability_tertile(gates["val"]["adjusted_u"], cutpoints)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fixed, single unified model/seed/test pass. All later diagnostics reuse these logits.
    raw_predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {variant: {} for variant in VARIANTS}
    test_probabilities: dict[str, dict[int, np.ndarray]] = {variant: {} for variant in VARIANTS}
    model_records: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            model, record = load_model(variant, seed, protocol, device)
            gate = None
            if variant == "B_always_future":
                gate = np.ones(len(test["sample_index"]), dtype=np.float32)
            elif variant == "C_motion_gate":
                gate = gates["test"]["motion"]
            elif variant == "D_reliability_gate":
                gate = gates["test"]["reliability"]
            outputs = predict(model, variant, test, gate, device)
            raw_predictions[variant][seed] = outputs
            test_probabilities[variant][seed] = probability(outputs["final_logit"], float(record["temperature"]))
            model_records[f"{variant}_seed{seed}"] = record
            del model
        if seed % 2 == 0:
            print(json.dumps({"test_pass_seed_complete": seed, "protocol_sha256": protocol_sha}, ensure_ascii=False), flush=True)

    test_metrics: dict[str, Any] = {"protocol_sha256": protocol_sha, "split": "official_test", "models": {}}
    for variant in VARIANTS:
        by_seed: dict[str, Any] = {}
        for seed in SEEDS:
            record = model_records[f"{variant}_seed{seed}"]
            by_seed[str(seed)] = {
                **binary_metrics(labels, test_probabilities[variant][seed], float(record["threshold"])),
                "temperature": float(record["temperature"]),
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
        aggregate: dict[str, Any] = {}
        for metric in ("roc_auc", "brier", "ece_15_equal_width", "balanced_accuracy", "f1_positive", "negative_recall_specificity"):
            aggregate[metric] = mean_sample_std([by_seed[str(seed)][metric] for seed in SEEDS])
        test_metrics["models"][variant] = {"seeds": by_seed, "mean_sample_std": aggregate}
    write_json(OUTPUT_ROOT / "test_metrics.json", test_metrics)

    comparisons: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    for short_a, short_b in (("A", "D"), ("B", "D"), ("C", "D"), ("A", "B"), ("B", "C")):
        variant_a, variant_b = SHORT_TO_FULL[short_a], SHORT_TO_FULL[short_b]
        key = f"{short_b}-{short_a}"
        rows = {}
        for seed in SEEDS:
            threshold_a = model_records[f"{variant_a}_seed{seed}"]["threshold"]
            threshold_b = model_records[f"{variant_b}_seed{seed}"]["threshold"]
            rows[str(seed)] = paired_delta_metrics(labels, test_probabilities[variant_a][seed], test_probabilities[variant_b][seed], threshold_a, threshold_b)
            if (short_a, short_b) in BOOTSTRAP_COMPARISONS:
                bootstrap[f"{key}_seed{seed}"] = cluster_bootstrap_paired_delta(
                    labels, test_probabilities[variant_a][seed], test_probabilities[variant_b][seed], scene_ids,
                    repetitions=2000, seed=9124,
                )
        aggregate = {metric: mean_sample_std([rows[str(seed)][metric] for seed in SEEDS]) for metric in rows["42"]}
        comparisons[key] = {"per_seed": rows, "mean_sample_std": aggregate}
    write_json(OUTPUT_ROOT / "paired_deltas.json", {"protocol_sha256": protocol_sha, "comparisons": comparisons})
    write_json(OUTPUT_ROOT / "cluster_bootstrap.json", {"protocol_sha256": protocol_sha, "comparisons": bootstrap})

    strata_metrics: dict[str, Any] = {}
    for stratum_index, stratum_name in enumerate(("low_uncertainty", "medium_uncertainty", "high_uncertainty")):
        mask = test_tertiles == stratum_index
        y = labels[mask]
        rows: dict[str, Any] = {"sample_count": int(mask.sum()), "models": {}, "deltas": {}}
        for variant in VARIANTS:
            rows["models"][variant] = {}
            for seed in SEEDS:
                p = test_probabilities[variant][seed][mask]
                if np.unique(y).size == 2:
                    rows["models"][variant][str(seed)] = {
                        "roc_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, p)),
                        "brier": float(__import__("sklearn.metrics", fromlist=["brier_score_loss"]).brier_score_loss(y, p)),
                    }
                else:
                    rows["models"][variant][str(seed)] = {"roc_auc": None, "brier": float(np.mean((y - p) ** 2))}
        for short_a, short_b in (("A", "B"), ("A", "D"), ("B", "D")):
            a, b = SHORT_TO_FULL[short_a], SHORT_TO_FULL[short_b]
            rows["deltas"][f"{short_b}-{short_a}"] = {}
            for seed in SEEDS:
                pa, pb = test_probabilities[a][seed][mask], test_probabilities[b][seed][mask]
                if np.unique(y).size == 2:
                    auc_delta = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, pb) - __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y, pa))
                else:
                    auc_delta = None
                brier_delta = float(np.mean((y - pb) ** 2) - np.mean((y - pa) ** 2))
                rows["deltas"][f"{short_b}-{short_a}"][str(seed)] = {"delta_roc_auc": auc_delta, "delta_brier": brier_delta}
        strata_metrics[stratum_name] = rows
    write_json(OUTPUT_ROOT / "reliability_strata.json", {"protocol_sha256": protocol_sha, "cutpoints": cutpoints.tolist(), "strata": strata_metrics})

    val_labels = val["intent_label"].astype(np.int64)
    val_predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {variant: {} for variant in VARIANTS}
    for seed in SEEDS:
        for variant in VARIANTS:
            model, record = load_model(variant, seed, protocol, device)
            gate = None if variant == "A_observed_only" else (
                np.ones(len(val["sample_index"]), dtype=np.float32) if variant == "B_always_future" else
                gates["val"]["motion"] if variant == "C_motion_gate" else gates["val"]["reliability"]
            )
            val_predictions[variant][seed] = predict(model, variant, val, gate, device)
            del model
    val_gate_diag = paired_gate_diagnostics("official_val", val["crossing_label"].astype(np.int64), gates["val"], val_predictions, val_tertiles)
    test_gate_diag = paired_gate_diagnostics("official_test", crossing_labels, gates["test"], raw_predictions, test_tertiles)
    write_json(OUTPUT_ROOT / "gate_diagnostics.json", {
        "protocol_sha256": protocol_sha,
        "motion_gate_validation": gate_summary(gates["val"]["motion"]),
        "motion_gate_test": gate_summary(gates["test"]["motion"]),
        "reliability_gate_validation": gate_summary(gates["val"]["reliability"]),
        "reliability_gate_test": gate_summary(gates["test"]["reliability"]),
        "validation": val_gate_diag,
        "test": test_gate_diag,
        "D_negligible_residual_warning": {
            str(seed): bool(np.mean(np.abs(raw_predictions["D_reliability_gate"][seed]["gate"] * raw_predictions["D_reliability_gate"][seed]["delta_logit"])) < 1e-3)
            for seed in SEEDS
        },
    })

    # Within train-defined motion deciles, shuffle only D's reliability gate.
    motion_decile_cutpoints = train_defined_deciles(train["observed_motion_pixel"])
    test_deciles = np.searchsorted(motion_decile_cutpoints, test["observed_motion_pixel"], side="right")
    shuffle_report: dict[str, Any] = {}
    for seed in SEEDS:
        values = raw_predictions["D_reliability_gate"][seed]
        temp = float(model_records[f"D_reliability_gate_seed{seed}"]["temperature"])
        real_p = test_probabilities["D_reliability_gate"][seed]
        rng = np.random.RandomState(9124)
        auc_values: list[float] = []
        brier_values: list[float] = []
        for _ in range(200):
            shuffled_gate = values["gate"].copy()
            for group in range(10):
                indexes = np.flatnonzero(test_deciles == group)
                if len(indexes) > 1:
                    shuffled_gate[indexes] = shuffled_gate[rng.permutation(indexes)]
            shuffled_logits = values["base_logit"] + shuffled_gate * values["delta_logit"]
            shuffled_p = probability(shuffled_logits, temp)
            if np.unique(labels).size == 2:
                auc_values.append(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, shuffled_p)))
            brier_values.append(float(np.mean((labels - shuffled_p) ** 2)))
        shuffle_report[str(seed)] = {
            "real_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, real_p)),
            "real_brier": float(np.mean((labels - real_p) ** 2)),
            "shuffled_auc_mean_std": mean_sample_std(auc_values),
            "shuffled_brier_mean_std": mean_sample_std(brier_values),
            "real_minus_shuffled_auc_mean": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, real_p) - np.mean(auc_values)) if auc_values else None,
            "real_minus_shuffled_brier_mean": float(np.mean((labels - real_p) ** 2) - np.mean(brier_values)),
            "permutations": 200,
            "motion_decile_cutpoints_train_oof": motion_decile_cutpoints.tolist(),
            "gate_shuffle_preserves_base_logit_and_future_prediction": True,
        }
    write_json(OUTPUT_ROOT / "gate_shuffle.json", {"protocol_sha256": protocol_sha, "diagnostic_only": True, "results": shuffle_report})

    # A final released test feature cache with labels is created only after protocol verification.
    released = {key: value for key, value in test.items()}
    released["intent_label"] = labels
    released["crossing_label"] = crossing_labels
    np.savez_compressed(CACHE_ROOT / "test_features.npz", **released)
    write_json(OUTPUT_ROOT / "test_features_released_after_protocol.json", {
        "protocol_sha256": protocol_sha,
        "unlabeled_feature_cache_sha256": sha256_file(CACHE_ROOT / "test_features_unlabeled.npz"),
        "released_labeled_feature_cache_sha256": sha256_file(CACHE_ROOT / "test_features.npz"),
        "test_intent_labels_read_after_protocol_freeze": True,
        "source_test_npz_sha256": protocol["data"]["test_npz_sha256"],
        "sample_count": len(labels),
        "intent_label_counts": {str(value): int(np.sum(labels == value)) for value in np.unique(labels)},
        "crossing_label_counts": {str(value): int(np.sum(crossing_labels == value)) for value in np.unique(crossing_labels)},
    })
    print(json.dumps({"official_test_evaluation_complete": True, "protocol_sha256": protocol_sha, "metrics": str(OUTPUT_ROOT / "test_metrics.json")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
