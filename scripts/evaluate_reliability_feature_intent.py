#!/usr/bin/env python3
"""Run the single frozen-protocol test evaluation for reliability features."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reliability_feature_intent_utils import (
    PRIMARY_COMPARISONS,
    SEEDS,
    VARIANT_FEATURES,
    VARIANT_NAMES,
    add_normalized_reliability,
    apply_motion_adjustment,
    binary_metrics,
    bootstrap_comparison,
    get_model_inputs,
    load_json,
    load_npz,
    metric_pair_delta,
    sha256_file,
    summarize_seed_metrics,
    write_json,
)
from src.models.reliability_feature_intent import ReliabilityFeatureIntent


SOURCE_ROOT = PROJECT_ROOT / "results/reliability_gated_intent_15x15"
OUTPUT_ROOT = PROJECT_ROOT / "results/reliability_feature_intent_15x15"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/reliability_feature_intent_15x15"
MODELS = ("A", "B", "C", "D", "E", "D_no_future")


def sigmoid(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))


@torch.no_grad()
def predict(
    model: ReliabilityFeatureIntent,
    data: dict[str, np.ndarray],
    variant: str,
    device: torch.device,
) -> np.ndarray:
    obs, future, reliability = get_model_inputs(data, variant)
    tensors = [torch.from_numpy(obs).to(device)]
    future_tensor = torch.from_numpy(future).to(device) if future is not None else None
    reliability_tensor = (
        torch.from_numpy(reliability).to(device) if reliability is not None else None
    )
    model.eval()
    batch_size = 512
    logits: list[np.ndarray] = []
    for start in range(0, len(obs), batch_size):
        stop = min(start + batch_size, len(obs))
        result = model(
            tensors[0][start:stop],
            future_tensor[start:stop] if future_tensor is not None else None,
            reliability_tensor[start:stop] if reliability_tensor is not None else None,
        )
        logits.append(result["final_logit"].cpu().numpy())
    return np.concatenate(logits).astype(np.float64)


def check_frozen_protocol() -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol_path = OUTPUT_ROOT / "protocol_frozen.json"
    sha_path = OUTPUT_ROOT / "protocol_frozen.sha256"
    if not protocol_path.is_file() or not sha_path.is_file():
        raise FileNotFoundError("frozen protocol and SHA record must exist before test access")
    protocol_sha = sha256_file(protocol_path)
    recorded_sha = sha_path.read_text(encoding="utf-8").split()[0]
    if protocol_sha != recorded_sha:
        raise ValueError("frozen protocol SHA record mismatch")
    protocol = load_json(protocol_path)
    if protocol.get("test_protocol_frozen_before_labeled_test_labels_read") is not True:
        raise ValueError("protocol does not certify pre-test freeze")
    normalization_path = OUTPUT_ROOT / "feature_normalization.json"
    if sha256_file(normalization_path) != protocol["feature_normalization"]["sha256"]:
        raise ValueError("feature normalization changed after protocol freeze")
    normalization = load_json(normalization_path)
    if normalization.get("fit_split") != "official_train_oof":
        raise ValueError("normalization was not fit on official train OOF")
    return protocol, normalization, protocol_sha


def test_feature_data(
    protocol: dict[str, Any], normalization: dict[str, Any]
) -> tuple[dict[str, np.ndarray], str]:
    labeled_path = SOURCE_ROOT / "cache/test_features.npz"
    expected_sha = protocol["data"]["labeled_test_cache_sha256_released_after_prior_protocol"]
    actual_sha = sha256_file(labeled_path)
    if actual_sha != expected_sha:
        raise ValueError("labeled test feature cache SHA does not match the frozen protocol")
    release = load_json(SOURCE_ROOT / "test_features_released_after_protocol.json")
    if release.get("released_labeled_feature_cache_sha256") != actual_sha:
        raise ValueError("test feature release record SHA mismatch")
    data = load_npz(labeled_path)
    required = {
        "sample_index",
        "scene_id",
        "target_obs",
        "future_pred_mean",
        "u_mean_pixel",
        "observed_motion_pixel",
        "intent_label",
    }
    if not required.issubset(data):
        raise ValueError(f"labeled test cache missing fields: {sorted(required - data.keys())}")
    forbidden = {"future_gt", "ADE", "FDE", "ade", "fde"} & data.keys()
    if forbidden:
        raise ValueError(f"forbidden ground-truth/error fields in intention feature cache: {sorted(forbidden)}")
    if len(data["sample_index"]) != protocol["data"]["test_sample_count_from_unlabeled_cache"]:
        raise ValueError("labeled test cache sample count differs from frozen unlabeled cache")
    if not np.isin(data["intent_label"], [0, 1]).all():
        raise ValueError("test intent labels are not binary")
    transform = load_json(SOURCE_ROOT / "reliability_transform.json")
    if sha256_file(SOURCE_ROOT / "reliability_transform.json") != protocol["motion_adjustment"]["transform_sha256"]:
        raise ValueError("previously frozen motion-adjustment transform changed")
    data["adjusted_u"] = apply_motion_adjustment(
        data["observed_motion_pixel"], data["u_mean_pixel"], transform["polynomial"]
    )
    data = add_normalized_reliability(data, normalization)
    return data, actual_sha


def evaluate_seed_models(
    data: dict[str, np.ndarray], protocol: dict[str, Any], device: torch.device
) -> tuple[dict[str, dict[int, np.ndarray]], dict[str, dict[int, np.ndarray]], dict[str, dict[str, Any]]]:
    probabilities: dict[str, dict[int, np.ndarray]] = {model: {} for model in MODELS}
    logits_by_model: dict[str, dict[int, np.ndarray]] = {model: {} for model in MODELS}
    record_lookup: dict[str, dict[str, Any]] = protocol["selected_models"]
    for seed in SEEDS:
        for variant in MODELS:
            record = record_lookup[f"{variant}_seed{seed}"]
            checkpoint_path = PROJECT_ROOT / record["checkpoint"]
            if sha256_file(checkpoint_path) != record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint SHA mismatch: {checkpoint_path}")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = ReliabilityFeatureIntent(variant)
            model.load_state_dict(payload["model_state"], strict=True)
            model.to(device)
            logits = predict(model, data, variant, device)
            logits_by_model[variant][seed] = logits
            probabilities[variant][seed] = sigmoid(logits, record["temperature"])
            del model, payload
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(json.dumps({"test_seed_complete": seed}, ensure_ascii=False), flush=True)
    return probabilities, logits_by_model, record_lookup


def summarize_strata(
    data: dict[str, np.ndarray],
    probabilities: dict[str, dict[int, np.ndarray]],
    records: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    cutpoints = np.asarray(
        protocol["reliability_tertile_cutpoints_train_oof_adjusted_u"], dtype=np.float64
    )
    strata_index = np.searchsorted(cutpoints, data["adjusted_u"], side="right")
    labels = data["intent_label"].astype(np.int64)
    result: dict[str, Any] = {
        "cutpoints_train_oof_adjusted_u": cutpoints.tolist(),
        "meaning": {
            "high_reliability": "lower adjusted_u residual; comparatively lower uncertainty and more reliable trajectory feature",
            "medium_reliability": "middle adjusted_u tertile",
            "low_reliability": "higher adjusted_u residual; comparatively higher uncertainty",
        },
        "test_percentile_normalization_used": False,
        "strata": {},
    }
    for index, name in enumerate(("high_reliability", "medium_reliability", "low_reliability")):
        mask = strata_index == index
        y = labels[mask]
        model_result: dict[str, Any] = {}
        for variant in MODELS:
            per_seed: dict[str, Any] = {}
            for seed in SEEDS:
                p = probabilities[variant][seed][mask]
                auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else None
                brier = float(brier_score_loss(y, p))
                per_seed[str(seed)] = {"roc_auc": auc, "brier": brier}
            model_result[variant] = {
                "per_seed": per_seed,
                "mean_sample_std": {
                    metric: {
                        "mean": float(np.mean(values)),
                        "sample_std": float(np.std(values, ddof=1)),
                        "n": len(values),
                    }
                    for metric in ("roc_auc", "brier")
                    if (values := [per_seed[str(seed)][metric] for seed in SEEDS]).count(None) == 0
                } | {
                    "brier": {
                        "mean": float(np.mean([per_seed[str(seed)]["brier"] for seed in SEEDS])),
                        "sample_std": float(np.std([per_seed[str(seed)]["brier"] for seed in SEEDS], ddof=1)),
                        "n": len(SEEDS),
                    }
                },
            }
        deltas: dict[str, Any] = {}
        for baseline, candidate in (("B", "D"), ("A", "D"), ("C", "D")):
            values: dict[str, Any] = {}
            for seed in SEEDS:
                pa = probabilities[baseline][seed][mask]
                pb = probabilities[candidate][seed][mask]
                delta_auc = (
                    float(roc_auc_score(y, pb) - roc_auc_score(y, pa))
                    if np.unique(y).size == 2
                    else None
                )
                values[str(seed)] = {
                    "delta_roc_auc": delta_auc,
                    "delta_brier": float(brier_score_loss(y, pb) - brier_score_loss(y, pa)),
                }
            deltas[f"{candidate}-{baseline}"] = values
        result["strata"][name] = {
            "sample_count": int(mask.sum()),
            "negative_count": int(np.sum(y == 0)),
            "positive_count": int(np.sum(y == 1)),
            "models": model_result,
            "deltas": deltas,
        }
    result["high_uncertainty_group_primary"] = result["strata"]["low_reliability"]
    return result


def main() -> None:
    protocol_path = OUTPUT_ROOT / "protocol_frozen.json"
    protocol, normalization, protocol_sha = check_frozen_protocol()
    if any(
        (OUTPUT_ROOT / filename).exists()
        for filename in ("test_metrics.json", "paired_delta.json", "cluster_bootstrap.json")
    ):
        raise FileExistsError("test evaluation outputs already exist; refusing a second test pass")

    # Only now, after checking the immutable protocol, open the released labeled test archive.
    data, test_cache_sha = test_feature_data(protocol, normalization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probabilities, logits_by_model, records = evaluate_seed_models(data, protocol, device)
    labels = data["intent_label"].astype(np.int64)

    test_metrics: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "test_cache_sha256": test_cache_sha,
        "test_labels_accessed_after_protocol_freeze": True,
        "split": "official_test",
        "models": {},
    }
    for variant in MODELS:
        by_seed: dict[str, Any] = {}
        for seed in SEEDS:
            record = records[f"{variant}_seed{seed}"]
            threshold = float(record["threshold"])
            by_seed[str(seed)] = {
                **binary_metrics(labels, probabilities[variant][seed], threshold),
                "temperature": float(record["temperature"]),
                "validation_threshold": threshold,
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
        test_metrics["models"][variant] = {
            "variant_name": VARIANT_NAMES[variant],
            "seeds": by_seed,
            "mean_sample_std": summarize_seed_metrics(by_seed),
        }
    write_json(OUTPUT_ROOT / "test_metrics.json", test_metrics)

    paired: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "direction": "candidate minus baseline; negative delta_brier is better",
        "comparisons": {},
    }
    bootstrap: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "bootstrap_unit": "scene_id video cluster",
        "paired_sampling": True,
        "comparisons": {},
    }
    for baseline, candidate in PRIMARY_COMPARISONS:
        key = f"{candidate}-{baseline}"
        per_seed: dict[str, Any] = {}
        bootstrap_per_seed: dict[str, Any] = {}
        for seed in SEEDS:
            p_base = probabilities[baseline][seed]
            p_candidate = probabilities[candidate][seed]
            base_record = records[f"{baseline}_seed{seed}"]
            candidate_record = records[f"{candidate}_seed{seed}"]
            per_seed[str(seed)] = metric_pair_delta(
                labels,
                p_base,
                p_candidate,
                float(base_record["threshold"]),
                float(candidate_record["threshold"]),
            )
            bootstrap_per_seed[str(seed)] = bootstrap_comparison(
                labels, p_base, p_candidate, data["scene_id"]
            )
        metric_keys = tuple(next(iter(per_seed.values())).keys())
        paired["comparisons"][key] = {
            "per_seed": per_seed,
            "mean_sample_std": {
                metric: {
                    "mean": float(np.mean([per_seed[str(seed)][metric] for seed in SEEDS])),
                    "sample_std": float(np.std([per_seed[str(seed)][metric] for seed in SEEDS], ddof=1)),
                    "n": len(SEEDS),
                }
                for metric in metric_keys
            },
        }
        bootstrap["comparisons"][key] = {
            "per_seed": bootstrap_per_seed,
            "mean_seed_delta": {
                metric: float(np.mean([per_seed[str(seed)][metric] for seed in SEEDS]))
                for metric in ("delta_roc_auc", "delta_brier")
            },
        }
    # Explicit aliases document the requested raw-vs-adjusted ablation naming.
    paired["requested_ablation_aliases"] = {"D_raw": "C", "D_adjusted": "D", "raw_vs_adjusted": "D-C"}
    write_json(OUTPUT_ROOT / "paired_delta.json", paired)
    write_json(OUTPUT_ROOT / "cluster_bootstrap.json", bootstrap)

    strata = summarize_strata(data, probabilities, records, protocol)
    strata["protocol_sha256"] = protocol_sha
    write_json(OUTPUT_ROOT / "reliability_strata.json", strata)

    access_record = {
        "protocol_sha256": protocol_sha,
        "test_cache_sha256": test_cache_sha,
        "test_cache_matches_frozen_protocol": True,
        "labels_read_after_protocol_freeze": True,
        "labels_used_for_training_selection_or_normalization": False,
        "sample_count": int(len(labels)),
        "evaluation_pass": "single unified A/B/C/D/E/D_no_future pass",
    }
    write_json(OUTPUT_ROOT / "test_access_record.json", access_record)
    print(
        json.dumps(
            {
                "test_evaluation_complete": True,
                "protocol_sha256": protocol_sha,
                "test_sample_count": len(labels),
                "test_metrics": str(OUTPUT_ROOT / "test_metrics.json"),
                "bootstrap_comparisons": list(bootstrap["comparisons"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
