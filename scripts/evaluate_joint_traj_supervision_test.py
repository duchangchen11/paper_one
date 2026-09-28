#!/usr/bin/env python3
"""One-time, post-freeze official test evaluation for J0 and frozen J100."""

from __future__ import annotations

import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_utils import (
    CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, map_video_ids, sha256_file, write_json,
)
from scripts.train_joint_transformer_gate import compute_metrics
from src.models.joint_transformer_gate import JointTransformerSceneGate

PROTOCOL_PATH = RESULTS_ROOT / "protocol_frozen.json"
PROTOCOL_SHA_PATH = RESULTS_ROOT / "protocol_frozen.sha256"
ACCESS_PATH = RESULTS_ROOT / "official_test_access_record.json"


def preflight() -> tuple[dict[str, Any], str]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SHA_PATH.is_file():
        raise RuntimeError("Official test evaluation is forbidden until protocol_frozen.json and its SHA exist")
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    actual_protocol_sha = hashlib.sha256(protocol_bytes).hexdigest()
    recorded_protocol_sha = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").split()[0]
    if actual_protocol_sha != recorded_protocol_sha:
        raise RuntimeError("Frozen protocol SHA256 mismatch")
    protocol = json.loads(protocol_bytes)
    if protocol.get("status") != "frozen_before_first_access_to_test_archive":
        raise RuntimeError("Protocol is not in the frozen pre-test state")
    access = protocol.get("test_access_before_freeze", {})
    if access.get("test_dataset_loaded_or_evaluated_by_this_protocol") is not False or access.get("test_archive_opened_or_newly_hashed_by_this_protocol") is not False:
        raise RuntimeError("Protocol records pre-freeze test access")
    if ACCESS_PATH.exists():
        raise RuntimeError("The one-time official test access record already exists; refusing a second evaluation")
    hashes = protocol["sha256"]
    for section in ("config_sources_and_code", "training_and_validation_data_only", "pretest_and_comparator_artifacts"):
        for relative, expected in hashes[section].items():
            path = ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"Frozen artifact changed after protocol freeze: {relative}")
    return protocol, actual_protocol_sha


def make_model(state: dict[str, torch.Tensor], hidden_dim: int, gate_mode: str) -> JointTransformerSceneGate:
    scene_dim = int(state["scene_encoder.0.weight"].shape[1])
    pred_len = int(state["traj_head.3.weight"].shape[0] // 2)
    max_obs_len = int(state["position_embedding"].shape[1])
    model = JointTransformerSceneGate(
        input_dim=8, scene_dim=scene_dim, hidden_dim=hidden_dim,
        pred_len=pred_len, gate_mode=gate_mode, max_obs_len=max_obs_len,
    )
    model.load_state_dict(state, strict=True)
    return model


def official_eval(
    model: JointTransformerSceneGate,
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    prior_weight: float,
    traj_weight: float,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    sample_count = len(arrays["intent_label"])
    labels: list[float] = []
    logits: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    total_loss = 0.0
    total_items = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, sample_count, batch_size):
            stop = min(start + batch_size, sample_count)
            sl = slice(start, stop)
            to_tensor = lambda key: torch.from_numpy(np.asarray(arrays[key][sl]).astype(np.float32, copy=False)).to(device)
            target = torch.cat([to_tensor("target_obs"), to_tensor("target_abs_obs")], dim=-1)
            future = to_tensor("future_gt")
            output = model(
                target,
                to_tensor("neighbor_obs"),
                to_tensor("neighbor_mask"),
                to_tensor("neighbor_visible_mask"),
                to_tensor("scene_feat"),
            )
            label = torch.from_numpy(np.asarray(arrays["intent_label"][sl]).astype(np.float32, copy=False)).to(device)
            loss = F.binary_cross_entropy_with_logits(output["intent_logit"], label)
            loss = loss + prior_weight * F.binary_cross_entropy_with_logits(output["prior_logit"], label)
            loss = loss + traj_weight * F.smooth_l1_loss(output["future_pred"], future)
            count = stop - start
            total_loss += float(loss) * count
            total_items += count
            labels.extend(label.cpu().numpy().astype(np.int64).tolist())
            logits.extend(output["intent_logit"].cpu().numpy().astype(np.float64).tolist())
            predictions.append(output["future_pred"].cpu().numpy())
            targets.append(np.asarray(arrays["future_gt"][sl], dtype=np.float32))
            scales.append(np.asarray(arrays["image_size"][sl], dtype=np.float32))
            gates.append(output["gate"].cpu().numpy())
            entropies.append(output["entropy"].cpu().numpy())
    metrics = compute_metrics(labels, logits, predictions, targets, scales, gates, entropies)
    metrics["loss"] = float(total_loss / total_items)
    pred = np.concatenate(predictions, axis=0)
    gt = np.concatenate(targets, axis=0)
    image_size = np.concatenate(scales, axis=0)
    error_pixel = np.linalg.norm((pred - gt) * image_size[:, None, :], axis=-1)
    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
    payload = {
        "labels": np.asarray(labels, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
        "logits": np.asarray(logits, dtype=np.float64),
        "raw_intent_logit": np.asarray(logits, dtype=np.float64),
        "probability": probability,
        "raw_sigmoid_probability": probability,
        "threshold": np.asarray(0.5, dtype=np.float32),
        "predicted_class": (probability >= 0.5).astype(np.int64),
        "future_pred": pred.astype(np.float32),
        "future_gt": gt.astype(np.float32),
        "image_size": image_size.astype(np.float32),
        "ade_pixel_per_sample": error_pixel.mean(axis=1).astype(np.float32),
        "fde_pixel_per_sample": error_pixel[:, -1].astype(np.float32),
    }
    return metrics, payload


def historical_metric_differences(actual: dict[str, Any], historical: dict[str, Any]) -> dict[str, float]:
    keys = (
        "intent_accuracy", "intent_balanced_accuracy", "intent_f1", "intent_brier", "intent_auc",
        "trajectory_ade_normalized", "trajectory_fde_normalized", "trajectory_ade_pixel",
        "trajectory_fde_pixel", "gate_mean", "entropy_mean", "loss",
    )
    return {key: float(actual[key]) - float(historical[key]) for key in keys if key in actual and key in historical}


def main() -> None:
    protocol, protocol_sha = preflight()
    config = json.loads((ROOT / "configs/joint_traj_supervision_attribution.json").read_text(encoding="utf-8"))
    training = config["training"]
    test_path = ROOT / protocol["test_access_before_freeze"]["planned_test_archive_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all model states and the previously frozen P1 ID reference before
    # touching test.npz, so any structural problem is caught without holdout use.
    models: dict[str, dict[str, JointTransformerSceneGate]] = {}
    historical_metrics: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        models[str(seed)] = {}
        for arm, path in (
            ("J0", CHECKPOINT_ROOT / f"j0_seed{seed}.pt"),
            ("J100", ROOT / f"checkpoints/joint_loss_balance/lambda100_seed{seed}.pt"),
        ):
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            models[str(seed)][arm] = make_model(checkpoint["model"], int(training["hidden_dim"]), str(training["gate_mode"])).to(device)
        old_metrics = json.loads((ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json").read_text(encoding="utf-8"))
        historical_metrics[str(seed)] = old_metrics["test"]
    p1_paths = {
        str(seed): ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/test_predictions.npz"
        for seed in SEEDS
    }
    p1_refs: dict[str, dict[str, np.ndarray]] = {}
    for seed, path in p1_paths.items():
        with np.load(path, allow_pickle=False) as archive:
            required = ("scene_id", "target_id", "obs_end_frame", "video_id")
            if any(key not in archive.files for key in required):
                raise RuntimeError(f"P1 prediction reference lacks join keys: {path}")
            p1_refs[seed] = {key: archive[key].copy() for key in required}

    if test_path.name != "test.npz" or not test_path.is_file():
        raise FileNotFoundError(f"Frozen test archive is missing: {test_path}")
    if ACCESS_PATH.exists():
        raise RuntimeError("Official test access was already consumed")
    access_record: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "status": "official_test_access_started; do_not_rerun",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_archive": str(test_path.relative_to(ROOT)),
        "test_archive_sha256": None,
        "test_sample_count": None,
        "seed_status": {},
        "test_evaluation_count": 1,
        "warning": "This marker is written before the sole test archive read. Any interruption is a consumed test access and must not be retried without explicit protocol amendment.",
    }
    write_json(ACCESS_PATH, access_record)

    # Single filesystem read; checksum and parse the same in-memory bytes.
    blob = test_path.read_bytes()
    test_sha = hashlib.sha256(blob).hexdigest()
    expected_hashes = set(protocol["test_access_before_freeze"]["expected_archive_sha256_reused_from_frozen_M0_P1_records"].values())
    if expected_hashes != {test_sha}:
        access_record.update({"status": "consumed_test_access_archive_hash_mismatch; do_not_rerun", "test_archive_sha256": test_sha, "expected_historical_test_archive_sha256": sorted(expected_hashes)})
        write_json(ACCESS_PATH, access_record)
        raise RuntimeError("Test archive SHA differs from frozen historical M0/P1 references; no inference performed")
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    del blob
    required = (
        "target_obs", "target_abs_obs", "future_gt", "neighbor_obs", "neighbor_mask",
        "neighbor_visible_mask", "intent_label", "scene_feat", "image_size",
        "scene_id", "target_id", "obs_end_frame",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        access_record.update({"status": "consumed_test_access_schema_error", "test_archive_sha256": test_sha, "missing_fields": missing})
        write_json(ACCESS_PATH, access_record)
        raise RuntimeError(f"Held-out archive missing required fields: {missing}")
    sample_count = len(arrays["intent_label"])
    if any(len(arrays[key]) != sample_count for key in required):
        access_record.update({"status": "consumed_test_access_length_error", "test_archive_sha256": test_sha})
        write_json(ACCESS_PATH, access_record)
        raise RuntimeError("Held-out arrays do not have consistent sample counts")

    final: dict[str, Any] = {"protocol_sha256": protocol_sha, "test_archive_sha256": test_sha, "test_sample_count": sample_count, "per_seed": {}}
    for seed in SEEDS:
        seed_key = str(seed)
        video_ids = map_video_ids(p1_refs[seed_key], arrays)
        final["per_seed"][seed_key] = {}
        for arm, arm_label, weight in (("J0", "J0", 0.0), ("J100", "J100", 100.0)):
            metrics, prediction_payload = official_eval(
                models[seed_key][arm], arrays,
                batch_size=int(training["batch_size"]),
                prior_weight=float(training["prior_weight"]), traj_weight=weight, device=device,
            )
            prediction_payload.update({
                "scene_id": np.asarray(arrays["scene_id"]).astype(str),
                "target_id": np.asarray(arrays["target_id"]).astype(str),
                "obs_end_frame": np.asarray(arrays["obs_end_frame"], dtype=np.int64),
                "video_id": video_ids.astype(str),
            })
            seed_dir = RESULTS_ROOT / "j0" / f"seed{seed}"
            if arm == "J0":
                metrics_path = seed_dir / "official_test_metrics.json"
                predictions_path = seed_dir / "test_predictions.npz"
            else:
                metrics_path = seed_dir / "j100_reference_test_metrics.json"
                predictions_path = seed_dir / "j100_reference_predictions.npz"
            write_json(metrics_path, metrics)
            np.savez_compressed(predictions_path, **prediction_payload)
            record: dict[str, Any] = {
                "metrics_path": str(metrics_path.relative_to(ROOT)),
                "metrics": metrics,
                "predictions_path": str(predictions_path.relative_to(ROOT)),
                "predictions_sha256": sha256_file(predictions_path),
                "probability_calibration": "none; raw sigmoid(logit), matching historical J100 evaluation",
                "classification_threshold": 0.5,
            }
            if arm == "J100":
                differences = historical_metric_differences(metrics, historical_metrics[seed_key])
                record["difference_from_historical_test_metrics"] = differences
                record["matches_historical_within_1e-5"] = all(abs(value) <= 1e-5 for value in differences.values())
                if not record["matches_historical_within_1e-5"]:
                    record["historical_verification_warning"] = "Re-evaluation differs from historical summary; preserve both and investigate without rerunning test."
            final["per_seed"][seed_key][arm_label] = record
        access_record["seed_status"][seed_key] = "evaluated_once"
        access_record.update({"test_archive_sha256": test_sha, "test_sample_count": sample_count})
        write_json(ACCESS_PATH, access_record)

    final["status"] = "official_test_evaluated_once"
    write_json(RESULTS_ROOT / "official_test_evaluation.json", final)
    access_record.update({"status": "complete_one_time_official_test_evaluation", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(ACCESS_PATH, access_record)
    print(json.dumps({"status": final["status"], "test_sample_count": sample_count, "test_archive_sha256": test_sha, "per_seed": {seed: {arm: row["metrics"] for arm, row in arms.items()} for seed, arms in final["per_seed"].items()}, "device": str(device)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
