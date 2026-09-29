#!/usr/bin/env python3
"""Consume the clean study's held-out test once, after protocol freeze."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_clean_utils import (
    ARMS, CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, map_video_ids,
    sha256_file, sha256_state, write_json,
)
from scripts.train_joint_transformer_gate import compute_metrics
from src.models.joint_transformer_gate import JointTransformerSceneGate

PROTOCOL_PATH = RESULTS_ROOT / "protocol_frozen.json"
PROTOCOL_SHA_PATH = RESULTS_ROOT / "protocol_frozen.sha256"
ACCESS_PATH = RESULTS_ROOT / "official_test_access_record.json"


def preflight() -> tuple[dict[str, Any], str]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SHA_PATH.is_file():
        raise RuntimeError("Clean test evaluation requires frozen protocol JSON and SHA256")
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    actual_sha = hashlib.sha256(protocol_bytes).hexdigest()
    declared_sha = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").split()[0]
    if actual_sha != declared_sha:
        raise RuntimeError("Clean protocol SHA256 mismatch")
    protocol = json.loads(protocol_bytes)
    if protocol.get("status") != "frozen_before_first_access_to_test_archive":
        raise RuntimeError("Protocol is not frozen before current study test access")
    if protocol.get("test_access_before_freeze", {}).get("clean_test_archive_opened_or_hashed") is not False:
        raise RuntimeError("Frozen protocol reports clean test access before freeze")
    if ACCESS_PATH.exists():
        raise RuntimeError("Clean test access marker already exists; refusing another evaluation")
    for relative, expected in protocol["sha256"]["frozen_pretest_artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen artifact changed: {relative}")
    for seed in SEEDS:
        for arm in ARMS:
            run_dir = RESULTS_ROOT / arm / f"seed{seed}"
            if (run_dir / "test_predictions.npz").exists() or (run_dir / "official_test_metrics.json").exists():
                raise RuntimeError(f"Test artifact exists before one-time evaluation: {run_dir}")
    return protocol, actual_sha


def build_model(state: dict[str, torch.Tensor], hidden_dim: int, gate_mode: str) -> JointTransformerSceneGate:
    scene_dim = int(state["scene_encoder.0.weight"].shape[1])
    pred_len = int(state["traj_head.3.weight"].shape[0] // 2)
    max_obs_len = int(state["position_embedding"].shape[1])
    model = JointTransformerSceneGate(
        input_dim=8, scene_dim=scene_dim, hidden_dim=hidden_dim,
        pred_len=pred_len, gate_mode=gate_mode, max_obs_len=max_obs_len,
    )
    model.load_state_dict(state, strict=True)
    return model


def evaluate_once(
    model: JointTransformerSceneGate,
    arrays: dict[str, np.ndarray],
    *, batch_size: int, device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    labels: list[int] = []
    logits: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    image_sizes: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["intent_label"]), batch_size):
            stop = min(start + batch_size, len(arrays["intent_label"]))
            sl = slice(start, stop)
            def tensor(key: str) -> torch.Tensor:
                return torch.from_numpy(np.asarray(arrays[key][sl], dtype=np.float32)).to(device)
            target = torch.cat([tensor("target_obs"), tensor("target_abs_obs")], dim=-1)
            output = model(
                target, tensor("neighbor_obs"), tensor("neighbor_mask"),
                tensor("neighbor_visible_mask"), tensor("scene_feat"),
            )
            labels.extend(arrays["intent_label"][sl].astype(np.int64).tolist())
            logits.extend(output["intent_logit"].cpu().numpy().astype(np.float64).tolist())
            predictions.append(output["future_pred"].cpu().numpy().astype(np.float32))
            targets.append(np.asarray(arrays["future_gt"][sl], dtype=np.float32))
            image_sizes.append(np.asarray(arrays["image_size"][sl], dtype=np.float32))
            gates.append(output["gate"].cpu().numpy())
            entropies.append(output["entropy"].cpu().numpy())
    metrics = compute_metrics(labels, logits, predictions, targets, image_sizes, gates, entropies)
    y = np.asarray(labels, dtype=np.int64)
    logit_array = np.asarray(logits, dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-logit_array))
    prediction = np.concatenate(predictions, axis=0)
    future_gt = np.concatenate(targets, axis=0)
    image_size = np.concatenate(image_sizes, axis=0)
    pixel_error = np.linalg.norm((prediction - future_gt) * image_size[:, None, :], axis=-1)
    payload = {
        "scene_id": np.asarray(arrays["scene_id"]).astype(str),
        "target_id": np.asarray(arrays["target_id"]).astype(str),
        "obs_end_frame": np.asarray(arrays["obs_end_frame"], dtype=np.int64),
        "intent_label": y,
        "intent_logit": logit_array,
        "intent_probability": probability,
        "predicted_class": (probability >= 0.5).astype(np.int64),
        "future_pred": prediction,
        "future_gt": future_gt,
        "image_size": image_size,
        "ade_by_sample_pixel": pixel_error.mean(axis=1).astype(np.float32),
        "fde_by_sample_pixel": pixel_error[:, -1].astype(np.float32),
    }
    return metrics, payload


def main() -> None:
    protocol, protocol_sha = preflight()
    test_relative = protocol["test_access_before_freeze"]["planned_test_archive_path"]
    test_path = ROOT / test_relative
    config = json.loads((ROOT / "configs/joint_traj_supervision_clean.json").read_text(encoding="utf-8"))
    settings = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load all selected states and historical ID-only reference artifacts before
    # opening this study's held-out archive, so structural errors fail preflight.
    models: dict[str, dict[str, JointTransformerSceneGate]] = {}
    for seed in SEEDS:
        models[str(seed)] = {}
        for arm in ARMS:
            path = CHECKPOINT_ROOT / "formal" / f"{arm}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            frozen = protocol["selected_checkpoints"][arm][str(seed)]
            if sha256_state(checkpoint["model"]) != frozen["selected_model_state_sha256"]:
                raise RuntimeError(f"Selected state hash mismatch for {arm}/seed{seed}")
            model = build_model(checkpoint["model"], int(settings["hidden_dim"]), str(settings["gate_mode"]))
            models[str(seed)][arm] = model.to(device)

    references: dict[str, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        reference_path = ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/test_predictions.npz"
        with np.load(reference_path, allow_pickle=False) as archive:
            required = ("scene_id", "target_id", "obs_end_frame", "video_id")
            if any(key not in archive.files for key in required):
                raise RuntimeError(f"Frozen sample-ID reference is incomplete: {reference_path}")
            references[str(seed)] = {key: archive[key].copy() for key in required}

    if not test_path.is_file():
        raise FileNotFoundError(f"Frozen held-out archive is missing: {test_path}")
    access = {
        "protocol_sha256": protocol_sha,
        "status": "official_test_access_started_do_not_rerun",
        "test_access_count": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_archive": test_relative,
        "test_archive_sha256": None,
        "sample_count": None,
        "per_seed_arm_status": {},
        "warning": "Marker is written before the single read. Any interruption consumes this test access; do not rerun.",
    }
    write_json(ACCESS_PATH, access)

    # Exactly one read of the held-out NPZ; all six selected models use this
    # same in-memory sample array and their outputs are saved for paired analysis.
    blob = test_path.read_bytes()
    archive_sha = hashlib.sha256(blob).hexdigest()
    expected = protocol["test_access_before_freeze"]["expected_archive_sha256_from_prior_frozen_protocol"]
    if archive_sha != expected:
        access.update({"status": "consumed_test_access_archive_hash_mismatch_do_not_rerun", "test_archive_sha256": archive_sha})
        write_json(ACCESS_PATH, access)
        raise RuntimeError("Test archive differs from the pre-existing frozen dataset fingerprint; no inference performed")
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    del blob
    required_arrays = (
        "target_obs", "target_abs_obs", "future_gt", "neighbor_obs", "neighbor_mask",
        "neighbor_visible_mask", "scene_feat", "intent_label", "image_size",
        "scene_id", "target_id", "obs_end_frame",
    )
    missing = [key for key in required_arrays if key not in arrays]
    if missing:
        access.update({"status": "consumed_test_access_schema_error_do_not_rerun", "test_archive_sha256": archive_sha, "missing_fields": missing})
        write_json(ACCESS_PATH, access)
        raise RuntimeError(f"Test archive lacks fields: {missing}")
    count = len(arrays["intent_label"])
    if any(len(arrays[key]) != count for key in required_arrays):
        access.update({"status": "consumed_test_access_length_error_do_not_rerun", "test_archive_sha256": archive_sha})
        write_json(ACCESS_PATH, access)
        raise RuntimeError("Test fields have inconsistent sample counts")

    ids = {
        key: np.asarray(arrays[key]).copy()
        for key in ("scene_id", "target_id", "obs_end_frame", "intent_label")
    }
    evaluation: dict[str, Any] = {
        "status": "official_test_evaluated_once",
        "protocol_sha256": protocol_sha,
        "test_archive_sha256": archive_sha,
        "test_sample_count": count,
        "prediction_pairing": "all six runs use identical ordered sample IDs, labels, future ground truth, and image sizes",
        "per_seed": {},
    }
    reference_seed = None
    for seed in SEEDS:
        seed_key = str(seed)
        video_ids = map_video_ids(references[seed_key], arrays)
        evaluation["per_seed"][seed_key] = {}
        for arm, traj_weight in ARMS.items():
            metrics, predictions = evaluate_once(
                models[seed_key][arm], arrays, batch_size=int(settings["batch_size"]), device=device,
            )
            predictions["video_id"] = video_ids.astype(str)
            if reference_seed is None:
                reference_seed = {key: predictions[key].copy() for key in (
                    "scene_id", "target_id", "obs_end_frame", "intent_label", "future_gt", "image_size",
                )}
            else:
                for key, reference in reference_seed.items():
                    if not np.array_equal(reference, predictions[key]):
                        access.update({"status": "consumed_test_access_pairing_error_do_not_rerun", "test_archive_sha256": archive_sha})
                        write_json(ACCESS_PATH, access)
                        raise RuntimeError(f"Test sample pairing failed for {arm}/seed{seed}: {key}")
            run_dir = RESULTS_ROOT / arm / f"seed{seed}"
            prediction_path = run_dir / "test_predictions.npz"
            metrics_path = run_dir / "official_test_metrics.json"
            np.savez_compressed(prediction_path, **predictions)
            write_json(metrics_path, {
                "seed": seed, "arm": arm, "traj_weight": traj_weight,
                "metrics": metrics,
                "raw_probability": "sigmoid(intent_logit); no calibration",
                "classification_threshold": 0.5,
                "selected_checkpoint_epoch": protocol["selected_checkpoints"][arm][seed_key]["selected_epoch"],
                "protocol_sha256": protocol_sha,
                "test_archive_sha256": archive_sha,
                "prediction_file": prediction_path.name,
            })
            evaluation["per_seed"][seed_key][arm] = {
                "metrics": metrics,
                "prediction_path": str(prediction_path.relative_to(ROOT)),
                "prediction_sha256": sha256_file(prediction_path),
                "metrics_path": str(metrics_path.relative_to(ROOT)),
            }
            access["per_seed_arm_status"][f"{arm}/seed{seed}"] = "evaluated_once"
            access.update({"test_archive_sha256": archive_sha, "sample_count": count})
            write_json(ACCESS_PATH, access)

    write_json(RESULTS_ROOT / "official_test_evaluation.json", evaluation)
    access.update({"status": "complete_one_time_official_test_evaluation", "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    write_json(ACCESS_PATH, access)
    print(json.dumps({
        "status": evaluation["status"], "test_sample_count": count,
        "test_archive_sha256": archive_sha,
        "per_seed_auc": {seed: {arm: row["metrics"]["intent_auc"] for arm, row in arms.items()} for seed, arms in evaluation["per_seed"].items()},
        "device": str(device),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
