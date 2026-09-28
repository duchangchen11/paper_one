#!/usr/bin/env python3
"""Metadata-only recovery for the frozen M1 evaluator's missing video_id field.

The original frozen evaluator remains unchanged. This script is permitted only
when its first post-freeze attempt stopped at the archive-field check before
computing model metrics or writing predictions. It uses the exact (scene,
target, observation-end-frame) rows in frozen P1 outputs to attach video_id.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import evaluate_pretrained_clone_intent_test as frozen_eval
from scripts.pretrained_clone_intent_utils import RESULTS_ROOT, load_config, sha256_file, write_json
from scripts.trajectory_preserving_utils import SEEDS, intention_metrics, probabilities_from_logits, trajectory_metrics

ADDENDUM_PATH = RESULTS_ROOT / "protocol_frozen_addendum.json"
ADDENDUM_SHA_PATH = RESULTS_ROOT / "protocol_frozen_addendum.sha256"
ACCESS_PATH = RESULTS_ROOT / "official_test_access_record.json"
TEST_PATH = ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz"
IDENTITY_FIELDS = ("scene_id", "target_id", "obs_end_frame")
REQUIRED = (
    "target_obs", "target_abs_obs", "scene_feat", "intent_label", "future_gt", "image_size",
    "scene_id", "target_id", "obs_end_frame",
)


def row_key(arrays: dict[str, np.ndarray], index: int, fields=IDENTITY_FIELDS) -> tuple[str, ...]:
    return tuple(str(np.asarray(arrays[field]).reshape(-1)[index]) for field in fields)


def map_video_ids(reference: dict[str, np.ndarray], test_arrays: dict[str, np.ndarray]) -> np.ndarray:
    ref_keys = [row_key(reference, i) for i in range(len(reference["scene_id"]))]
    test_keys = [row_key(test_arrays, i) for i in range(len(test_arrays["scene_id"]))]
    if len(set(ref_keys)) != len(ref_keys) or len(set(test_keys)) != len(test_keys):
        raise ValueError("Duplicate scene/target/frame key prevents safe video_id mapping")
    lookup = {key: str(reference["video_id"][i]) for i, key in enumerate(ref_keys)}
    if set(lookup) != set(test_keys):
        raise ValueError("Test IDs do not exactly match the frozen P1 sample identities")
    return np.asarray([lookup[key] for key in test_keys], dtype=str)


def load_reference(seed: int) -> dict[str, np.ndarray]:
    path = ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/test_predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def write_addendum(protocol_hash: str, prior_attempt: dict[str, Any]) -> str:
    if ADDENDUM_PATH.exists() or ADDENDUM_SHA_PATH.exists():
        raise RuntimeError("Evaluator recovery addendum already exists")
    payload = {
        "addendum_id": "M1-evaluator-metadata-field-correction-v1",
        "parent_protocol_sha256": protocol_hash,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "reason": "The frozen test archive does not contain video_id; its IDs are reconstructed only by exact join to pre-registered, hash-verified P1 outputs.",
        "analysis_protocol_changed": False,
        "model_or_checkpoint_changed": False,
        "selection_or_calibration_changed": False,
        "test_labels_or_metrics_examined_during_failed_attempt": False,
        "failed_attempt": prior_attempt,
        "video_id_recovery": {
            "source": "frozen P1_target_only test_predictions.npz",
            "join_key": list(IDENTITY_FIELDS),
            "requirements": ["keys unique on both sides", "exact key-set match", "label parity checked before metric computation"],
        },
        "recovery_evaluator_sha256": sha256_file(Path(__file__)),
        "parent_evaluator_sha256": sha256_file(ROOT / "scripts/evaluate_pretrained_clone_intent_test.py"),
        "official_test_evaluation_count": 1,
    }
    write_json(ADDENDUM_PATH, payload)
    digest = hashlib.sha256(ADDENDUM_PATH.read_bytes()).hexdigest()
    ADDENDUM_SHA_PATH.write_text(f"{digest}  protocol_frozen_addendum.json\n", encoding="utf-8")
    return digest


def main() -> None:
    if not ACCESS_PATH.is_file():
        raise RuntimeError("Recovery is only allowed after the recorded failed frozen-evaluator attempt")
    previous = json.loads(ACCESS_PATH.read_text(encoding="utf-8"))
    if previous.get("test_archive_loaded") is not False or previous.get("evaluation_count") != 1:
        raise RuntimeError("Prior attempt state is not the expected pre-metrics field-validation failure")
    if ADDENDUM_PATH.exists() or ADDENDUM_SHA_PATH.exists():
        raise RuntimeError("Recovery addendum already exists; refusing repeat test access")
    for seed in SEEDS:
        if (RESULTS_ROOT / f"seed{seed}/test_predictions.npz").exists():
            raise RuntimeError(f"Seed {seed} already has test predictions")
        metrics = json.loads((RESULTS_ROOT / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
        if metrics.get("test") is not None:
            raise RuntimeError(f"Seed {seed} already has official test metrics")

    protocol, protocol_hash = frozen_eval.verify_protocol()
    models = frozen_eval.load_frozen_models(protocol)
    # The failed attempt only exposed required archive member names and stopped
    # before model inference or labels/metrics were accessed.
    prior_attempt = {
        "protocol_sha256": protocol_hash,
        "archive_opened_after_freeze": True,
        "archive_field_validation_failed": True,
        "missing_member": "video_id",
        "model_inference_started": False,
        "test_labels_or_metrics_computed": False,
        "predictions_written": False,
        "evaluation_count": 0,
    }
    addendum_hash = write_addendum(protocol_hash, prior_attempt)

    # Keep an explicit attempt log; the only official inference pass remains
    # the evaluation below, with no changes to frozen weights or decisions.
    write_json(RESULTS_ROOT / "failed_test_evaluation_attempt.json", prior_attempt)
    raw_bytes = TEST_PATH.read_bytes()
    test_hash = hashlib.sha256(raw_bytes).hexdigest()
    with np.load(io.BytesIO(raw_bytes), allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED) - set(archive.files))
        if missing:
            raise RuntimeError(f"Test archive lacks required model/data fields: {missing}")
        arrays = {name: archive[name].copy() for name in REQUIRED}
    expected_test_hash = protocol.get("test_archive_sha256_reference_from_prior_reports")
    if expected_test_hash and test_hash != expected_test_hash:
        raise RuntimeError("Held-out test archive hash differs from frozen M0/P1 reference")

    labels = arrays["intent_label"].astype(np.int64).reshape(-1)
    image_size = arrays["image_size"].astype(np.float32)
    ground_truth = arrays["future_gt"].astype(np.float32)
    video_ids: dict[int, np.ndarray] = {}
    references: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        reference = load_reference(seed)
        video_ids[seed] = map_video_ids(reference, arrays)
        references[seed] = reference
        if not np.array_equal(reference["intent_label"].astype(np.int64), labels):
            raise RuntimeError(f"Test labels/order do not match frozen P1 rows for seed {seed}")

    predictions = frozen_eval.predict_all(models, arrays, 512)
    config = load_config()
    for seed in SEEDS:
        model_prediction = predictions[seed]
        fit = models[seed][1]
        logits = model_prediction["logit"].astype(np.float64).reshape(-1)
        future = model_prediction["future"].astype(np.float32)
        raw_prob = probabilities_from_logits(logits)
        calibrated = probabilities_from_logits(logits, fit["temperature"])
        predicted = (calibrated >= fit["threshold"]).astype(np.int64)
        calibrated_metrics = intention_metrics(labels, logits, temperature=fit["temperature"], threshold=fit["threshold"])
        raw_metrics = intention_metrics(labels, logits)
        trajectory = trajectory_metrics(future, ground_truth, image_size)
        reference = references[seed]
        max_future_diff = float(np.max(np.abs(future - reference["future_prediction"].astype(np.float32))))
        if max_future_diff >= float(config["trajectory_preservation"]["max_abs_future_prediction_difference"]):
            raise RuntimeError(f"Trajectory output no longer matches frozen P1 for seed {seed}: {max_future_diff}")
        run_dir = RESULTS_ROOT / f"seed{seed}"
        prediction_path = run_dir / "test_predictions.npz"
        np.savez_compressed(
            prediction_path,
            scene_id=arrays["scene_id"], video_id=video_ids[seed], target_id=arrays["target_id"],
            obs_end_frame=arrays["obs_end_frame"], frame=arrays["obs_end_frame"],
            intent_label=labels, raw_logit=logits.astype(np.float32), raw_probability=raw_prob.astype(np.float32),
            calibrated_probability=calibrated.astype(np.float32), threshold=np.asarray(fit["threshold"], dtype=np.float32),
            predicted_label=predicted, future_prediction=future, future_ground_truth=ground_truth, image_size=image_size,
            ade_by_sample_pixel=np.linalg.norm((future-ground_truth)*image_size[:,None,:],axis=-1).mean(axis=-1).astype(np.float32),
            fde_by_sample_pixel=np.linalg.norm((future-ground_truth)*image_size[:,None,:],axis=-1)[:,-1].astype(np.float32),
        )
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["test"] = {
            "intent": calibrated_metrics,
            "intent_raw_threshold_0_5": raw_metrics,
            "trajectory": trajectory,
            "sample_count": len(labels),
            "checkpoint_sha256": protocol["selected_checkpoints"][str(seed)]["checkpoint_sha256"],
            "protocol_sha256": protocol_hash,
            "protocol_addendum_sha256": addendum_hash,
            "test_npz_sha256": test_hash,
            "calibration_source": "validation only",
            "paired_p1_order_verified": True,
            "video_id_source": "frozen P1 predictions joined on scene_id,target_id,obs_end_frame",
            "max_abs_future_prediction_difference_vs_p1": max_future_diff,
            "predictions_file": prediction_path.name,
            "predictions_file_sha256": sha256_file(prediction_path),
        }
        metrics["test_evaluation_status"] = "evaluated_once_after_protocol_freeze_with_metadata_addendum"
        write_json(metrics_path, metrics)
        write_json(run_dir / "test_access_record.json", {
            "loaded_after_frozen_protocol": True, "test_access_started": True, "test_archive_loaded": True,
            "protocol_sha256": protocol_hash, "protocol_addendum_sha256": addendum_hash,
            "test_npz_sha256": test_hash, "sample_count": len(labels), "official_evaluation_count": 1,
        })
        print(json.dumps({"seed": seed, "test": metrics["test"]}, ensure_ascii=False, indent=2), flush=True)

    write_json(ACCESS_PATH, {
        "protocol_sha256": protocol_hash,
        "protocol_addendum_sha256": addendum_hash,
        "test_archive_open_attempts": 2,
        "failed_field_validation_attempt": 1,
        "official_test_evaluation_count": 1,
        "test_archive_loaded": True,
        "test_npz_sha256": test_hash,
        "sample_count": len(labels),
        "seeds_evaluated": list(SEEDS),
    })


if __name__ == "__main__":
    main()
