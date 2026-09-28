#!/usr/bin/env python3
"""One-shot M1 official test evaluation, permitted only after protocol freeze."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrained_clone_intent_utils import RESULTS_ROOT, build_pretrained_clone, load_config, sha256_file, write_json
from scripts.trajectory_preserving_utils import SEEDS, intention_metrics, probabilities_from_logits, trajectory_metrics

PROTOCOL_PATH = RESULTS_ROOT / "protocol_frozen.json"
PROTOCOL_SHA_PATH = RESULTS_ROOT / "protocol_frozen.sha256"
TEST_PATH = ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz"
REQUIRED_ARRAYS = (
    "target_obs", "target_abs_obs", "scene_feat", "intent_label", "future_gt", "image_size",
    "scene_id", "video_id", "target_id", "obs_end_frame",
)


class TestArrays(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.target = torch.from_numpy(np.concatenate([arrays["target_obs"], arrays["target_abs_obs"]], axis=-1).astype(np.float32))
        self.scene = torch.from_numpy(arrays["scene_feat"].astype(np.float32))
        self.label = torch.from_numpy(arrays["intent_label"].astype(np.float32))
        self.future = torch.from_numpy(arrays["future_gt"].astype(np.float32))
        self.size = torch.from_numpy(arrays["image_size"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int):
        return self.target[index], self.scene[index], self.label[index], self.future[index], self.size[index]


def verify_protocol() -> tuple[dict[str, Any], str]:
    payload = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").strip().split()[0]
    if digest != expected:
        raise RuntimeError("Frozen M1 protocol checksum mismatch")
    protocol = json.loads(payload)
    if protocol.get("frozen") is not True or protocol.get("test_access_before_freeze") is not False:
        raise RuntimeError("M1 test protocol is not validly frozen")
    for relative, expected_hash in protocol["source_sha256"].items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    for relative, expected_hash in protocol["training_validation_data_sha256"].items():
        if sha256_file(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Frozen train/validation data changed: {relative}")
    for run in protocol["immutable_baseline_artifacts"].values():
        for row in run.values():
            if sha256_file(ROOT / row["metrics_path"]) != row["metrics_sha256"]:
                raise RuntimeError(f"Frozen baseline metrics changed: {row['metrics_path']}")
            if sha256_file(ROOT / row["predictions_path"]) != row["predictions_sha256"]:
                raise RuntimeError(f"Frozen baseline predictions changed: {row['predictions_path']}")
    return protocol, digest


def load_frozen_models(protocol: dict[str, Any]) -> dict[int, tuple[torch.nn.Module, dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for seed in SEEDS:
        row = protocol["selected_checkpoints"][str(seed)]
        checkpoint_path = ROOT / row["checkpoint"]
        if sha256_file(checkpoint_path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"Selected M1 checkpoint hash mismatch for seed {seed}")
        model, _ = build_pretrained_clone(seed, device=device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if int(checkpoint["selected_epoch"]) != int(row["selected_epoch"]):
            raise RuntimeError(f"Selected epoch mismatch for seed {seed}")
        model.intention_branch.load_state_dict(checkpoint["intention_model"], strict=True)
        model.eval()
        validation = json.loads((RESULTS_ROOT / f"seed{seed}/metrics_validation.json").read_text(encoding="utf-8"))
        if sha256_file(RESULTS_ROOT / f"seed{seed}/metrics_validation.json") != row["validation_metrics_sha256"]:
            raise RuntimeError(f"Frozen calibration artifact changed for seed {seed}")
        calibration = validation["selected_validation_calibration"]
        loaded[seed] = (model, {
            "device": device,
            "temperature": float(calibration["temperature"]),
            "threshold": float(calibration["threshold"]),
        })
    return loaded


def predict_all(models, arrays: dict[str, np.ndarray], batch_size: int) -> dict[int, dict[str, np.ndarray]]:
    dataset = TestArrays(arrays)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: dict[int, dict[str, list[np.ndarray]]] = {
        seed: {"logit": [], "future": []} for seed in models
    }
    for target, scene, _label, _future, _size in loader:
        for seed, (model, fit) in models.items():
            with torch.no_grad():
                target_device = target.to(fit["device"])
                scene_device = scene.to(fit["device"])
                intent = model.forward_intention(target_device)["intent_logit"]
                future = model.forward_trajectory(target_device, scene_device)
            outputs[seed]["logit"].append(intent.cpu().numpy())
            outputs[seed]["future"].append(future.cpu().numpy())
    return {
        seed: {name: np.concatenate(chunks, axis=0) for name, chunks in values.items()}
        for seed, values in outputs.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true", help="verify all frozen inputs without reading test.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    access_record = RESULTS_ROOT / "official_test_access_record.json"
    if not args.preflight_only and access_record.exists():
        raise RuntimeError("M1 official test access has already started; refusing a second access")
    protocol, protocol_hash = verify_protocol()
    if set(protocol["seeds"]) != set(SEEDS):
        raise RuntimeError("Frozen protocol does not include exactly the registered seeds")
    models = load_frozen_models(protocol)
    if args.preflight_only:
        print(json.dumps({"preflight": "passed", "test_archive_opened": False, "protocol_sha256": protocol_hash}, indent=2))
        return
    if any((RESULTS_ROOT / f"seed{seed}/test_predictions.npz").exists() for seed in SEEDS):
        raise RuntimeError("M1 test prediction output already exists; refusing to overwrite")

    # Make the single disk read explicit: hash bytes in memory, then let NumPy
    # decode the same bytes. A durable marker prevents accidental repeat use.
    access_record.write_text(json.dumps({
        "protocol_sha256": protocol_hash,
        "test_access_started": True,
        "test_archive_loaded": False,
        "evaluation_count": 1,
        "seed_set": list(SEEDS),
    }, indent=2) + "\n", encoding="utf-8")
    test_bytes = TEST_PATH.read_bytes()
    test_hash = hashlib.sha256(test_bytes).hexdigest()
    with np.load(io.BytesIO(test_bytes), allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise RuntimeError(f"Frozen test archive is missing required arrays: {missing}")
        arrays = {key: archive[key].copy() for key in REQUIRED_ARRAYS}
    if protocol.get("test_archive_sha256_reference_from_prior_reports") and test_hash != protocol["test_archive_sha256_reference_from_prior_reports"]:
        raise RuntimeError("Current test archive hash differs from the pre-registered M0/P1 test archive")
    predictions = predict_all(models, arrays, args.batch_size)
    labels = arrays["intent_label"].astype(np.int64).reshape(-1)
    image_size = arrays["image_size"].astype(np.float32)
    ground_truth = arrays["future_gt"].astype(np.float32)
    identity = {key: arrays[key] for key in ("scene_id", "video_id", "target_id", "obs_end_frame")}
    for seed in SEEDS:
        model_output = predictions[seed]
        fit = models[seed][1]
        logits = model_output["logit"].astype(np.float64).reshape(-1)
        future = model_output["future"].astype(np.float32)
        raw_probability = probabilities_from_logits(logits)
        calibrated = probabilities_from_logits(logits, fit["temperature"])
        predicted = (calibrated >= fit["threshold"]).astype(np.int64)
        intent_calibrated = intention_metrics(labels, logits, temperature=fit["temperature"], threshold=fit["threshold"])
        intent_raw = intention_metrics(labels, logits)
        trajectory = trajectory_metrics(future, ground_truth, image_size)

        # The pre-existing P1 artifact is an exact-order trajectory reference.
        p1_path = ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/test_predictions.npz"
        with np.load(p1_path, allow_pickle=False) as p1:
            if len(p1["intent_label"]) != len(labels) or not np.array_equal(p1["intent_label"], labels):
                raise RuntimeError(f"P1/M1 test label order mismatch for seed {seed}")
            for key, values in identity.items():
                if key not in p1.files or not np.array_equal(p1[key].astype(str), values.astype(str)):
                    raise RuntimeError(f"P1/M1 paired sample IDs differ for seed {seed}: {key}")
            p1_future = p1["future_prediction"].astype(np.float32)
        max_future_diff = float(np.max(np.abs(future - p1_future)))
        if max_future_diff >= float(load_config()["trajectory_preservation"]["max_abs_future_prediction_difference"]):
            raise RuntimeError(f"M1 trajectory prediction differs from P1 reference for seed {seed}: {max_future_diff}")

        run_dir = RESULTS_ROOT / f"seed{seed}"
        prediction_path = run_dir / "test_predictions.npz"
        np.savez_compressed(
            prediction_path,
            **identity,
            frame=arrays["obs_end_frame"],
            intent_label=labels,
            raw_logit=logits.astype(np.float32),
            raw_probability=raw_probability.astype(np.float32),
            calibrated_probability=calibrated.astype(np.float32),
            threshold=np.asarray(fit["threshold"], dtype=np.float32),
            predicted_label=predicted,
            future_prediction=future,
            future_ground_truth=ground_truth,
            image_size=image_size,
            ade_by_sample_pixel=(np.linalg.norm((future - ground_truth) * image_size[:, None, :], axis=-1).mean(axis=-1)).astype(np.float32),
            fde_by_sample_pixel=(np.linalg.norm((future - ground_truth) * image_size[:, None, :], axis=-1)[:, -1]).astype(np.float32),
        )
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["test"] = {
            "intent": intent_calibrated,
            "intent_raw_threshold_0_5": intent_raw,
            "trajectory": trajectory,
            "sample_count": len(labels),
            "checkpoint_sha256": protocol["selected_checkpoints"][str(seed)]["checkpoint_sha256"],
            "protocol_sha256": protocol_hash,
            "test_npz_sha256": test_hash,
            "calibration_source": "validation only",
            "paired_p1_order_verified": True,
            "max_abs_future_prediction_difference_vs_p1": max_future_diff,
            "predictions_file": "test_predictions.npz",
            "predictions_file_sha256": sha256_file(prediction_path),
        }
        metrics["test_evaluation_status"] = "evaluated_once_after_protocol_freeze"
        write_json(metrics_path, metrics)
        write_json(run_dir / "test_access_record.json", {
            "loaded_after_frozen_protocol": True,
            "test_access_started": True,
            "test_archive_loaded": True,
            "protocol_sha256": protocol_hash,
            "test_npz_sha256": test_hash,
            "sample_count": len(labels),
            "evaluation_count": 1,
        })
        print(json.dumps({"seed": seed, "test": metrics["test"]}, ensure_ascii=False, indent=2), flush=True)

    write_json(access_record, {
        "protocol_sha256": protocol_hash,
        "test_access_started": True,
        "test_archive_loaded": True,
        "test_npz_sha256": test_hash,
        "sample_count": len(labels),
        "evaluation_count": 1,
        "seeds_evaluated": list(SEEDS),
    })


if __name__ == "__main__":
    main()
