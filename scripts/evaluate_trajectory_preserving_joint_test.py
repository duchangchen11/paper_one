#!/usr/bin/env python3
"""One-time official test evaluator guarded by the frozen P1/P2 protocol."""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.trajectory_preserving_utils import (
    backbone_sha256,
    intention_metrics,
    load_seed_backbone,
    probabilities_from_logits,
    sha256_file,
    trajectory_metrics,
)

METHODS = ("P1_target_only", "P2_target_scene")
SEEDS = (42, 123, 2024)
REQUIRED_TEST_ARRAYS = (
    "target_obs",
    "target_abs_obs",
    "scene_feat",
    "intent_label",
    "future_gt",
    "image_size",
    "scene_id",
    "target_id",
    "obs_end_frame",
)


def verify_frozen_protocol(protocol_path: Path, checksum_path: Path) -> tuple[dict[str, Any], str]:
    payload = protocol_path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    expected_hash = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    if actual_hash != expected_hash:
        raise RuntimeError("Protocol checksum mismatch; refusing to access test data")
    protocol = json.loads(payload.decode("utf-8"))
    if protocol.get("frozen") is not True or protocol.get("test_access_before_freeze") is not False:
        raise RuntimeError("A valid frozen protocol is required before loading test data")
    for relative_path, expected_source_hash in protocol["source_sha256"].items():
        if sha256_file(ROOT / relative_path) != expected_source_hash:
            raise RuntimeError(f"Frozen source hash mismatch for {relative_path}")
    return protocol, actual_hash


def load_test_archive_after_freeze(
    test_path: Path, protocol_path: Path, checksum_path: Path
) -> tuple[dict[str, np.ndarray], str, str]:
    """The sole test archive loader; protocol/source checks happen before np.load."""
    protocol, protocol_hash = verify_frozen_protocol(protocol_path, checksum_path)
    test_hash = sha256_file(test_path)
    with np.load(test_path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_TEST_ARRAYS) - set(archive.files))
        if missing:
            raise RuntimeError(f"Frozen test archive lacks required arrays: {missing}")
        arrays = {key: archive[key].copy() for key in REQUIRED_TEST_ARRAYS}
    return arrays, protocol_hash, test_hash


class FrozenTestDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.target = torch.from_numpy(
            np.concatenate([arrays["target_obs"], arrays["target_abs_obs"]], axis=-1).astype(np.float32)
        )
        self.scene_feat = torch.from_numpy(arrays["scene_feat"].astype(np.float32))
        self.intent_label = torch.from_numpy(arrays["intent_label"].astype(np.float32))
        self.future_gt = torch.from_numpy(arrays["future_gt"].astype(np.float32))
        self.image_size = torch.from_numpy(arrays["image_size"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.intent_label)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target": self.target[index],
            "scene_feat": self.scene_feat[index],
            "intent_label": self.intent_label[index],
            "future_gt": self.future_gt[index],
            "image_size": self.image_size[index],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    # Refuse a second attempt as soon as any official test access was initiated.
    metrics_path = args.output_root / "metrics.json"
    access_record_path = args.output_root / "test_access_record.json"
    if access_record_path.exists():
        raise RuntimeError("Official test access was already initiated for this seed/method")
    if metrics_path.is_file():
        existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if existing_metrics.get("test") is not None:
            raise RuntimeError("This seed/method already has official test results; refusing to rerun")

    # Validate freeze, source code and matched checkpoint before opening test.npz.
    protocol, protocol_hash = verify_frozen_protocol(args.protocol, args.protocol_sha256)
    if protocol.get("frozen") is not True or set(protocol.get("seeds", [])) != set(SEEDS):
        raise RuntimeError("Frozen protocol does not contain all registered seeds")
    if set(protocol.get("methods", [])) != set(METHODS):
        raise RuntimeError("Frozen protocol does not contain both registered methods")
    checkpoint_key = str(args.seed)
    frozen_checkpoint = protocol["selected_checkpoints"][args.method][checkpoint_key]
    if sha256_file(args.checkpoint) != frozen_checkpoint["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint SHA256 differs from the frozen validation-selected checkpoint")
    args.output_root.mkdir(parents=True, exist_ok=True)
    access_record_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_hash,
                "method": args.method,
                "seed": args.seed,
                "test_access_started": True,
                "test_archive_loaded": False,
                "evaluation_count": 1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    test_path = ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz"
    arrays, loaded_protocol_hash, test_hash = load_test_archive_after_freeze(
        test_path, args.protocol, args.protocol_sha256
    )
    if loaded_protocol_hash != protocol_hash:
        raise RuntimeError("Protocol hash changed during test evaluation")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != args.seed or checkpoint["method"] != args.method:
        raise RuntimeError("Requested seed/method does not match the selected checkpoint")
    intent_input = "target" if args.method == "P1_target_only" else "target_scene"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, load_report, trajectory_checkpoint_path, trajectory_checkpoint_sha = load_seed_backbone(
        args.seed,
        intent_input,
        device=device,
        input_dim=int(arrays["target_obs"].shape[-1] + arrays["target_abs_obs"].shape[-1]),
        scene_dim=int(arrays["scene_feat"].shape[-1]),
        observed_length=int(arrays["target_obs"].shape[1]),
        prediction_length=int(arrays["future_gt"].shape[1]),
    )
    if not load_report["complete"]:
        raise RuntimeError("Trajectory checkpoint loading report is incomplete")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if backbone_sha256(model) != checkpoint["trajectory_backbone_sha256"]:
        raise RuntimeError("Frozen trajectory backbone hash changed since intention training")
    if trajectory_checkpoint_sha != checkpoint["trajectory_checkpoint_sha256"]:
        raise RuntimeError("Selected model no longer matches its same-seed trajectory checkpoint")

    dataset = FrozenTestDataset(arrays)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    logits, labels, trajectory_prediction = [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["target"].to(device), batch["scene_feat"].to(device))
            logits.append(output["intent_logit"].cpu().numpy())
            labels.append(batch["intent_label"].numpy())
            trajectory_prediction.append(output["future_pred"].cpu().numpy())
    logits_array = np.concatenate(logits).astype(np.float64)
    labels_array = np.concatenate(labels).astype(np.int64)
    future_prediction = np.concatenate(trajectory_prediction).astype(np.float32)
    future_gt = arrays["future_gt"].astype(np.float32)
    image_size = arrays["image_size"].astype(np.float32)
    # Calibration is selected with validation logits and stored in the run metrics.
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    calibration = run_metrics["selected_validation_calibration"]
    temperature = float(calibration["temperature"])
    threshold = float(calibration["threshold"])
    test_intention = intention_metrics(
        labels_array, logits_array, temperature=temperature, threshold=threshold
    )
    test_trajectory = trajectory_metrics(future_prediction, future_gt, image_size)
    probability = probabilities_from_logits(logits_array, temperature)

    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_ids = arrays["scene_id"].astype(str)
    np.savez_compressed(
        args.output_root / "test_predictions.npz",
        scene_id=scene_ids,
        video_id=scene_ids,
        target_id=arrays["target_id"].astype(str),
        obs_end_frame=arrays["obs_end_frame"],
        intent_label=labels_array,
        intent_probability=probability.astype(np.float32),
        future_prediction=future_prediction,
        future_ground_truth=future_gt,
        image_size=image_size,
        ade_by_sample_pixel=np.linalg.norm(
            (future_prediction - future_gt) * image_size[:, None, :], axis=-1
        ).mean(axis=1).astype(np.float32),
        fde_by_sample_pixel=np.linalg.norm(
            (future_prediction - future_gt) * image_size[:, None, :], axis=-1
        )[:, -1].astype(np.float32),
    )
    result = {
        "intent": test_intention,
        "trajectory": test_trajectory,
        "sample_count": len(labels_array),
        "checkpoint_sha256": frozen_checkpoint["checkpoint_sha256"],
        "protocol_sha256": protocol_hash,
        "test_npz_sha256_after_freeze": test_hash,
        "calibration_source": "validation only",
        "trajectory_checkpoint": str(trajectory_checkpoint_path.relative_to(ROOT)),
        "trajectory_checkpoint_sha256": trajectory_checkpoint_sha,
        "predictions_file": "test_predictions.npz",
    }
    run_metrics["test"] = result
    run_metrics["test_evaluation_status"] = "evaluated_once_after_protocol_freeze"
    write_payload = json.dumps(run_metrics, ensure_ascii=False, indent=2) + "\n"
    metrics_path.write_text(write_payload, encoding="utf-8")
    access_record_path.write_text(
        json.dumps(
            {
                "loaded_after_frozen_protocol": True,
                "test_access_started": True,
                "test_archive_loaded": True,
                "protocol_sha256": protocol_hash,
                "test_npz_sha256": test_hash,
                "sample_count": len(labels_array),
                "evaluation_count": 1,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"method": args.method, "seed": args.seed, "test": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
