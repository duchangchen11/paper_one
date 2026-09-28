#!/usr/bin/env python3
"""Evaluate frozen DGB checkpoints on JAAD test and save row-aligned predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_joint_transformer_gate import compute_metrics
from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_protocol(protocol_path: Path, checksum_path: Path) -> tuple[dict, str]:
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    actual = sha256_file(protocol_path)
    if actual != expected:
        raise RuntimeError("Frozen protocol SHA256 does not match its sidecar")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("frozen") is not True:
        raise RuntimeError("Protocol is not marked frozen; refusing to access test data")
    for relative_path, expected_hash in protocol["source_sha256"].items():
        if sha256_file(PROJECT_ROOT / relative_path) != expected_hash:
            raise RuntimeError(f"Frozen source hash mismatch: {relative_path}")
    return protocol, actual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2024), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    # Verify freeze and source integrity before opening test.npz.
    protocol, protocol_hash = verify_frozen_protocol(args.protocol, args.protocol_sha256)
    if args.seed not in protocol.get("seeds", []):
        raise RuntimeError("Requested seed is not covered by the frozen protocol")
    expected_checkpoint_hash = protocol["validation_only_review"][str(args.seed)]["checkpoint_sha256"]
    if sha256_file(args.checkpoint) != expected_checkpoint_hash:
        raise RuntimeError("Checkpoint differs from the validation-selected frozen checkpoint")
    test_path = args.data_root / "test.npz"
    expected_test_hash = protocol["data_sha256"]["data/processed/jaad_sequences_scene_15x15/test.npz"]
    if sha256_file(test_path) != expected_test_hash:
        raise RuntimeError("JAAD test archive differs from the frozen protocol data hash")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_args = checkpoint["args"]
    if not isinstance(training_args, dict):
        training_args = vars(training_args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JAADSequenceDataset(test_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(dataset.scene_feat.shape[-1]),
        hidden_dim=int(training_args["hidden_dim"]),
        pred_len=dataset.future_gt.shape[1],
        gate_mode=training_args["gate_mode"],
        max_obs_len=dataset.target_obs.shape[1],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    logits: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            output = model(
                target,
                batch["neighbor_obs"].to(device),
                batch["neighbor_mask"].to(device),
                batch["neighbor_visible_mask"].to(device),
                batch["scene_feat"].to(device),
            )
            logits.append(output["intent_logit"].cpu().numpy())
            predictions.append(output["future_pred"].cpu().numpy())
            labels.append(batch["intent_label"].numpy())

    raw = np.load(test_path, allow_pickle=False)
    y_logits = np.concatenate(logits)
    y_labels = np.concatenate(labels)
    y_prediction = np.concatenate(predictions)
    future_gt = raw["future_gt"].astype(np.float32)
    image_size = raw["image_size"].astype(np.float32)
    scale = np.broadcast_to(image_size[:, None, :], future_gt.shape)
    pixel_error = np.linalg.norm((y_prediction - future_gt) * scale, axis=-1)
    metric_result = compute_metrics(
        y_labels,
        y_logits,
        [y_prediction],
        [future_gt],
        [image_size],
        [np.zeros_like(y_labels)],
        [np.zeros_like(y_labels)],
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    scene_ids = raw["scene_id"].astype(str)
    np.savez_compressed(
        args.output_root / "test_predictions.npz",
        scene_id=scene_ids,
        # In the processed JAAD archives scene_id is the source video identifier.
        video_id=scene_ids,
        target_id=raw["target_id"].astype(str),
        obs_end_frame=raw["obs_end_frame"],
        intent_label=y_labels.astype(np.int64),
        intent_probability=(1.0 / (1.0 + np.exp(-np.clip(y_logits, -80.0, 80.0)))).astype(np.float32),
        future_prediction=y_prediction.astype(np.float32),
        future_ground_truth=future_gt,
        image_size=image_size,
        ade_by_sample_pixel=pixel_error.mean(axis=1).astype(np.float32),
        fde_by_sample_pixel=pixel_error[:, -1].astype(np.float32),
    )
    metrics_path = args.output_root / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["test"] = metric_result
    metrics["test_evaluation_status"] = "evaluated_after_frozen_protocol"
    metrics["test_protocol_sha256"] = protocol_hash
    metrics["test_predictions"] = "test_predictions.npz"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"seed": metrics["seed"], "test": metric_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
