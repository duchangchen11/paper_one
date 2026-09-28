#!/usr/bin/env python3
"""One-time post-freeze official evaluation for all M0 seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trajectory_preserving_utils import (
    SEEDS,
    intention_metrics,
    probabilities_from_logits,
    sha256_file,
)
from scripts.train_intention_scratch_matched import model_sha256
from src.models.intention_scratch_transformer import IntentionScratchTransformer

RESULTS = ROOT / "results/intention_scratch_matched"
REQUIRED_ARRAYS = (
    "target_obs",
    "target_abs_obs",
    "intent_label",
    "scene_id",
    "target_id",
    "obs_end_frame",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_protocol(protocol_path: Path, checksum_path: Path) -> tuple[dict[str, Any], str]:
    encoded = protocol_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    if digest != expected:
        raise RuntimeError("Frozen protocol checksum mismatch; refusing test access")
    protocol = json.loads(encoded.decode("utf-8"))
    if protocol.get("frozen") is not True or protocol.get("m0_test_accessed_before_freeze") is not False:
        raise RuntimeError("M0 must have a valid frozen protocol before test access")
    for relative_path, expected_sha in protocol["source_sha256"].items():
        if sha256_file(ROOT / relative_path) != expected_sha:
            raise RuntimeError(f"Frozen source hash mismatch: {relative_path}")
    for relative_path, expected_sha in protocol["data_and_frozen_input_sha256"].items():
        if sha256_file(ROOT / relative_path) != expected_sha:
            raise RuntimeError(f"Frozen train/val/baseline input hash mismatch: {relative_path}")
    return protocol, digest


def _test_arrays_after_freeze(path: Path) -> tuple[dict[str, np.ndarray], str]:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise RuntimeError(f"Test archive lacks required arrays: {missing}")
        arrays = {key: archive[key].copy() for key in REQUIRED_ARRAYS}
    return arrays, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=RESULTS / "protocol_frozen.json",
    )
    parser.add_argument(
        "--protocol-sha256",
        type=Path,
        default=RESULTS / "protocol_frozen.sha256",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    access_record_path = RESULTS / "test_access_record.json"
    if access_record_path.exists():
        raise RuntimeError("M0 official test access already started; refusing a second evaluation")
    protocol, protocol_sha = verify_protocol(args.protocol, args.protocol_sha256)
    if tuple(protocol.get("seeds", [])) != SEEDS:
        raise RuntimeError("Frozen protocol seed list does not match the registered M0 seeds")
    p1_identity: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        run_dir = RESULTS / f"seed{seed}"
        metrics = load_json(run_dir / "metrics.json")
        if metrics.get("test") is not None or (run_dir / "test_predictions.npz").exists():
            raise RuntimeError(f"M0 seed {seed} already has test outputs")
        frozen = protocol["selected_checkpoints"][str(seed)]
        if sha256_file(ROOT / frozen["path"]) != frozen["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen selected checkpoint changed for seed {seed}")
        checkpoint = torch.load(ROOT / frozen["path"], map_location="cpu", weights_only=False)
        scratch = IntentionScratchTransformer(
            input_dim=8,
            d_model=int(protocol["model"]["transformer"]["hidden_dimension"]),
            nhead=int(protocol["model"]["transformer"]["heads"]),
            num_layers=int(protocol["model"]["transformer"]["layers"]),
            dropout=float(protocol["model"]["transformer"]["dropout"]),
            max_obs_len=15,
        )
        scratch.load_state_dict(checkpoint["model"], strict=True)
        if model_sha256(scratch) != frozen["selected_model_state_sha256"]:
            raise RuntimeError(f"Frozen M0 state hash mismatch for seed {seed}")
        p1_info = protocol["p1_baseline_artifacts"][str(seed)]
        p1_archive_path = ROOT / p1_info["predictions_path"]
        if sha256_file(p1_archive_path) != p1_info["predictions_sha256"]:
            raise RuntimeError(f"Frozen P1 prediction artifact hash mismatch, seed {seed}")
        with np.load(p1_archive_path, allow_pickle=False) as p1:
            p1_identity[seed] = {
                "scene_id": p1["scene_id"].astype(str),
                "target_id": p1["target_id"].astype(str),
                "obs_end_frame": p1["obs_end_frame"].astype(np.int64),
                "intent_label": p1["intent_label"].astype(np.int64),
            }

    access_record_path.parent.mkdir(parents=True, exist_ok=True)
    access_record_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_sha,
                "m0_test_access_started": True,
                "test_archive_loaded": False,
                "evaluation_count": 1,
                "seeds": list(SEEDS),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # This is the only code path that opens test.npz, after protocol and source
    # validation. All seeds share one in-memory read of the official split.
    test_path = ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz"
    arrays, test_sha = _test_arrays_after_freeze(test_path)
    if arrays["target_obs"].shape[-1] != 4 or arrays["target_abs_obs"].shape[-1] != 4:
        raise RuntimeError("Test input dimensions differ from frozen P1/M0 input definition")
    labels = arrays["intent_label"].astype(np.int64).reshape(-1)
    target_history = np.concatenate([arrays["target_obs"], arrays["target_abs_obs"]], axis=-1).astype(np.float32)
    if target_history.shape[1:] != (15, 8):
        raise RuntimeError(f"Test input shape {target_history.shape[1:]} differs from the frozen [15,8] protocol")
    scene_ids = arrays["scene_id"].astype(str)
    target_ids = arrays["target_id"].astype(str)
    frames = arrays["obs_end_frame"].astype(np.int64)
    if not (len(labels) == len(target_history) == len(scene_ids) == len(target_ids) == len(frames)):
        raise RuntimeError("Test metadata lengths do not match input sample count")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_tensor = torch.from_numpy(target_history)
    run_summaries: dict[str, Any] = {}
    for seed in SEEDS:
        run_dir = RESULTS / f"seed{seed}"
        checkpoint_path = ROOT / protocol["selected_checkpoints"][str(seed)]["path"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("pretrained_checkpoint_loaded") is not False:
            raise RuntimeError(f"M0 seed {seed} is not a scratch-initialized checkpoint")
        model = IntentionScratchTransformer(
            input_dim=8,
            d_model=int(protocol["model"]["transformer"]["hidden_dimension"]),
            nhead=int(protocol["model"]["transformer"]["heads"]),
            num_layers=int(protocol["model"]["transformer"]["layers"]),
            dropout=float(protocol["model"]["transformer"]["dropout"]),
            max_obs_len=15,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval()
        if model_sha256(model) != protocol["selected_checkpoints"][str(seed)]["selected_model_state_sha256"]:
            raise RuntimeError(f"M0 selected model state hash mismatch, seed {seed}")

        logits_parts: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(target_tensor), args.batch_size):
                output = model(target_tensor[start : start + args.batch_size].to(device))
                logits_parts.append(output["intent_logit"].cpu().numpy())
        logits = np.concatenate(logits_parts).astype(np.float64)

        metrics = load_json(run_dir / "metrics.json")
        calibration = metrics["selected_validation_calibration"]
        temperature = float(calibration["temperature"])
        threshold = float(calibration["threshold"])
        raw_probability = probabilities_from_logits(logits)
        calibrated_probability = probabilities_from_logits(logits, temperature)
        raw_metrics = intention_metrics(labels, logits, threshold=0.5)
        calibrated_metrics = intention_metrics(
            labels, logits, temperature=temperature, threshold=threshold
        )

        # Validate exact paired sample order against the previously frozen P1
        # baseline artifact; it is a saved prediction file, not a test archive.
        p1 = p1_identity[seed]
        same_ids = (
            np.array_equal(scene_ids, p1["scene_id"])
            and np.array_equal(target_ids, p1["target_id"])
            and np.array_equal(frames, p1["obs_end_frame"])
            and np.array_equal(labels, p1["intent_label"])
        )
        if not same_ids:
            raise RuntimeError(f"P1 and M0 held-out sample identities differ for seed {seed}")

        predicted_label = (calibrated_probability >= threshold).astype(np.int64)
        prediction_path = run_dir / "test_predictions.npz"
        np.savez_compressed(
            prediction_path,
            scene_id=scene_ids,
            video_id=scene_ids,
            target_id=target_ids,
            frame=frames,
            obs_end_frame=frames,
            intent_label=labels,
            raw_logit=logits.astype(np.float32),
            raw_probability=raw_probability.astype(np.float32),
            calibrated_probability=calibrated_probability.astype(np.float32),
            threshold=np.asarray(threshold, dtype=np.float32),
            predicted_label=predicted_label,
        )
        test_result = {
            "intent": calibrated_metrics,
            "intent_raw_uncalibrated_threshold_0_5": raw_metrics,
            "sample_count": int(len(labels)),
            "test_npz_sha256_after_protocol_freeze": test_sha,
            "protocol_sha256": protocol_sha,
            "calibration_source": "validation only",
            "p1_matched_test_sample_order_verified": True,
            "predictions_file": "test_predictions.npz",
            "predictions_file_sha256": sha256_file(prediction_path),
        }
        metrics["test"] = test_result
        metrics["test_evaluation_status"] = "evaluated_once_after_protocol_freeze"
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_summaries[str(seed)] = test_result

    access_record_path.write_text(
        json.dumps(
            {
                "protocol_sha256": protocol_sha,
                "m0_test_access_started": True,
                "test_archive_loaded": True,
                "test_npz_sha256": test_sha,
                "sample_count": int(len(labels)),
                "evaluation_count": 1,
                "evaluated_seeds": list(SEEDS),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"protocol_sha256": protocol_sha, "test_npz_sha256": test_sha, "test_archive_loads": 1, "runs": run_summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
