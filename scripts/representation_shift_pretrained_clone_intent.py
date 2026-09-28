#!/usr/bin/env python3
"""Compare pretrained and M1 target-context representations on validation only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrained_clone_intent_utils import (
    RESULTS_ROOT,
    build_pretrained_clone,
    load_config,
    sha256_file,
    write_json,
)
from scripts.trajectory_preserving_utils import SEEDS


def feature_summary(features: np.ndarray) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_sample_l2_norm": float(np.linalg.norm(values, axis=1).mean()),
        "std_sample_l2_norm": float(np.linalg.norm(values, axis=1).std()),
        "per_dimension_variance": values.var(axis=0).tolist(),
        "mean_per_dimension_variance": float(values.var(axis=0).mean()),
    }


def linear_cka(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    return float(np.linalg.norm(cross, ord="fro") ** 2 / denominator) if denominator else 0.0


def contexts(model: torch.nn.Module, target: torch.Tensor, *, trajectory: bool) -> np.ndarray:
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(target), 256):
            batch = target[start : start + 256]
            if trajectory:
                embedded = model.input_projection(batch) + model.position_embedding[:, : batch.shape[1]]
                encoded = model.temporal_encoder(embedded)
                feature = encoded[:, -1]
            else:
                feature = model.encode_target(batch)
            output.append(feature.detach().cpu().numpy())
    return np.concatenate(output, axis=0)


def main() -> None:
    config = load_config()
    requested = int(config["representation_shift"]["validation_samples"])
    val_path = ROOT / config["training"]["validation_split"]
    # This is explicitly validation data, not the held-out test archive.
    with np.load(val_path, allow_pickle=False) as archive:
        target = np.concatenate([archive["target_obs"], archive["target_abs_obs"]], axis=-1).astype(np.float32)
    count = min(requested, len(target))
    if count < requested:
        raise RuntimeError(f"Representation audit needs {requested} validation samples, found {count}")
    target_tensor = torch.from_numpy(target[:count])

    payload: dict[str, Any] = {
        "comparison": "original pretrained trajectory target_context vs selected M1 intention target_context",
        "split": "validation",
        "sample_count": count,
        "selection_use": False,
        "per_seed": {},
    }
    for seed in SEEDS:
        model, _ = build_pretrained_clone(seed, device="cpu")
        metrics = __import__("json").loads((RESULTS_ROOT / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
        checkpoint_path = ROOT / metrics["checkpoint"]
        if sha256_file(checkpoint_path) != metrics["checkpoint_sha256"]:
            raise RuntimeError(f"M1 checkpoint hash mismatch for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.intention_branch.load_state_dict(checkpoint["intention_model"], strict=True)
        original = contexts(model.trajectory_branch, target_tensor, trajectory=True)
        adapted = contexts(model.intention_branch, target_tensor, trajectory=False)
        cosine = np.sum(original * adapted, axis=1) / np.maximum(
            np.linalg.norm(original, axis=1) * np.linalg.norm(adapted, axis=1), 1e-12
        )
        payload["per_seed"][str(seed)] = {
            "selected_epoch": int(checkpoint["selected_epoch"]),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "original_pretrained_target_context": feature_summary(original),
            "m1_finetuned_target_context": feature_summary(adapted),
            "per_sample_cosine_similarity": {
                "mean": float(cosine.mean()),
                "std": float(cosine.std()),
                "min": float(cosine.min()),
                "max": float(cosine.max()),
            },
            "linear_cka": linear_cka(original, adapted),
        }
    write_json(RESULTS_ROOT / "representation_shift.json", payload)
    print(__import__("json").dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
