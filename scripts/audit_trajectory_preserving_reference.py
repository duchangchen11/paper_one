#!/usr/bin/env python3
"""Inventory existing matched-seed trajectory-only checkpoints without retraining."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SEEDS = (42, 123, 2024)

from src.models.trajectory_transformer import SceneTrajectoryTransformer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    output = ROOT / "results/trajectory_preserving_joint"
    output.mkdir(parents=True, exist_ok=True)
    train_path = ROOT / "data/processed/jaad_sequences_scene_15x15/train.npz"
    with np.load(train_path, allow_pickle=False) as train:
        obs_len = int(train["target_obs"].shape[1])
        target_obs_dim = int(train["target_obs"].shape[-1])
        target_abs_dim = int(train["target_abs_obs"].shape[-1])
        scene_dim = int(train["scene_feat"].shape[-1])
        pred_len = int(train["future_gt"].shape[1])

    references = {}
    for seed in SEEDS:
        checkpoint_path = ROOT / "checkpoints" / f"trajectory_transformer_scene_15x15_seed{seed}.pt"
        metrics_path = ROOT / "results" / f"trajectory_transformer_scene_15x15_seed{seed}" / "metrics.json"
        if not checkpoint_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Missing existing trajectory-only artifact for seed {seed}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_seed = int(payload["args"]["seed"])
        if checkpoint_seed != seed:
            raise RuntimeError(f"Seed mismatch: requested {seed}, checkpoint metadata says {checkpoint_seed}")
        config = payload["args"]
        model = SceneTrajectoryTransformer(
            input_dim=target_obs_dim + target_abs_dim,
            scene_dim=scene_dim,
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            pred_len=pred_len,
            max_obs_len=obs_len,
        )
        incompatible = model.load_state_dict(payload["model"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"Strict standalone checkpoint load failed for seed {seed}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_record = next(row for row in metrics["history"] if row["epoch"] == metrics["best_epoch"])
        references[str(seed)] = {
            "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metrics_file": str(metrics_path.relative_to(ROOT)),
            "metrics_file_sha256": sha256_file(metrics_path),
            "scene_mode": config.get("scene_mode", "real"),
            "architecture": {
                "hidden_dim": int(config["d_model"]),
                "transformer_layers": int(config["num_layers"]),
                "attention_heads": 4,
                "observed_length": obs_len,
                "prediction_length": pred_len,
                "target_obs_dimension": target_obs_dim,
                "target_abs_obs_dimension": target_abs_dim,
                "trajectory_input_dimension": target_obs_dim + target_abs_dim,
                "scene_feature_dimension": scene_dim,
            },
            "checkpoint_tensor_count": len(payload["model"]),
            "checkpoint_selection": metrics.get("checkpoint_selection", "lowest validation pixel ADE"),
            "best_epoch": int(metrics["best_epoch"]),
            "validation": best_record["val"],
            "existing_official_test_reference": metrics["test"],
            "checkpoint_sha_verified_by_strict_model_load": True,
        }

    reference = {
        "source_commit": "671a3ca18986b552717fb3543755a61d1316f874",
        "model_class": "src.models.trajectory_transformer.SceneTrajectoryTransformer",
        "seeds": list(SEEDS),
        "reference_metric_source": "pre-existing results from the standalone trajectory-only experiments; no trajectory-only retraining was performed in this phase",
        "input_semantics": "concat(target_obs, target_abs_obs), feature dimension 8; scene_feat dimension 512",
        "prediction_semantics": "normalized future coordinates [batch, 15, 2]; ADE/FDE in the cited metrics are pixels",
        "per_seed": references,
    }
    (output / "trajectory_reference.json").write_text(
        json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    mapping = f"""# Trajectory-only architecture mapping

Reference source: `src/models/trajectory_transformer.py`, class
`SceneTrajectoryTransformer`; training implementation:
`scripts/train_trajectory_transformer.py`.

## Forward data path

1. Inputs are `target_obs` with shape `[B, 15, 4]`, `target_abs_obs` with shape
   `[B, 15, 4]`, and `scene_feat` with shape `[B, 512]`. The trainer concatenates
   the two target histories along their feature axis to form `[B, 15, 8]`.
2. `input_projection`: linear projection `8 → 128`; add the learned positional
   embedding `[1, max_obs_len, 128]`.
3. `temporal_encoder`: three-layer, four-head Transformer encoder; take the last
   time-step output as `target_context` `[B, 128]`.
4. `scene_encoder`: `Linear(512,128) → LayerNorm(128) → GELU → Dropout(0.1)`;
   output `scene_context` `[B,128]`.
5. The original decoder input is exactly
   `traj_input = concat(target_context, scene_context)` `[B,256]`.
6. `decoder`: `Linear(256,128) → GELU → Dropout(0.1) → Linear(128,30)`;
   reshape to normalized future coordinates `[B,15,2]`.

Thus the reference forward is exactly:

```text
target_context = temporal_encoder(input_projection(target_history) + position_embedding)[:, -1]
scene_context = scene_encoder(scene_feat)
traj_input = concat(target_context, scene_context)
future_pred = decoder(traj_input).reshape(B, 15, 2)
```

## Preservation design

The new model will contain the original `SceneTrajectoryTransformer` as a
backbone and evaluate these same modules in the same order. Its original decoder
will receive only `[target_context, scene_context]`. Neither social features,
intent logits, nor an intention-fused representation will enter the trajectory
path. The frozen backbone will remain in evaluation mode (dropout disabled) even
while the new intention head is trained.

P1 uses only `target_context` for intention. P2 uses
`concat(target_context, scene_context)`. The new intention branch is the only
trainable part. An initialization equivalence test over at least 1,024
validation samples is required before either intention run may begin; its result
is recorded separately in `equivalence_test.json`.

## Per-seed pretrained checkpoints

Checkpoint paths, SHA256 values, selected checkpoint validation metrics, and
existing standalone test ADE/FDE values are recorded in
`trajectory_reference.json`. Each P1/P2 seed must load only the checkpoint with
the same seed.
"""
    (output / "architecture_mapping.md").write_text(mapping, encoding="utf-8")
    print(json.dumps({"reference": str(output / "trajectory_reference.json"), "architecture_mapping": str(output / "architecture_mapping.md"), "seeds": list(SEEDS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
