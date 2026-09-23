#!/usr/bin/env python3
"""Evaluate real-scene checkpoints with real and zero scene inputs (diagnostic only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_trajectory_transformer import SequenceWithImageSize, run_epoch
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--checkpoint-root", type=Path, default=PROJECT_ROOT / "checkpoints")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/scene_ablation_15x15/trajectory_zero_scene_inference_diagnostic.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = SequenceWithImageSize(args.data_root / "test.npz")
    loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    results = {}
    for seed in args.seeds:
        checkpoint_path = args.checkpoint_root / f"trajectory_transformer_scene_15x15_seed{seed}.pt"
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = payload["model"]
        saved_args = payload.get("args", {})
        model = SceneTrajectoryTransformer(
            input_dim=8,
            scene_dim=test_set.dataset.scene_feat.shape[-1],
            d_model=int(saved_args.get("d_model", 128)),
            nhead=4,
            num_layers=int(saved_args.get("num_layers", 3)),
            pred_len=test_set.dataset.future_gt.shape[1],
            dropout=0.1,
            max_obs_len=test_set.dataset.target_obs.shape[1],
        ).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            real = run_epoch(model, loader, device, scene_mode="real")
            zero = run_epoch(model, loader, device, scene_mode="zero")
        results[str(seed)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "real_scene_test": real,
            "zero_scene_input_test": zero,
            "zero_minus_real": {
                key: zero[key] - real[key]
                for key in (
                    "trajectory_ade_pixel",
                    "trajectory_fde_pixel",
                    "trajectory_ade_normalized",
                    "trajectory_fde_normalized",
                )
            },
        }
    output = {
        "purpose": "inference diagnostic only; not the formally retrained zero-scene trajectory baseline",
        "scene_modes": "same real-scene trained checkpoint evaluated with real scene features and torch.zeros_like(scene_feat)",
        "seeds": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
