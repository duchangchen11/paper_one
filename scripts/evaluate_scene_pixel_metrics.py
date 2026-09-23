#!/usr/bin/env python3
"""Evaluate a scene-aware checkpoint in image-pixel ADE/FDE units."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.scene_social_gate import SceneSocialGate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JAADSequenceDataset(args.data_root / f"{args.split}.npz")
    raw = np.load(args.data_root / f"{args.split}.npz", allow_pickle=False)
    image_size = torch.from_numpy(raw["image_size"].astype(np.float32))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    state = checkpoint["model"]
    input_dim = int(state["target_encoder.weight_ih_l0"].shape[1])
    hidden_dim = int(saved_args.get("hidden_dim", state["target_encoder.weight_ih_l0"].shape[0] // 3))
    gate_mode = saved_args.get("gate_mode", "uncertainty")
    model = SceneSocialGate(
        input_dim=input_dim,
        scene_dim=int(dataset.scene_feat.shape[-1]),
        hidden_dim=hidden_dim,
        pred_len=dataset.future_gt.shape[1],
        gate_mode=gate_mode,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    errors_norm = []
    errors_pixel = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            target = batch["target_obs"]
            if input_dim == 8:
                target = torch.cat([target, batch["target_abs_obs"]], dim=-1)
            batch_size = target.shape[0]
            output = model(
                target.to(device),
                batch["neighbor_obs"].to(device),
                batch["neighbor_mask"].to(device),
                batch["neighbor_visible_mask"].to(device),
                batch["scene_feat"].to(device),
            )
            pred = output["future_pred"].cpu()
            gt = batch["future_gt"]
            scale = image_size[offset : offset + batch_size]
            offset += batch_size
            errors_norm.append(torch.linalg.vector_norm(pred - gt, dim=-1))
            errors_pixel.append(torch.linalg.vector_norm((pred - gt) * scale[:, None, :], dim=-1))

    error_norm = torch.cat(errors_norm)
    error_pixel = torch.cat(errors_pixel)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "device": str(device),
        "num_samples": int(len(dataset)),
        "coordinate_unit": "pixel",
        "trajectory_ade_pixel": float(error_pixel.mean()),
        "trajectory_fde_pixel": float(error_pixel[:, -1].mean()),
        "trajectory_ade_pixel_median": float(error_pixel.mean(dim=1).median()),
        "trajectory_fde_pixel_median": float(error_pixel[:, -1].median()),
        "trajectory_ade_normalized": float(error_norm.mean()),
        "trajectory_fde_normalized": float(error_norm[:, -1].mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
