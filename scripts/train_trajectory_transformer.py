#!/usr/bin/env python3
"""Train/evaluate a scene-aware Transformer trajectory baseline."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.trajectory_transformer import SceneTrajectoryTransformer


class SequenceWithImageSize(Dataset):
    """Keep each sample's pixel scale attached when a loader shuffles samples."""

    def __init__(self, path: Path) -> None:
        self.dataset = JAADSequenceDataset(path)
        with np.load(path, allow_pickle=False) as raw:
            self.image_size = torch.from_numpy(raw["image_size"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        item["image_size"] = self.image_size[index]
        return item


def apply_scene_mode(scene_feat: torch.Tensor, scene_mode: str) -> torch.Tensor:
    if scene_mode == "real":
        return scene_feat
    if scene_mode == "zero":
        return torch.zeros_like(scene_feat)
    raise ValueError(f"Unsupported scene_mode: {scene_mode}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, device, optimizer=None, scene_mode="real"):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions, targets, scales = [], [], []
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        scene = apply_scene_mode(batch["scene_feat"].to(device), scene_mode)
        future = batch["future_gt"].to(device)
        pred = model(target, scene)
        loss = nn.functional.smooth_l1_loss(pred, future)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = target.shape[0]
        total_loss += loss.item() * count
        predictions.append(pred.detach().cpu())
        targets.append(future.detach().cpu())
        scales.append(batch["image_size"])
    pred = torch.cat(predictions)
    gt = torch.cat(targets)
    scale = torch.cat(scales)
    error_norm = torch.linalg.vector_norm(pred - gt, dim=-1)
    error_pixel = torch.linalg.vector_norm((pred - gt) * scale[:, None, :], dim=-1)
    return {
        "loss": float(total_loss / len(loader.dataset)),
        "trajectory_ade_normalized": float(error_norm.mean()),
        "trajectory_fde_normalized": float(error_norm[:, -1].mean()),
        "trajectory_ade_pixel": float(error_pixel.mean()),
        "trajectory_fde_pixel": float(error_pixel[:, -1].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scene-mode", choices=("real", "zero"), default="real")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_set = SequenceWithImageSize(args.data_root / "train.npz")
    val_set = SequenceWithImageSize(args.data_root / "val.npz")
    test_set = SequenceWithImageSize(args.data_root / "test.npz")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=int(train_set.dataset.scene_feat.shape[-1]),
        d_model=args.d_model,
        num_layers=args.num_layers,
        pred_len=train_set.dataset.future_gt.shape[1],
        max_obs_len=train_set.dataset.target_obs.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, device, optimizer, args.scene_mode)
        with torch.no_grad():
            val = run_epoch(model, val_loader, device, scene_mode=args.scene_mode)
        scheduler.step(val["trajectory_ade_pixel"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train,
            "val": val,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val["trajectory_ade_pixel"] < best_val:
            best_val = val["trajectory_ade_pixel"]
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "args": vars(args), "scene_mode": args.scene_mode},
                args.checkpoint,
            )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        val = run_epoch(model, val_loader, device, scene_mode=args.scene_mode)
        test = run_epoch(model, test_loader, device, scene_mode=args.scene_mode)
    result = {
        "device": str(device),
        "seed": args.seed,
        "scene_mode": args.scene_mode,
        "checkpoint_selection": "lowest validation pixel ADE",
        "best_epoch": best_epoch,
        "best_val_ade_pixel": best_val,
        "val": val,
        "history": history,
        "test": test,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
