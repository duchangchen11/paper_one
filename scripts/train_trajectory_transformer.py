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
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, image_sizes, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    predictions, targets, scales = [], [], []
    offset = 0
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        scene = batch["scene_feat"].to(device)
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
        scales.append(image_sizes[offset : offset + count])
        offset += count
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
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    train_set = JAADSequenceDataset(args.data_root / "train.npz")
    val_set = JAADSequenceDataset(args.data_root / "val.npz")
    test_set = JAADSequenceDataset(args.data_root / "test.npz")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    train_raw = np.load(args.data_root / "train.npz", allow_pickle=False)
    val_raw = np.load(args.data_root / "val.npz", allow_pickle=False)
    test_raw = np.load(args.data_root / "test.npz", allow_pickle=False)
    train_sizes = torch.from_numpy(train_raw["image_size"].astype(np.float32))
    val_sizes = torch.from_numpy(val_raw["image_size"].astype(np.float32))
    test_sizes = torch.from_numpy(test_raw["image_size"].astype(np.float32))
    model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=int(train_set.scene_feat.shape[-1]),
        d_model=args.d_model,
        num_layers=args.num_layers,
        pred_len=train_set.future_gt.shape[1],
        max_obs_len=train_set.target_obs.shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_val = float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, train_sizes, device, optimizer)
        with torch.no_grad():
            val = run_epoch(model, val_loader, val_sizes, device)
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
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.checkpoint)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test = run_epoch(model, test_loader, test_sizes, device)
    result = {
        "device": str(device),
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_ade_pixel": best_val,
        "history": history,
        "test": test,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
