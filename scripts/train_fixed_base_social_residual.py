#!/usr/bin/env python3
"""Train a neighbor-only logit residual while freezing the shared base model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_fixed_base_intent import (
    SequenceWithImageSize,
    classification_metrics,
    ece_10,
    load_backbone,
    set_seed,
)
from src.models.fixed_base_social_residual import FixedBaseIntentModel, FixedBaseSocialResidual


def distribution(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64).reshape(-1)
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
    }


def load_base_model(
    base_checkpoint_path: Path,
    dataset: SequenceWithImageSize,
    device: torch.device,
) -> FixedBaseIntentModel:
    payload = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    trajectory_path = Path(payload["trajectory_checkpoint"])
    if not trajectory_path.is_absolute():
        trajectory_path = PROJECT_ROOT / trajectory_path
    sample = dataset[0]
    target = torch.cat([sample["target_obs"], sample["target_abs_obs"]], dim=-1)
    backbone, _ = load_backbone(
        trajectory_path,
        input_dim=target.shape[-1],
        scene_dim=sample["scene_feat"].numel(),
        pred_len=sample["future_gt"].shape[0],
        max_obs_len=target.shape[0],
        map_location="cpu",
    )
    base_model = FixedBaseIntentModel(backbone)
    base_model.load_state_dict(payload["model"], strict=True)
    base_model.freeze_base_classifier()
    base_model.to(device).eval()
    return base_model


def collect(model, loader, device, optimizer=None, residual_reg_weight=0.0):
    training = optimizer is not None
    model.train(training)
    total_loss = total_bce = total_reg = 0.0
    total_count = 0
    labels = []
    final_logits = []
    base_logits = []
    deltas = []
    gates = []
    effective_gates = []
    entropies = []
    future_preds = []
    future_targets = []
    image_sizes = []
    criterion = torch.nn.BCEWithLogitsLoss()
    for batch in loader:
        label = batch["intent_label"].to(device)
        if torch.any((label < 0) | (label > 1)):
            raise ValueError("Clean intent data must contain only binary labels")
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        output = model(
            target,
            batch["scene_feat"].to(device),
            batch["neighbor_obs"].to(device),
            batch["neighbor_mask"].to(device),
        )
        bce = criterion(output["final_logit"], label)
        residual_reg = output["delta_logit"].square().mean()
        loss = bce + residual_reg_weight * residual_reg
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            optimizer.step()
        count = len(label)
        total_count += count
        total_loss += float(loss.detach()) * count
        total_bce += float(bce.detach()) * count
        total_reg += float(residual_reg.detach()) * count
        labels.append(label.detach().cpu().numpy())
        final_logits.append(output["final_logit"].detach().cpu().numpy())
        base_logits.append(output["base_logit"].detach().cpu().numpy())
        deltas.append(output["delta_logit"].detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
        effective_gates.append(output["effective_gate"].detach().cpu().numpy())
        entropies.append(output["base_entropy"].detach().cpu().numpy())
        future_preds.append(output["future_pred"].detach().cpu().numpy())
        future_targets.append(batch["future_gt"].numpy())
        image_sizes.append(batch["image_size"].numpy())

    y = np.concatenate(labels)
    logits = np.concatenate(final_logits)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    base_logit = np.concatenate(base_logits)
    base_probability = 1.0 / (1.0 + np.exp(-np.clip(base_logit, -80, 80)))
    delta = np.concatenate(deltas)
    gate = np.concatenate(gates)
    effective_gate = np.concatenate(effective_gates)
    entropy = np.concatenate(entropies)
    pred = np.concatenate(future_preds)
    future = np.concatenate(future_targets)
    scales = np.concatenate(image_sizes)
    pixel_error = np.linalg.norm((pred - future) * scales[:, None, :], axis=-1)

    metrics: dict[str, Any] = classification_metrics(y, probabilities)
    metrics["ece_10"] = ece_10(y, probabilities)
    metrics.update(
        {
            "loss": total_loss / total_count,
            "bce_loss": total_bce / total_count,
            "residual_regularization": total_reg / total_count,
            "base_auc": classification_metrics(y, base_probability)["auc"],
            "trajectory_ade_pixel": float(pixel_error.mean()),
            "trajectory_fde_pixel": float(pixel_error[:, -1].mean()),
            "gate_distribution": distribution(gate),
            "effective_gate_distribution": distribution(effective_gate),
            "entropy_distribution": distribution(entropy),
            "delta_logit_distribution": distribution(delta),
            "delta_logit_abs_mean": float(np.abs(delta).mean()),
            "logit_change_distribution": distribution(logits - base_logit),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-mode", choices=("always", "uncertainty"), required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--residual-reg-weight", type=float, default=1e-3)
    parser.add_argument("--social-scale", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_set = SequenceWithImageSize(args.data_root / "train.npz")
    val_set = SequenceWithImageSize(args.data_root / "val.npz")
    test_set = SequenceWithImageSize(args.data_root / "test.npz")
    labels = train_set.dataset.intent_label.to(torch.int64)
    if torch.any((labels < 0) | (labels > 1)):
        raise ValueError("Clean training labels must be binary; ambiguous supervision is prohibited")
    counts = torch.bincount(labels, minlength=2).float()
    weights = torch.where(labels == 0, 1.0 / counts[0], 1.0 / counts[1])
    sampler = WeightedRandomSampler(
        weights.double(), len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    base_model = load_base_model(args.base_checkpoint, train_set, device)
    model = FixedBaseSocialResidual(
        base_model,
        gate_mode=args.gate_mode,
        social_scale=args.social_scale,
    ).to(device)
    if not model.all_base_parameters_frozen:
        raise RuntimeError("Trajectory Transformer and base intent classifier must both be frozen")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    best_auc = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = collect(
            model, train_loader, device, optimizer, args.residual_reg_weight
        )
        with torch.no_grad():
            val_metrics = collect(model, val_loader, device)
        scheduler.step(val_metrics["auc"])
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "base_checkpoint": str(args.base_checkpoint),
                    "best_validation_auc": best_auc,
                    # .cpu() would move this registered buffer off the active device.
                    "base_temperature": float(model.base_model.temperature.detach().item()),
                },
                args.checkpoint,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    with torch.no_grad():
        test_metrics = collect(model, test_loader, device)
    result = {
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "checkpoint": str(args.checkpoint),
        "base_checkpoint": str(args.base_checkpoint),
        "base_temperature": float(model.base_model.temperature.detach().item()),
        "trajectory_and_base_classifier_frozen": model.all_base_parameters_frozen,
        "best_epoch": best_epoch,
        "best_validation_auc": best_auc,
        "ambiguous_supervision_used": False,
        "training_protocol": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "residual_reg_weight": args.residual_reg_weight,
            "social_scale": args.social_scale,
            "sampling": "inverse-frequency weighted random sampler",
            "trajectory_loss": False,
        },
        "history": history,
        "test": test_metrics,
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
