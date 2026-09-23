#!/usr/bin/env python3
"""Train a joint Transformer trajectory and scene-social intention model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, logits, predictions, targets, image_sizes, gates, entropies):
    y_true = np.asarray(labels, dtype=np.int64)
    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    prediction = (probability >= 0.5).astype(np.int64)
    pred = np.concatenate(predictions)
    gt = np.concatenate(targets)
    scale = np.concatenate(image_sizes)
    error_norm = np.linalg.norm(pred - gt, axis=-1)
    error_pixel = np.linalg.norm((pred - gt) * scale[:, None, :], axis=-1)
    result = {
        "intent_accuracy": float(accuracy_score(y_true, prediction)),
        "intent_balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "intent_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "intent_brier": float(brier_score_loss(y_true, probability)),
        "trajectory_ade_normalized": float(error_norm.mean()),
        "trajectory_fde_normalized": float(error_norm[:, -1].mean()),
        "trajectory_ade_pixel": float(error_pixel.mean()),
        "trajectory_fde_pixel": float(error_pixel[:, -1].mean()),
        "gate_mean": float(np.concatenate(gates).mean()),
        "entropy_mean": float(np.concatenate(entropies).mean()),
    }
    if len(np.unique(y_true)) == 2:
        result["intent_auc"] = float(roc_auc_score(y_true, probability))
    return result


def run_epoch(model, loader, image_sizes, device, optimizer, prior_weight, traj_weight, ambiguous_weight=0.0):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = total_items = 0.0
    labels, logits, predictions, targets, scales, gates, entropies = [], [], [], [], [], [], []
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        output = model(
            target,
            batch["neighbor_obs"].to(device),
            batch["neighbor_mask"].to(device),
            batch["neighbor_visible_mask"].to(device),
            batch["scene_feat"].to(device),
        )
        label = batch["intent_label"].to(device)
        loss = loss_fn(output["intent_logit"], label)
        loss = loss + prior_weight * loss_fn(output["prior_logit"], label)
        loss = loss + traj_weight * nn.functional.smooth_l1_loss(output["future_pred"], batch["future_gt"].to(device))
        if ambiguous_weight:
            loss = ambiguous_weight * 0.5 * (
                output["prior_logit"].square().mean() + output["intent_logit"].square().mean()
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = target.shape[0]
        total_items += count
        total_loss += loss.item() * count
        labels.extend(label.detach().cpu().numpy().tolist())
        logits.extend(output["intent_logit"].detach().cpu().numpy().tolist())
        predictions.append(output["future_pred"].detach().cpu().numpy())
        targets.append(batch["future_gt"].numpy())
        scales.append(image_sizes[len(np.concatenate(targets)) - count : len(np.concatenate(targets))])
        gates.append(output["gate"].detach().cpu().numpy())
        entropies.append(output["entropy"].detach().cpu().numpy())
    metrics = compute_metrics(labels, logits, predictions, targets, scales, gates, entropies)
    metrics["loss"] = float(total_loss / total_items)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ambiguous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--init-trajectory-checkpoint", type=Path, default=None)
    parser.add_argument("--gate-mode", choices=("uncertainty", "always", "none"), default="uncertainty")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument("--ambiguous-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_set = JAADSequenceDataset(args.data_root / "train.npz")
    val_set = JAADSequenceDataset(args.data_root / "val.npz")
    test_set = JAADSequenceDataset(args.data_root / "test.npz")
    ambiguous_set = JAADSequenceDataset(args.ambiguous_root / "train.npz")
    counts = torch.bincount(train_set.intent_label.to(torch.int64), minlength=2).float()
    weights = torch.where(train_set.intent_label == 0, 1.0 / counts[0], 1.0 / counts[1])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=WeightedRandomSampler(weights.double(), len(train_set), replacement=True))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    ambiguous_loader = DataLoader(ambiguous_set, batch_size=args.batch_size, shuffle=True)
    raw = {split: np.load(args.data_root / f"{split}.npz", allow_pickle=False) for split in ("train", "val", "test")}
    sizes = {split: torch.from_numpy(raw[split]["image_size"].astype(np.float32)) for split in raw}
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(train_set.scene_feat.shape[-1]),
        hidden_dim=args.hidden_dim,
        pred_len=train_set.future_gt.shape[1],
        gate_mode=args.gate_mode,
        max_obs_len=train_set.target_obs.shape[1],
    ).to(device)
    if args.init_trajectory_checkpoint is not None:
        pretrained = torch.load(args.init_trajectory_checkpoint, map_location="cpu", weights_only=False)["model"]
        current = model.state_dict()
        prefix_map = {
            "input_projection.": "target_projection.",
            "temporal_encoder.": "target_encoder.",
            "decoder.": "traj_head.",
        }
        loaded = []
        for key, value in pretrained.items():
            mapped_key = key
            for source_prefix, target_prefix in prefix_map.items():
                if key.startswith(source_prefix):
                    mapped_key = target_prefix + key[len(source_prefix) :]
                    break
            if mapped_key in current and current[mapped_key].shape == value.shape:
                current[mapped_key] = value
                loaded.append(mapped_key)
        model.load_state_dict(current)
        print(f"initialized_trajectory_parameters={len(loaded)} from={args.init_trajectory_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_score = -float("inf")
    best_epoch = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        amb_iter = iter(ambiguous_loader)
        for batch in train_loader:
            try:
                amb_batch = next(amb_iter)
            except StopIteration:
                amb_iter = iter(ambiguous_loader)
                amb_batch = next(amb_iter)
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            output = model(target, batch["neighbor_obs"].to(device), batch["neighbor_mask"].to(device), batch["neighbor_visible_mask"].to(device), batch["scene_feat"].to(device))
            label = batch["intent_label"].to(device)
            loss = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], label)
            loss = loss + args.prior_weight * nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], label)
            loss = loss + args.traj_weight * nn.functional.smooth_l1_loss(output["future_pred"], batch["future_gt"].to(device))
            amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1).to(device)
            amb_output = model(amb_target, amb_batch["neighbor_obs"].to(device), amb_batch["neighbor_mask"].to(device), amb_batch["neighbor_visible_mask"].to(device), amb_batch["scene_feat"].to(device))
            loss = loss + args.ambiguous_weight * 0.5 * (amb_output["prior_logit"].square().mean() + amb_output["intent_logit"].square().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        with torch.no_grad():
            val = run_epoch(model, val_loader, sizes["val"], device, None, args.prior_weight, args.traj_weight)
        score = val["intent_auc"] + 0.1 * val["intent_f1"] - 0.01 * val["trajectory_ade_pixel"]
        scheduler.step(score)
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "val": val}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test = run_epoch(model, test_loader, sizes["test"], device, None, args.prior_weight, args.traj_weight)
    result = {"device": str(device), "seed": args.seed, "gate_mode": args.gate_mode, "best_epoch": best_epoch, "history": history, "test": test}
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
