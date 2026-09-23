"""Train and evaluate a frozen-scene-feature social interaction model."""

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
from src.models.scene_social_gate import SceneSocialGate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classification_metrics(labels, logits):
    y_true = np.asarray(labels, dtype=np.int64)
    probability = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    prediction = (probability >= 0.5).astype(np.int64)
    result = {
        "intent_accuracy": float(accuracy_score(y_true, prediction)),
        "intent_balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "intent_f1": float(f1_score(y_true, prediction, zero_division=0)),
        "intent_brier": float(brier_score_loss(y_true, probability)),
    }
    if len(np.unique(y_true)) == 2:
        result["intent_auc"] = float(roc_auc_score(y_true, probability))
    return result


def run_epoch(model, loader, optimizer, device, traj_weight, prior_weight):
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = total_items = 0.0
    labels, logits, predictions, targets, gates, entropies = [], [], [], [], [], []
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        scene_feat = batch["scene_feat"].to(device)
        future_gt = batch["future_gt"].to(device)
        output = model(
            target,
            batch["neighbor_obs"].to(device),
            batch["neighbor_mask"].to(device),
            batch["neighbor_visible_mask"].to(device),
            scene_feat,
        )
        label = batch["intent_label"].to(device)
        intent_loss = loss_fn(output["intent_logit"], label)
        prior_loss = loss_fn(output["prior_logit"], label)
        traj_loss = nn.functional.smooth_l1_loss(output["future_pred"], future_gt)
        loss = intent_loss + prior_weight * prior_loss + traj_weight * traj_loss
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
        targets.append(future_gt.detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
        entropies.append(output["entropy"].detach().cpu().numpy())
    pred = np.concatenate(predictions)
    gt = np.concatenate(targets)
    error = np.linalg.norm(pred - gt, axis=-1)
    result = classification_metrics(labels, logits)
    result.update({
        "loss": float(total_loss / total_items),
        "trajectory_ade_normalized": float(error.mean()),
        "trajectory_fde_normalized": float(error[:, -1].mean()),
        "gate_mean": float(np.concatenate(gates).mean()),
        "entropy_mean": float(np.concatenate(entropies).mean()),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ambiguous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gate-mode", choices=("uncertainty", "always", "none"), default="always")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--ambiguous-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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
    sampler = WeightedRandomSampler(weights.double(), len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    ambiguous_loader = DataLoader(ambiguous_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = SceneSocialGate(input_dim=8, hidden_dim=args.hidden_dim, pred_len=12, gate_mode=args.gate_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    history, best_score, best_epoch, stale = [], -float("inf"), 0, 0
    ambiguous_iterator = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_items = 0.0
        # Train the main clean batches and add ambiguous uncertainty supervision.
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
            loss = loss_fn(output["intent_logit"], label) + args.prior_weight * loss_fn(output["prior_logit"], label)
            loss = loss + args.traj_weight * nn.functional.smooth_l1_loss(output["future_pred"], batch["future_gt"].to(device))
            amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1).to(device)
            amb_output = model(amb_target, amb_batch["neighbor_obs"].to(device), amb_batch["neighbor_mask"].to(device), amb_batch["neighbor_visible_mask"].to(device), amb_batch["scene_feat"].to(device))
            loss = loss + args.ambiguous_weight * 0.5 * (amb_output["prior_logit"].square().mean() + amb_output["intent_logit"].square().mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = target.shape[0]
            total_items += count
            total_loss += loss.item() * count
        with torch.no_grad():
            val = run_epoch(model, val_loader, None, device, args.traj_weight, args.prior_weight)
        score = val["intent_auc"] + 0.2 * val["intent_f1"] - 0.1 * val["trajectory_ade_normalized"]
        scheduler.step(score)
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": total_loss / total_items, "val": val}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "args": vars(args), "best_score": score}, args.checkpoint)
        else:
            stale += 1
            if stale >= 4:
                break
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test = run_epoch(model, test_loader, None, device, args.traj_weight, args.prior_weight)
    result = {"device": str(device), "seed": args.seed, "gate_mode": args.gate_mode, "best_epoch": best_epoch, "best_score": float(best_score), "history": history, "test": test}
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
