"""Train and evaluate the uncertainty-aware social-gating model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.uncertainty_social_gate import UncertaintySocialGate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def classification_metrics(labels: list[float], logits: list[float]) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    y_pred = (y_prob >= 0.5).astype(np.int64)
    result = {
        "intent_accuracy": float(accuracy_score(y_true, y_pred)),
        "intent_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "intent_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "intent_brier": float(brier_score_loss(y_true, y_prob)),
    }
    ece = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, 11)[:-1], np.linspace(0.0, 1.0, 11)[1:]):
        mask = (y_prob >= lower) & (y_prob < upper if upper < 1.0 else y_prob <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(y_prob[mask].mean()) - float(y_true[mask].mean()))
    result["intent_ece_10"] = float(ece)
    if len(np.unique(y_true)) == 2:
        result["intent_auc"] = float(roc_auc_score(y_true, y_prob))
    return result


def run_epoch(
    model,
    loader,
    optimizer,
    intent_loss_fn,
    device,
    traj_weight,
    prior_weight,
    target_features,
    ambiguous_loader=None,
    ambiguous_weight=0.0,
):
    training = optimizer is not None
    model.train(training)
    total_loss = total_intent = total_traj = total_items = 0.0
    labels, logits, predictions, targets = [], [], [], []
    gates, entropies = [], []
    ambiguous_iterator = iter(ambiguous_loader) if training and ambiguous_loader is not None else None
    for batch in loader:
        target_obs = batch["target_obs"]
        if target_features == "relative_abs":
            target_obs = torch.cat([target_obs, batch["target_abs_obs"]], dim=-1)
        target_obs = target_obs.to(device, non_blocking=True)
        future_gt = batch["future_gt"].to(device, non_blocking=True)
        neighbor_obs = batch["neighbor_obs"].to(device, non_blocking=True)
        neighbor_mask = batch["neighbor_mask"].to(device, non_blocking=True)
        visible_mask = batch["neighbor_visible_mask"].to(device, non_blocking=True)
        label = batch["intent_label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = model(target_obs, neighbor_obs, neighbor_mask, visible_mask)
            intent_loss = intent_loss_fn(output["intent_logit"], label)
            prior_loss = intent_loss_fn(output["prior_logit"], label)
            traj_loss = nn.functional.smooth_l1_loss(output["future_pred"], future_gt)
            loss = intent_loss + prior_weight * prior_loss + traj_weight * traj_loss
            if ambiguous_iterator is not None and ambiguous_weight > 0.0:
                try:
                    ambiguous_batch = next(ambiguous_iterator)
                except StopIteration:
                    ambiguous_iterator = iter(ambiguous_loader)
                    ambiguous_batch = next(ambiguous_iterator)
                ambiguous_target = ambiguous_batch["target_obs"]
                if target_features == "relative_abs":
                    ambiguous_target = torch.cat(
                        [ambiguous_target, ambiguous_batch["target_abs_obs"]], dim=-1
                    )
                ambiguous_output = model(
                    ambiguous_target.to(device, non_blocking=True),
                    ambiguous_batch["neighbor_obs"].to(device, non_blocking=True),
                    ambiguous_batch["neighbor_mask"].to(device, non_blocking=True),
                    ambiguous_batch["neighbor_visible_mask"].to(device, non_blocking=True),
                )
                # crossing=-1 is not a negative label; it supervises high uncertainty.
                ambiguous_loss = 0.5 * (
                    ambiguous_output["prior_logit"].square().mean()
                    + ambiguous_output["intent_logit"].square().mean()
                )
                loss = loss + ambiguous_weight * ambiguous_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        count = target_obs.shape[0]
        total_items += count
        total_loss += loss.item() * count
        total_intent += intent_loss.item() * count
        total_traj += traj_loss.item() * count
        labels.extend(label.detach().cpu().numpy().tolist())
        logits.extend(output["intent_logit"].detach().cpu().numpy().tolist())
        predictions.append(output["future_pred"].detach().cpu().numpy())
        targets.append(future_gt.detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
        entropies.append(output["entropy"].detach().cpu().numpy())
    pred = np.concatenate(predictions, axis=0)
    gt = np.concatenate(targets, axis=0)
    point_error = np.linalg.norm(pred - gt, axis=-1)
    metrics = classification_metrics(labels, logits)
    metrics.update(
        {
            "loss": float(total_loss / total_items),
            "intent_loss": float(total_intent / total_items),
            "trajectory_loss": float(total_traj / total_items),
            "trajectory_ade_normalized": float(point_error.mean()),
            "trajectory_fde_normalized": float(point_error[:, -1].mean()),
            "gate_mean": float(np.concatenate(gates).mean()),
            "entropy_mean": float(np.concatenate(entropies).mean()),
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/uncertainty_social_gate")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/uncertainty_social_gate_best.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--target-features", choices=("relative", "relative_abs"), default="relative_abs")
    parser.add_argument("--gate-mode", choices=("uncertainty", "always", "none"), default="uncertainty")
    parser.add_argument("--ambiguous-root", type=Path, default=None)
    parser.add_argument("--ambiguous-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_set = JAADSequenceDataset(args.data_root / "train.npz")
    val_set = JAADSequenceDataset(args.data_root / "val.npz")
    test_set = JAADSequenceDataset(args.data_root / "test.npz")
    ambiguous_set = (
        JAADSequenceDataset(args.ambiguous_root / "train.npz")
        if args.ambiguous_root is not None
        else None
    )
    class_counts = torch.bincount(train_set.intent_label.to(torch.int64), minlength=2).float()
    sample_weights = torch.where(train_set.intent_label == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
    sampler = WeightedRandomSampler(sample_weights.double(), len(train_set), replacement=True)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
    ambiguous_loader = (
        DataLoader(ambiguous_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        if ambiguous_set is not None
        else None
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    input_dim = 8 if args.target_features == "relative_abs" else 4
    model = UncertaintySocialGate(input_dim=input_dim, hidden_dim=args.hidden_dim, pred_len=train_set.future_gt.shape[1], gate_mode=args.gate_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    intent_loss_fn = nn.BCEWithLogitsLoss()
    history = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            intent_loss_fn,
            device,
            args.traj_weight,
            args.prior_weight,
            args.target_features,
            ambiguous_loader,
            args.ambiguous_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                None,
                intent_loss_fn,
                device,
                args.traj_weight,
                args.prior_weight,
                args.target_features,
            )
        score = val_metrics["intent_auc"] + 0.2 * val_metrics["intent_f1"] - 0.1 * val_metrics["trajectory_ade_normalized"]
        scheduler.step(score)
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "best_score": best_score}, args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            None,
            intent_loss_fn,
            device,
            args.traj_weight,
            args.prior_weight,
            args.target_features,
        )
    result = {
        "device": str(device),
        "seed": args.seed,
        "target_features": args.target_features,
        "gate_mode": args.gate_mode,
        "ambiguous_root": str(args.ambiguous_root) if args.ambiguous_root else None,
        "ambiguous_weight": args.ambiguous_weight if ambiguous_set is not None else 0.0,
        "class_counts": {"negative": int(class_counts[0]), "positive": int(class_counts[1])},
        "sampling": "inverse-frequency weighted random sampler",
        "best_epoch": best_epoch,
        "best_score": float(best_score),
        "history": history,
        "test": test_metrics,
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
