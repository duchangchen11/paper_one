"""Train and evaluate the first target-only JAAD baseline."""

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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.baseline_target_only import TargetOnlyBaseline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(
    labels: list[float], logits: list[float], future_pred: list[np.ndarray], future_gt: list[np.ndarray]
) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = 1.0 / (1.0 + np.exp(-np.asarray(logits)))
    y_pred = (y_prob >= 0.5).astype(np.int64)
    pred = np.concatenate(future_pred, axis=0)
    gt = np.concatenate(future_gt, axis=0)
    point_error = np.linalg.norm(pred - gt, axis=-1)
    metrics = {
        "intent_accuracy": float(accuracy_score(y_true, y_pred)),
        "intent_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "intent_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "intent_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "intent_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "trajectory_ade_normalized": float(point_error.mean()),
        "trajectory_fde_normalized": float(point_error[:, -1].mean()),
    }
    if len(np.unique(y_true)) == 2:
        metrics["intent_auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    intent_loss_fn: nn.Module,
    device: torch.device,
    traj_weight: float,
    target_features: str,
) -> tuple[dict[str, float], dict[str, torch.Tensor] | None]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_intent_loss = 0.0
    total_traj_loss = 0.0
    total_items = 0
    labels: list[float] = []
    logits: list[float] = []
    future_pred: list[np.ndarray] = []
    future_gt: list[np.ndarray] = []

    for batch in loader:
        target_obs = batch["target_obs"]
        if target_features == "relative_abs":
            target_obs = torch.cat([target_obs, batch["target_abs_obs"]], dim=-1)
        target_obs = target_obs.to(device, non_blocking=True)
        gt = batch["future_gt"].to(device, non_blocking=True)
        label = batch["intent_label"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            output = model(target_obs)
            intent_loss = intent_loss_fn(output["intent_logit"], label)
            traj_loss = nn.functional.smooth_l1_loss(output["future_pred"], gt)
            loss = intent_loss + traj_weight * traj_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        count = target_obs.shape[0]
        total_items += count
        total_loss += loss.item() * count
        total_intent_loss += intent_loss.item() * count
        total_traj_loss += traj_loss.item() * count
        labels.extend(label.detach().cpu().numpy().tolist())
        logits.extend(output["intent_logit"].detach().cpu().numpy().tolist())
        future_pred.append(output["future_pred"].detach().cpu().numpy())
        future_gt.append(gt.detach().cpu().numpy())

    metrics = compute_metrics(labels, logits, future_pred, future_gt)
    metrics.update(
        {
            "loss": total_loss / total_items,
            "intent_loss": total_intent_loss / total_items,
            "trajectory_loss": total_traj_loss / total_items,
        }
    )
    return metrics, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/baseline_target_only")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/baseline_target_only_best.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument("--target-features", choices=("relative", "relative_abs"), default="relative_abs")
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
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    negative = float((train_set.intent_label == 0).sum())
    positive = float((train_set.intent_label == 1).sum())
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    intent_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    input_dim = 8 if args.target_features == "relative_abs" else 4
    model = TargetOnlyBaseline(input_dim=input_dim, hidden_dim=args.hidden_dim, pred_len=train_set.future_gt.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    history = []
    best_score = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, optimizer, intent_loss_fn, device, args.traj_weight, args.target_features)
        with torch.no_grad():
            val_metrics, _ = run_epoch(model, val_loader, None, intent_loss_fn, device, args.traj_weight, args.target_features)
        score = val_metrics["intent_f1"] + 0.1 * val_metrics["intent_auc"] - 0.1 * val_metrics["trajectory_ade_normalized"]
        scheduler.step(score)
        record = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "best_score": best_score}, args.checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test_metrics, _ = run_epoch(model, test_loader, None, intent_loss_fn, device, args.traj_weight, args.target_features)
    result = {
        "device": str(device),
        "seed": args.seed,
        "target_features": args.target_features,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "class_counts": {"negative": int(negative), "positive": int(positive)},
        "pos_weight": float(pos_weight.item()),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "test": test_metrics,
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
