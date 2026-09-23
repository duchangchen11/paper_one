"""Pilot training for explicit non-crossing/crossing/ambiguous intent states."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.tristate_social_gate import TriStateSocialGate


class TriStateDataset(Dataset):
    def __init__(self, dataset: JAADSequenceDataset, state: int | None = None):
        self.dataset = dataset
        self.state = state

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        label = int(item["intent_label"].item()) if self.state is None else self.state
        if label == -1:
            label = 2
        item["intent_label"] = torch.tensor(label, dtype=torch.long)
        return item


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(labels, logits):
    y_true = np.asarray(labels, dtype=np.int64)
    probabilities = torch.softmax(torch.from_numpy(np.asarray(logits)), dim=-1).numpy()
    predictions = probabilities.argmax(axis=1)
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "ambiguous_f1": float(f1_score(y_true == 2, predictions == 2, zero_division=0)),
    }
    if len(np.unique(y_true)) == 3:
        result["ambiguous_auc"] = float(roc_auc_score(y_true == 2, probabilities[:, 2]))
    return result


def run_epoch(model, loader, optimizer, device, traj_weight, prior_weight):
    training = optimizer is not None
    model.train(training)
    ce = nn.CrossEntropyLoss()
    total_loss = total_items = 0.0
    labels, logits, predictions, targets, entropies, gates = [], [], [], [], [], []
    for batch in loader:
        target = batch["target_obs"].to(device)
        future_gt = batch["future_gt"].to(device)
        output = model(
            target,
            batch["neighbor_obs"].to(device),
            batch["neighbor_mask"].to(device),
            batch["neighbor_visible_mask"].to(device),
        )
        label = batch["intent_label"].to(device)
        intent_loss = ce(output["intent_logits"], label)
        prior_loss = ce(output["prior_logits"], label)
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
        logits.extend(output["intent_logits"].detach().cpu().numpy().tolist())
        predictions.append(output["future_pred"].detach().cpu().numpy())
        targets.append(future_gt.detach().cpu().numpy())
        entropies.append(output["entropy"].detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
    pred = np.concatenate(predictions)
    gt = np.concatenate(targets)
    error = np.linalg.norm(pred - gt, axis=-1)
    result = metrics(labels, logits)
    result.update({
        "loss": float(total_loss / total_items),
        "ade": float(error.mean()),
        "fde": float(error[:, -1].mean()),
        "entropy_mean": float(np.concatenate(entropies).mean()),
        "gate_mean": float(np.concatenate(gates).mean()),
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences")
    parser.add_argument("--ambiguous-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_ambiguous")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/tristate_pilot")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/tristate_pilot.pt")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    def combined(split):
        clean = TriStateDataset(JAADSequenceDataset(args.data_root / f"{split}.npz"))
        ambiguous = TriStateDataset(JAADSequenceDataset(args.ambiguous_root / f"{split}.npz"), state=2)
        return ConcatDataset([clean, ambiguous])

    train_set = combined("train")
    val_set = combined("val")
    test_set = combined("test")
    labels = torch.tensor([train_set[i]["intent_label"] for i in range(len(train_set))])
    counts = torch.bincount(labels, minlength=3).float()
    weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(weights.double(), len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    model = TriStateSocialGate(input_dim=4, hidden_dim=args.hidden_dim, pred_len=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best_score, best_epoch = [], -float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        train = run_epoch(model, train_loader, optimizer, device, args.traj_weight, args.prior_weight)
        with torch.no_grad():
            val = run_epoch(model, val_loader, None, device, args.traj_weight, args.prior_weight)
        score = val["ambiguous_auc"] + val["macro_f1"] - 0.1 * val["ade"]
        history.append({"epoch": epoch, "train": train, "val": val})
        print(json.dumps(history[-1], ensure_ascii=False))
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "args": vars(args), "best_score": score}, args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        test = run_epoch(model, test_loader, None, device, args.traj_weight, args.prior_weight)
    result = {
        "device": str(device), "seed": args.seed, "class_counts": counts.tolist(),
        "best_epoch": best_epoch, "best_score": float(best_score),
        "history": history, "test": test,
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
