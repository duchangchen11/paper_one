#!/usr/bin/env python3
"""Train one no-social intent model on top of a frozen trajectory Transformer."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_residual_social_joint import load_backbone
from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.fixed_base_social_residual import FixedBaseIntentModel


class SequenceWithImageSize(Dataset):
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ece_10(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= lower) & (
            probabilities < upper if index < 9 else probabilities <= upper
        )
        if mask.any():
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return ece


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10": ece_10(labels, probabilities),
    }


def make_model(
    dataset: SequenceWithImageSize,
    trajectory_checkpoint: Path,
    device: torch.device,
) -> FixedBaseIntentModel:
    item = dataset[0]
    target = torch.cat([item["target_obs"], item["target_abs_obs"]], dim=-1)
    backbone, _ = load_backbone(
        trajectory_checkpoint,
        input_dim=target.shape[-1],
        scene_dim=item["scene_feat"].numel(),
        pred_len=item["future_gt"].shape[0],
        max_obs_len=target.shape[0],
        map_location="cpu",
    )
    model = FixedBaseIntentModel(backbone).to(device)
    if not model.trajectory_backbone_frozen:
        raise RuntimeError("Trajectory Transformer must remain frozen during base intent training")
    return model


@torch.no_grad()
def collect(model: FixedBaseIntentModel, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list[np.ndarray]] = {
        key: []
        for key in ("labels", "raw_logits", "calibrated_logits", "future_pred", "future_gt", "image_size")
    }
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        output = model(target, batch["scene_feat"].to(device))
        values["labels"].append(batch["intent_label"].numpy())
        values["raw_logits"].append(output["base_raw_logit"].cpu().numpy())
        values["calibrated_logits"].append(output["base_logit"].cpu().numpy())
        values["future_pred"].append(output["future_pred"].cpu().numpy())
        values["future_gt"].append(batch["future_gt"].numpy())
        values["image_size"].append(batch["image_size"].numpy())
    return {key: np.concatenate(items, axis=0) for key, items in values.items()}


def trajectory_metrics(outputs: dict[str, np.ndarray]) -> dict[str, float]:
    error = outputs["future_pred"] - outputs["future_gt"]
    pixel_error = np.linalg.norm(error * outputs["image_size"][:, None, :], axis=-1)
    normalized_error = np.linalg.norm(error, axis=-1)
    return {
        "ade_pixel": float(pixel_error.mean()),
        "fde_pixel": float(pixel_error[:, -1].mean()),
        "ade_normalized": float(normalized_error.mean()),
        "fde_normalized": float(normalized_error[:, -1].mean()),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float64, device=device)
    labels_t = torch.as_tensor(labels, dtype=torch.float64, device=device)
    log_temperature = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = nn.functional.binary_cross_entropy_with_logits(logits_t / temperature, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())


def run_train_epoch(
    model: FixedBaseIntentModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    labels, logits = [], []
    for batch in loader:
        label = batch["intent_label"].to(device)
        if torch.any((label < 0) | (label > 1)):
            raise ValueError("Clean intent training data must contain only labels 0 and 1")
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        output = model(target, batch["scene_feat"].to(device))
        loss = nn.functional.binary_cross_entropy_with_logits(output["base_raw_logit"], label)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
        count = len(label)
        total_loss += float(loss.detach()) * count
        total_count += count
        labels.append(label.detach().cpu().numpy())
        logits.append(output["base_raw_logit"].detach().cpu().numpy())
    y = np.concatenate(labels)
    raw_logits = np.concatenate(logits)
    metrics = classification_metrics(y, 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -80, 80))))
    metrics["loss"] = total_loss / total_count
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--trajectory-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_set = SequenceWithImageSize(args.data_root / "train.npz")
    val_set = SequenceWithImageSize(args.data_root / "val.npz")
    test_set = SequenceWithImageSize(args.data_root / "test.npz")
    train_labels = train_set.dataset.intent_label.to(torch.int64)
    if torch.any((train_labels < 0) | (train_labels > 1)):
        raise ValueError("Clean train split unexpectedly contains non-binary labels")
    class_counts = torch.bincount(train_labels, minlength=2).float()
    sample_weights = torch.where(train_labels == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
    sampler = WeightedRandomSampler(
        sample_weights.double(),
        len(train_set),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model = make_model(train_set, args.trajectory_checkpoint, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
    best_auc = -float("inf")
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(model, train_loader, device, optimizer)
        val_outputs = collect(model, val_loader, device)
        val_metrics = classification_metrics(
            val_outputs["labels"], 1.0 / (1.0 + np.exp(-np.clip(val_outputs["raw_logits"], -80, 80)))
        )
        score = val_metrics["auc"]
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_validation_auc": best_auc,
                    "trajectory_checkpoint": str(args.trajectory_checkpoint),
                    "temperature": 1.0,
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
    val_outputs = collect(model, val_loader, device)
    val_before = classification_metrics(
        val_outputs["labels"], 1.0 / (1.0 + np.exp(-np.clip(val_outputs["raw_logits"], -80, 80)))
    )
    temperature = fit_temperature(val_outputs["raw_logits"], val_outputs["labels"], device)
    model.temperature.fill_(temperature)
    model.freeze_base_classifier()
    val_after = classification_metrics(
        val_outputs["labels"], 1.0 / (1.0 + np.exp(-np.clip(val_outputs["raw_logits"] / temperature, -80, 80)))
    )
    test_outputs = collect(model, test_loader, device)
    test_before = classification_metrics(
        test_outputs["labels"], 1.0 / (1.0 + np.exp(-np.clip(test_outputs["raw_logits"], -80, 80)))
    )
    test_after = classification_metrics(
        test_outputs["labels"],
        1.0 / (1.0 + np.exp(-np.clip(test_outputs["raw_logits"] / temperature, -80, 80))),
    )
    checkpoint_payload = {
        **saved,
        "model": model.state_dict(),
        "temperature": temperature,
        "calibration_fit_split": "val",
        "val_calibration_before": val_before,
        "val_calibration_after": val_after,
    }
    torch.save(checkpoint_payload, args.checkpoint)
    result = {
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "trajectory_checkpoint": str(args.trajectory_checkpoint),
        "trajectory_backbone_frozen": model.trajectory_backbone_frozen,
        "base_classifier_frozen_after_training": model.base_classifier_frozen,
        "best_epoch": best_epoch,
        "best_validation_auc_before_calibration": best_auc,
        "temperature": temperature,
        "calibration_fit_split": "val",
        "class_counts": {"negative": int(class_counts[0]), "positive": int(class_counts[1])},
        "ambiguous_supervision_used": False,
        "history": history,
        "validation_calibration": {"before": val_before, "after": val_after},
        "test_calibration": {"before": test_before, "after": test_after},
        "test_trajectory": trajectory_metrics(test_outputs),
        "test": {
            **test_after,
            **trajectory_metrics(test_outputs),
        },
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "temperature": temperature, "test": result["test"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
