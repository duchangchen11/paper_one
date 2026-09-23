#!/usr/bin/env python3
"""Train Stage A/B social residual heads around a frozen Transformer checkpoint."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.residual_social_joint import ResidualSocialJointModel
from src.models.trajectory_transformer import SceneTrajectoryTransformer


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


def load_backbone(
    checkpoint_path: Path,
    *,
    input_dim: int,
    scene_dim: int,
    pred_len: int,
    max_obs_len: int,
    map_location: torch.device | str,
) -> tuple[SceneTrajectoryTransformer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    saved_args = checkpoint.get("args", {})
    state = checkpoint["model"]
    d_model = int(saved_args.get("d_model", state["input_projection.weight"].shape[0]))
    num_layers = int(
        saved_args.get(
            "num_layers",
            1 + max(
                int(key.split(".layers.")[1].split(".")[0])
                for key in state
                if key.startswith("temporal_encoder.layers.")
            ),
        )
    )
    backbone = SceneTrajectoryTransformer(
        input_dim=input_dim,
        scene_dim=scene_dim,
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        pred_len=pred_len,
        dropout=0.1,
        max_obs_len=max_obs_len,
    )
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone, saved_args


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    y_true = labels.astype(np.int64)
    y_prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    y_pred = (y_prob >= 0.5).astype(np.int64)
    result = {
        "intent_accuracy": float(accuracy_score(y_true, y_pred)),
        "intent_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "intent_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "intent_brier": float(brier_score_loss(y_true, y_prob)),
    }
    if np.unique(y_true).size == 2:
        result["intent_auc"] = float(roc_auc_score(y_true, y_prob))
    return result


def distribution_summary(values: np.ndarray) -> dict[str, float]:
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


def forward_batch(model, batch: dict[str, torch.Tensor], device: torch.device):
    target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(
        device, non_blocking=True
    )
    output = model(
        target,
        batch["neighbor_obs"].to(device, non_blocking=True),
        batch["neighbor_mask"].to(device, non_blocking=True),
        batch["neighbor_visible_mask"].to(device, non_blocking=True),
        batch["scene_feat"].to(device, non_blocking=True),
    )
    return target, output


def run_epoch(
    model: ResidualSocialJointModel,
    loader: DataLoader,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    stage: str,
    prior_weight: float,
    traj_weight: float,
    residual_reg_weight: float,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_intent_loss = total_prior_loss = 0.0
    total_traj_loss = total_residual_penalty = total_items = 0
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    prior_logits: list[np.ndarray] = []
    point_errors: list[np.ndarray] = []
    point_errors_pixel: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    entropies: list[np.ndarray] = []
    criterion = nn.BCEWithLogitsLoss()

    for batch in loader:
        label = batch["intent_label"].to(device, non_blocking=True)
        if torch.any((label < 0) | (label > 1)):
            raise ValueError("Clean train/val/test data must contain only intent labels 0 or 1")
        future_gt = batch["future_gt"].to(device, non_blocking=True)
        target, output = forward_batch(model, batch, device)
        intent_loss = criterion(output["intent_logit"], label)
        prior_loss = criterion(output["prior_logit"], label)
        loss = intent_loss + prior_weight * prior_loss
        traj_loss = target.new_zeros(())
        residual_penalty = output["delta_future"].square().mean()
        if stage == "B":
            traj_loss = nn.functional.smooth_l1_loss(output["future_pred"], future_gt)
            loss = loss + traj_weight * traj_loss + residual_reg_weight * residual_penalty

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            optimizer.step()

        batch_size = target.shape[0]
        total_items += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_intent_loss += float(intent_loss.detach()) * batch_size
        total_prior_loss += float(prior_loss.detach()) * batch_size
        total_traj_loss += float(traj_loss.detach()) * batch_size
        total_residual_penalty += float(residual_penalty.detach()) * batch_size
        labels.append(label.detach().cpu().numpy())
        logits.append(output["intent_logit"].detach().cpu().numpy())
        prior_logits.append(output["prior_logit"].detach().cpu().numpy())
        gates.append(output["gate"].detach().cpu().numpy())
        entropies.append(output["normalized_entropy"].detach().cpu().numpy())
        error_norm = torch.linalg.vector_norm(output["future_pred"] - future_gt, dim=-1)
        scale = batch["image_size"].to(device, non_blocking=True)
        error_pixel = torch.linalg.vector_norm(
            (output["future_pred"] - future_gt) * scale[:, None, :], dim=-1
        )
        point_errors.append(error_norm.detach().cpu().numpy())
        point_errors_pixel.append(error_pixel.detach().cpu().numpy())

    y_true = np.concatenate(labels)
    logit_array = np.concatenate(logits)
    prior_logit_array = np.concatenate(prior_logits)
    error_norm_array = np.concatenate(point_errors)
    error_pixel_array = np.concatenate(point_errors_pixel)
    gate_array = np.concatenate(gates)
    entropy_array = np.concatenate(entropies)
    metrics: dict[str, Any] = classification_metrics(y_true, logit_array)
    metrics["prior_" + "intent_auc"] = classification_metrics(y_true, prior_logit_array).get(
        "intent_auc", float("nan")
    )
    metrics.update(
        {
            "loss": total_loss / total_items,
            "intent_loss": total_intent_loss / total_items,
            "prior_intent_loss": total_prior_loss / total_items,
            "trajectory_loss": total_traj_loss / total_items,
            "residual_regularization": total_residual_penalty / total_items,
            "trajectory_ade_normalized": float(error_norm_array.mean()),
            "trajectory_fde_normalized": float(error_norm_array[:, -1].mean()),
            "trajectory_ade_pixel": float(error_pixel_array.mean()),
            "trajectory_fde_pixel": float(error_pixel_array[:, -1].mean()),
            "gate_mean": float(gate_array.mean()),
            "gate_std": float(gate_array.std(ddof=1)),
            "gate_distribution": distribution_summary(gate_array),
            "entropy_mean": float(entropy_array.mean()),
            "entropy_std": float(entropy_array.std(ddof=1)),
        }
    )
    return metrics


def make_model(
    dataset: SequenceWithImageSize,
    checkpoint_path: Path,
    device: torch.device,
    *,
    gate_mode: str,
    trajectory_residual_scale: float,
    enable_trajectory_residual: bool,
) -> ResidualSocialJointModel:
    sample = dataset[0]
    target = torch.cat([sample["target_obs"], sample["target_abs_obs"]], dim=-1)
    backbone, _ = load_backbone(
        checkpoint_path,
        input_dim=target.shape[-1],
        scene_dim=sample["scene_feat"].numel(),
        pred_len=sample["future_gt"].shape[0],
        max_obs_len=target.shape[0],
        map_location="cpu",
    )
    model = ResidualSocialJointModel(
        input_dim=target.shape[-1],
        scene_dim=sample["scene_feat"].numel(),
        d_model=backbone.input_projection.out_features,
        pred_len=sample["future_gt"].shape[0],
        max_obs_len=target.shape[0],
        gate_mode=gate_mode,
        trajectory_residual_scale=trajectory_residual_scale,
        enable_trajectory_residual=enable_trajectory_residual,
        trajectory_backbone=backbone,
    ).to(device)
    if not model.trajectory_backbone_frozen:
        raise RuntimeError("Pretrained trajectory backbone was not fully frozen")
    return model


def evaluate_frozen_baseline(
    checkpoint_path: Path,
    data_root: Path,
    split: str,
    output_path: Path,
    *,
    device: torch.device,
) -> dict[str, Any]:
    dataset = SequenceWithImageSize(data_root / f"{split}.npz")
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    sample = dataset[0]
    target_sample = torch.cat([sample["target_obs"], sample["target_abs_obs"]], dim=-1)
    backbone, _ = load_backbone(
        checkpoint_path,
        input_dim=target_sample.shape[-1],
        scene_dim=sample["scene_feat"].numel(),
        pred_len=sample["future_gt"].shape[0],
        max_obs_len=target_sample.shape[0],
        map_location=device,
    )
    backbone.to(device).eval()
    predictions, targets, sizes = [], [], []
    with torch.no_grad():
        for batch in loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            scene = batch["scene_feat"].to(device)
            predictions.append(backbone(target, scene).cpu())
            targets.append(batch["future_gt"])
            sizes.append(batch["image_size"])
    pred = torch.cat(predictions)
    future = torch.cat(targets)
    image_size = torch.cat(sizes)
    error_normalized = torch.linalg.vector_norm(pred - future, dim=-1)
    error_pixel = torch.linalg.vector_norm(
        (pred - future) * image_size[:, None, :], dim=-1
    )
    result = {
        "sample_count": len(dataset),
        "trajectory_ade_normalized": float(error_normalized.mean()),
        "trajectory_fde_normalized": float(error_normalized[:, -1].mean()),
        "trajectory_ade_pixel": float(error_pixel.mean()),
        "trajectory_fde_pixel": float(error_pixel[:, -1].mean()),
    }
    result.update(
        {
            "seed": 123,
            "split": split,
            "checkpoint": str(checkpoint_path),
            "baseline_type": "pretrained SceneTrajectoryTransformer, frozen and unmodified",
            "coordinate_unit": "pixel",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def baseline_reproduction_main(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result = evaluate_frozen_baseline(
        args.trajectory_checkpoint,
        args.data_root,
        "test",
        args.output_root / "baseline_reproduction_seed123.json",
        device=device,
    )
    expected_ade = 11.0584754944
    expected_fde = 19.5931129456
    result["reference"] = {"ade_pixel": expected_ade, "fde_pixel": expected_fde}
    result["difference_from_reference"] = {
        "ade_pixel": result["trajectory_ade_pixel"] - expected_ade,
        "fde_pixel": result["trajectory_fde_pixel"] - expected_fde,
    }
    result["within_0_01_pixel_tolerance"] = bool(
        abs(result["difference_from_reference"]["ade_pixel"]) <= 0.01
        and abs(result["difference_from_reference"]["fde_pixel"]) <= 0.01
    )
    output_path = args.output_root / "baseline_reproduction_seed123.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["within_0_01_pixel_tolerance"]:
        raise SystemExit("Seed123 baseline reproduction differs by more than 0.01 pixel; stop before training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--trajectory-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--stage", choices=("A", "B"), default="A")
    parser.add_argument("--gate-mode", choices=("none", "always", "uncertainty"), default="uncertainty")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument("--residual-reg-weight", type=float, default=0.01)
    parser.add_argument("--trajectory-residual-scale", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--baseline-reproduction-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline_reproduction_only:
        baseline_reproduction_main(args)
        return

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or (
        PROJECT_ROOT
        / "checkpoints"
        / f"residual_social_stage{args.stage}_{args.gate_mode}_seed{args.seed}.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_set = SequenceWithImageSize(args.data_root / "train.npz")
    val_set = SequenceWithImageSize(args.data_root / "val.npz")
    test_set = SequenceWithImageSize(args.data_root / "test.npz")
    labels = train_set.dataset.intent_label.to(torch.int64)
    if torch.any((labels < 0) | (labels > 1)):
        raise ValueError("Main clean training data contains non-binary intent labels")
    class_counts = torch.bincount(labels, minlength=2).float()
    weights = torch.where(labels == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        weights.double(), len(train_set), replacement=True, generator=generator
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory)

    baseline_val_path = args.output_root / "frozen_baseline_val.json"
    baseline_val = evaluate_frozen_baseline(
        args.trajectory_checkpoint, args.data_root, "val", baseline_val_path, device=device
    )
    model = make_model(
        train_set,
        args.trajectory_checkpoint,
        device,
        gate_mode=args.gate_mode,
        trajectory_residual_scale=args.trajectory_residual_scale,
        enable_trajectory_residual=(args.stage == "B"),
    )
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    history = []
    best_auc = -float("inf")
    best_epoch: int | None = None
    stale_epochs = 0
    baseline_val_ade = baseline_val["trajectory_ade_pixel"]

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            stage=args.stage,
            prior_weight=args.prior_weight,
            traj_weight=args.traj_weight,
            residual_reg_weight=args.residual_reg_weight,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                stage=args.stage,
                prior_weight=args.prior_weight,
                traj_weight=args.traj_weight,
                residual_reg_weight=args.residual_reg_weight,
            )

        trajectory_eligible = (
            args.stage == "A"
            or val_metrics["trajectory_ade_pixel"] <= baseline_val_ade * 1.05
        )
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
            "trajectory_constraint_satisfied": trajectory_eligible,
            "val_ade_limit_pixel": baseline_val_ade * 1.05,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        score = val_metrics["intent_auc"] if trajectory_eligible else -float("inf")
        if trajectory_eligible:
            scheduler.step(score)
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_val_intent_auc": best_auc,
                    "baseline_val_ade_pixel": baseline_val_ade,
                    "trajectory_ade_constraint_pixel": baseline_val_ade * 1.05,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint_selected = checkpoint_path.is_file() and best_epoch is not None
    if checkpoint_selected:
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            stage=args.stage,
            prior_weight=args.prior_weight,
            traj_weight=args.traj_weight,
            residual_reg_weight=args.residual_reg_weight,
        )

    result = {
        "stage": args.stage,
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "trajectory_residual_enabled": args.stage == "B",
        "trajectory_backbone_frozen": model.trajectory_backbone_frozen,
        "trajectory_checkpoint": str(args.trajectory_checkpoint),
        "checkpoint": str(checkpoint_path) if checkpoint_selected else None,
        "checkpoint_selected": checkpoint_selected,
        "best_epoch": best_epoch,
        "best_validation_intent_auc": best_auc if checkpoint_selected else None,
        "val_trajectory_ade_constraint_pixel": baseline_val_ade * 1.05,
        "baseline_val": baseline_val,
        "class_counts": {"negative": int(class_counts[0]), "positive": int(class_counts[1])},
        "ambiguous_supervision_used": False,
        "training_protocol": {
            "prior_weight": args.prior_weight,
            "traj_weight": args.traj_weight if args.stage == "B" else 0.0,
            "residual_reg_weight": args.residual_reg_weight if args.stage == "B" else 0.0,
            "trajectory_residual_scale": args.trajectory_residual_scale,
            "batch_size": args.batch_size,
            "epochs_requested": args.epochs,
            "learning_rate": args.learning_rate,
            "sampling": "inverse-frequency weighted random sampler",
        },
        "history": history,
        "test": test_metrics,
    }
    (args.output_root / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"best_epoch": best_epoch, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
