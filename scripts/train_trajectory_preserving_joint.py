#!/usr/bin/env python3
"""Train P1/P2 intention heads over a strictly frozen trajectory backbone."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reliability_gated_intent_utils import choose_balanced_accuracy_threshold, fit_temperature
from scripts.trajectory_preserving_utils import (
    SEEDS,
    TrajectoryIntentDataset,
    backbone_sha256,
    intention_metrics,
    load_seed_backbone,
    probabilities_from_logits,
    set_seed,
    trajectory_metrics,
)

CONFIG_PATH = ROOT / "configs/trajectory_preserving_joint.json"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def evaluate_validation(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_prediction: list[np.ndarray] = []
    all_future: list[np.ndarray] = []
    all_sizes: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["target"].to(device), batch["scene_feat"].to(device))
            all_logits.append(output["intent_logit"].cpu().numpy())
            all_labels.append(batch["intent_label"].numpy())
            all_prediction.append(output["future_pred"].cpu().numpy())
            all_future.append(batch["future_gt"].numpy())
            all_sizes.append(batch["image_size"].numpy())
    logits = np.concatenate(all_logits).astype(np.float64)
    labels = np.concatenate(all_labels).astype(np.int64)
    prediction = np.concatenate(all_prediction).astype(np.float32)
    future = np.concatenate(all_future).astype(np.float32)
    image_size = np.concatenate(all_sizes).astype(np.float32)
    binary = intention_metrics(labels, logits, threshold=0.5)
    traj = trajectory_metrics(prediction, future, image_size)
    return {**binary, **traj}, logits, labels, prediction, future, image_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("P1_target_only", "P2_target_scene"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = args.seed
    intent_input = "target" if args.method == "P1_target_only" else "target_scene"
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Deliberately load only train.npz and val.npz. Test data is exclusively
    # handled by the separate post-freeze evaluator.
    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    train_set = TrajectoryIntentDataset(data_root / "train.npz")
    val_set = TrajectoryIntentDataset(data_root / "val.npz")
    train_config = config["training"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    labels = train_set.intent_label
    positive_count = int((labels > 0.5).sum())
    negative_count = int((labels <= 0.5).sum())
    class_weights = {
        "positive": len(labels) / (2.0 * positive_count),
        "negative": len(labels) / (2.0 * negative_count),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "sampling": "natural shuffle=True; inverse-frequency weighted BCE",
    }

    model, loading_report, trajectory_checkpoint, trajectory_checkpoint_sha = load_seed_backbone(
        seed,
        intent_input,
        device=device,
        input_dim=int(train_set.target.shape[-1]),
        scene_dim=int(train_set.scene_feat.shape[-1]),
        observed_length=int(train_set.target.shape[1]),
        prediction_length=int(train_set.future_gt.shape[1]),
    )
    if not loading_report["complete"]:
        raise RuntimeError("Trajectory checkpoint was not fully restored")
    backbone_before = backbone_sha256(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    expected_trainable = [
        name for name in model.state_dict() if name.startswith(("intent_adapter.", "intent_head."))
    ]
    if trainable_names != expected_trainable:
        raise RuntimeError(f"Unexpected trainable scope: {trainable_names}")
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise RuntimeError("Trajectory backbone must be frozen before training")

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    best_auc = -float("inf")
    best_brier = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    parameter_hash_history = [{"stage": "before_training", "epoch": 0, "sha256": backbone_before}]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(train_config["maximum_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            scene = batch["scene_feat"].to(device)
            y = batch["intent_label"].to(device)
            output = model(target, scene)
            per_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                output["intent_logit"], y, reduction="none"
            )
            weights = torch.where(
                y > 0.5,
                torch.as_tensor(class_weights["positive"], dtype=y.dtype, device=device),
                torch.as_tensor(class_weights["negative"], dtype=y.dtype, device=device),
            )
            loss = (per_sample_loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                float(train_config["gradient_clip_norm"]),
            )
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(y)
            train_sample_count += len(y)

        if backbone_sha256(model) != backbone_before:
            parameter_hash_history.append(
                {"stage": "after_epoch", "epoch": epoch, "sha256": backbone_sha256(model)}
            )
            write_json(
                args.output_root / "parameter_hashes.json",
                {
                    "seed": seed,
                    "method": args.method,
                    "hash_before": backbone_before,
                    "hash_history": parameter_hash_history,
                    "unchanged": False,
                },
            )
            raise RuntimeError(f"Frozen trajectory backbone changed during epoch {epoch}")
        parameter_hash_history.append(
            {"stage": "after_epoch", "epoch": epoch, "sha256": backbone_before}
        )

        val_metrics, val_logits, val_labels, _, _, _ = evaluate_validation(model, val_loader, device)
        reference_path = ROOT / "results/trajectory_preserving_joint/trajectory_reference.json"
        reference = json.loads(reference_path.read_text(encoding="utf-8"))["per_seed"][str(seed)]["validation"]
        ade_drift = val_metrics["ade_pixel"] - float(reference["trajectory_ade_pixel"])
        fde_drift = val_metrics["fde_pixel"] - float(reference["trajectory_fde_pixel"])
        if abs(ade_drift) >= 0.05 or abs(fde_drift) >= 0.05:
            raise RuntimeError(
                f"Trajectory validation drift exceeded 0.05 px for {args.method} seed {seed}: "
                f"delta ADE/FDE={ade_drift:.6f}/{fde_drift:.6f}"
            )

        epoch_row = {
            "epoch": epoch,
            "train_weighted_bce": train_loss_sum / train_sample_count,
            "validation": val_metrics,
            "trajectory_parameter_sha256": backbone_sha256(model),
            "trajectory_parameter_hash_unchanged": backbone_sha256(model) == backbone_before,
            "trajectory_ade_difference_vs_reference_pixel": ade_drift,
            "trajectory_fde_difference_vs_reference_pixel": fde_drift,
        }
        history.append(epoch_row)
        write_json(args.output_root / "validation_history.json", {"seed": seed, "method": args.method, "epochs": history})

        auc = val_metrics["roc_auc"]
        brier = val_metrics["brier"]
        tolerance = float(train_config["selection_tolerance"])
        select = auc > best_auc + tolerance
        if abs(auc - best_auc) <= tolerance and brier < best_brier:
            select = True
        print(
            json.dumps(
                {
                    "method": args.method,
                    "seed": seed,
                    "epoch": epoch,
                    "train_weighted_bce": epoch_row["train_weighted_bce"],
                    "val_auc": auc,
                    "val_brier": brier,
                    "val_ade_pixel": val_metrics["ade_pixel"],
                    "val_fde_pixel": val_metrics["fde_pixel"],
                    "trajectory_hash_unchanged": epoch_row["trajectory_parameter_hash_unchanged"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if select:
            best_auc, best_brier, best_epoch = auc, brier, epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "seed": seed,
                    "method": args.method,
                    "intent_input": intent_input,
                    "trajectory_checkpoint": str(trajectory_checkpoint.relative_to(ROOT)),
                    "trajectory_checkpoint_sha256": trajectory_checkpoint_sha,
                    "trajectory_backbone_sha256": backbone_before,
                    "config": config,
                    "selected_epoch": epoch,
                    "selection_auc": auc,
                    "selection_brier": brier,
                },
                args.checkpoint,
            )

    best = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best["model"], strict=True)
    model.eval()
    selected_val, logits, y_true, _, _, _ = evaluate_validation(model, val_loader, device)
    temperature = fit_temperature(logits, y_true)
    val_probability = probabilities_from_logits(logits, temperature)
    threshold = choose_balanced_accuracy_threshold(val_probability, y_true)
    calibrated_validation = intention_metrics(
        y_true, logits, temperature=temperature, threshold=threshold
    )
    if backbone_sha256(model) != backbone_before:
        raise RuntimeError("Selected checkpoint does not preserve the frozen trajectory state")

    metrics = {
        "method": args.method,
        "seed": seed,
        "intent_input": intent_input,
        "trajectory_checkpoint": str(trajectory_checkpoint.relative_to(ROOT)),
        "trajectory_checkpoint_sha256": trajectory_checkpoint_sha,
        "trajectory_backbone_sha256_before_training": backbone_before,
        "trajectory_backbone_sha256_after_training": backbone_sha256(model),
        "trajectory_backbone_frozen": True,
        "weight_loading_report": loading_report,
        "trainable_parameter_names": trainable_names,
        "best_epoch": best_epoch,
        "checkpoint_selection": config["training"]["checkpoint_selection"],
        "selected_validation_raw_metrics": selected_val,
        "selected_validation_calibration": {
            "temperature": temperature,
            "temperature_fit_split": "validation",
            "threshold": threshold,
            "threshold_fit_split": "validation balanced accuracy",
            "metrics": calibrated_validation,
        },
        "training": {
            **train_config,
            "class_weights": class_weights,
            "optimizer": "AdamW",
            "sampling": "natural shuffle=True",
            "loss": "inverse-frequency weighted BCE on intention only; trajectory output is forward-only monitor",
        },
        "history": history,
        "test": None,
        "test_evaluation_status": "withheld_until_protocol_freeze",
    }
    write_json(args.output_root / "metrics.json", metrics)
    write_json(
        args.output_root / "parameter_hashes.json",
        {
            "seed": seed,
            "method": args.method,
            "hash_before": backbone_before,
            "hash_history": parameter_hash_history,
            "hash_after": backbone_sha256(model),
            "unchanged_every_epoch": all(row["sha256"] == backbone_before for row in parameter_hash_history),
            "final_hash_unchanged": backbone_sha256(model) == backbone_before,
        },
    )
    print(
        json.dumps(
            {
                "method": args.method,
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_raw": selected_val,
                "validation_calibrated": calibrated_validation,
                "test": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
