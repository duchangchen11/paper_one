#!/usr/bin/env python3
"""Train the input-matched M0 intention Transformer from random initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reliability_gated_intent_utils import choose_balanced_accuracy_threshold, fit_temperature
from scripts.trajectory_preserving_utils import (
    SEEDS,
    TrajectoryIntentDataset,
    intention_metrics,
    probabilities_from_logits,
    set_seed,
    sha256_file,
)
from src.models.intention_scratch_transformer import IntentionScratchTransformer

CONFIG_PATH = ROOT / "configs/intention_scratch_matched.json"
RESULTS_ROOT = ROOT / "results/intention_scratch_matched"
CHECKPOINT_ROOT = ROOT / "checkpoints/intention_scratch_matched"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_sha256(model: nn.Module) -> str:
    return state_sha256(model.state_dict())


def evaluate(model: IntentionScratchTransformer, loader: DataLoader, device: torch.device):
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["target"].to(device))
            logits.append(output["intent_logit"].cpu().numpy())
            labels.append(batch["intent_label"].numpy())
    logits_array = np.concatenate(logits).astype(np.float64)
    labels_array = np.concatenate(labels).astype(np.int64)
    return intention_metrics(labels_array, logits_array), logits_array, labels_array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = args.seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Explicit train/val paths only. TrajectoryIntentDataset rejects test.npz.
    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    train_set = TrajectoryIntentDataset(data_root / "train.npz")
    val_set = TrajectoryIntentDataset(data_root / "val.npz")
    train_cfg = config["training"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
    )

    labels = train_set.intent_label.numpy().astype(np.int64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if not positive_count or not negative_count:
        raise RuntimeError("Both intention classes must occur in the training split")
    class_weights = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive": len(labels) / (2.0 * positive_count),
        "negative": len(labels) / (2.0 * negative_count),
        "sampling": "natural shuffle=True; inverse-frequency weighted BCE; no WeightedRandomSampler",
    }

    model = IntentionScratchTransformer(
        input_dim=int(train_set.target.shape[-1]),
        d_model=int(config["model"]["transformer"]["hidden_dimension"]),
        nhead=int(config["model"]["transformer"]["heads"]),
        num_layers=int(config["model"]["transformer"]["layers"]),
        dropout=float(config["model"]["transformer"]["dropout"]),
        max_obs_len=int(train_set.target.shape[1]),
    ).to(device)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Every M0 parameter must be trainable from scratch")
    if hasattr(model, "scene_encoder") or hasattr(model, "decoder"):
        raise RuntimeError("M0 must not contain scene or trajectory decoder modules")

    initial_sha = model_sha256(model)
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    initialization_report = {
        "seed": seed,
        "architecture": config["model"],
        "input_shape": list(train_set.target.shape[1:]),
        "parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_count_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_sha256_before_training": initial_sha,
        "pretrained_checkpoint_loaded": False,
        "pretrained_checkpoint_path": None,
        "initialization": "PyTorch random initialization; positional embedding retains the same zero initialization as P1's trajectory encoder",
    }
    run_dir = RESULTS_ROOT / f"seed{seed}"
    checkpoint_path = CHECKPOINT_ROOT / f"M0_scratch_seed{seed}.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "initialization_report.json", initialization_report)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    best_auc = -float("inf")
    best_brier = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    tolerance = float(train_cfg["selection_tolerance"])

    for epoch in range(1, int(train_cfg["maximum_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            target_labels = batch["intent_label"].to(device)
            logits = model(target)["intent_logit"]
            per_sample_loss = nn.functional.binary_cross_entropy_with_logits(
                logits, target_labels, reduction="none"
            )
            weights = torch.where(
                target_labels > 0.5,
                torch.as_tensor(class_weights["positive"], dtype=target_labels.dtype, device=device),
                torch.as_tensor(class_weights["negative"], dtype=target_labels.dtype, device=device),
            )
            loss = (per_sample_loss * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target_labels)
            sample_count += len(target_labels)

        val_metrics, _, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_weighted_bce": loss_sum / sample_count,
            "validation": val_metrics,
            "model_sha256_after_epoch": model_sha256(model),
        }
        history.append(row)
        write_json(run_dir / "validation_history.json", {"seed": seed, "epochs": history})

        auc = val_metrics["roc_auc"]
        brier = val_metrics["brier"]
        select = auc > best_auc + tolerance
        if abs(auc - best_auc) <= tolerance and brier < best_brier:
            select = True
        print(
            json.dumps(
                {
                    "method": "M0_scratch_matched",
                    "seed": seed,
                    "epoch": epoch,
                    "train_weighted_bce": row["train_weighted_bce"],
                    "val_auc": auc,
                    "val_brier": brier,
                    "val_f1": val_metrics["f1"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_accuracy": val_metrics["accuracy"],
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
                    "method": "M0_scratch_matched",
                    "architecture": config["model"],
                    "input_definition": config["input"],
                    "initialization_sha256": initial_sha,
                    "pretrained_checkpoint_loaded": False,
                    "selected_epoch": epoch,
                    "selection_auc": auc,
                    "selection_brier": brier,
                    "config": config,
                },
                checkpoint_path,
            )

    best_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model"], strict=True)
    selected_val, val_logits, val_labels = evaluate(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    val_probability = probabilities_from_logits(val_logits, temperature)
    threshold = choose_balanced_accuracy_threshold(val_probability, val_labels)
    calibrated_metrics = intention_metrics(
        val_labels, val_logits, temperature=temperature, threshold=threshold
    )

    metrics = {
        "method": "M0_scratch_matched",
        "seed": seed,
        "input_definition": config["input"],
        "pretrained_checkpoint_loaded": False,
        "initialization_sha256": initial_sha,
        "selected_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "selected_checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count_total": initialization_report["parameter_count_total"],
        "parameter_count_trainable": initialization_report["parameter_count_trainable"],
        "trainable_parameter_names": trainable_names,
        "best_epoch": best_epoch,
        "checkpoint_selection": train_cfg["checkpoint_selection"],
        "selected_validation_raw_metrics": selected_val,
        "selected_validation_calibration": {
            "temperature": temperature,
            "temperature_fit_split": "validation",
            "threshold": threshold,
            "threshold_fit_split": "validation balanced accuracy",
            "metrics": calibrated_metrics,
        },
        "training": {
            **train_cfg,
            "class_weights": class_weights,
            "optimizer": "AdamW",
            "sampling": "natural shuffle=True",
            "loss": "inverse-frequency weighted BCE on intention only",
            "data_splits_loaded": ["train", "val"],
            "test_accessed": False,
        },
        "history": history,
        "test": None,
        "test_evaluation_status": "withheld_until_protocol_freeze",
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "metrics_validation.json",
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "selected_validation_raw_metrics": selected_val,
            "selected_validation_calibration": metrics["selected_validation_calibration"],
            "all_epochs": history,
            "test": None,
        },
    )
    print(json.dumps({"method": "M0_scratch_matched", "seed": seed, "best_epoch": best_epoch, "validation": selected_val, "test": None}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
