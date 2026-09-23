#!/usr/bin/env python3
"""Train/evaluate one matched intent readout for the 15x15 scene ablation."""

from __future__ import annotations

import argparse
import hashlib
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

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.scene_ablation_intent import SceneAblationIntentModel
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state_dict(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_backbone(checkpoint_path: Path, sample: dict[str, torch.Tensor]) -> SceneTrajectoryTransformer:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["model"]
    saved_args = payload.get("args", {})
    d_model = int(saved_args.get("d_model", state["input_projection.weight"].shape[0]))
    num_layers = int(
        saved_args.get(
            "num_layers",
            1
            + max(
                int(key.split(".layers.")[1].split(".")[0])
                for key in state
                if key.startswith("temporal_encoder.layers.")
            ),
        )
    )
    target = torch.cat([sample["target_obs"], sample["target_abs_obs"]], dim=-1)
    backbone = SceneTrajectoryTransformer(
        input_dim=target.shape[-1],
        scene_dim=sample["scene_feat"].numel(),
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        pred_len=sample["future_gt"].shape[0],
        dropout=0.1,
        max_obs_len=target.shape[0],
    )
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(logits, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def ece_10(labels: np.ndarray, probabilities: np.ndarray) -> float:
    y = labels.astype(np.int64)
    p = np.clip(probabilities.astype(np.float64), 0.0, 1.0)
    value = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lower) & (p < upper if index < 9 else p <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return value


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    y = labels.astype(np.int64)
    p = np.clip(probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    prediction = (p >= threshold).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10": ece_10(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


def fit_temperature(logits: np.ndarray, labels: np.ndarray, device: torch.device) -> float:
    x = torch.as_tensor(logits, dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.float64, device=device)
    log_t = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(x / log_t.exp().clamp(0.05, 20.0), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0).cpu())


def fit_temperature_bias(
    logits: np.ndarray, labels: np.ndarray, device: torch.device
) -> tuple[float, float]:
    x = torch.as_tensor(logits, dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.float64, device=device)
    log_t = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_t, bias], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            x / log_t.exp().clamp(0.05, 20.0) + bias, y
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        float(log_t.detach().exp().clamp(0.05, 20.0).cpu()),
        float(bias.detach().cpu()),
    )


def apply_calibration(logits: np.ndarray, calibration: dict[str, float | str]) -> np.ndarray:
    temperature = float(calibration["temperature"])
    bias = float(calibration.get("bias", 0.0))
    return sigmoid(np.asarray(logits) / temperature + bias)


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    p = np.unique(np.clip(probabilities.astype(np.float64), 0.0, 1.0))
    thresholds = np.unique(
        np.concatenate(([0.0, 1.0], p, (p[:-1] + p[1:]) / 2.0))
    )
    y = labels.astype(np.int64)
    order = np.argsort(probabilities, kind="mergesort")
    sorted_p = probabilities[order]
    sorted_y = y[order]
    prefix_pos = np.concatenate(([0], np.cumsum(sorted_y)))
    boundary = np.searchsorted(sorted_p, thresholds, side="left")
    true_pos = int(y.sum()) - prefix_pos[boundary]
    false_pos = (len(y) - boundary) - true_pos
    true_neg = len(y) - int(y.sum()) - false_pos
    tpr = true_pos / max(int(y.sum()), 1)
    tnr = true_neg / max(len(y) - int(y.sum()), 1)
    scores = (tpr + tnr) / 2.0
    best = float(scores.max())
    tied = thresholds[np.isclose(scores, best, rtol=0.0, atol=1e-12)]
    chosen = float(tied[np.argmin(np.abs(tied - 0.5))])
    return chosen, best


@torch.no_grad()
def collect(
    model: SceneAblationIntentModel,
    loader: DataLoader,
    device: torch.device,
    *,
    scene_override: str = "real",
    zero_scene_context: bool = False,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in ("labels", "raw_logits", "future_pred", "future_gt", "image_size")
    }
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        scene = batch["scene_feat"].to(device)
        if scene_override == "zero":
            scene = torch.zeros_like(scene)
        elif scene_override != "real":
            raise ValueError(f"Unsupported scene_override: {scene_override}")
        output = model(target, scene, zero_scene_context=zero_scene_context)
        collected["labels"].append(batch["intent_label"].numpy())
        collected["raw_logits"].append(output["base_raw_logit"].cpu().numpy())
        collected["future_pred"].append(output["future_pred"].cpu().numpy())
        collected["future_gt"].append(batch["future_gt"].numpy())
        collected["image_size"].append(batch["image_size"].numpy())
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def run_train_epoch(
    model: SceneAblationIntentModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    labels, logits = [], []
    total_loss = 0.0
    count = 0
    for batch in loader:
        y = batch["intent_label"].to(device)
        if torch.any((y < 0) | (y > 1)):
            raise ValueError("Clean intent training data must contain only labels 0 and 1")
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        output = model(target, batch["scene_feat"].to(device))
        loss = nn.functional.binary_cross_entropy_with_logits(output["base_raw_logit"], y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        optimizer.step()
        n = len(y)
        count += n
        total_loss += float(loss.detach()) * n
        labels.append(y.detach().cpu().numpy())
        logits.append(output["base_raw_logit"].detach().cpu().numpy())
    y_all = np.concatenate(labels).astype(np.int64)
    p_all = sigmoid(np.concatenate(logits))
    return {**classification_metrics(y_all, p_all, 0.5), "loss": total_loss / count}


def trajectory_metrics(values: dict[str, np.ndarray]) -> dict[str, float]:
    error = values["future_pred"] - values["future_gt"]
    pixel = np.linalg.norm(error * values["image_size"][:, None, :], axis=-1)
    normalized = np.linalg.norm(error, axis=-1)
    return {
        "ade_pixel": float(pixel.mean()),
        "fde_pixel": float(pixel[:, -1].mean()),
        "ade_normalized": float(normalized.mean()),
        "fde_normalized": float(normalized[:, -1].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--trajectory-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--reuse-checkpoint", type=Path, default=None)
    parser.add_argument("--scene-mode", choices=("target_only", "target_scene"), required=True)
    parser.add_argument("--seed", type=int, required=True)
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
    metrics_path = args.output_root / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {metrics_path}")
    for path in (args.trajectory_checkpoint, args.reuse_checkpoint):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

    train_set = SequenceWithImageSize(args.data_root / "train.npz")
    val_set = SequenceWithImageSize(args.data_root / "val.npz")
    test_set = SequenceWithImageSize(args.data_root / "test.npz")
    train_labels = train_set.dataset.intent_label.to(torch.int64)
    if torch.any((train_labels < 0) | (train_labels > 1)):
        raise ValueError("Clean train split unexpectedly contains non-binary labels")
    class_counts = torch.bincount(train_labels, minlength=2).float()
    sample_weights = torch.where(train_labels == 0, 1.0 / class_counts[0], 1.0 / class_counts[1])
    sampler = WeightedRandomSampler(
        sample_weights.double(), len(train_set), replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    backbone = load_backbone(args.trajectory_checkpoint, train_set[0]).to(device)
    model = SceneAblationIntentModel(backbone, scene_mode=args.scene_mode).to(device)
    if not model.trajectory_backbone_frozen:
        raise RuntimeError("Frozen trajectory backbone invariant failed")

    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_auc = float("-inf")
    if args.reuse_checkpoint is not None:
        if args.scene_mode != "target_scene" or args.seed != 123:
            raise ValueError("Only target_scene seed 123 may reuse the historical pilot checkpoint")
        saved = torch.load(args.reuse_checkpoint, map_location=device, weights_only=False)
        if Path(str(saved.get("trajectory_checkpoint", ""))).name != args.trajectory_checkpoint.name:
            raise ValueError("Reused intent checkpoint references a different trajectory checkpoint")
        model.load_state_dict(saved["model"], strict=True)
        legacy_metrics_path = PROJECT_ROOT / "results/fixed_base_intent_seed123/metrics.json"
        legacy_metrics = json.loads(legacy_metrics_path.read_text(encoding="utf-8"))
        if legacy_metrics.get("trajectory_checkpoint") != str(saved.get("trajectory_checkpoint")):
            raise ValueError("Historical metrics do not confirm the reused trajectory checkpoint")
        if legacy_metrics.get("class_counts") != {
            "negative": int(class_counts[0]),
            "positive": int(class_counts[1]),
        }:
            raise ValueError("Historical intent class counts do not match the current clean train split")
        best_epoch = int(legacy_metrics.get("best_epoch", 0))
        best_auc = float(saved.get("best_validation_auc", float("nan")))
        training_source = "reused fixed_base_intent_seed123; protocol and checkpoint audited"
    else:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required unless --reuse-checkpoint is supplied")
        if args.checkpoint.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.checkpoint}")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=1
        )
        stale = 0
        training_source = "trained in this scene-ablation protocol"
        for epoch in range(1, args.epochs + 1):
            train_metrics = run_train_epoch(model, train_loader, device, optimizer)
            val_values = collect(model, val_loader, device)
            val_probability = sigmoid(val_values["raw_logits"])
            val_metrics = classification_metrics(val_values["labels"], val_probability, 0.5)
            scheduler.step(val_metrics["auc"])
            record = {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                        "best_validation_auc": best_auc,
                        "trajectory_checkpoint": str(args.trajectory_checkpoint),
                        "scene_mode": args.scene_mode,
                    },
                    args.checkpoint,
                )
            else:
                stale += 1
                if stale >= args.patience:
                    break
        saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)

    val_values = collect(model, val_loader, device)
    test_values = collect(model, test_loader, device)
    val_raw = val_values["raw_logits"]
    val_labels = val_values["labels"].astype(np.int64)
    temp = fit_temperature(val_raw, val_labels, device)
    temp_bias, bias = fit_temperature_bias(val_raw, val_labels, device)
    calibration_candidates = {
        "temperature": {"method": "temperature", "temperature": temp, "bias": 0.0},
        "temperature_bias": {"method": "temperature+bias", "temperature": temp_bias, "bias": bias},
    }
    for candidate in calibration_candidates.values():
        candidate["validation"] = classification_metrics(
            val_labels, apply_calibration(val_raw, candidate), 0.5
        )
    # Primary selection is minimum validation Brier; exact/negligible ties use ECE,
    # then prefer the simpler temperature-only calibration.
    chosen_name = min(
        calibration_candidates,
        key=lambda name: (
            calibration_candidates[name]["validation"]["brier"],
            calibration_candidates[name]["validation"]["ece_10"],
            name != "temperature",
        ),
    )
    calibration = calibration_candidates[chosen_name]
    val_prob = apply_calibration(val_raw, calibration)
    threshold, validation_best_bacc = select_threshold(val_labels, val_prob)

    test_labels = test_values["labels"].astype(np.int64)
    test_prob = apply_calibration(test_values["raw_logits"], calibration)
    test_selected_metrics = classification_metrics(test_labels, test_prob, threshold)
    test_half_metrics = classification_metrics(test_labels, test_prob, 0.5)
    result: dict[str, Any] = {
        "seed": args.seed,
        "scene_mode": args.scene_mode,
        "training_source": training_source,
        "device": str(device),
        "trajectory_checkpoint": str(args.trajectory_checkpoint),
        "trajectory_checkpoint_sha256": sha256_file(args.trajectory_checkpoint),
        "trajectory_backbone_state_sha256": sha256_state_dict(model.trajectory_backbone.state_dict()),
        "intent_checkpoint": str(args.reuse_checkpoint or args.checkpoint),
        "intent_checkpoint_sha256": sha256_file(args.reuse_checkpoint or args.checkpoint),
        "trajectory_backbone_frozen": model.trajectory_backbone_frozen,
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
        "architecture": {
            "intent_head": "base_fusion(2*d_model->d_model)+GELU+Dropout+base_head(d_model->1)",
            "d_model": model.d_model,
            "classifier_input_dim": model.base_fusion[0].in_features,
            "total_parameter_count": sum(p.numel() for p in model.parameters()),
            "intent_head_parameter_count": sum(
                p.numel() for name, p in model.named_parameters()
                if name.startswith("base_fusion.") or name.startswith("base_head.")
            ),
        },
        "dataset_sha256": {
            split: sha256_file(args.data_root / f"{split}.npz")
            for split in ("train", "val", "test")
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "checkpoint_selection": "highest validation AUC",
            "metadata_source": "reused checkpoint metadata" if args.reuse_checkpoint else "this run",
        },
        "best_epoch": best_epoch,
        "best_validation_auc_raw": best_auc,
        "class_counts": {"negative": int(class_counts[0]), "positive": int(class_counts[1])},
        "sampling": {
            "method": "inverse-frequency WeightedRandomSampler",
            "replacement": True,
            "num_samples": len(train_set),
            "seed": args.seed,
        },
        "ambiguous_supervision_used": False,
        "trajectory_loss_used": False,
        "social_loss_used": False,
        "history": history,
        "validation_calibration_candidates": calibration_candidates,
        "selected_calibration": calibration,
        "calibration_selection_rule": "minimum validation Brier; tie-break by validation ECE, then temperature-only",
        "selected_threshold": threshold,
        "threshold_selection": {
            "split": "validation",
            "objective": "maximum balanced accuracy",
            "validation_balanced_accuracy": validation_best_bacc,
            "tie_break": "threshold nearest 0.5",
        },
        "validation_selected_calibration_metrics": classification_metrics(val_labels, val_prob, threshold),
        "test": {
            **test_selected_metrics,
            "selected_threshold": threshold,
            "balanced_accuracy_threshold_0_5": test_half_metrics["balanced_accuracy"],
            "f1_threshold_0_5": test_half_metrics["f1"],
            "trajectory": trajectory_metrics(test_values),
        },
    }
    if args.scene_mode == "target_scene":
        permutation = np.random.default_rng(9124).permutation(len(test_set))
        permuted_set = _SceneOverrideDataset(test_set, permutation)
        permuted_loader = DataLoader(permuted_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        permuted_values = collect(model, permuted_loader, device)
        zero_context_values = collect(model, test_loader, device, zero_scene_context=True)
        result["scene_permutation_diagnostic"] = {
            "seed": 9124,
            "real_scene_auc": test_selected_metrics["auc"],
            "shuffled_scene_auc": classification_metrics(
                test_labels, apply_calibration(permuted_values["raw_logits"], calibration), threshold
            )["auc"],
            "zero_scene_auc": classification_metrics(
                test_labels, apply_calibration(zero_context_values["raw_logits"], calibration), threshold
            )["auc"],
            "note": "permutation changes only the scene embeddings at intent readout; target and labels stay fixed",
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"seed": args.seed, "scene_mode": args.scene_mode, "test": result["test"]}, ensure_ascii=False, indent=2), flush=True)


class _SceneOverrideDataset(Dataset):
    """Re-index scene embeddings only, preserving each test target and label."""

    def __init__(self, source: SequenceWithImageSize, scene_indices: np.ndarray) -> None:
        self.source = source
        self.scene_indices = np.asarray(scene_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.source[index]
        other = self.source[int(self.scene_indices[index])]
        item["scene_feat"] = other["scene_feat"]
        return item


if __name__ == "__main__":
    main()
