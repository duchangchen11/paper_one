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


SHARED_GRADIENT_PREFIXES = (
    "target_projection.",
    "position_embedding",
    "target_encoder.",
    "neighbor_encoder.",
    "scene_encoder.",
    "proposal_fusion.",
    "proposal_head.",
    "gate.",
    "fusion.",
)


def balanced_sample_indices(labels: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Create a fixed, approximately class-balanced audit subset without using global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = labels.to(torch.int64).cpu()
    classes = [torch.where(labels == cls)[0] for cls in (0, 1)]
    if any(indices.numel() == 0 for indices in classes):
        raise ValueError("Gradient audit requires both intent classes in the training split")
    per_class = count // 2
    selected = []
    for indices in classes:
        order = torch.randperm(indices.numel(), generator=generator)
        take = min(per_class, indices.numel())
        chosen = indices[order[:take]]
        if take < per_class:
            extra = indices[torch.randint(indices.numel(), (per_class - take,), generator=generator)]
            chosen = torch.cat([chosen, extra])
        selected.append(chosen)
    result = torch.cat(selected)
    if result.numel() < count:
        extra = torch.randint(labels.numel(), (count - result.numel(),), generator=generator)
        result = torch.cat([result, extra])
    return result[torch.randperm(result.numel(), generator=generator)][:count]


def stack_dataset_batch(dataset: JAADSequenceDataset, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    keys = (
        "target_obs",
        "target_abs_obs",
        "future_gt",
        "neighbor_obs",
        "neighbor_mask",
        "neighbor_visible_mask",
        "scene_feat",
        "intent_label",
    )
    return {key: getattr(dataset, key)[indices] for key in keys}


def _flat_gradient_norm(gradients, parameters) -> tuple[torch.Tensor, torch.Tensor]:
    squared_norm = None
    flattened = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            value = torch.zeros_like(parameter, dtype=torch.float32).reshape(-1)
        else:
            value = gradient.detach().float().reshape(-1)
        flattened.append(value)
        term = torch.sum(value * value)
        squared_norm = term if squared_norm is None else squared_norm + term
    if not flattened:
        raise RuntimeError("No gradients were produced for the shared parameter set")
    vector = torch.cat(flattened)
    return torch.sqrt(squared_norm), vector


def measure_shared_gradient_balance(
    model: JointTransformerSceneGate,
    main_batch: dict[str, torch.Tensor],
    ambiguous_batch: dict[str, torch.Tensor],
    device: torch.device,
    prior_weight: float,
    trajectory_weight: float,
    ambiguous_weight: float,
) -> dict[str, float | int]:
    """Measure task gradients at a fixed audit batch without affecting training RNG/state."""
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    was_training = model.training
    try:
        # GRU backward is kept in train mode. Restore RNG afterwards so dropout
        # here cannot perturb the next epoch's data order or stochastic masks.
        model.train()
        main_target = torch.cat([main_batch["target_obs"], main_batch["target_abs_obs"]], dim=-1).to(device)
        output = model(
            main_target,
            main_batch["neighbor_obs"].to(device),
            main_batch["neighbor_mask"].to(device),
            main_batch["neighbor_visible_mask"].to(device),
            main_batch["scene_feat"].to(device),
        )
        labels = main_batch["intent_label"].to(device)
        intent_main = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], labels)
        intent_prior = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], labels)
        trajectory_raw = nn.functional.smooth_l1_loss(
            output["future_pred"], main_batch["future_gt"].to(device)
        )

        ambiguous_target = torch.cat(
            [ambiguous_batch["target_obs"], ambiguous_batch["target_abs_obs"]], dim=-1
        ).to(device)
        ambiguous_output = model(
            ambiguous_target,
            ambiguous_batch["neighbor_obs"].to(device),
            ambiguous_batch["neighbor_mask"].to(device),
            ambiguous_batch["neighbor_visible_mask"].to(device),
            ambiguous_batch["scene_feat"].to(device),
        )
        ambiguity_raw = 0.5 * (
            ambiguous_output["prior_logit"].square().mean()
            + ambiguous_output["intent_logit"].square().mean()
        )
        intent_objective = (
            intent_main + prior_weight * intent_prior + ambiguous_weight * ambiguity_raw
        )

        parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(SHARED_GRADIENT_PREFIXES)
        ]
        intent_gradients = torch.autograd.grad(
            intent_objective, parameters, retain_graph=True, allow_unused=True
        )
        trajectory_gradients = torch.autograd.grad(
            trajectory_raw, parameters, allow_unused=True
        )
        intent_norm, intent_vector = _flat_gradient_norm(intent_gradients, parameters)
        trajectory_norm_raw, trajectory_vector = _flat_gradient_norm(trajectory_gradients, parameters)
        trajectory_norm_weighted = trajectory_norm_raw * trajectory_weight
        cosine = torch.dot(intent_vector, trajectory_vector) / (
            intent_norm * trajectory_norm_raw
        ).clamp_min(torch.finfo(intent_vector.dtype).eps)
        ratio = intent_norm / trajectory_norm_weighted.clamp_min(1e-20)

        return {
            "audit_main_samples": int(labels.numel()),
            "audit_ambiguous_samples": int(ambiguous_batch["intent_label"].numel()),
            "shared_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            "intent_gradient_norm": float(intent_norm.detach().cpu()),
            "trajectory_gradient_norm_unweighted": float(trajectory_norm_raw.detach().cpu()),
            "trajectory_gradient_norm_weighted": float(trajectory_norm_weighted.detach().cpu()),
            "intent_over_weighted_trajectory_gradient_ratio": float(ratio.detach().cpu()),
            "gradient_cosine_intent_vs_trajectory": float(cosine.detach().cpu()),
        }
    finally:
        model.train(was_training)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


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
    audit_count = min(args.batch_size, len(train_set))
    audit_main_indices = balanced_sample_indices(
        train_set.intent_label, audit_count, seed=args.seed + 1_000_003
    )
    audit_ambiguous_generator = torch.Generator(device="cpu").manual_seed(args.seed + 2_000_003)
    audit_ambiguous_indices = torch.randperm(
        len(ambiguous_set), generator=audit_ambiguous_generator
    )[: min(audit_count, len(ambiguous_set))]
    audit_main_batch = stack_dataset_batch(train_set, audit_main_indices)
    audit_ambiguous_batch = stack_dataset_batch(ambiguous_set, audit_ambiguous_indices)
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
    gradient_history = {
        "diagnostic_only": True,
        "optimizer_created_for_diagnostics": False,
        "parameters_updated_by_diagnostics": False,
        "gradient_scope": "shared target, scene, social, proposal, gate, and fusion parameters; task output heads excluded",
        "gradient_sampling": "one fixed class-balanced main training subset and one fixed ambiguous-training subset per run, measured after each epoch; diagnostic RNG state is restored",
        "gradient_ratio_definition": "intent_gradient_norm / (lambda_trajectory * unweighted_trajectory_gradient_norm)",
        "seed": args.seed,
        "lambda_trajectory": args.traj_weight,
        "epochs": [],
    }
    gradient_history_path = args.output_root / "gradient_history.json"
    for epoch in range(1, args.epochs + 1):
        model.train()
        amb_iter = iter(ambiguous_loader)
        epoch_sums = {
            key: torch.zeros((), device=device)
            for key in (
                "main_intent_bce",
                "weighted_proposal_intent_bce",
                "weighted_ambiguity_regularizer",
                "intent_loss",
                "trajectory_loss",
                "weighted_trajectory_loss",
                "total_loss",
            )
        }
        epoch_items = 0
        for batch in train_loader:
            try:
                amb_batch = next(amb_iter)
            except StopIteration:
                amb_iter = iter(ambiguous_loader)
                amb_batch = next(amb_iter)
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            output = model(target, batch["neighbor_obs"].to(device), batch["neighbor_mask"].to(device), batch["neighbor_visible_mask"].to(device), batch["scene_feat"].to(device))
            label = batch["intent_label"].to(device)
            main_intent_bce = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], label)
            proposal_intent_bce = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], label)
            trajectory_loss = nn.functional.smooth_l1_loss(output["future_pred"], batch["future_gt"].to(device))
            amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1).to(device)
            amb_output = model(amb_target, amb_batch["neighbor_obs"].to(device), amb_batch["neighbor_mask"].to(device), amb_batch["neighbor_visible_mask"].to(device), amb_batch["scene_feat"].to(device))
            ambiguity_regularizer = 0.5 * (
                amb_output["prior_logit"].square().mean()
                + amb_output["intent_logit"].square().mean()
            )
            intent_loss = (
                main_intent_bce
                + args.prior_weight * proposal_intent_bce
                + args.ambiguous_weight * ambiguity_regularizer
            )
            # Preserve the baseline objective's operation/order: supervised
            # intent + weighted trajectory, then the ambiguity regularizer.
            loss = main_intent_bce + args.prior_weight * proposal_intent_bce
            loss = loss + args.traj_weight * trajectory_loss
            weighted_ambiguity_regularizer = args.ambiguous_weight * ambiguity_regularizer
            loss = loss + weighted_ambiguity_regularizer
            count = target.shape[0]
            epoch_items += count
            epoch_sums["main_intent_bce"] += main_intent_bce.detach() * count
            epoch_sums["weighted_proposal_intent_bce"] += args.prior_weight * proposal_intent_bce.detach() * count
            epoch_sums["weighted_ambiguity_regularizer"] += weighted_ambiguity_regularizer.detach() * count
            epoch_sums["intent_loss"] += intent_loss.detach() * count
            epoch_sums["trajectory_loss"] += trajectory_loss.detach() * count
            epoch_sums["weighted_trajectory_loss"] += args.traj_weight * trajectory_loss.detach() * count
            epoch_sums["total_loss"] += loss.detach() * count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        train_metrics = {
            key: float((value / epoch_items).detach().cpu())
            for key, value in epoch_sums.items()
        }
        train_metrics["sample_count"] = epoch_items
        gradient_metrics = measure_shared_gradient_balance(
            model,
            audit_main_batch,
            audit_ambiguous_batch,
            device,
            prior_weight=args.prior_weight,
            trajectory_weight=args.traj_weight,
            ambiguous_weight=args.ambiguous_weight,
        )
        gradient_record = {
            "epoch": epoch,
            "lambda_trajectory": args.traj_weight,
            "intent_loss": train_metrics["intent_loss"],
            "trajectory_loss": train_metrics["trajectory_loss"],
            "total_loss": train_metrics["total_loss"],
            **gradient_metrics,
        }
        gradient_history["epochs"].append(gradient_record)
        gradient_history_path.write_text(
            json.dumps(gradient_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with torch.no_grad():
            val = run_epoch(model, val_loader, sizes["val"], device, None, args.prior_weight, args.traj_weight)
        score = val["intent_auc"] + 0.1 * val["intent_f1"] - 0.01 * val["trajectory_ade_pixel"]
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "gradient": gradient_record,
            "val": val,
        }
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
    result = {
        "device": str(device),
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "prior_weight": args.prior_weight,
        "traj_weight": args.traj_weight,
        "ambiguous_weight": args.ambiguous_weight,
        "best_epoch": best_epoch,
        "history": history,
        "test": test,
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
