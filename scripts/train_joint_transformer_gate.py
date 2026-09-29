#!/usr/bin/env python3
"""Train a joint Transformer trajectory and scene-social intention model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


COMPONENT_ABLATIONS = (
    "full",
    "no_scene",
    "no_social",
    "no_proposal_loss",
    "no_adaptive_gate",
    "no_ambiguity",
)


def component_loss_weights(
    component_ablation: str, prior_weight: float, ambiguous_weight: float
) -> tuple[float, float]:
    """Return loss weights for one predeclared component intervention."""
    if component_ablation not in COMPONENT_ABLATIONS:
        raise ValueError(f"Unknown component ablation: {component_ablation}")
    if component_ablation == "no_proposal_loss":
        prior_weight = 0.0
    if component_ablation == "no_ambiguity":
        ambiguous_weight = 0.0
    return float(prior_weight), float(ambiguous_weight)


def requested_split_names(skip_test: bool) -> tuple[str, ...]:
    """Select splits without opening test data during training-only phases."""
    return ("train", "val") if skip_test else ("train", "val", "test")


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


def intent_auc_selection_decision(
    current_auc: float,
    current_brier: float,
    best_auc_seen: float | None,
    selected_auc: float | None,
    selected_brier: float | None,
    tolerance: float = 1e-4,
) -> tuple[bool, str, float]:
    """Select by raw validation AUC; within tolerance, prefer lower raw Brier.

    best_auc_seen is a high-water mark. The selected checkpoint may be at most
    ``tolerance`` below it when the Brier tie-break chooses another epoch.
    """
    if not math.isfinite(current_auc) or not math.isfinite(current_brier):
        raise ValueError("intent_auc selection requires finite AUC and Brier")
    new_high = best_auc_seen is None or current_auc > best_auc_seen
    next_best_auc = current_auc if best_auc_seen is None else max(best_auc_seen, current_auc)
    if selected_auc is None or selected_brier is None:
        return True, "first_valid_checkpoint", next_best_auc
    if new_high and current_auc - selected_auc > tolerance:
        return True, "higher_auc_outside_tie_tolerance", next_best_auc
    if abs(next_best_auc - current_auc) <= tolerance and current_brier < selected_brier:
        return True, "auc_within_1e-4_tie_lower_brier", next_best_auc
    if new_high and current_auc - selected_auc <= tolerance and current_auc - selected_auc >= -tolerance:
        return False, "auc_within_1e-4_tie_brier_not_lower", next_best_auc
    return False, "lower_auc_or_outside_tie_tolerance", next_best_auc


def validation_scheduler_monitor(selection_mode: str, validation_metrics: dict) -> tuple[str, float]:
    if selection_mode == "intent_auc":
        return "intent_auc", float(validation_metrics["intent_auc"])
    if selection_mode == "composite":
        value = validation_metrics["intent_auc"] + 0.1 * validation_metrics["intent_f1"] - 0.01 * validation_metrics["trajectory_ade_pixel"]
        return "composite_auc_f1_ade", float(value)
    raise ValueError(f"Unknown selection mode: {selection_mode}")


class FingerprintingWeightedRandomSampler(WeightedRandomSampler):
    """Same inverse-class weighted sampling, with a reproducibility fingerprint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_indices: list[int] | None = None
        self.last_sha256: str | None = None

    def __iter__(self):
        indices = list(super().__iter__())
        self.last_indices = indices
        digest = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
        self.last_sha256 = digest
        return iter(indices)


class DynamicGradientBalance:
    """Log-space EMA controller for the shared intent/trajectory gradient ratio."""

    def __init__(
        self,
        initial_lambda: float = 100.0,
        target_ratio: float = 20.0,
        beta: float = 0.9,
        lambda_min: float = 10.0,
        lambda_max: float = 300.0,
        warmup_epochs: int = 1,
        update_interval: int = 10,
        eps: float = 1e-12,
    ) -> None:
        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")
        if not (0.0 < lambda_min <= initial_lambda <= lambda_max):
            raise ValueError("lambda bounds must contain the initial lambda")
        if target_ratio <= 0 or update_interval < 1 or warmup_epochs < 0 or eps <= 0:
            raise ValueError("target_ratio, update_interval, and eps must be positive")
        self.lambda_value = float(initial_lambda)
        self.target_ratio = float(target_ratio)
        self.beta = float(beta)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.warmup_epochs = int(warmup_epochs)
        self.update_interval = int(update_interval)
        self.eps = float(eps)
        self.update_attempts = 0
        self.update_count = 0
        self.skipped_invalid_count = 0
        self.lower_bound_hits = 0
        self.upper_bound_hits = 0

    def should_measure(self, batch_index: int) -> bool:
        return batch_index > 0 and batch_index % self.update_interval == 0

    def observe(
        self, epoch: int, batch_index: int, intent_norm: float, trajectory_norm: float
    ) -> dict[str, float | int | bool | str]:
        """Record one aligned gradient sample and, when allowed, update lambda.

        The updated value applies starting with the *next* training batch. Epoch 1
        is a fixed-lambda warm-up, though its scheduled gradient samples are logged.
        """
        before = self.lambda_value
        record: dict[str, float | int | bool | str] = {
            "epoch": int(epoch),
            "batch_index": int(batch_index),
            "lambda_used": before,
            "lambda_next": before,
            "controller_update": False,
            "update_status": "warmup" if epoch <= self.warmup_epochs else "interval_sample",
        }
        if not self.should_measure(batch_index):
            raise ValueError("observe must be called only at a configured measurement interval")
        values_finite = math.isfinite(intent_norm) and math.isfinite(trajectory_norm)
        if epoch <= self.warmup_epochs:
            record["update_status"] = "warmup"
            return record
        self.update_attempts += 1
        if (
            not values_finite
            or intent_norm <= self.eps
            or trajectory_norm <= self.eps
        ):
            self.skipped_invalid_count += 1
            record["update_status"] = "invalid_gradient_norm"
            return record

        raw_ratio = intent_norm / (trajectory_norm + self.eps)
        lambda_target = raw_ratio / self.target_ratio
        if not math.isfinite(lambda_target) or lambda_target <= 0:
            self.skipped_invalid_count += 1
            record["update_status"] = "invalid_lambda_target"
            return record

        smoothed_log_lambda = (
            self.beta * math.log(before)
            + (1.0 - self.beta) * math.log(lambda_target)
        )
        clipped_log_lambda = min(
            math.log(self.lambda_max), max(math.log(self.lambda_min), smoothed_log_lambda)
        )
        if clipped_log_lambda >= math.log(self.lambda_max):
            next_lambda = self.lambda_max
        elif clipped_log_lambda <= math.log(self.lambda_min):
            next_lambda = self.lambda_min
        else:
            next_lambda = math.exp(clipped_log_lambda)
        hit_min = next_lambda <= self.lambda_min * (1.0 + 1e-12)
        hit_max = next_lambda >= self.lambda_max * (1.0 - 1e-12)
        self.lambda_value = float(next_lambda)
        self.update_count += 1
        self.lower_bound_hits += int(hit_min)
        self.upper_bound_hits += int(hit_max)
        record.update(
            {
                "lambda_target": float(lambda_target),
                "lambda_next": self.lambda_value,
                "controller_update": True,
                "update_status": "clipped_min" if hit_min else "clipped_max" if hit_max else "updated",
                "hit_lambda_min": hit_min,
                "hit_lambda_max": hit_max,
            }
        )
        return record


def shared_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return the exact shared parameter scope used by the preceding audit."""
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(SHARED_GRADIENT_PREFIXES)
    ]


def measure_aligned_task_gradients(
    intent_loss: torch.Tensor,
    trajectory_loss: torch.Tensor,
    named_parameters: list[tuple[str, nn.Parameter]],
) -> dict[str, float]:
    """Measure task gradients on one existing graph without touching .grad buffers."""
    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise RuntimeError("No shared parameters selected for dynamic gradient balance")
    intent_gradients = torch.autograd.grad(
        intent_loss, parameters, retain_graph=True, allow_unused=True
    )
    trajectory_gradients = torch.autograd.grad(
        trajectory_loss, parameters, retain_graph=True, allow_unused=True
    )
    intent_norm, intent_vector = _flat_gradient_norm(intent_gradients, parameters)
    trajectory_norm, trajectory_vector = _flat_gradient_norm(trajectory_gradients, parameters)
    eps = torch.finfo(intent_vector.dtype).eps
    cosine = torch.dot(intent_vector, trajectory_vector) / (
        intent_norm * trajectory_norm
    ).clamp_min(eps)
    intent_norm_value = float(intent_norm.detach().cpu())
    trajectory_norm_value = float(trajectory_norm.detach().cpu())
    return {
        "intent_gradient_norm": intent_norm_value,
        "trajectory_gradient_norm_unweighted": trajectory_norm_value,
        "raw_gradient_ratio": intent_norm_value / (trajectory_norm_value + 1e-12),
        "gradient_cosine_intent_vs_trajectory": float(cosine.detach().cpu()),
    }


def compose_training_objective(
    main_intent_bce: torch.Tensor,
    proposal_intent_bce: torch.Tensor,
    trajectory_loss: torch.Tensor,
    ambiguity_regularizer: torch.Tensor,
    prior_weight: float,
    trajectory_weight: float,
    ambiguous_weight: float,
) -> torch.Tensor:
    """Keep the legacy fixed-objective summation order for reproducibility."""
    loss = main_intent_bce + prior_weight * proposal_intent_bce
    loss = loss + trajectory_weight * trajectory_loss
    loss = loss + ambiguous_weight * ambiguity_regularizer
    return loss


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def summarize_dynamic_epoch(
    epoch: int,
    epoch_start_lambda: float,
    epoch_end_lambda: float,
    lambda_values: list[float],
    gradient_samples: list[dict[str, float | int | bool | str]],
    update_records: list[dict[str, float | int | bool | str]],
    train_metrics: dict[str, float | int],
    controller: DynamicGradientBalance,
) -> dict[str, object]:
    def values(key: str) -> list[float]:
        return [float(sample[key]) for sample in gradient_samples]

    def stats(data: list[float]) -> dict[str, float]:
        if not data:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "median": float(np.median(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

    weighted_ratios = values("weighted_gradient_ratio")
    lambda_stats = stats(lambda_values)
    actual_updates = [r for r in update_records if r.get("controller_update")]
    lower_hits = sum(bool(r.get("hit_lambda_min", False)) for r in actual_updates)
    upper_hits = sum(bool(r.get("hit_lambda_max", False)) for r in actual_updates)
    update_count = len(actual_updates)
    gradient_summary = {
        "measurement_count": len(gradient_samples),
        "mean_intent_gradient_norm": float(np.mean(values("intent_gradient_norm"))) if gradient_samples else 0.0,
        "mean_trajectory_gradient_norm_unweighted": float(np.mean(values("trajectory_gradient_norm_unweighted"))) if gradient_samples else 0.0,
        "mean_trajectory_gradient_norm_weighted": float(np.mean(values("weighted_trajectory_gradient_norm"))) if gradient_samples else 0.0,
        "mean_raw_gradient_ratio": float(np.mean(values("raw_gradient_ratio"))) if gradient_samples else 0.0,
        "weighted_gradient_ratio": stats(weighted_ratios),
        "gradient_cosine_mean": float(np.mean(values("gradient_cosine_intent_vs_trajectory"))) if gradient_samples else 0.0,
    }
    return {
        "epoch": epoch,
        "train_intent_loss": float(train_metrics["intent_loss"]),
        "raw_trajectory_loss": float(train_metrics["trajectory_loss"]),
        "weighted_trajectory_loss": float(train_metrics["weighted_trajectory_loss"]),
        "total_loss": float(train_metrics["total_loss"]),
        "gradient_statistics": gradient_summary,
        "lambda_statistics": {
            **lambda_stats,
            "training_batch_count": len(lambda_values),
            "start": float(epoch_start_lambda),
            "end": float(epoch_end_lambda),
        },
        "lambda_per_training_batch": [float(value) for value in lambda_values],
        "controller_updates": {
            "measurement_attempts": len(update_records),
            "updates_applied": update_count,
            "invalid_skips": sum(r.get("update_status", "").startswith("invalid") for r in update_records),
            "lower_bound_hits": lower_hits,
            "upper_bound_hits": upper_hits,
            "lower_bound_hit_percent": 100.0 * lower_hits / update_count if update_count else 0.0,
            "upper_bound_hit_percent": 100.0 * upper_hits / update_count if update_count else 0.0,
        },
        "gradient_samples": gradient_samples,
        "update_records": update_records,
        "cumulative_controller_state": {
            "update_attempts": controller.update_attempts,
            "updates_applied": controller.update_count,
            "invalid_skips": controller.skipped_invalid_count,
            "lower_bound_hits": controller.lower_bound_hits,
            "upper_bound_hits": controller.upper_bound_hits,
        },
    }


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
    total_loss = total_intent_objective = total_trajectory_loss = total_items = 0.0
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
        main_intent_bce = loss_fn(output["intent_logit"], label)
        proposal_intent_bce = loss_fn(output["prior_logit"], label)
        intent_objective = main_intent_bce + prior_weight * proposal_intent_bce
        trajectory_loss = nn.functional.smooth_l1_loss(output["future_pred"], batch["future_gt"].to(device))
        loss = intent_objective + traj_weight * trajectory_loss
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
        total_intent_objective += float(intent_objective.detach()) * count
        total_trajectory_loss += float(trajectory_loss.detach()) * count
        labels.extend(label.detach().cpu().numpy().tolist())
        logits.extend(output["intent_logit"].detach().cpu().numpy().tolist())
        predictions.append(output["future_pred"].detach().cpu().numpy())
        targets.append(batch["future_gt"].numpy())
        scales.append(image_sizes[len(np.concatenate(targets)) - count : len(np.concatenate(targets))])
        gates.append(output["gate"].detach().cpu().numpy())
        entropies.append(output["entropy"].detach().cpu().numpy())
    metrics = compute_metrics(labels, logits, predictions, targets, scales, gates, entropies)
    metrics["loss"] = float(total_loss / total_items)
    metrics["intent_objective"] = float(total_intent_objective / total_items)
    metrics["trajectory_loss"] = float(total_trajectory_loss / total_items)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--ambiguous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-state-checkpoint", type=Path, default=None)
    parser.add_argument("--init-trajectory-checkpoint", type=Path, default=None)
    parser.add_argument("--gate-mode", choices=("uncertainty", "always", "none"), default="uncertainty")
    parser.add_argument("--component-ablation", choices=COMPONENT_ABLATIONS, default="full")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    parser.add_argument("--traj-weight", type=float, default=1.0)
    parser.add_argument(
        "--traj-weight-mode", choices=("fixed", "dynamic_gradient"), default="fixed"
    )
    parser.add_argument("--ambiguous-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--selection-mode", choices=("composite", "intent_auc"), default="composite")
    parser.add_argument("--selection-tolerance", type=float, default=1e-4)
    parser.add_argument("--dgb-target-ratio", type=float, default=20.0)
    parser.add_argument("--dgb-beta", type=float, default=0.9)
    parser.add_argument("--dgb-lambda-min", type=float, default=10.0)
    parser.add_argument("--dgb-lambda-max", type=float, default=300.0)
    parser.add_argument("--dgb-warmup-epochs", type=int, default=1)
    parser.add_argument("--dgb-update-interval", type=int, default=10)
    parser.add_argument("--skip-test", action="store_true", help="Do not load or evaluate the test split")
    parser.add_argument("--smoke-test", action="store_true", help="Validate the two-epoch DGB smoke protocol")
    args = parser.parse_args()
    if args.traj_weight_mode == "dynamic_gradient" and args.traj_weight <= 0:
        parser.error("dynamic_gradient requires --traj-weight to provide a positive initial lambda")
    if args.initial_state_checkpoint is not None and args.init_trajectory_checkpoint is not None:
        parser.error("--initial-state-checkpoint and --init-trajectory-checkpoint are mutually exclusive")
    if args.selection_tolerance < 0:
        parser.error("--selection-tolerance must be non-negative")
    if args.smoke_test and (args.traj_weight_mode != "dynamic_gradient" or args.seed != 123 or args.epochs != 2):
        parser.error("--smoke-test requires dynamic_gradient, seed 123, and exactly 2 epochs")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    train_set = JAADSequenceDataset(args.data_root / "train.npz")
    val_set = JAADSequenceDataset(args.data_root / "val.npz")
    test_set = None if args.skip_test else JAADSequenceDataset(args.data_root / "test.npz")
    ambiguous_set = JAADSequenceDataset(args.ambiguous_root / "train.npz")
    counts = torch.bincount(train_set.intent_label.to(torch.int64), minlength=2).float()
    weights = torch.where(train_set.intent_label == 0, 1.0 / counts[0], 1.0 / counts[1])
    train_sampler = FingerprintingWeightedRandomSampler(weights.double(), len(train_set), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = (
        None if test_set is None else DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    )
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
    split_names = requested_split_names(args.skip_test)
    raw = {split: np.load(args.data_root / f"{split}.npz", allow_pickle=False) for split in split_names}
    sizes = {split: torch.from_numpy(raw[split]["image_size"].astype(np.float32)) for split in raw}
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(train_set.scene_feat.shape[-1]),
        hidden_dim=args.hidden_dim,
        pred_len=train_set.future_gt.shape[1],
        gate_mode=args.gate_mode,
        max_obs_len=train_set.target_obs.shape[1],
        component_flags_all_enabled=args.component_ablation == "full",
        component_ablation=None if args.component_ablation == "full" else args.component_ablation,
    ).to(device)
    effective_prior_weight, effective_ambiguous_weight = component_loss_weights(
        args.component_ablation, args.prior_weight, args.ambiguous_weight
    )
    declared_initial_state_sha256 = None
    if args.initial_state_checkpoint is not None:
        initial_payload = torch.load(args.initial_state_checkpoint, map_location="cpu", weights_only=False)
        initial_state = initial_payload["model"] if "model" in initial_payload else initial_payload
        model.load_state_dict(initial_state, strict=True)
        declared_initial_state_sha256 = initial_payload.get("sha256") if isinstance(initial_payload, dict) else None
    initial_state_sha256 = state_dict_sha256(model.state_dict())
    if declared_initial_state_sha256 is not None and initial_state_sha256 != declared_initial_state_sha256:
        raise RuntimeError("Loaded initial-state weights do not match their declared SHA256")
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

    # Deterministic validation-only probe. This never opens the official-test split.
    was_training = model.training
    model.eval()
    probe_batch = stack_dataset_batch(val_set, list(range(min(8, len(val_set)))))
    probe_target = torch.cat([probe_batch["target_obs"], probe_batch["target_abs_obs"]], dim=-1).to(device)
    with torch.no_grad():
        probe_output = model(
            probe_target,
            probe_batch["neighbor_obs"].to(device),
            probe_batch["neighbor_mask"].to(device),
            probe_batch["neighbor_visible_mask"].to(device),
            probe_batch["scene_feat"].to(device),
        )
    component_probe = {
        "component_ablation": args.component_ablation,
        "validation_probe_samples": int(probe_target.shape[0]),
        "intent_logits_finite": bool(torch.isfinite(probe_output["intent_logit"]).all().item()),
        "proposal_logits_finite": bool(torch.isfinite(probe_output["prior_logit"]).all().item()),
        "future_predictions_finite": bool(torch.isfinite(probe_output["future_pred"]).all().item()),
        "proposal_output_present": "prior_logit" in probe_output,
        "effective_scene_context_mean_abs": float(probe_output["effective_scene_context"].abs().mean().cpu()),
        "effective_social_context_mean_abs": float(probe_output["effective_social_context"].abs().mean().cpu()),
        "scene_downstream_tensor_zero": bool(
            args.component_ablation != "no_scene"
            or torch.count_nonzero(probe_output["effective_scene_context"]).item() == 0
        ),
        "social_downstream_tensor_zero": bool(
            args.component_ablation != "no_social"
            or torch.count_nonzero(probe_output["effective_social_context"]).item() == 0
        ),
        "fixed_neutral_gate": bool(
            args.component_ablation != "no_adaptive_gate"
            or torch.allclose(probe_output["gate"], torch.full_like(probe_output["gate"], 0.5))
        ),
        "both_gate_fusion_contexts_present": bool(
            args.component_ablation != "no_adaptive_gate"
            or (
                "effective_scene_context" in probe_output
                and "effective_social_context" in probe_output
                and hasattr(model, "gate")
            )
        ),
        "effective_prior_weight": effective_prior_weight,
        "effective_ambiguous_weight": effective_ambiguous_weight,
        "weighted_proposal_loss_zero": effective_prior_weight == 0.0,
        "weighted_ambiguity_contribution_zero": effective_ambiguous_weight == 0.0,
        "validation_only_probe": True,
        "test_split_loaded": not args.skip_test,
    }
    model.train(was_training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_score = -float("inf")
    best_auc_seen = None
    selected_validation_auc = None
    selected_validation_brier = None
    best_epoch = 0
    history = []
    gradient_history = {
        "diagnostic_only": args.traj_weight_mode == "fixed",
        "optimizer_created_for_diagnostics": False,
        "parameters_updated_by_diagnostics": False,
        "gradient_scope": "same shared target, scene, social, proposal, gate, and fusion parameter prefixes as the preceding audit; intent_head and traj_head excluded",
        "gradient_sampling": "fixed mode: prior fixed class-balanced diagnostic subset after each epoch; DGB mode: every 10th actual training batch on its existing forward graph",
        "gradient_ratio_definition": "intent_gradient_norm / (lambda_trajectory * unweighted_trajectory_gradient_norm)",
        "seed": args.seed,
        "lambda_trajectory_initial": args.traj_weight,
        "trajectory_weight_mode": args.traj_weight_mode,
        "epochs": [],
    }
    gradient_history_path = args.output_root / "gradient_history.json"
    controller = None
    shared_parameters = None
    if args.traj_weight_mode == "dynamic_gradient":
        controller = DynamicGradientBalance(
            initial_lambda=args.traj_weight,
            target_ratio=args.dgb_target_ratio,
            beta=args.dgb_beta,
            lambda_min=args.dgb_lambda_min,
            lambda_max=args.dgb_lambda_max,
            warmup_epochs=args.dgb_warmup_epochs,
            update_interval=args.dgb_update_interval,
        )
        shared_parameters = shared_named_parameters(model)
        parameter_manifest_path = PROJECT_ROOT / "results/joint_dynamic_balance/shared_parameter_names.json"
        parameter_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        parameter_manifest_path.write_text(
            json.dumps(
                {
                    "source_commit": "ad05ef51711d74f423cecd0b7e8d89debb0a3ae4",
                    "prefixes": list(SHARED_GRADIENT_PREFIXES),
                    "excluded_prefixes": ["intent_head.", "traj_head."],
                    "parameter_count": len(shared_parameters),
                    "scalar_count": int(sum(parameter.numel() for _, parameter in shared_parameters)),
                    "parameter_names": [name for name, _ in shared_parameters],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        gradient_history.update(
            {
                "controller": {
                    "name": "DGB-20",
                    "target_ratio": args.dgb_target_ratio,
                    "beta": args.dgb_beta,
                    "lambda_min": args.dgb_lambda_min,
                    "lambda_max": args.dgb_lambda_max,
                    "warmup_epochs": args.dgb_warmup_epochs,
                    "update_interval_training_batches": args.dgb_update_interval,
                    "eps": controller.eps,
                    "ema": "log-space; update after the measured batch and apply lambda to the next batch",
                },
                "shared_parameter_names_path": str(parameter_manifest_path),
            }
        )
    training_steps = 0
    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        amb_iter = iter(ambiguous_loader)
        epoch_start_lambda = controller.lambda_value if controller is not None else args.traj_weight
        epoch_lambda_values: list[float] = []
        epoch_gradient_samples: list[dict[str, float | int | bool | str]] = []
        epoch_update_records: list[dict[str, float | int | bool | str]] = []
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
        for batch_index, batch in enumerate(train_loader, start=1):
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
                + effective_prior_weight * proposal_intent_bce
                + effective_ambiguous_weight * ambiguity_regularizer
            )
            weighted_ambiguity_regularizer = effective_ambiguous_weight * ambiguity_regularizer
            lambda_used = controller.lambda_value if controller is not None else args.traj_weight
            measured_gradient = None
            update_record = None
            if controller is not None:
                epoch_lambda_values.append(float(lambda_used))
                if controller.should_measure(batch_index):
                    # Clear the prior batch's .grad buffers before measuring. autograd.grad
                    # itself does not write them; the same retained graph is then used by
                    # the ordinary total-loss backward below.
                    optimizer.zero_grad(set_to_none=True)
                    measured_gradient = measure_aligned_task_gradients(
                        intent_loss, trajectory_loss, shared_parameters
                    )
                    update_record = controller.observe(
                        epoch,
                        batch_index,
                        measured_gradient["intent_gradient_norm"],
                        measured_gradient["trajectory_gradient_norm_unweighted"],
                    )
                    measured_gradient.update(
                        {
                            "epoch": epoch,
                            "batch_index": batch_index,
                            "lambda_used": float(lambda_used),
                            "weighted_trajectory_gradient_norm": float(
                                lambda_used * measured_gradient["trajectory_gradient_norm_unweighted"]
                            ),
                            "weighted_gradient_ratio": float(
                                measured_gradient["intent_gradient_norm"]
                                / (
                                    lambda_used
                                    * measured_gradient["trajectory_gradient_norm_unweighted"]
                                    + controller.eps
                                )
                            ),
                            "lambda_next": float(update_record["lambda_next"]),
                            "controller_update": bool(update_record["controller_update"]),
                        }
                    )
                    epoch_gradient_samples.append(measured_gradient)
                    epoch_update_records.append(update_record)
                loss = compose_training_objective(
                    main_intent_bce,
                    proposal_intent_bce,
                    trajectory_loss,
                    ambiguity_regularizer,
                    effective_prior_weight,
                    lambda_used,
                    effective_ambiguous_weight,
                )
            else:
                # Preserve the legacy fixed-mode summation and backward path exactly.
                loss = main_intent_bce + effective_prior_weight * proposal_intent_bce
                loss = loss + args.traj_weight * trajectory_loss
                loss = loss + weighted_ambiguity_regularizer
            count = target.shape[0]
            epoch_items += count
            epoch_sums["main_intent_bce"] += main_intent_bce.detach() * count
            epoch_sums["weighted_proposal_intent_bce"] += effective_prior_weight * proposal_intent_bce.detach() * count
            epoch_sums["weighted_ambiguity_regularizer"] += weighted_ambiguity_regularizer.detach() * count
            epoch_sums["intent_loss"] += intent_loss.detach() * count
            epoch_sums["trajectory_loss"] += trajectory_loss.detach() * count
            epoch_sums["weighted_trajectory_loss"] += lambda_used * trajectory_loss.detach() * count
            epoch_sums["total_loss"] += loss.detach() * count
            if measured_gradient is None:
                optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_steps += 1
        train_metrics = {
            key: float((value / epoch_items).detach().cpu())
            for key, value in epoch_sums.items()
        }
        train_metrics["sample_count"] = epoch_items
        train_metrics["sampler_sha256"] = train_sampler.last_sha256
        train_metrics["first_sample_indices"] = (train_sampler.last_indices or [])[:32]
        if controller is None:
            gradient_metrics = measure_shared_gradient_balance(
                model,
                audit_main_batch,
                audit_ambiguous_batch,
                device,
                prior_weight=effective_prior_weight,
                trajectory_weight=args.traj_weight,
                ambiguous_weight=effective_ambiguous_weight,
            )
            gradient_record = {
                "epoch": epoch,
                "lambda_trajectory": args.traj_weight,
                "intent_loss": train_metrics["intent_loss"],
                "trajectory_loss": train_metrics["trajectory_loss"],
                "total_loss": train_metrics["total_loss"],
                **gradient_metrics,
            }
        else:
            gradient_record = summarize_dynamic_epoch(
                epoch=epoch,
                epoch_start_lambda=epoch_start_lambda,
                epoch_end_lambda=controller.lambda_value,
                lambda_values=epoch_lambda_values,
                gradient_samples=epoch_gradient_samples,
                update_records=epoch_update_records,
                train_metrics=train_metrics,
                controller=controller,
            )
            gradient_record["lambda_trajectory"] = controller.lambda_value
        gradient_history["epochs"].append(gradient_record)
        gradient_history_path.write_text(
            json.dumps(gradient_history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with torch.no_grad():
            val_lambda = controller.lambda_value if controller is not None else args.traj_weight
            val = run_epoch(model, val_loader, sizes["val"], device, None, effective_prior_weight, val_lambda)
        scheduler_monitor_metric, scheduler_monitor_value = validation_scheduler_monitor(args.selection_mode, val)
        if args.selection_mode == "intent_auc":
            scheduler.step(scheduler_monitor_value)
            selected_this_epoch, tie_break_reason, best_auc_seen = intent_auc_selection_decision(
                float(val["intent_auc"]), float(val["intent_brier"]), best_auc_seen,
                selected_validation_auc, selected_validation_brier, args.selection_tolerance,
            )
            selection_score = scheduler_monitor_value
            if selected_this_epoch:
                best_epoch = epoch
                selected_validation_auc = float(val["intent_auc"])
                selected_validation_brier = float(val["intent_brier"])
                torch.save({"model": model.state_dict(), "args": vars(args)}, args.checkpoint)
        else:
            selection_score = scheduler_monitor_value
            scheduler.step(scheduler_monitor_value)
            selected_this_epoch = bool(selection_score > best_score)
            tie_break_reason = "higher_composite_score" if selected_this_epoch else "composite_score_not_higher"
            if selected_this_epoch:
                best_score = float(selection_score)
                best_epoch = epoch
                selected_validation_auc = float(val["intent_auc"])
                selected_validation_brier = float(val["intent_brier"])
                torch.save({"model": model.state_dict(), "args": vars(args)}, args.checkpoint)
        if args.selection_mode == "intent_auc" and selected_this_epoch:
            best_score = float(selection_score)
        record = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "learning_rate_next": float(optimizer.param_groups[0]["lr"]),
            "train": train_metrics,
            "gradient": gradient_record,
            "val": val,
            "scheduler_monitor_metric": scheduler_monitor_metric,
            "scheduler_monitor_value": scheduler_monitor_value,
            "selection": {
                "mode": args.selection_mode,
                "primary_metric": "raw_validation_intent_auc" if args.selection_mode == "intent_auc" else "composite_auc_f1_ade",
                "current_validation_auc": float(val["intent_auc"]),
                "current_validation_brier": float(val["intent_brier"]),
                "best_validation_auc_seen": best_auc_seen,
                "selected_checkpoint_validation_auc": selected_validation_auc,
                "selected_checkpoint_validation_brier": selected_validation_brier,
                "selected_checkpoint": bool(selected_this_epoch),
                "tie_break_reason": tie_break_reason,
                "selection_used_trajectory_metric": args.selection_mode != "intent_auc",
                "tolerance": args.selection_tolerance if args.selection_mode == "intent_auc" else None,
            },
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
    if not args.skip_test:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            test = run_epoch(model, test_loader, sizes["test"], device, None, effective_prior_weight, args.traj_weight)
    else:
        test = None
    final_state_sha256 = state_dict_sha256(model.state_dict())
    result = {
        "device": str(device),
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "prior_weight": args.prior_weight,
        "traj_weight": args.traj_weight,
        "ambiguous_weight": args.ambiguous_weight,
        "effective_prior_weight": effective_prior_weight,
        "effective_ambiguous_weight": effective_ambiguous_weight,
        "component_ablation": args.component_ablation,
        "component_flags_all_enabled": args.component_ablation == "full",
        "component_probe": component_probe,
        "selection_mode": args.selection_mode,
        "selection_tolerance": args.selection_tolerance,
        "scheduler_monitor_metric": "intent_auc" if args.selection_mode == "intent_auc" else "composite_auc_f1_ade",
        "best_epoch": best_epoch,
        "best_validation_auc_seen": best_auc_seen,
        "selected_checkpoint_validation_auc": selected_validation_auc,
        "selected_checkpoint_validation_brier": selected_validation_brier,
        "history": history,
        "test": test,
        "test_evaluation_status": "withheld_until_protocol_freeze" if args.skip_test else "evaluated",
        "training_steps": training_steps,
        "initial_model_state_sha256": initial_state_sha256,
        "final_model_state_sha256": final_state_sha256,
        "dynamic_controller_final_lambda": None if controller is None else controller.lambda_value,
        "dynamic_controller_cumulative_updates": None if controller is None else {
            "attempts": controller.update_attempts,
            "applied": controller.update_count,
            "invalid_skips": controller.skipped_invalid_count,
            "lower_bound_hits": controller.lower_bound_hits,
            "upper_bound_hits": controller.upper_bound_hits,
        },
    }
    (args.output_root / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "test": test, "test_evaluation_status": result["test_evaluation_status"]}, ensure_ascii=False, indent=2))
    if args.smoke_test:
        epoch_records = gradient_history["epochs"]
        epoch1_lambdas = epoch_records[0]["lambda_statistics"]
        epoch2_lambdas = epoch_records[1]["lambda_statistics"]
        all_lambdas = [
            float(value)
            for epoch_record in epoch_records
            for value in (
                epoch_record["lambda_statistics"]["min"],
                epoch_record["lambda_statistics"]["max"],
                epoch_record["lambda_statistics"]["start"],
                epoch_record["lambda_statistics"]["end"],
            )
        ]
        all_norms = [
            float(sample[key])
            for epoch_record in epoch_records
            for sample in epoch_record["gradient_samples"]
            for key in (
                "intent_gradient_norm",
                "trajectory_gradient_norm_unweighted",
                "weighted_trajectory_gradient_norm",
                "weighted_gradient_ratio",
                "gradient_cosine_intent_vs_trajectory",
            )
        ]
        checks = {
            "epoch1_lambda_fixed_at_initial_100": epoch1_lambdas["min"] == 100.0 and epoch1_lambdas["max"] == 100.0,
            "epoch2_lambda_changed": not math.isclose(epoch2_lambdas["end"], 100.0, rel_tol=0.0, abs_tol=1e-9),
            "lambda_within_10_300": all(10.0 <= value <= 300.0 for value in all_lambdas),
            "finite_lambda_and_gradient_statistics": all(math.isfinite(value) for value in all_lambdas + all_norms),
            "ordinary_backward_and_optimizer_steps_completed": training_steps > 0,
            "model_parameters_changed": initial_state_sha256 != final_state_sha256,
            "test_split_not_loaded_or_evaluated": args.skip_test and test is None,
            "epoch2_controller_updated": epoch_records[1]["controller_updates"]["updates_applied"] > 0,
        }
        report = {
            "status": "passed" if all(checks.values()) else "failed",
            "seed": args.seed,
            "epochs": args.epochs,
            "device": str(device),
            "checks": checks,
            "epoch1_lambda": epoch1_lambdas,
            "epoch2_lambda": epoch2_lambdas,
            "training_steps": training_steps,
            "initial_model_state_sha256": initial_state_sha256,
            "final_model_state_sha256": final_state_sha256,
            "fixed_mode_objective_unit_test_required": True,
            "test_evaluation_status": result["test_evaluation_status"],
        }
        (args.output_root / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if report["status"] != "passed":
            raise RuntimeError(f"DGB smoke checks failed: {checks}")


if __name__ == "__main__":
    main()
