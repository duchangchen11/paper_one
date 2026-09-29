#!/usr/bin/env python3
"""Validation-only Pareto, representation, and selected-gradient diagnostics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_clean_utils import (
    ARMS, CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, SHARED_PREFIXES, load_config,
    sha256_state, write_json,
)
from scripts.train_joint_transformer_gate import set_seed
from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate


def make_model(dataset: JAADSequenceDataset, hidden_dim: int, gate_mode: str) -> JointTransformerSceneGate:
    return JointTransformerSceneGate(
        input_dim=8, scene_dim=int(dataset.scene_feat.shape[-1]), hidden_dim=hidden_dim,
        pred_len=int(dataset.future_gt.shape[1]), gate_mode=gate_mode,
        max_obs_len=int(dataset.target_obs.shape[1]),
    )


def feature_stats(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    variance = values.var(axis=0)
    return {
        "sample_count": int(values.shape[0]),
        "feature_width": int(values.shape[1]),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "mean_l2_norm": float(norms.mean()),
        "std_l2_norm": float(norms.std()),
        "per_dimension_variance": variance.tolist(),
        "mean_per_dimension_variance": float(variance.mean()),
    }


def feature_similarity(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(current, dtype=np.float64)
    cosine = np.sum(x * y, axis=1) / np.maximum(
        np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12
    )
    xc, yc = x - x.mean(axis=0, keepdims=True), y - y.mean(axis=0, keepdims=True)
    cross, xx, yy = xc.T @ yc, xc.T @ xc, yc.T @ yc
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    cka = float(np.linalg.norm(cross, ord="fro") ** 2 / denominator) if denominator else 0.0
    return {"cosine_mean": float(cosine.mean()), "cosine_std": float(cosine.std()), "linear_cka": cka}


@torch.inference_mode()
def extract_features(
    model: JointTransformerSceneGate,
    dataset: JAADSequenceDataset,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    model.eval()
    target_chunks: list[np.ndarray] = []
    fused_chunks: list[np.ndarray] = []
    target_outputs: list[torch.Tensor] = []
    fused_outputs: list[torch.Tensor] = []
    h_target = model.target_encoder.register_forward_hook(
        lambda _module, _inputs, output: target_outputs.append(output.detach())
    )
    h_fusion = model.fusion.register_forward_hook(
        lambda _module, _inputs, output: fused_outputs.append(output.detach())
    )
    try:
        for start in range(0, len(indices), batch_size):
            ix = indices[start : start + batch_size]
            target = torch.cat([dataset.target_obs[ix], dataset.target_abs_obs[ix]], dim=-1).to(device)
            before_target, before_fused = len(target_outputs), len(fused_outputs)
            model(
                target,
                dataset.neighbor_obs[ix].to(device),
                dataset.neighbor_mask[ix].to(device),
                dataset.neighbor_visible_mask[ix].to(device),
                dataset.scene_feat[ix].to(device),
            )
            if len(target_outputs) != before_target + 1 or len(fused_outputs) != before_fused + 1:
                raise RuntimeError("Expected one target/fusion feature output per validation batch")
            target_chunks.append(target_outputs[-1][:, -1].cpu().numpy().copy())
            fused_chunks.append(fused_outputs[-1].cpu().numpy().copy())
    finally:
        h_target.remove()
        h_fusion.remove()
    return {
        "target_encoder_last": np.concatenate(target_chunks),
        "fused_representation": np.concatenate(fused_chunks),
    }


def pareto_history() -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": "validation",
        "selection_use": False,
        "definition": "A point is Pareto-efficient if no epoch has at least its AUC and at most its ADE with one strict improvement.",
        "per_seed": {},
    }
    tradeoff_any = False
    for seed in SEEDS:
        metrics_path = RESULTS_ROOT / "J100_clean" / f"seed{seed}" / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        points = [
            {
                "epoch": int(row["epoch"]),
                "validation_auc": float(row["val"]["intent_auc"]),
                "validation_ade_pixel": float(row["val"]["trajectory_ade_pixel"]),
                "validation_fde_pixel": float(row["val"]["trajectory_fde_pixel"]),
            }
            for row in metrics["history"]
        ]
        for point in points:
            auc, ade = point["validation_auc"], point["validation_ade_pixel"]
            point["pareto_efficient"] = not any(
                other["validation_auc"] >= auc
                and other["validation_ade_pixel"] <= ade
                and (other["validation_auc"] > auc or other["validation_ade_pixel"] < ade)
                for other in points
            )
        tradeoff = any(
            (left["validation_auc"] > right["validation_auc"] and left["validation_ade_pixel"] > right["validation_ade_pixel"])
            or (left["validation_auc"] < right["validation_auc"] and left["validation_ade_pixel"] < right["validation_ade_pixel"])
            for i, left in enumerate(points) for right in points[i + 1 :]
        )
        tradeoff_any |= tradeoff
        result["per_seed"][str(seed)] = {
            "selected_epoch": int(metrics["best_epoch"]),
            "selected_validation_auc": float(metrics["selected_checkpoint_validation_auc"]),
            "selected_validation_brier": float(metrics["selected_checkpoint_validation_brier"]),
            "auc_ade_tradeoff_observed": tradeoff,
            "pareto_points": [row for row in points if row["pareto_efficient"]],
            "all_epochs": points,
        }
    result["auc_ade_tradeoff_observed_any_seed"] = tradeoff_any
    result["interpretation"] = "Descriptive validation-only epoch trade-off; not used for checkpoint selection or test tuning."
    write_json(RESULTS_ROOT / "pareto_history.json", result)
    return result


def representation_diagnostic(
    config: dict[str, Any], val_set: JAADSequenceDataset, device: torch.device,
) -> dict[str, Any]:
    requested = 1000
    indices = torch.linspace(0, len(val_set) - 1, min(requested, len(val_set))).round().long()
    index_sha = hashlib.sha256(indices.numpy().astype(np.int64).tobytes()).hexdigest()
    training = config["training"]
    result: dict[str, Any] = {
        "split": "validation",
        "sample_count": int(len(indices)),
        "index_selection": "rounded evenly-spaced row indices, fixed and shared across seeds/arms/initialization",
        "sample_indices_sha256": index_sha,
        "feature_definitions": {
            "target_encoder_last": "last temporal target-encoder token",
            "fused_representation": "output of the shared fusion block, consumed by intention and trajectory heads",
        },
        "selection_use": False,
        "per_seed": {},
    }
    for seed in SEEDS:
        initial_payload = torch.load(CHECKPOINT_ROOT / f"initial_state_seed{seed}.pt", map_location="cpu", weights_only=False)
        initial_model = make_model(val_set, int(training["hidden_dim"]), str(training["gate_mode"]))
        initial_model.load_state_dict(initial_payload["model"], strict=True)
        init_features = extract_features(initial_model.to(device), val_set, indices, device)
        result["per_seed"][str(seed)] = {"initial_state_sha256": sha256_state(initial_payload["model"]), "features": {}}
        del initial_model
        arm_features: dict[str, dict[str, np.ndarray]] = {}
        for arm, weight in ARMS.items():
            run_metrics = json.loads((RESULTS_ROOT / arm / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
            checkpoint = torch.load(
                CHECKPOINT_ROOT / "formal" / f"{arm}_seed{seed}.pt", map_location="cpu", weights_only=False
            )
            if int(run_metrics["best_epoch"]) < 1 or float(run_metrics["traj_weight"]) != weight:
                raise RuntimeError(f"Invalid selected checkpoint metadata for {arm}/seed{seed}")
            model = make_model(val_set, int(training["hidden_dim"]), str(training["gate_mode"]))
            model.load_state_dict(checkpoint["model"], strict=True)
            features = extract_features(model.to(device), val_set, indices, device)
            arm_features[arm] = features
            del model
        for name in init_features:
            result["per_seed"][str(seed)]["features"][name] = {
                "initialization_statistics": feature_stats(init_features[name]),
                "arms": {
                    arm: {
                        "statistics": feature_stats(arm_features[arm][name]),
                        "similarity_to_initialization": feature_similarity(init_features[name], arm_features[arm][name]),
                    }
                    for arm in ARMS
                },
                "J100_vs_J0": feature_similarity(arm_features["J100_clean"][name], arm_features["J0_clean"][name]),
            }
        del init_features
        del arm_features
    result["interpretation"] = "Descriptive endpoint features only; these selected-checkpoint comparisons do not establish a training-time causal mechanism."
    write_json(RESULTS_ROOT / "representation_diagnostic.json", result)
    return result


def gradient_for_checkpoint(
    model: JointTransformerSceneGate,
    val_set: JAADSequenceDataset,
    ambiguous_set: JAADSequenceDataset,
    *, seed: int, prior_weight: float, ambiguous_weight: float, traj_weight: float,
) -> dict[str, Any]:
    model.train()
    device = next(model.parameters()).device
    count = min(512, len(val_set), len(ambiguous_set))
    val_batch = {key: getattr(val_set, key)[:count].to(device) for key in (
        "target_obs", "target_abs_obs", "future_gt", "neighbor_obs", "neighbor_mask",
        "neighbor_visible_mask", "scene_feat", "intent_label",
    )}
    amb_batch = {key: getattr(ambiguous_set, key)[:count].to(device) for key in (
        "target_obs", "target_abs_obs", "future_gt", "neighbor_obs", "neighbor_mask",
        "neighbor_visible_mask", "scene_feat", "intent_label",
    )}
    target = torch.cat([val_batch["target_obs"], val_batch["target_abs_obs"]], dim=-1)
    output = model(target, val_batch["neighbor_obs"], val_batch["neighbor_mask"], val_batch["neighbor_visible_mask"], val_batch["scene_feat"])
    bce = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], val_batch["intent_label"])
    prior = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], val_batch["intent_label"])
    intent = bce + prior_weight * prior
    amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1)
    amb_out = model(amb_target, amb_batch["neighbor_obs"], amb_batch["neighbor_mask"], amb_batch["neighbor_visible_mask"], amb_batch["scene_feat"])
    ambiguity = 0.5 * (amb_out["prior_logit"].square().mean() + amb_out["intent_logit"].square().mean())
    intent = intent + ambiguous_weight * ambiguity
    trajectory = nn.functional.smooth_l1_loss(output["future_pred"], val_batch["future_gt"])
    params = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith(SHARED_PREFIXES)]
    gi = torch.autograd.grad(intent, params, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(trajectory, params, allow_unused=True)

    def flat_norm(grads: tuple[torch.Tensor | None, ...]) -> tuple[float, torch.Tensor]:
        pieces = [torch.zeros_like(p, dtype=torch.float32).reshape(-1) if g is None else g.detach().float().reshape(-1) for g, p in zip(grads, params)]
        vector = torch.cat(pieces)
        return float(torch.linalg.vector_norm(vector)), vector

    gi_norm, gi_vec = flat_norm(gi)
    gt_norm, gt_vec = flat_norm(gt)
    cosine = float(torch.dot(gi_vec, gt_vec) / max(gi_norm * gt_norm, 1e-12))
    weighted = float(traj_weight * gt_norm)
    return {
        "seed": seed,
        "validation_batch_count": count,
        "ambiguous_regularizer_batch_count": count,
        "ambiguous_regularizer_source": "fixed first rows of ambiguous train split; follows the training objective definition",
        "selected_checkpoint_gradient_is_endpoint_diagnostic_only": True,
        "intent_objective": float(intent.detach()),
        "raw_trajectory_loss": float(trajectory.detach()),
        "intent_shared_gradient_norm": gi_norm,
        "raw_trajectory_shared_gradient_norm": gt_norm,
        "weighted_trajectory_shared_gradient_norm": weighted,
        "intent_vs_raw_trajectory_cosine": cosine,
        "intent_over_weighted_trajectory_gradient_ratio": None if weighted == 0 else gi_norm / weighted,
        "trajectory_weight": traj_weight,
    }


def gradient_diagnostic(config: dict[str, Any], val_set: JAADSequenceDataset, ambiguous_set: JAADSequenceDataset, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "split": "fixed validation batch for intent/trajectory; fixed ambiguous-train batch for ambiguity regularization",
        "selection_use": False,
        "interpretation": "Selected-checkpoint gradient is a descriptive endpoint diagnostic only and does not represent gradients across training.",
        "per_seed": {},
    }
    for seed in SEEDS:
        result["per_seed"][str(seed)] = {}
        for arm, weight in ARMS.items():
            checkpoint = torch.load(CHECKPOINT_ROOT / "formal" / f"{arm}_seed{seed}.pt", map_location="cpu", weights_only=False)
            model = make_model(val_set, int(config["training"]["hidden_dim"]), str(config["training"]["gate_mode"]))
            model.load_state_dict(checkpoint["model"], strict=True)
            set_seed(seed + 9_100_000)
            result["per_seed"][str(seed)][arm] = gradient_for_checkpoint(
                model.to(device), val_set, ambiguous_set, seed=seed,
                prior_weight=float(config["training"]["prior_weight"]),
                ambiguous_weight=float(config["training"]["ambiguous_weight"]), traj_weight=weight,
            )
            del model
    write_json(RESULTS_ROOT / "gradient_diagnostic.json", result)
    return result


def main() -> None:
    config = load_config()
    val = JAADSequenceDataset(ROOT / config["training"]["data_root"] / "val.npz")
    ambiguous = JAADSequenceDataset(ROOT / config["training"]["ambiguous_root"] / "train.npz")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pareto = pareto_history()
    representation = representation_diagnostic(config, val, device)
    gradients = gradient_diagnostic(config, val, ambiguous, device)
    print(json.dumps({
        "validation_only": True,
        "test_split_loaded": False,
        "pareto_tradeoff_observed": pareto["auc_ade_tradeoff_observed_any_seed"],
        "representation_samples": representation["sample_count"],
        "gradient_diagnostic_seeds": list(gradients["per_seed"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
