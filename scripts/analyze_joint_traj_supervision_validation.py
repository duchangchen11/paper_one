#!/usr/bin/env python3
"""Pre-freeze J0/J100 validation diagnostics: selection, drift, and gradients."""

from __future__ import annotations

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

from scripts.audit_joint_model_gradients import balanced_indices, stack_batch
from scripts.joint_traj_supervision_utils import (
    RESULTS_ROOT, SEEDS, SHARED_PREFIXES, feature_similarity, feature_summary,
    load_config, linear_norm, sha256_state, write_json,
)
from scripts.train_joint_transformer_gate import set_seed
from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate


def create_model(dataset: JAADSequenceDataset, seed: int, gate_mode: str) -> JointTransformerSceneGate:
    set_seed(seed)
    return JointTransformerSceneGate(
        input_dim=8, scene_dim=int(dataset.scene_feat.shape[-1]), hidden_dim=128,
        pred_len=int(dataset.future_gt.shape[1]), gate_mode=gate_mode,
        max_obs_len=int(dataset.target_obs.shape[1]),
    )


@torch.no_grad()
def shared_contexts(
    model: JointTransformerSceneGate,
    dataset: JAADSequenceDataset,
    indices: torch.Tensor,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    model.eval()
    captured_fused: list[torch.Tensor] = []
    hook = model.fusion.register_forward_hook(lambda _module, _inputs, output: captured_fused.append(output.detach().cpu()))
    target_rows: list[np.ndarray] = []
    fused_rows: list[np.ndarray] = []
    try:
        for start in range(0, len(indices), batch_size):
            ix = indices[start : start + batch_size]
            target = torch.cat([dataset.target_obs[ix], dataset.target_abs_obs[ix]], dim=-1)
            before = len(captured_fused)
            model(
                target,
                dataset.neighbor_obs[ix],
                dataset.neighbor_mask[ix],
                dataset.neighbor_visible_mask[ix],
                dataset.scene_feat[ix],
            )
            if len(captured_fused) != before + 1:
                raise RuntimeError("Expected one fused representation per validation batch")
            target_context = model.target_encoder(
                model.target_projection(target) + model.position_embedding[:, : target.shape[1]]
            )[:, -1]
            target_rows.append(target_context.cpu().numpy())
            fused_rows.append(captured_fused[-1].numpy())
    finally:
        hook.remove()
    return {
        "target_encoder_last": np.concatenate(target_rows, axis=0),
        "fused_decoder_context": np.concatenate(fused_rows, axis=0),
    }


def grad_diagnostic(
    model: JointTransformerSceneGate,
    main_batch: dict[str, torch.Tensor],
    amb_batch: dict[str, torch.Tensor],
    *, prior_weight: float, ambiguous_weight: float, traj_weight: float,
) -> dict[str, Any]:
    model.train()
    target = torch.cat([main_batch["target_obs"], main_batch["target_abs_obs"]], dim=-1)
    output = model(target, main_batch["neighbor_obs"], main_batch["neighbor_mask"], main_batch["neighbor_visible_mask"], main_batch["scene_feat"])
    labels = main_batch["intent_label"]
    bce = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], labels)
    proposal = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], labels)
    traj = nn.functional.smooth_l1_loss(output["future_pred"], main_batch["future_gt"])
    amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1)
    amb_out = model(amb_target, amb_batch["neighbor_obs"], amb_batch["neighbor_mask"], amb_batch["neighbor_visible_mask"], amb_batch["scene_feat"])
    ambiguity = 0.5 * (amb_out["prior_logit"].square().mean() + amb_out["intent_logit"].square().mean())
    intent_obj = bce + prior_weight * proposal + ambiguous_weight * ambiguity
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and n.startswith(SHARED_PREFIXES)]
    params = [p for _, p in named]
    gi = torch.autograd.grad(intent_obj, params, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(traj, params, allow_unused=True)
    intent_norm, intent_vec = linear_norm(gi, params)
    trajectory_norm, trajectory_vec = linear_norm(gt, params)
    cosine = float(torch.dot(intent_vec, trajectory_vec) / max(intent_norm * trajectory_norm, 1e-12))
    weighted_norm = float(traj_weight * trajectory_norm)
    return {
        "traj_weight": traj_weight,
        "intent_objective": float(intent_obj.detach()),
        "raw_trajectory_loss": float(traj.detach()),
        "intent_gradient_norm_shared": intent_norm,
        "trajectory_gradient_norm_shared_unweighted": trajectory_norm,
        "trajectory_gradient_norm_shared_weighted": weighted_norm,
        "weighted_intent_to_trajectory_gradient_ratio": (intent_norm / weighted_norm) if weighted_norm > 0 else None,
        "raw_intent_to_trajectory_gradient_ratio": intent_norm / max(trajectory_norm, 1e-12),
        "gradient_cosine_intent_vs_trajectory": cosine,
        "shared_parameter_count": sum(p.numel() for p in params),
    }


def main() -> None:
    config = load_config()
    val_set = JAADSequenceDataset(ROOT / config["training"]["data_root"] / "val.npz")
    ambiguous_set = JAADSequenceDataset(ROOT / config["training"]["ambiguous_root"] / "train.npz")
    j100_metrics_by_seed: dict[str, dict[str, Any]] = {}
    j0_metrics_by_seed: dict[str, dict[str, Any]] = {}
    j100_checkpoints: dict[str, dict[str, Any]] = {}
    j0_checkpoints: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        j100_metrics_by_seed[str(seed)] = json.loads((ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json").read_text(encoding="utf-8"))
        j0_metrics_by_seed[str(seed)] = json.loads((RESULTS_ROOT / "j0" / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))
        j100_checkpoints[str(seed)] = torch.load(ROOT / f"checkpoints/joint_loss_balance/lambda100_seed{seed}.pt", map_location="cpu", weights_only=False)
        j0_checkpoints[str(seed)] = torch.load(ROOT / "checkpoints/joint_traj_supervision_attribution" / f"j0_seed{seed}.pt", map_location="cpu", weights_only=False)

    selection: dict[str, Any] = {}
    representation: dict[str, Any] = {
        "split": "validation",
        "sample_count": min(1000, len(val_set)),
        "indices": "evenly spaced indices fixed before test; identical for J0/J100/initialization",
        "feature_definitions": {
            "target_encoder_last": "last temporal target-encoder token; direct trajectory decoder input",
            "fused_decoder_context": "output of shared fusion block; joint trajectory decoder input",
        },
        "selection_use": False,
        "per_seed": {},
    }
    gradients: dict[str, Any] = {
        "split": "fixed validation batch for intent/trajectory; fixed ambiguous-train batch for ambiguity regularizer",
        "selection_use": False,
        "per_seed": {},
    }
    val_indices = torch.linspace(0, len(val_set) - 1, min(1000, len(val_set))).round().long()
    cfg = config["training"]
    for seed in SEEDS:
        j0_metrics, j100_metrics = j0_metrics_by_seed[str(seed)], j100_metrics_by_seed[str(seed)]
        j0_history = j0_metrics["history"]
        scores = [row["val"]["intent_auc"] + 0.1 * row["val"]["intent_f1"] - 0.01 * row["val"]["trajectory_ade_pixel"] for row in j0_history]
        aucs = [row["val"]["intent_auc"] for row in j0_history]
        composite_epoch = int(np.argmax(scores) + 1)
        auc_epoch = int(np.argmax(aucs) + 1)
        selected_val = j0_history[int(j0_metrics["best_epoch"]) - 1]["val"]
        selection[str(seed)] = {
            "composite_best_epoch": int(j0_metrics["best_epoch"]),
            "recomputed_composite_best_epoch": composite_epoch,
            "validation_auc_best_epoch": auc_epoch,
            "same_epoch": composite_epoch == auc_epoch,
            "selected_checkpoint_val_auc": float(selected_val["intent_auc"]),
            "selected_checkpoint_val_f1": float(selected_val["intent_f1"]),
            "selected_checkpoint_val_ade_pixel": float(selected_val["trajectory_ade_pixel"]),
            "auc_best_epoch_metrics": j0_history[auc_epoch - 1]["val"],
            "selection_rule": "validation intent_auc + 0.1*intent_f1 - 0.01*trajectory_ade_pixel",
        }
        init_model = create_model(val_set, seed, cfg["gate_mode"])
        init_hash = sha256_state(init_model.state_dict())
        if init_hash != j0_metrics["initial_model_state_sha256"]:
            raise RuntimeError(f"J0 seed{seed} did not begin from the deterministic initialization")
        # Historical J100 metrics predate initialization-hash logging. Its source
        # commit uses the same seed/model-construction sequence; record this as
        # reconstructed, not as a directly artifact-verified hash.
        source_audit = {
            "J0_initialization_hash_directly_saved": True,
            "J100_initialization_hash_directly_saved": False,
            "J100_initialization_reconstruction": "same seed and unchanged model construction point verified against J100 source commit; original initial hash was not persisted",
        }
        j0_model = create_model(val_set, seed, cfg["gate_mode"])
        j0_model.load_state_dict(j0_checkpoints[str(seed)]["model"], strict=True)
        j100_model = create_model(val_set, seed, cfg["gate_mode"])
        j100_model.load_state_dict(j100_checkpoints[str(seed)]["model"], strict=True)
        initial_features = shared_contexts(init_model, val_set, val_indices)
        j0_features = shared_contexts(j0_model, val_set, val_indices)
        j100_features = shared_contexts(j100_model, val_set, val_indices)
        representation["per_seed"][str(seed)] = {"initialization_state_sha256": init_hash, "initialization_provenance": source_audit, "features": {}}
        for feature_name in initial_features:
            representation["per_seed"][str(seed)]["features"][feature_name] = {
                "J0": {
                    "statistics": feature_summary(j0_features[feature_name]),
                    "similarity_to_initialization": feature_similarity(initial_features[feature_name], j0_features[feature_name]),
                },
                "J100": {
                    "statistics": feature_summary(j100_features[feature_name]),
                    "similarity_to_initialization": feature_similarity(initial_features[feature_name], j100_features[feature_name]),
                },
                "J0_vs_J100": feature_similarity(j100_features[feature_name], j0_features[feature_name]),
            }

        val_idx = balanced_indices(val_set.intent_label, int(cfg["batch_size"]), seed + 3_000_003)
        amb_gen = torch.Generator(device="cpu").manual_seed(seed + 4_000_003)
        amb_idx = torch.randperm(len(ambiguous_set), generator=amb_gen)[: min(int(cfg["batch_size"]), len(ambiguous_set))]
        val_batch = stack_batch(val_set, val_idx)
        amb_batch = stack_batch(ambiguous_set, amb_idx)
        set_seed(seed + 5_000_003)
        j0_gradient = grad_diagnostic(j0_model, val_batch, amb_batch, prior_weight=float(cfg["prior_weight"]), ambiguous_weight=float(cfg["ambiguous_weight"]), traj_weight=0.0)
        set_seed(seed + 5_000_003)
        j100_gradient = grad_diagnostic(j100_model, val_batch, amb_batch, prior_weight=float(cfg["prior_weight"]), ambiguous_weight=float(cfg["ambiguous_weight"]), traj_weight=100.0)
        gradients["per_seed"][str(seed)] = {
            "matched_dropout_seed": seed + 5_000_003,
            "J0": j0_gradient,
            "J100": j100_gradient,
        }
        gradients["per_seed"][str(seed)]["gradient_cosine_change_J100_minus_J0"] = (
            gradients["per_seed"][str(seed)]["J100"]["gradient_cosine_intent_vs_trajectory"]
            - gradients["per_seed"][str(seed)]["J0"]["gradient_cosine_intent_vs_trajectory"]
        )

    write_json(RESULTS_ROOT / "checkpoint_selection_sensitivity.json", {
        "per_seed": selection,
        "composite_vs_auc_epoch_agreement_count": sum(row["same_epoch"] for row in selection.values()),
        "selection_use": "The composite rule alone selects the official J0 checkpoint; AUC-best epochs are read-only sensitivity analysis.",
        "test_split_loaded": False,
    })
    write_json(RESULTS_ROOT / "representation_drift.json", representation)
    write_json(RESULTS_ROOT / "gradient_diagnostics.json", gradients)
    print(json.dumps({"selection": selection, "representation_written": True, "gradient_diagnostics_written": True, "test_split_loaded": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
