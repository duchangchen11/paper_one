#!/usr/bin/env python3
"""Read-only diagnostics for saved trajectory-only and joint checkpoints.

This script never creates an optimizer or updates model parameters. It computes
loss/gradient diagnostics at the supplied saved weights and feature statistics
on a deterministic subset. Per-epoch gradient results require per-epoch
checkpoints; the training script currently saves only its selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def balanced_indices(labels: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    """Return a deterministic approximately class-balanced sample."""
    generator = torch.Generator().manual_seed(seed)
    labels = labels.to(torch.int64).cpu()
    classes = [torch.where(labels == label)[0] for label in (0, 1)]
    if any(indices.numel() == 0 for indices in classes):
        return torch.linspace(0, labels.numel() - 1, min(count, labels.numel())).long()
    per_class = max(1, count // 2)
    selected = []
    for indices in classes:
        order = torch.randperm(indices.numel(), generator=generator)
        chosen = indices[order[: min(per_class, indices.numel())]]
        if chosen.numel() < per_class:
            extra = indices[torch.randint(indices.numel(), (per_class - chosen.numel(),), generator=generator)]
            chosen = torch.cat([chosen, extra])
        selected.append(chosen)
    result = torch.cat(selected)
    return result[torch.randperm(result.numel(), generator=generator)][:count]


def evenly_spaced_indices(size: int, count: int) -> torch.Tensor:
    count = min(size, count)
    return torch.linspace(0, size - 1, count).round().long()


def stack_batch(dataset: JAADSequenceDataset, indices: torch.Tensor) -> dict[str, torch.Tensor]:
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


def make_joint_model(
    checkpoint: dict,
    dataset: JAADSequenceDataset,
    device: torch.device,
) -> JointTransformerSceneGate:
    saved_args = checkpoint.get("args", {})
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(dataset.scene_feat.shape[-1]),
        hidden_dim=int(saved_args.get("hidden_dim", 128)),
        pred_len=int(dataset.future_gt.shape[1]),
        gate_mode=str(saved_args.get("gate_mode", "uncertainty")),
        max_obs_len=int(dataset.target_obs.shape[1]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def make_trajectory_model(
    checkpoint: dict,
    dataset: JAADSequenceDataset,
    device: torch.device,
) -> SceneTrajectoryTransformer:
    saved_args = checkpoint.get("args", {})
    model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=int(dataset.scene_feat.shape[-1]),
        d_model=int(saved_args.get("d_model", 128)),
        num_layers=int(saved_args.get("num_layers", 3)),
        pred_len=int(dataset.future_gt.shape[1]),
        max_obs_len=int(dataset.target_obs.shape[1]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model


def feature_stats(tensor: torch.Tensor) -> dict[str, float]:
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    return {
        "sample_count": int(flat.shape[0]),
        "feature_width": int(flat.shape[1]),
        "l2_norm_mean": float(norms.mean().cpu()),
        "l2_norm_std": float(norms.std(unbiased=False).cpu()),
        "feature_mean": float(flat.mean().cpu()),
        "feature_variance_mean_over_dimensions": float(flat.var(dim=0, unbiased=False).mean().cpu()),
    }


@torch.no_grad()
def representation_analysis(
    trajectory_model: SceneTrajectoryTransformer,
    joint_model: JointTransformerSceneGate,
    dataset: JAADSequenceDataset,
    device: torch.device,
    sample_count: int,
    batch_size: int,
) -> dict[str, dict[str, float]]:
    trajectory_model.eval()
    joint_model.eval()
    indices = evenly_spaced_indices(len(dataset), sample_count)
    trajectory_target_features = []
    trajectory_decoder_features = []
    joint_target_features = []
    joint_decoder_features = []

    for start in range(0, indices.numel(), batch_size):
        batch = stack_batch(dataset, indices[start : start + batch_size])
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        scene = batch["scene_feat"].to(device)

        traj_encoded = trajectory_model.temporal_encoder(
            trajectory_model.input_projection(target)
            + trajectory_model.position_embedding[:, : target.shape[1]]
        )
        traj_target = traj_encoded[:, -1]
        traj_scene = trajectory_model.scene_encoder(scene)
        traj_decoder = torch.cat([traj_target, traj_scene], dim=-1)

        joint_encoded = joint_model.target_encoder(
            joint_model.target_projection(target)
            + joint_model.position_embedding[:, : target.shape[1]]
        )
        joint_target = joint_encoded[:, -1]
        captured: dict[str, torch.Tensor] = {}
        def capture_fused(_module, _inputs, output):
            captured["fused"] = output

        hook = joint_model.fusion.register_forward_hook(capture_fused)
        joint_model(
            target,
            batch["neighbor_obs"].to(device),
            batch["neighbor_mask"].to(device),
            batch["neighbor_visible_mask"].to(device),
            scene,
        )
        hook.remove()
        joint_decoder = torch.cat([captured["fused"], joint_target], dim=-1)

        trajectory_target_features.append(traj_target.cpu())
        trajectory_decoder_features.append(traj_decoder.cpu())
        joint_target_features.append(joint_target.cpu())
        joint_decoder_features.append(joint_decoder.cpu())

    return {
        "trajectory_only_target_encoder": feature_stats(torch.cat(trajectory_target_features)),
        "joint_target_encoder": feature_stats(torch.cat(joint_target_features)),
        "trajectory_only_decoder_input": feature_stats(torch.cat(trajectory_decoder_features)),
        "joint_trajectory_decoder_input": feature_stats(torch.cat(joint_decoder_features)),
    }


def flatten_gradients(gradients: Iterable[torch.Tensor | None], parameters: list[torch.Tensor]) -> torch.Tensor:
    flattened = []
    for gradient, parameter in zip(gradients, parameters):
        flattened.append(
            torch.zeros_like(parameter, memory_format=torch.contiguous_format).reshape(-1)
            if gradient is None
            else gradient.detach().reshape(-1)
        )
    return torch.cat(flattened)


def gradient_diagnostics(
    joint_model: JointTransformerSceneGate,
    main_batch: dict[str, torch.Tensor],
    ambiguous_batch: dict[str, torch.Tensor],
    device: torch.device,
    prior_weight: float,
    traj_weight: float,
    ambiguous_weight: float,
) -> dict[str, float | int]:
    # GRU/CuDNN requires training mode for backward; no optimizer is used and
    # no parameter or optimizer state is updated by this diagnostic.
    joint_model.train()
    main_target = torch.cat([main_batch["target_obs"], main_batch["target_abs_obs"]], dim=-1).to(device)
    main_output = joint_model(
        main_target,
        main_batch["neighbor_obs"].to(device),
        main_batch["neighbor_mask"].to(device),
        main_batch["neighbor_visible_mask"].to(device),
        main_batch["scene_feat"].to(device),
    )
    labels = main_batch["intent_label"].to(device)
    intent_main = nn.functional.binary_cross_entropy_with_logits(main_output["intent_logit"], labels)
    intent_prior = nn.functional.binary_cross_entropy_with_logits(main_output["prior_logit"], labels)
    trajectory = nn.functional.smooth_l1_loss(main_output["future_pred"], main_batch["future_gt"].to(device))

    ambiguous_target = torch.cat(
        [ambiguous_batch["target_obs"], ambiguous_batch["target_abs_obs"]], dim=-1
    ).to(device)
    ambiguous_output = joint_model(
        ambiguous_target,
        ambiguous_batch["neighbor_obs"].to(device),
        ambiguous_batch["neighbor_mask"].to(device),
        ambiguous_batch["neighbor_visible_mask"].to(device),
        ambiguous_batch["scene_feat"].to(device),
    )
    ambiguity_regularizer = 0.5 * (
        ambiguous_output["prior_logit"].square().mean()
        + ambiguous_output["intent_logit"].square().mean()
    )
    intent_total = intent_main + prior_weight * intent_prior + ambiguous_weight * ambiguity_regularizer
    trajectory_weighted = traj_weight * trajectory

    shared_prefixes = (
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
    shared_named = [
        (name, parameter)
        for name, parameter in joint_model.named_parameters()
        if parameter.requires_grad and name.startswith(shared_prefixes)
    ]
    parameters = [parameter for _, parameter in shared_named]
    intent_grad = torch.autograd.grad(intent_total, parameters, retain_graph=True, allow_unused=True)
    trajectory_grad = torch.autograd.grad(trajectory_weighted, parameters, allow_unused=True)
    intent_vector = flatten_gradients(intent_grad, parameters)
    trajectory_vector = flatten_gradients(trajectory_grad, parameters)
    intent_norm = torch.linalg.vector_norm(intent_vector)
    trajectory_norm = torch.linalg.vector_norm(trajectory_vector)
    cosine = torch.dot(intent_vector, trajectory_vector) / (
        intent_norm * trajectory_norm
    ).clamp_min(torch.finfo(intent_vector.dtype).eps)

    return {
        "shared_parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "sample_count_main": int(labels.numel()),
        "sample_count_ambiguous": int(ambiguous_batch["intent_label"].numel()),
        "intent_main_bce": float(intent_main.detach().cpu()),
        "intent_prior_bce_unweighted": float(intent_prior.detach().cpu()),
        "prior_weight": float(prior_weight),
        "intent_prior_bce_weighted": float((prior_weight * intent_prior).detach().cpu()),
        "ambiguity_regularizer_unweighted": float(ambiguity_regularizer.detach().cpu()),
        "ambiguity_weight": float(ambiguous_weight),
        "ambiguity_regularizer_weighted": float((ambiguous_weight * ambiguity_regularizer).detach().cpu()),
        "intent_objective_total": float(intent_total.detach().cpu()),
        "trajectory_smooth_l1_unweighted": float(trajectory.detach().cpu()),
        "trajectory_weight": float(traj_weight),
        "trajectory_smooth_l1_weighted": float(trajectory_weighted.detach().cpu()),
        "intent_to_weighted_trajectory_loss_ratio": float(
            (intent_total / trajectory_weighted.clamp_min(1e-20)).detach().cpu()
        ),
        "intent_shared_gradient_norm": float(intent_norm.detach().cpu()),
        "weighted_trajectory_shared_gradient_norm": float(trajectory_norm.detach().cpu()),
        "weighted_trajectory_to_intent_gradient_norm_ratio": float(
            (trajectory_norm / intent_norm.clamp_min(1e-20)).detach().cpu()
        ),
        "gradient_cosine_intent_vs_trajectory": float(cosine.detach().cpu()),
    }


def parse_epoch_checkpoints(items: list[str]) -> dict[int, Path]:
    result = {}
    for item in items:
        epoch_text, separator, path_text = item.partition("=")
        if not separator:
            raise ValueError(f"Expected EPOCH=PATH, got {item!r}")
        result[int(epoch_text)] = resolve_project_path(path_text)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/jaad_sequences_scene_15x15"))
    parser.add_argument("--ambiguous-root", type=Path, default=Path("data/processed/jaad_ambiguous_scene_15x15"))
    parser.add_argument("--joint-checkpoint", type=Path, default=Path("checkpoints/joint_transformer_gate_15x15_seed123.pt"))
    parser.add_argument("--trajectory-checkpoint", type=Path, default=Path("checkpoints/trajectory_transformer_scene_15x15_seed123.pt"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--gradient-samples", type=int, default=512)
    parser.add_argument("--feature-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--epoch-checkpoint",
        action="append",
        default=[],
        metavar="EPOCH=PATH",
        help="Optional saved joint checkpoint for a specific epoch; repeat as needed.",
    )
    args = parser.parse_args()

    data_root = resolve_project_path(args.data_root)
    ambiguous_root = resolve_project_path(args.ambiguous_root)
    joint_path = resolve_project_path(args.joint_checkpoint)
    trajectory_path = resolve_project_path(args.trajectory_checkpoint)
    joint_checkpoint = torch.load(joint_path, map_location="cpu", weights_only=False)
    trajectory_checkpoint = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    joint_args = joint_checkpoint.get("args", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    train_set = JAADSequenceDataset(data_root / "train.npz")
    val_set = JAADSequenceDataset(data_root / "val.npz")
    split_set = JAADSequenceDataset(data_root / f"{args.split}.npz")
    ambiguous_set = JAADSequenceDataset(ambiguous_root / "train.npz")
    joint_model = make_joint_model(joint_checkpoint, train_set, device)
    trajectory_model = make_trajectory_model(trajectory_checkpoint, train_set, device)

    gradient_indices = balanced_indices(train_set.intent_label, args.gradient_samples, seed=123)
    validation_gradient_indices = balanced_indices(val_set.intent_label, args.gradient_samples, seed=124)
    ambiguous_indices = evenly_spaced_indices(len(ambiguous_set), min(args.gradient_samples, len(ambiguous_set)))
    train_batch = stack_batch(train_set, gradient_indices)
    validation_batch = stack_batch(val_set, validation_gradient_indices)
    ambiguous_batch = stack_batch(ambiguous_set, ambiguous_indices)
    train_gradient_result = gradient_diagnostics(
        joint_model,
        train_batch,
        ambiguous_batch,
        device,
        prior_weight=float(joint_args.get("prior_weight", 0.5)),
        traj_weight=float(joint_args.get("traj_weight", 1.0)),
        ambiguous_weight=float(joint_args.get("ambiguous_weight", 0.2)),
    )
    validation_gradient_result = gradient_diagnostics(
        joint_model,
        validation_batch,
        ambiguous_batch,
        device,
        prior_weight=float(joint_args.get("prior_weight", 0.5)),
        traj_weight=float(joint_args.get("traj_weight", 1.0)),
        ambiguous_weight=float(joint_args.get("ambiguous_weight", 0.2)),
    )

    metrics_path = resolve_project_path(joint_args.get("output_root", "results/joint_transformer_gate_15x15_seed123")) / "metrics.json"
    best_epoch = None
    if metrics_path.exists():
        best_epoch = json.loads(metrics_path.read_text(encoding="utf-8")).get("best_epoch")
    requested_epoch_checkpoints = parse_epoch_checkpoints(args.epoch_checkpoint)
    requested_epochs = (1, 5, 10)
    epoch_gradient_status = []
    for epoch in requested_epochs:
        checkpoint_path = requested_epoch_checkpoints.get(epoch)
        if checkpoint_path is None or not checkpoint_path.exists():
            epoch_gradient_status.append(
                {
                    "epoch": epoch,
                    "status": "missing_per_epoch_checkpoint",
                    "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
                }
            )
            continue
        epoch_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        epoch_args = epoch_checkpoint.get("args", {})
        epoch_model = make_joint_model(epoch_checkpoint, train_set, device)
        epoch_gradient_status.append(
            {
                "epoch": epoch,
                "status": "calculated_without_parameter_updates",
                "checkpoint": str(checkpoint_path),
                "train_balanced_subset": gradient_diagnostics(
                    epoch_model,
                    train_batch,
                    ambiguous_batch,
                    device,
                    prior_weight=float(epoch_args.get("prior_weight", 0.5)),
                    traj_weight=float(epoch_args.get("traj_weight", 1.0)),
                    ambiguous_weight=float(epoch_args.get("ambiguous_weight", 0.2)),
                ),
                "validation_balanced_subset": gradient_diagnostics(
                    epoch_model,
                    validation_batch,
                    ambiguous_batch,
                    device,
                    prior_weight=float(epoch_args.get("prior_weight", 0.5)),
                    traj_weight=float(epoch_args.get("traj_weight", 1.0)),
                    ambiguous_weight=float(epoch_args.get("ambiguous_weight", 0.2)),
                ),
            }
        )
        del epoch_model, epoch_checkpoint

    result = {
        "diagnostic_only": True,
        "optimizer_created": False,
        "parameters_updated": False,
        "device": str(device),
        "joint_checkpoint": str(joint_path),
        "joint_checkpoint_selected_epoch_from_metrics": best_epoch,
        "trajectory_checkpoint": str(trajectory_path),
        "gradient_sample": "deterministic class-balanced subsets of train and validation splits; train-mode forward for GRU backward; no optimizer/update",
        "feature_sample": f"{args.split} split, deterministic evenly-spaced indices",
        "gradient_at_saved_best_checkpoint": {
            "train_balanced_subset": train_gradient_result,
            "validation_balanced_subset": validation_gradient_result,
        },
        "requested_epoch_gradient_checkpoints": epoch_gradient_status,
        "representation_statistics": representation_analysis(
            trajectory_model,
            joint_model,
            split_set,
            device,
            sample_count=args.feature_samples,
            batch_size=args.batch_size,
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
