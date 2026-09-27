#!/usr/bin/env python3
"""Read-only gradient and representation audit for saved JAAD checkpoints.

No optimizer is constructed and no model parameter is updated. By default the
script measures gradients on ten class-balanced training batches at the saved
joint checkpoint, then compares encoder and trajectory-decoder features on a
random 1,000-sample subset of the training split.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def resolve_path(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else PROJECT_ROOT / path


def stack_batch(dataset: JAADSequenceDataset, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    names = (
        "target_obs", "target_abs_obs", "future_gt", "neighbor_obs",
        "neighbor_mask", "neighbor_visible_mask", "scene_feat", "intent_label",
    )
    return {name: getattr(dataset, name)[indices] for name in names}


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": mean(values),
        "std_sample_ddof_1": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def flatten_gradients(
    gradients: Iterable[torch.Tensor | None], parameters: list[torch.Tensor]
) -> torch.Tensor:
    flat = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            flat.append(torch.zeros_like(parameter).reshape(-1))
        else:
            flat.append(gradient.detach().reshape(-1))
    return torch.cat(flat)


def make_batches(
    dataset: JAADSequenceDataset,
    batch_size: int,
    batch_count: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Mirror the main trainer's inverse-frequency sampler and shuffled ambiguity set."""
    labels = dataset.intent_label.to(torch.int64)
    counts = torch.bincount(labels, minlength=2).float()
    if torch.any(counts == 0):
        raise ValueError("Both intent classes are required for the trainer-matched sampler")
    weights = torch.where(labels == 0, 1.0 / counts[0], 1.0 / counts[1])
    generator = torch.Generator().manual_seed(seed)
    sampled = torch.multinomial(
        weights, batch_count * batch_size, replacement=True, generator=generator
    ).reshape(batch_count, batch_size)
    return [stack_batch(dataset, row) for row in sampled]


def sample_ambiguous_batches(
    dataset: JAADSequenceDataset, batch_size: int, batch_count: int, seed: int
) -> list[dict[str, torch.Tensor]]:
    # The training loop shuffles the ambiguous set without replacement and
    # restarts its loader if it is exhausted. Reproduce that cycling behavior.
    generator = torch.Generator().manual_seed(seed)
    batches = []
    order = torch.randperm(len(dataset), generator=generator)
    offset = 0
    for _ in range(batch_count):
        if offset >= len(dataset):
            order = torch.randperm(len(dataset), generator=generator)
            offset = 0
        end = min(offset + batch_size, len(dataset))
        # Preserve the trainer's potentially short final batch; the next main
        # training step restarts a fresh shuffled ambiguous-data iterator.
        batches.append(stack_batch(dataset, order[offset:end]))
        offset = end
    return batches


def joint_forward(model: JointTransformerSceneGate, batch: dict[str, torch.Tensor], device: torch.device):
    target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
    return model(
        target,
        batch["neighbor_obs"].to(device),
        batch["neighbor_mask"].to(device),
        batch["neighbor_visible_mask"].to(device),
        batch["scene_feat"].to(device),
    )


def diagnose_gradients(
    model: JointTransformerSceneGate,
    main_batches: list[dict[str, torch.Tensor]],
    ambiguous_batches: list[dict[str, torch.Tensor]],
    device: torch.device,
    prior_weight: float,
    trajectory_weight: float,
    ambiguity_weight: float,
) -> dict:
    # cuDNN GRU backward requires train mode; dropout is active as during
    # training. No optimizer/zero_grad/step is used and parameters stay frozen.
    model.train()
    shared_prefixes = (
        "target_projection.", "position_embedding", "target_encoder.",
        "neighbor_encoder.", "scene_encoder.", "proposal_fusion.",
        "proposal_head.", "gate.", "fusion.",
    )
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(shared_prefixes)
    ]
    if not parameters:
        raise RuntimeError("No shared model parameters found")

    rows = []
    for index, (main_batch, ambiguous_batch) in enumerate(zip(main_batches, ambiguous_batches), 1):
        output = joint_forward(model, main_batch, device)
        labels = main_batch["intent_label"].to(device)
        intent_main = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], labels)
        intent_prior = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], labels)
        trajectory_raw = nn.functional.smooth_l1_loss(
            output["future_pred"], main_batch["future_gt"].to(device)
        )

        ambiguous_output = joint_forward(model, ambiguous_batch, device)
        ambiguity_raw = 0.5 * (
            ambiguous_output["prior_logit"].square().mean()
            + ambiguous_output["intent_logit"].square().mean()
        )
        intent_loss = intent_main + prior_weight * intent_prior + ambiguity_weight * ambiguity_raw
        trajectory_loss = trajectory_weight * trajectory_raw

        intent_grad = torch.autograd.grad(intent_loss, parameters, retain_graph=True, allow_unused=True)
        trajectory_grad = torch.autograd.grad(trajectory_loss, parameters, allow_unused=True)
        intent_vector = flatten_gradients(intent_grad, parameters)
        trajectory_vector = flatten_gradients(trajectory_grad, parameters)
        intent_norm = torch.linalg.vector_norm(intent_vector)
        trajectory_norm = torch.linalg.vector_norm(trajectory_vector)
        cosine = torch.dot(intent_vector, trajectory_vector) / (
            intent_norm * trajectory_norm
        ).clamp_min(torch.finfo(intent_vector.dtype).eps)

        rows.append({
            "batch": index,
            "main_sample_count": int(labels.numel()),
            "ambiguous_sample_count": int(ambiguous_batch["intent_label"].numel()),
            "intent_main_bce": float(intent_main.detach().cpu()),
            "intent_prior_bce_unweighted": float(intent_prior.detach().cpu()),
            "intent_prior_bce_weighted": float((prior_weight * intent_prior).detach().cpu()),
            "ambiguity_regularizer_unweighted": float(ambiguity_raw.detach().cpu()),
            "ambiguity_regularizer_weighted": float((ambiguity_weight * ambiguity_raw).detach().cpu()),
            "intent_objective_total": float(intent_loss.detach().cpu()),
            "trajectory_smooth_l1_unweighted": float(trajectory_raw.detach().cpu()),
            "trajectory_objective_weighted": float(trajectory_loss.detach().cpu()),
            "gradient_cosine_intent_vs_trajectory": float(cosine.detach().cpu()),
            "intent_gradient_norm": float(intent_norm.detach().cpu()),
            "trajectory_gradient_norm_weighted": float(trajectory_norm.detach().cpu()),
            "gradient_magnitude_ratio_intent_over_trajectory": float(
                (intent_norm / trajectory_norm.clamp_min(1e-20)).detach().cpu()
            ),
        })

    names = (
        "gradient_cosine_intent_vs_trajectory", "intent_gradient_norm",
        "trajectory_gradient_norm_weighted", "gradient_magnitude_ratio_intent_over_trajectory",
        "intent_objective_total", "trajectory_objective_weighted",
    )
    summaries = {name: summarize([row[name] for row in rows]) for name in names}
    return {
        "diagnostic_only": True,
        "optimizer_created": False,
        "parameters_updated": False,
        "batch_count": len(rows),
        "batch_size": rows[0]["main_sample_count"] if rows else 0,
        "shared_parameter_count": int(sum(p.numel() for p in parameters)),
        "shared_parameter_groups": list(shared_prefixes),
        "gradient_scope": "shared trunk/interaction/fusion parameters; task output heads excluded",
        "loss_definition": {
            "intent": "main BCE + prior_weight*prior BCE + ambiguity_weight*0.5*(prior_logit^2 + intent_logit^2)",
            "trajectory": "trajectory_weight*SmoothL1(normalized future coordinates)",
            "prior_weight": prior_weight,
            "trajectory_weight": trajectory_weight,
            "ambiguity_weight": ambiguity_weight,
        },
        "sampling": "10 independent 512-sample batches drawn with the trainer's inverse-class-frequency replacement sampler; ambiguous samples shuffled without replacement and cycled as in trainer",
        "summary": summaries,
        "per_batch": rows,
        "interpretation": {
            "cosine_below_zero": "local gradient conflict on shared parameters",
            "cosine_near_zero": "weak local gradient alignment",
            "cosine_above_zero": "local gradient alignment",
            "ratio_definition": "||g_intent|| / ||g_trajectory|| after the configured task weights",
        },
    }


def stats(tensor: torch.Tensor) -> dict[str, float | int]:
    flat = tensor.detach().float().reshape(tensor.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=1)
    return {
        "sample_count": int(flat.shape[0]),
        "feature_width": int(flat.shape[1]),
        "feature_mean": float(flat.mean().cpu()),
        "feature_std_population": float(flat.std(unbiased=False).cpu()),
        "mean_per_dimension_variance": float(flat.var(dim=0, unbiased=False).mean().cpu()),
        "feature_norm_mean": float(norms.mean().cpu()),
        "feature_norm_std_population": float(norms.std(unbiased=False).cpu()),
    }


@torch.no_grad()
def diagnose_features(
    trajectory_model: SceneTrajectoryTransformer,
    joint_model: JointTransformerSceneGate,
    dataset: JAADSequenceDataset,
    sample_count: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[: min(sample_count, len(dataset))]
    trajectory_target, trajectory_decoder, joint_target, joint_decoder = [], [], [], []
    trajectory_model.eval()
    joint_model.eval()

    for start in range(0, indices.numel(), batch_size):
        batch = stack_batch(dataset, indices[start : start + batch_size])
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        scene = batch["scene_feat"].to(device)
        traj_encoded = trajectory_model.temporal_encoder(
            trajectory_model.input_projection(target)
            + trajectory_model.position_embedding[:, : target.shape[1]]
        )
        traj_context = traj_encoded[:, -1]
        traj_scene = trajectory_model.scene_encoder(scene)

        captured: dict[str, torch.Tensor] = {}
        hook = joint_model.fusion.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("fused", output)
        )
        joint_output = joint_forward(joint_model, batch, device)
        hook.remove()
        joint_encoded = joint_model.target_encoder(
            joint_model.target_projection(target)
            + joint_model.position_embedding[:, : target.shape[1]]
        )
        joint_context = joint_encoded[:, -1]
        # Keep an assertion tied to the forward pass: trajectory receives the
        # shared fused context concatenated with its own target-encoder output.
        if "fused" not in captured or joint_output["future_pred"].shape[0] != target.shape[0]:
            raise RuntimeError("Failed to capture the joint trajectory feature flow")
        trajectory_target.append(traj_context.cpu())
        trajectory_decoder.append(torch.cat([traj_context, traj_scene], dim=-1).cpu())
        joint_target.append(joint_context.cpu())
        joint_decoder.append(torch.cat([captured["fused"], joint_context], dim=-1).cpu())

    return {
        "diagnostic_only": True,
        "split": "train",
        "sampling": "uniform random sample without replacement",
        "seed": seed,
        "requested_samples": sample_count,
        "actual_sample_count": int(indices.numel()),
        "statistics": {
            "trajectory_only_target_encoder_output": stats(torch.cat(trajectory_target)),
            "joint_target_encoder_output": stats(torch.cat(joint_target)),
            "trajectory_only_decoder_input_target_plus_scene": stats(torch.cat(trajectory_decoder)),
            "joint_trajectory_decoder_input_fused_plus_target": stats(torch.cat(joint_decoder)),
        },
        "caveat": "Different learned checkpoint weights and different decoder inputs are compared; larger variance/norm is not by itself evidence of better or collapsed representations.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/jaad_sequences_scene_15x15"))
    parser.add_argument("--ambiguous-root", type=Path, default=Path("data/processed/jaad_ambiguous_scene_15x15"))
    parser.add_argument("--joint-checkpoint", type=Path, default=Path("checkpoints/joint_transformer_gate_15x15_seed123.pt"))
    parser.add_argument("--trajectory-checkpoint", type=Path, default=Path("checkpoints/trajectory_transformer_scene_15x15_seed123.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/joint_audit"))
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--feature-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=9124)
    args = parser.parse_args()
    if args.batches <= 0 or args.batch_size <= 0 or args.feature_samples <= 0:
        parser.error("batches, batch-size and feature-samples must all be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = resolve_path(args.data_root)
    ambiguous_root = resolve_path(args.ambiguous_root)
    out_dir = resolve_path(args.output_dir)
    joint_path = resolve_path(args.joint_checkpoint)
    trajectory_path = resolve_path(args.trajectory_checkpoint)

    train_set = JAADSequenceDataset(data_root / "train.npz")
    ambiguous_set = JAADSequenceDataset(ambiguous_root / "train.npz")
    joint_checkpoint = torch.load(joint_path, map_location="cpu", weights_only=False)
    trajectory_checkpoint = torch.load(trajectory_path, map_location="cpu", weights_only=False)
    joint_args = joint_checkpoint.get("args", {})
    trajectory_args = trajectory_checkpoint.get("args", {})

    joint_model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(train_set.scene_feat.shape[-1]),
        hidden_dim=int(joint_args.get("hidden_dim", 128)),
        pred_len=int(train_set.future_gt.shape[1]),
        gate_mode=str(joint_args.get("gate_mode", "uncertainty")),
        max_obs_len=int(train_set.target_obs.shape[1]),
    ).to(device)
    joint_model.load_state_dict(joint_checkpoint["model"], strict=True)
    trajectory_model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=int(train_set.scene_feat.shape[-1]),
        d_model=int(trajectory_args.get("d_model", 128)),
        num_layers=int(trajectory_args.get("num_layers", 3)),
        pred_len=int(train_set.future_gt.shape[1]),
        max_obs_len=int(train_set.target_obs.shape[1]),
    ).to(device)
    trajectory_model.load_state_dict(trajectory_checkpoint["model"], strict=True)

    main_batches = make_batches(train_set, args.batch_size, args.batches, args.seed + 1)
    ambiguous_batches = sample_ambiguous_batches(
        ambiguous_set, args.batch_size, args.batches, args.seed + 2
    )
    gradient = diagnose_gradients(
        joint_model,
        main_batches,
        ambiguous_batches,
        device,
        prior_weight=float(joint_args.get("prior_weight", 0.5)),
        trajectory_weight=float(joint_args.get("traj_weight", 1.0)),
        ambiguity_weight=float(joint_args.get("ambiguous_weight", 0.2)),
    )
    features = diagnose_features(
        trajectory_model, joint_model, train_set, args.feature_samples,
        args.batch_size, device, args.seed + 3,
    )
    gradient.update({
        "device": str(device),
        "joint_checkpoint": str(joint_path.relative_to(PROJECT_ROOT)),
        "trajectory_checkpoint": str(trajectory_path.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
    })
    features.update({
        "device": str(device),
        "joint_checkpoint": str(joint_path.relative_to(PROJECT_ROOT)),
        "trajectory_checkpoint": str(trajectory_path.relative_to(PROJECT_ROOT)),
    })
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gradient_conflict.json").write_text(
        json.dumps(gradient, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "feature_statistics.json").write_text(
        json.dumps(features, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "gradient_conflict": gradient["summary"],
        "feature_statistics": features["statistics"],
        "output_dir": str(out_dir),
        "device": str(device),
        "optimizer_created": False,
        "parameters_updated": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
