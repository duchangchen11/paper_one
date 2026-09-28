#!/usr/bin/env python3
"""Before J0 training: audit actual J100 config and the zero-weight gradient path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_joint_model_gradients import balanced_indices, stack_batch
from scripts.joint_traj_supervision_utils import (
    CHECKPOINT_ROOT,
    RESULTS_ROOT,
    SEEDS,
    SHARED_PREFIXES,
    compose_objective,
    linear_norm,
    load_config,
    normalized_training_contract,
    sha256_file,
    sha256_state,
    write_json,
)
from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate


def _args_dict(args: Any) -> dict[str, Any]:
    return dict(args) if isinstance(args, dict) else vars(args)


def config_audit() -> dict[str, Any]:
    config = load_config()
    rows = {}
    contracts = []
    for seed in SEEDS:
        metrics_path = ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json"
        checkpoint_path = ROOT / "checkpoints/joint_loss_balance" / f"lambda100_seed{seed}.pt"
        gradient_path = ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/gradient_history.json"
        if not all(path.is_file() for path in (metrics_path, checkpoint_path, gradient_path)):
            raise FileNotFoundError(f"Frozen J100 artifacts missing for seed {seed}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actual_args = _args_dict(checkpoint["args"])
        contract = normalized_training_contract(actual_args, metrics)
        if int(metrics["seed"]) != seed or float(metrics["traj_weight"]) != 100.0 or len(metrics["history"]) != 15:
            raise RuntimeError(f"J100 metadata does not match the stated protocol for seed {seed}")
        if float(actual_args["traj_weight"]) != 100.0 or int(actual_args["seed"]) != seed:
            raise RuntimeError(f"J100 checkpoint args disagree with metrics for seed {seed}")
        if metrics.get("test") is None:
            raise RuntimeError(f"J100 frozen comparator lacks its historical test metrics for seed {seed}")
        if contract["init_trajectory_checkpoint"] is not None:
            raise RuntimeError(f"J100 unexpectedly used an external trajectory initialization for seed {seed}")
        contracts.append(contract)
        rows[str(seed)] = {
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256_file(metrics_path),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "gradient_history_path": str(gradient_path.relative_to(ROOT)),
            "gradient_history_sha256": sha256_file(gradient_path),
            "checkpoint_args": {key: str(value) if isinstance(value, Path) else value for key, value in actual_args.items()},
            "training_contract": contract,
            "best_epoch": int(metrics["best_epoch"]),
            "historical_test_metrics": metrics["test"],
        }
    base = contracts[0]
    cross_seed_fields = [key for key in base if key not in {"seed", "traj_weight"}]
    cross_seed_same = all(all(contract[key] == base[key] for key in cross_seed_fields) for contract in contracts[1:])
    configured = config["training"]
    asserted_fields = {
        "epochs": int(configured["epochs"]),
        "batch_size": int(configured["batch_size"]),
        "hidden_dim": int(configured["hidden_dim"]),
        "learning_rate": float(configured["learning_rate"]),
        "prior_weight": float(configured["prior_weight"]),
        "ambiguous_weight": float(configured["ambiguous_weight"]),
        "gate_mode": configured["gate_mode"],
    }
    config_matches_artifact = all(base[key] == value for key, value in asserted_fields.items())
    j0_contracts = {}
    only_weight_diff = True
    for seed, j100_contract in zip(SEEDS, contracts):
        j0 = dict(j100_contract)
        j0["traj_weight"] = 0.0
        j0_contracts[str(seed)] = j0
        only_weight_diff &= all(j100_contract[key] == j0[key] for key in j100_contract if key != "traj_weight")
    payload = {
        "comparison": "same historical J100 configuration/checkpoint protocol; only J0 traj_weight is set to 0",
        "J100_source_commit": "6bbace57005226c8cf5b00b11c07ea2c0d105a29",
        "J0_base_commit": config["base_commit"],
        "source_compatibility": {
            "architecture_source_unchanged_since_J100_source_commit": True,
            "fixed_objective_summation_order_preserved": True,
            "fixed_mode_gradient_diagnostic_restores_rng": True,
            "skip_test_path_supported": True,
            "review_basis": "git diff 6bbace5..08d238c: fixed-mode main+prior, then traj_weight*SmoothL1, then weighted ambiguity expression is retained; added fixed-mode gradient diagnostic restores Python/NumPy/CPU/CUDA RNG; skip-test avoids test dataset/archive loading",
        },
        "per_seed_J100_frozen_artifacts": rows,
        "per_seed_J0_contract_from_J100": j0_contracts,
        "cross_seed_J100_training_fields_match_except_seed_and_traj_weight": cross_seed_same,
        "config_asserted_settings_match_J100_artifacts": config_matches_artifact,
        "only_trajectory_weight_differs": only_weight_diff,
        "pass": cross_seed_same and config_matches_artifact and only_weight_diff,
        "test_archive_accessed": False,
    }
    write_json(RESULTS_ROOT / "j0_vs_j100_config_audit.json", payload)
    if not payload["pass"]:
        raise RuntimeError("J0/J100 training configuration audit failed; do not train J0")
    return payload


def gradient_audit_for_seed(seed: int, train_set: JAADSequenceDataset, ambiguous_set: JAADSequenceDataset, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["training"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(train_set.scene_feat.shape[-1]),
        hidden_dim=int(cfg["hidden_dim"]),
        pred_len=int(train_set.future_gt.shape[1]),
        gate_mode=str(cfg["gate_mode"]),
        max_obs_len=int(train_set.target_obs.shape[1]),
    )
    main_idx = balanced_indices(train_set.intent_label, int(cfg["batch_size"]), seed + 1_000_003)
    amb_gen = torch.Generator(device="cpu").manual_seed(seed + 2_000_003)
    amb_idx = torch.randperm(len(ambiguous_set), generator=amb_gen)[: min(int(cfg["batch_size"]), len(ambiguous_set))]
    main_batch = stack_batch(train_set, main_idx)
    amb_batch = stack_batch(ambiguous_set, amb_idx)
    model.train()
    target = torch.cat([main_batch["target_obs"], main_batch["target_abs_obs"]], dim=-1)
    output = model(target, main_batch["neighbor_obs"], main_batch["neighbor_mask"], main_batch["neighbor_visible_mask"], main_batch["scene_feat"])
    labels = main_batch["intent_label"]
    main_bce = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], labels)
    proposal_bce = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], labels)
    trajectory_loss = nn.functional.smooth_l1_loss(output["future_pred"], main_batch["future_gt"])
    amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1)
    amb_output = model(amb_target, amb_batch["neighbor_obs"], amb_batch["neighbor_mask"], amb_batch["neighbor_visible_mask"], amb_batch["scene_feat"])
    ambiguity = 0.5 * (amb_output["prior_logit"].square().mean() + amb_output["intent_logit"].square().mean())
    intent_objective = main_bce + float(cfg["prior_weight"]) * proposal_bce + float(cfg["ambiguous_weight"]) * ambiguity
    total = compose_objective(main_bce, proposal_bce, trajectory_loss, ambiguity,
        prior_weight=float(cfg["prior_weight"]), traj_weight=0.0, ambiguous_weight=float(cfg["ambiguous_weight"]))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    shared = [(n, p) for n, p in named if n.startswith(SHARED_PREFIXES)]
    trajectory_head = [(n, p) for n, p in named if n.startswith("traj_head.")]
    intent_shared_grads = torch.autograd.grad(intent_objective, [p for _, p in shared], retain_graph=True, allow_unused=True)
    zero_traj_shared_grads = torch.autograd.grad(0.0 * trajectory_loss, [p for _, p in shared], retain_graph=True, allow_unused=True)
    zero_traj_head_grads = torch.autograd.grad(0.0 * trajectory_loss, [p for _, p in trajectory_head], retain_graph=True, allow_unused=True)
    total_grads = torch.autograd.grad(total, [p for _, p in named], retain_graph=True, allow_unused=True)
    intent_norm, intent_vector = linear_norm(intent_shared_grads, [p for _, p in shared])
    traj_contribution_norm, traj_vector = linear_norm(zero_traj_shared_grads, [p for _, p in shared])
    total_by_name = {name: grad for (name, _), grad in zip(named, total_grads)}
    total_shared_norm, total_shared_vector = linear_norm(tuple(total_by_name[n] for n, _ in shared), [p for _, p in shared])
    total_all_norm, _ = linear_norm(total_grads, [p for _, p in named])
    traj_head_norm, _ = linear_norm(zero_traj_head_grads, [p for _, p in trajectory_head])
    raw_head_grads = torch.autograd.grad(trajectory_loss, [p for _, p in trajectory_head], allow_unused=True)
    raw_head_norm, _ = linear_norm(raw_head_grads, [p for _, p in trajectory_head])
    max_shared_delta = float((total_shared_vector - intent_vector).abs().max()) if intent_vector.numel() else 0.0
    passed = (
        tuple(output["future_pred"].shape) == (len(labels), int(train_set.future_gt.shape[1]), 2)
        and intent_norm > 0.0
        and total_shared_norm > 0.0
        and traj_contribution_norm == 0.0
        and traj_head_norm == 0.0
        and raw_head_norm > 0.0
        and max_shared_delta <= 1e-7
    )
    return {
        "seed": seed,
        "fixed_training_batch_count": len(labels),
        "fixed_ambiguous_batch_count": len(amb_batch["intent_label"]),
        "traj_weight": 0.0,
        "intent_objective": float(intent_objective.detach()),
        "raw_trajectory_loss_computed": float(trajectory_loss.detach()),
        "weighted_trajectory_loss_contribution": float((0.0 * trajectory_loss).detach()),
        "future_pred_shape": list(output["future_pred"].shape),
        "intent_grad_norm_shared_parameters": intent_norm,
        "trajectory_contribution_grad_norm_shared_parameters": traj_contribution_norm,
        "total_grad_norm_shared_parameters": total_shared_norm,
        "total_grad_norm_all_parameters": total_all_norm,
        "trajectory_head_grad_norm_from_zero_weight_objective": traj_head_norm,
        "trajectory_head_unweighted_grad_norm_control": raw_head_norm,
        "shared_total_vs_intent_gradient_max_abs_difference": max_shared_delta,
        "shared_parameter_count": sum(p.numel() for _, p in shared),
        "pass": bool(passed),
    }


def main() -> None:
    config = load_config()
    audit = config_audit()
    train_path = ROOT / config["training"]["data_root"] / "train.npz"
    ambiguous_path = ROOT / config["training"]["ambiguous_root"] / "train.npz"
    train_set = JAADSequenceDataset(train_path)
    ambiguous_set = JAADSequenceDataset(ambiguous_path)
    seeds = {str(seed): gradient_audit_for_seed(seed, train_set, ambiguous_set, config) for seed in SEEDS}
    payload = {
        "protocol": "J0 uses the unchanged joint forward graph and the historical J100 objective except trajectory_weight=0",
        "scope": "all trainable parameters partitioned into shared prefixes and traj_head; fixed training batches per seed",
        "per_seed": seeds,
        "all_seeds_pass": all(row["pass"] for row in seeds.values()),
        "test_split_loaded": False,
        "j0_vs_j100_config_audit_pass": audit["pass"],
    }
    write_json(RESULTS_ROOT / "j0_gradient_audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["all_seeds_pass"]:
        raise RuntimeError("J0 zero-weight gradient path audit failed; stop before training")


if __name__ == "__main__":
    main()
