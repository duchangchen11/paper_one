"""Shared construction, hashing, and drift helpers for the M1 experiment."""

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

from scripts.trajectory_preserving_utils import SEEDS, set_seed, sha256_file
from src.models.intention_scratch_transformer import IntentionScratchTransformer
from src.models.pretrained_clone_intent import (
    PretrainedCloneIntent,
    clone_target_encoder,
    compare_encoder_storage,
)
from src.models.trajectory_transformer import SceneTrajectoryTransformer

CONFIG_PATH = ROOT / "configs/pretrained_clone_intent.json"
CHECKPOINT_ROOT = ROOT / "checkpoints/pretrained_clone_intent"
RESULTS_ROOT = ROOT / "results/pretrained_clone_intent"
TRAJECTORY_CHECKPOINT_ROOT = ROOT / "checkpoints"


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_sha256(model: nn.Module) -> str:
    return state_sha256(model.state_dict())


def encoder_state_sha256(model: IntentionScratchTransformer) -> str:
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name == "position_embedding"
        or name.startswith(("input_projection.", "temporal_encoder."))
    }
    return state_sha256(state)


def trajectory_model_sha256(model: SceneTrajectoryTransformer) -> str:
    return state_sha256(model.state_dict())


def head_state_sha256(model: IntentionScratchTransformer) -> str:
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith(("intent_adapter.", "intent_head."))
    }
    return state_sha256(state)


def trajectory_checkpoint_path(seed: int) -> Path:
    return TRAJECTORY_CHECKPOINT_ROOT / f"trajectory_transformer_scene_15x15_seed{seed}.pt"


def build_pretrained_clone(
    seed: int,
    *,
    device: torch.device | str = "cpu",
) -> tuple[PretrainedCloneIntent, dict[str, Any]]:
    """Construct M1 with an M0-identical random head and copied target encoder."""
    config = load_config()
    model_cfg = config["model"]["intention_branch"]
    set_seed(seed)
    intention = IntentionScratchTransformer(
        input_dim=8,
        d_model=int(model_cfg["hidden_dimension"]),
        nhead=int(model_cfg["heads"]),
        num_layers=int(model_cfg["layers"]),
        dropout=float(model_cfg["dropout"]),
        max_obs_len=15,
    )
    initial_m0_equivalent_model_sha256 = model_sha256(intention)
    initial_head_sha256 = head_state_sha256(intention)

    checkpoint_path = trajectory_checkpoint_path(seed)
    checkpoint_sha = sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_seed = int(payload["args"]["seed"])
    if checkpoint_seed != seed:
        raise RuntimeError(f"Trajectory checkpoint seed {checkpoint_seed} does not match requested M1 seed {seed}")

    # Initializing the frozen branch must not consume the RNG stream used by
    # the M0-matched intention head or later dropout/DataLoader randomness.
    with torch.random.fork_rng(devices=[]):
        trajectory = SceneTrajectoryTransformer(
            input_dim=8,
            scene_dim=512,
            d_model=int(model_cfg["hidden_dimension"]),
            nhead=int(model_cfg["heads"]),
            num_layers=int(model_cfg["layers"]),
            pred_len=15,
            dropout=float(model_cfg["dropout"]),
            max_obs_len=15,
        )
    trajectory.load_state_dict(payload["model"], strict=True)
    trajectory.eval()

    clone_report = clone_target_encoder(trajectory, intention)
    if not clone_report["clone_pass"]:
        raise RuntimeError(f"Pretrained target encoder clone failed for seed {seed}: {clone_report}")
    independence = compare_encoder_storage(trajectory, intention)
    if (
        not independence["initial_value_equal"]
        or independence["same_parameter_object"]
        or independence["same_storage"]
    ):
        raise RuntimeError(f"Trajectory and intention encoders are not independent for seed {seed}: {independence}")

    model = PretrainedCloneIntent(trajectory, intention).to(device)
    model.trajectory_branch.eval()
    if any(parameter.requires_grad for parameter in model.trajectory_branch.parameters()):
        raise RuntimeError("M1 trajectory branch must be entirely frozen")
    if not all(parameter.requires_grad for parameter in model.intention_branch.parameters()):
        raise RuntimeError("Every M1 intention branch parameter must be trainable")

    report = {
        "seed": seed,
        "trajectory_checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "trajectory_checkpoint_sha256": checkpoint_sha,
        "trajectory_checkpoint_args_seed": checkpoint_seed,
        "random_m0_equivalent_initial_state_sha256_before_clone": initial_m0_equivalent_model_sha256,
        "intention_head_initialization_sha256": initial_head_sha256,
        "intention_model_state_sha256_after_clone": model_sha256(model.intention_branch),
        "intention_encoder_sha256_after_clone": encoder_state_sha256(model.intention_branch),
        "trajectory_branch_sha256": trajectory_model_sha256(model.trajectory_branch),
        "clone_report": clone_report,
        "parameter_independence": compare_encoder_storage(
            model.trajectory_branch, model.intention_branch
        ),
        "trajectory_frozen": all(
            not parameter.requires_grad for parameter in model.trajectory_branch.parameters()
        ),
        "trajectory_eval_mode": not model.trajectory_branch.training,
        "intention_all_trainable": all(
            parameter.requires_grad for parameter in model.intention_branch.parameters()
        ),
    }
    return model, report


def encoder_drift(
    reference: dict[str, torch.Tensor],
    current: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Relative L2 and cosine drift for the cloned target encoder, grouped by layer."""
    names = [
        name
        for name in reference
        if name == "position_embedding"
        or name.startswith(("input_projection.", "temporal_encoder."))
    ]
    missing = sorted(set(names) - set(current))
    if missing:
        raise RuntimeError(f"Current M1 state lacks pretrained encoder tensors: {missing}")

    def summarize(keys: list[str]) -> dict[str, float]:
        ref = torch.cat([reference[key].detach().cpu().float().reshape(-1) for key in keys])
        cur = torch.cat([current[key].detach().cpu().float().reshape(-1) for key in keys])
        diff = cur - ref
        denom = float(torch.linalg.vector_norm(ref))
        relative = float(torch.linalg.vector_norm(diff)) / denom if denom > 0.0 else float(torch.linalg.vector_norm(diff))
        denom_cos = float(torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(cur))
        cosine = float(torch.dot(ref, cur)) / denom_cos if denom_cos > 0.0 else 1.0
        return {
            "relative_l2_parameter_change": relative,
            "cosine_similarity": cosine,
            "absolute_l2_parameter_change": float(torch.linalg.vector_norm(diff)),
            "reference_l2_norm": denom,
        }

    groups: dict[str, list[str]] = {
        "input_projection": [name for name in names if name.startswith("input_projection.")],
        "position_embedding": [name for name in names if name == "position_embedding"],
    }
    for name in names:
        if name.startswith("temporal_encoder.layers."):
            layer = name.split(".")[2]
            groups.setdefault(f"temporal_encoder.layer_{layer}", []).append(name)
    return {
        "encoder_parameter_tensor_count": len(names),
        "overall": summarize(names),
        "per_layer": {group: summarize(keys) for group, keys in groups.items() if keys},
        "definition": "relative_l2=||current-reference||_2/||reference||_2; cosine over flattened tensors",
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bootstrap_delta_auc_ci(bootstrap: dict[str, Any]) -> tuple[float, float]:
    ci = bootstrap["delta_roc_auc"]["ci_percentile_95"]
    return float(ci["lower_95"]), float(ci["upper_95"])


def finite_json_float(value: float) -> float:
    """Convert numpy/torch scalar values for JSON serialization."""
    return float(np.asarray(value).reshape(()))
