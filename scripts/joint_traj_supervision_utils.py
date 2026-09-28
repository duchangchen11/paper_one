"""Shared, testable helpers for the controlled J0 vs J100 comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/joint_traj_supervision_attribution.json"
RESULTS_ROOT = ROOT / "results/joint_traj_supervision_attribution"
CHECKPOINT_ROOT = ROOT / "checkpoints/joint_traj_supervision_attribution"
SEEDS = (42, 123, 2024)
SHARED_PREFIXES = (
    "target_projection.", "position_embedding", "target_encoder.", "neighbor_encoder.",
    "scene_encoder.", "proposal_fusion.", "proposal_head.", "gate.", "fusion.",
)
ID_FIELDS = ("scene_id", "target_id", "obs_end_frame")


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def compose_objective(
    main_bce: torch.Tensor,
    proposal_bce: torch.Tensor,
    trajectory_loss: torch.Tensor,
    ambiguity_regularizer: torch.Tensor,
    *,
    prior_weight: float,
    traj_weight: float,
    ambiguous_weight: float,
) -> torch.Tensor:
    """Mirror the trainer's fixed-mode summation order, including lambda=0."""
    loss = main_bce + prior_weight * proposal_bce
    loss = loss + traj_weight * trajectory_loss
    return loss + ambiguous_weight * ambiguity_regularizer


def normalized_training_contract(args: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract all training choices from the actual saved J100 run artifacts."""
    get = lambda key, default=None: args.get(key, metrics.get(key, default))
    return {
        "seed": int(get("seed")),
        "data_root": str(get("data_root")),
        "ambiguous_root": str(get("ambiguous_root")),
        "epochs": int(get("epochs")),
        "batch_size": int(get("batch_size")),
        "hidden_dim": int(get("hidden_dim")),
        "learning_rate": float(get("learning_rate")),
        "prior_weight": float(get("prior_weight")),
        "traj_weight": float(get("traj_weight")),
        "ambiguous_weight": float(get("ambiguous_weight")),
        "gate_mode": str(get("gate_mode")),
        "init_trajectory_checkpoint": str(get("init_trajectory_checkpoint")) if get("init_trajectory_checkpoint") else None,
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5, "patience": 2},
        "gradient_clip_norm": 5.0,
        "train_sampler": "WeightedRandomSampler(inverse class counts,replacement=True,num_samples=len(train))",
        "ambiguous_sampler": "DataLoader(shuffle=True)",
        "checkpoint_selection": "validation intent_auc + 0.1*intent_f1 - 0.01*trajectory_ade_pixel",
        "objective": "main BCE + prior_weight*proposal BCE + traj_weight*SmoothL1(future_pred,future_gt) + ambiguous_weight*0.5*(ambiguous prior_logit^2 mean + intent_logit^2 mean)",
        "trajectory_weight_mode": "fixed",
    }


def only_trajectory_weight_differs(j100: dict[str, Any], j0: dict[str, Any]) -> bool:
    if set(j100) != set(j0):
        return False
    return all(j100[key] == j0[key] for key in j100 if key != "traj_weight") and j0["traj_weight"] == 0.0 and j100["traj_weight"] == 100.0


def guard_pretest_split(path: Path) -> None:
    if path.name == "test.npz":
        raise RuntimeError("test.npz cannot be loaded before the frozen official evaluator")


def sample_key(arrays: dict[str, np.ndarray], index: int) -> tuple[str, ...]:
    return tuple(str(np.asarray(arrays[key]).reshape(-1)[index]) for key in ID_FIELDS)


def map_video_ids(reference: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> np.ndarray:
    ref_keys = [sample_key(reference, i) for i in range(len(reference["scene_id"]))]
    target_keys = [sample_key(target, i) for i in range(len(target["scene_id"]))]
    if len(set(ref_keys)) != len(ref_keys) or len(set(target_keys)) != len(target_keys):
        raise ValueError("Sample join keys must be unique")
    if set(ref_keys) != set(target_keys):
        raise ValueError("Held-out sample IDs do not match frozen P1 references")
    lookup = {key: str(reference["video_id"][idx]) for idx, key in enumerate(ref_keys)}
    return np.asarray([lookup[key] for key in target_keys], dtype=str)


def feature_summary(features: np.ndarray) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    return {
        "sample_count": int(values.shape[0]),
        "feature_width": int(values.shape[1]),
        "feature_mean": float(values.mean()),
        "feature_std": float(values.std()),
        "l2_norm_mean": float(norms.mean()),
        "l2_norm_std": float(norms.std()),
        "per_dimension_variance": values.var(axis=0).tolist(),
        "mean_per_dimension_variance": float(values.var(axis=0).mean()),
    }


def feature_similarity(reference: np.ndarray, current: np.ndarray) -> dict[str, float]:
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(current, dtype=np.float64)
    cosine = np.sum(x * y, axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
    xc, yc = x - x.mean(axis=0, keepdims=True), y - y.mean(axis=0, keepdims=True)
    cross, xx, yy = xc.T @ yc, xc.T @ xc, yc.T @ yc
    denom = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    cka = float(np.linalg.norm(cross, ord="fro") ** 2 / denom) if denom else 0.0
    return {"cosine_mean": float(cosine.mean()), "cosine_std": float(cosine.std()), "linear_cka": cka}


def validate_metrics_payload(payload: dict[str, Any], *, seed: int, traj_weight: float, epoch_count: int) -> None:
    required = {"seed", "traj_weight", "best_epoch", "history", "test", "initial_model_state_sha256"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Training metrics missing required fields: {missing}")
    if int(payload["seed"]) != seed or float(payload["traj_weight"]) != traj_weight:
        raise ValueError("Run seed or trajectory weight does not match the frozen arm")
    if len(payload["history"]) != epoch_count or not (1 <= int(payload["best_epoch"]) <= epoch_count):
        raise ValueError("Training metrics history or selected epoch is incomplete")
    if traj_weight == 0 and payload["test"] is not None:
        raise ValueError("J0 training must withhold test results until protocol freeze")


def linear_norm(gradients: tuple[torch.Tensor | None, ...], parameters: list[torch.Tensor]) -> tuple[float, torch.Tensor]:
    flattened = []
    for grad, param in zip(gradients, parameters):
        flattened.append(torch.zeros_like(param, dtype=torch.float32).reshape(-1) if grad is None else grad.detach().float().reshape(-1))
    vector = torch.cat(flattened) if flattened else torch.empty(0)
    return float(torch.linalg.vector_norm(vector).detach().cpu()), vector
