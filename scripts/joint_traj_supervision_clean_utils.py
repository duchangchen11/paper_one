"""Shared utilities for the clean intent-only validation attribution study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results/joint_traj_supervision_clean"
CHECKPOINT_ROOT = ROOT / "checkpoints/joint_traj_supervision_clean"
SEEDS = (42, 123, 2024)
ARMS = {"J0_clean": 0.0, "J100_clean": 100.0}
SELECTION_TOLERANCE = 1e-4
SHARED_PREFIXES = (
    "target_projection.", "position_embedding", "target_encoder.", "neighbor_encoder.",
    "scene_encoder.", "proposal_fusion.", "proposal_head.", "gate.", "fusion.",
)


def config_path() -> Path:
    return ROOT / "configs/joint_traj_supervision_clean.json"


def load_config() -> dict[str, Any]:
    return json.loads(config_path().read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


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


def flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[torch.Tensor]) -> torch.Tensor:
    chunks = [
        torch.zeros_like(param, dtype=torch.float32).reshape(-1)
        if grad is None else grad.detach().float().reshape(-1)
        for grad, param in zip(grads, params)
    ]
    return torch.cat(chunks) if chunks else torch.empty(0)


def gradient_norm(grads: tuple[torch.Tensor | None, ...], params: list[torch.Tensor]) -> tuple[float, torch.Tensor]:
    vector = flatten_grads(grads, params)
    return float(torch.linalg.vector_norm(vector)), vector


def guard_pretest_split(path: Path) -> None:
    if path.name.lower() == "test.npz":
        raise RuntimeError("test.npz is prohibited before the frozen official evaluator")


def sample_key(arrays: dict[str, np.ndarray], index: int) -> tuple[str, str, int]:
    return (
        str(np.asarray(arrays["scene_id"]).reshape(-1)[index]),
        str(np.asarray(arrays["target_id"]).reshape(-1)[index]),
        int(np.asarray(arrays["obs_end_frame"]).reshape(-1)[index]),
    )


def map_video_ids(reference: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> np.ndarray:
    ref_keys = [sample_key(reference, i) for i in range(len(reference["scene_id"]))]
    target_keys = [sample_key(target, i) for i in range(len(target["scene_id"]))]
    if len(set(ref_keys)) != len(ref_keys) or len(set(target_keys)) != len(target_keys):
        raise ValueError("Sample join keys must be unique")
    if set(ref_keys) != set(target_keys):
        raise ValueError("Held-out sample IDs do not match frozen P1 references")
    lookup = {key: str(reference["video_id"][idx]) for idx, key in enumerate(ref_keys)}
    return np.asarray([lookup[key] for key in target_keys], dtype=str)


def normalized_clean_contract(
    source_args: dict[str, Any], *, seed: int, traj_weight: float, initial_state_path: str,
) -> dict[str, Any]:
    def get(name: str, default: Any = None) -> Any:
        value = source_args.get(name, default)
        return str(value) if isinstance(value, Path) else value

    return {
        "seed": int(seed),
        "data_root": str(get("data_root")),
        "ambiguous_root": str(get("ambiguous_root")),
        "epochs": int(get("epochs")),
        "batch_size": int(get("batch_size")),
        "hidden_dim": int(get("hidden_dim")),
        "learning_rate": float(get("learning_rate")),
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "prior_weight": float(get("prior_weight")),
        "ambiguous_weight": float(get("ambiguous_weight")),
        "gate_mode": str(get("gate_mode")),
        "traj_weight": float(traj_weight),
        "trajectory_weight_mode": "fixed",
        "init_trajectory_checkpoint": None,
        "initial_state_checkpoint": initial_state_path,
        "gradient_clip_norm": 5.0,
        "scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.5, "patience": 2, "monitor": "intent_auc"},
        "checkpoint_selection": {"primary": "raw_validation_intent_auc", "tie_break": "lower_raw_validation_brier_if_auc_difference_at_most_1e-4", "tolerance": SELECTION_TOLERANCE},
        "train_sampler": "WeightedRandomSampler(inverse class counts,replacement=True,num_samples=len(train))",
        "ambiguous_sampler": "DataLoader(shuffle=True)",
        "selection_mode": "intent_auc",
        "selection_tolerance": SELECTION_TOLERANCE,
        "calibration": "none; raw sigmoid probability; threshold=0.5",
    }


def only_traj_weight_differs(j0: dict[str, Any], j100: dict[str, Any]) -> bool:
    return (
        set(j0) == set(j100)
        and j0["traj_weight"] == 0.0
        and j100["traj_weight"] == 100.0
        and all(j0[key] == j100[key] for key in j0 if key != "traj_weight")
    )


def sampler_sequences_match(history_a: list[dict[str, Any]], history_b: list[dict[str, Any]]) -> bool:
    if len(history_a) != len(history_b):
        return False
    for left, right in zip(history_a, history_b):
        if left["epoch"] != right["epoch"]:
            return False
        if left["train"].get("sampler_sha256") != right["train"].get("sampler_sha256"):
            return False
        if left["train"].get("first_sample_indices") != right["train"].get("first_sample_indices"):
            return False
    return True
