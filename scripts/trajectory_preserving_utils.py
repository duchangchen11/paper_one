"""Shared utilities for the frozen trajectory-preserving intention experiment."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from torch.utils.data import Dataset

from src.models.trajectory_preserving_joint import TrajectoryPreservingJoint

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024)
CHECKPOINT_ROOT = ROOT / "checkpoints/trajectory_preserving_joint"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def backbone_sha256(model: TrajectoryPreservingJoint) -> str:
    return tensor_state_sha256(model.backbone.state_dict())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone_state(
    model: TrajectoryPreservingJoint, source_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    target_state = model.backbone.state_dict()
    missing = sorted(set(target_state) - set(source_state))
    unexpected = sorted(set(source_state) - set(target_state))
    shape_mismatch = [
        {
            "name": name,
            "checkpoint_shape": list(source_state[name].shape),
            "expected_shape": list(target_state[name].shape),
        }
        for name in sorted(set(source_state) & set(target_state))
        if source_state[name].shape != target_state[name].shape
    ]
    loadable = {
        name: source_state[name]
        for name in sorted(set(source_state) & set(target_state))
        if source_state[name].shape == target_state[name].shape
    }
    if missing or unexpected or shape_mismatch or len(loadable) != len(target_state):
        report = {
            "total_expected_tensors": len(target_state),
            "loaded_tensors": len(loadable),
            "missing_tensors": missing,
            "unexpected_tensors": unexpected,
            "shape_mismatch_tensors": shape_mismatch,
            "loaded_tensor_names": sorted(loadable),
            "complete": False,
        }
        raise RuntimeError(f"Trajectory backbone mapping was incomplete: {json.dumps(report)}")
    model.backbone.load_state_dict(loadable, strict=True)
    return {
        "total_expected_tensors": len(target_state),
        "loaded_tensors": len(loadable),
        "missing_tensors": [],
        "unexpected_tensors": [],
        "shape_mismatch_tensors": [],
        "loaded_tensor_names": sorted(loadable),
        "intent_branch_tensors_initialized_new": sorted(
            name for name in model.state_dict() if name.startswith(("intent_adapter.", "intent_head."))
        ),
        "complete": True,
    }


def load_seed_backbone(
    seed: int,
    intent_input: str,
    *,
    device: torch.device | str = "cpu",
    input_dim: int = 8,
    scene_dim: int = 512,
    observed_length: int = 15,
    prediction_length: int = 15,
) -> tuple[TrajectoryPreservingJoint, dict[str, Any], Path, str]:
    path = ROOT / "checkpoints" / f"trajectory_transformer_scene_15x15_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Matched trajectory-only checkpoint is missing: {path}")
    checkpoint_sha = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["args"]["seed"]) != seed:
        raise RuntimeError(f"Checkpoint seed mismatch for seed {seed}: {path}")
    config = payload["args"]
    model = TrajectoryPreservingJoint(
        input_dim=input_dim,
        scene_dim=scene_dim,
        d_model=int(config["d_model"]),
        nhead=4,
        num_layers=int(config["num_layers"]),
        pred_len=prediction_length,
        dropout=0.1,
        max_obs_len=observed_length,
        intent_input=intent_input,
    )
    loading_report = load_backbone_state(model, payload["model"])
    model.to(device)
    model.backbone.eval()
    return model, loading_report, path, checkpoint_sha


class TrajectoryIntentDataset(Dataset):
    """Load explicitly requested train/val split arrays; never discovers test."""

    def __init__(self, path: Path) -> None:
        if path.name == "test.npz":
            raise RuntimeError("Test split may only be loaded by the post-freeze evaluator")
        with np.load(path, allow_pickle=False) as archive:
            self.target = torch.from_numpy(
                np.concatenate([archive["target_obs"], archive["target_abs_obs"]], axis=-1).astype(np.float32)
            )
            self.scene_feat = torch.from_numpy(archive["scene_feat"].astype(np.float32))
            self.intent_label = torch.from_numpy(archive["intent_label"].astype(np.float32))
            self.future_gt = torch.from_numpy(archive["future_gt"].astype(np.float32))
            self.image_size = torch.from_numpy(archive["image_size"].astype(np.float32))

    def __len__(self) -> int:
        return self.intent_label.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target": self.target[index],
            "scene_feat": self.scene_feat[index],
            "intent_label": self.intent_label[index],
            "future_gt": self.future_gt[index],
            "image_size": self.image_size[index],
        }


def probabilities_from_logits(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    return 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))


def intention_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    *,
    temperature: float = 1.0,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = probabilities_from_logits(logits, temperature)
    prediction = (probability >= threshold).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def trajectory_metrics(
    prediction: np.ndarray, target: np.ndarray, image_size: np.ndarray
) -> dict[str, float]:
    error = (np.asarray(prediction) - np.asarray(target)) * np.asarray(image_size)[:, None, :]
    distance = np.linalg.norm(error, axis=-1)
    return {
        "ade_pixel": float(distance.mean()),
        "fde_pixel": float(distance[:, -1].mean()),
    }
