#!/usr/bin/env python3
"""Four-phase, official-train-only video-holdout replication of trajectory reliability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_trajectory_reliability import prediction_diagnostics  # noqa: E402
from scripts.evaluate_trajectory_reliability_deconfounding import (  # noqa: E402
    BOOTSTRAP_SEED,
    PRIMARY_SCORE,
    RISK_COVERAGES,
    apply_motion_adjustment,
    assign_quantile_strata,
    cluster_bootstrap_metrics,
    global_risk_curves,
    partial_spearman,
    score_metrics,
    stratified_risk_curve,
    within_motion_permutation_test,
)
from src.models.trajectory_transformer import SceneTrajectoryTransformer  # noqa: E402


PRIMARY_SCORE = "u_mean"
SCENE_MODE = "zero"
RANDOM_SEED = 314159
SEEDS = (42, 123, 2024)
SPLIT_FRACTIONS = {"internal_train": 0.70, "internal_val": 0.15}
PREDICTION_LENGTH = 15
OBSERVATION_LENGTH = 15
MODEL_CONFIG = {
    "input_dim": 8,
    "d_model": 128,
    "num_layers": 3,
    "nhead": 4,
    "pred_len": PREDICTION_LENGTH,
    "dropout": 0.1,
    "max_obs_len": OBSERVATION_LENGTH,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "epochs": 20,
    "gradient_clip_norm": 5.0,
    "loss": "SmoothL1Loss on normalized future coordinates",
    "optimizer": "AdamW",
    "scheduler": "ReduceLROnPlateau(mode=min,factor=0.5,patience=2), driven by internal_val pixel ADE",
}
RESULT_ROOT = PROJECT_ROOT / "results/reliability_internal_replication"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
TRAIN_PATH = PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15/train.npz"
EXPECTED_MANIFEST_SHA256 = "41270ddb5ec35f777f37956377e4e530381935d1ae44bccc0c74b9c058aaa0ea"
EXPECTED_PROTOCOL_SHA256 = "777b9131e163176bc0eaa7eca4997784ddb32a1ea8195a1c1eae842d29b7675b"
EXPECTED_TRAIN_NPZ_SHA256 = "6da2d1d40a918a218a7d054407d95f2791e2338a760b4836f2ca6d06c59ea19a"
EXPECTED_CHECKPOINT_SHA256 = {
    "42": "0f9efd94e0412c6710299d7f0d323deb7a37c0ea4588014e6ff18422f2512f9e",
    "123": "cf9c84f5e806c3ca99e75a8a6fc34d3649794389dc1859780a99a7a405cb91d3",
    "2024": "0f74da9131a5f2b4f97d71411149e3b5677e118584c305f24266d80b53fe6190",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: dict[str, Any], exclude: tuple[str, ...] = ()) -> str:
    payload = {key: item for key, item in value.items() if key not in exclude}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically replace one JSON artifact after it has been fully serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ordered_sample_metadata_sha256(
    scene_ids: np.ndarray, target_ids: np.ndarray, obs_end_frame: np.ndarray
) -> dict[str, Any]:
    scenes = np.asarray(scene_ids).astype(str).reshape(-1)
    targets = np.asarray(target_ids).astype(str).reshape(-1)
    frames = np.asarray(obs_end_frame).reshape(-1)
    if not (len(scenes) == len(targets) == len(frames)):
        raise ValueError("scene_id, target_id, and obs_end_frame must have matching lengths")
    rows = [
        {"scene_id": scene, "target_id": target, "obs_end_frame": int(frame)}
        for scene, target, frame in zip(scenes, targets, frames)
    ]
    canonical = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {
        "sample_count": len(rows),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "rows": rows,
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_video_ids(scene_ids: np.ndarray, seed: int = RANDOM_SEED) -> dict[str, list[str]]:
    unique_ids = sorted(set(np.asarray(scene_ids).astype(str).reshape(-1).tolist()))
    if len(unique_ids) < 3:
        raise ValueError("Need at least three videos for internal train/val/holdout splitting")
    permutation = np.random.default_rng(seed).permutation(len(unique_ids))
    shuffled = [unique_ids[int(index)] for index in permutation]
    train_count = int(round(len(shuffled) * SPLIT_FRACTIONS["internal_train"]))
    val_count = int(round(len(shuffled) * SPLIT_FRACTIONS["internal_val"]))
    train_count = min(max(train_count, 1), len(shuffled) - 2)
    val_count = min(max(val_count, 1), len(shuffled) - train_count - 1)
    return {
        "internal_train": shuffled[:train_count],
        "internal_val": shuffled[train_count : train_count + val_count],
        "internal_holdout": shuffled[train_count + val_count :],
    }


def indices_for_split(scene_ids: np.ndarray, split_scene_ids: list[str]) -> np.ndarray:
    scene_ids = np.asarray(scene_ids).astype(str).reshape(-1)
    allowed = np.asarray(split_scene_ids, dtype=str)
    indices = np.flatnonzero(np.isin(scene_ids, allowed))
    actual = set(scene_ids[indices].tolist())
    if actual != set(split_scene_ids):
        missing = sorted(set(split_scene_ids) - actual)
        raise ValueError(f"Split manifest includes scenes absent from the archive: {missing[:5]}")
    return indices


def select_best_epoch(history: list[dict[str, Any]]) -> int:
    """Select solely by internal_val pixel ADE, ignoring any holdout-like fields."""
    if not history:
        raise ValueError("Epoch history cannot be empty")
    return int(min(
        history,
        key=lambda row: float(row["internal_val"]["trajectory_ade_pixel"]),
    )["epoch"])


def freeze_val_thresholds(val_ade: np.ndarray, val_fde: np.ndarray) -> dict[str, Any]:
    """Freeze high-error pixel thresholds from internal_val only."""
    return {
        "high_ade_pixel_threshold": float(np.quantile(val_ade, 0.8)),
        "high_fde_pixel_threshold": float(np.quantile(val_fde, 0.8)),
        "source_split": "internal_val",
        "quantile": 0.8,
        "label_rule": "error >= frozen threshold",
        "holdout_top20_used_for_primary_label": False,
    }


def make_manifest_payload(
    scene_ids: np.ndarray,
    target_ids: np.ndarray,
    train_sha256: str,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    scenes = np.asarray(scene_ids).astype(str).reshape(-1)
    targets = np.asarray(target_ids).astype(str).reshape(-1)
    if len(scenes) != len(targets):
        raise ValueError("scene_id and target_id arrays differ in length")
    partitions = split_video_ids(scenes, seed)
    scene_sets = {name: set(values) for name, values in partitions.items()}
    overlaps = {
        "internal_train_internal_val": len(scene_sets["internal_train"] & scene_sets["internal_val"]),
        "internal_train_internal_holdout": len(scene_sets["internal_train"] & scene_sets["internal_holdout"]),
        "internal_val_internal_holdout": len(scene_sets["internal_val"] & scene_sets["internal_holdout"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Video-level split overlap detected: {overlaps}")
    if set.union(*scene_sets.values()) != set(np.unique(scenes).tolist()):
        raise RuntimeError("Video-level split does not cover all train videos exactly")

    split_stats = {}
    for name, allowed in scene_sets.items():
        sample_mask = np.isin(scenes, list(allowed))
        split_target_ids = targets[sample_mask]
        split_scene_ids_array = scenes[sample_mask]
        target_pairs = np.char.add(np.char.add(split_scene_ids_array, "::"), split_target_ids)
        split_stats[name] = {
            "scene_ids": partitions[name],
            "video_count": int(len(allowed)),
            "sample_count": int(sample_mask.sum()),
            "unique_target_count": int(np.unique(split_target_ids).size),
            "unique_scene_target_track_count": int(np.unique(target_pairs).size),
        }
    return {
        "protocol": "video-level random split within official train only",
        "random_seed": int(seed),
        "source_split": "official train only",
        "source_file": "data/processed/jaad_sequences_scene_15x15/train.npz",
        "train_npz_sha256": train_sha256,
        "total_sample_count": int(len(scenes)),
        "total_unique_videos": int(np.unique(scenes).size),
        "split_fractions": {"internal_train": 0.70, "internal_val": 0.15, "internal_holdout": 0.15},
        "rounding_rule": "round(0.70*N) train; round(0.15*N) val; remaining videos holdout",
        "splits": split_stats,
        "scene_overlap_counts": overlaps,
        "official_validation_or_test_accessed": False,
        "holdout_metrics_accessed": False,
    }


def prepare_manifest(train_path: Path, result_root: Path) -> dict[str, Any]:
    manifest_path = result_root / "split_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Split manifest already exists and is immutable: {manifest_path}")
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    with np.load(train_path, allow_pickle=False) as archive:
        # PHASE A reads only grouping/track metadata from official train.
        scene_ids = archive["scene_id"].astype(str)
        target_ids = archive["target_id"].astype(str)
    payload = make_manifest_payload(scene_ids, target_ids, sha256_file(train_path))
    payload["manifest_sha256"] = canonical_sha256(payload)
    result_root.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, payload)
    file_hash = sha256_file(manifest_path)
    (result_root / "split_manifest.sha256").write_text(
        f"canonical_content_sha256={payload['manifest_sha256']}\nfile_sha256={file_hash}\n",
        encoding="utf-8",
    )
    payload["manifest_file_sha256"] = file_hash
    return payload


def load_manifest(result_root: Path, train_path: Path) -> dict[str, Any]:
    path = result_root / "split_manifest.json"
    manifest = read_json(path)
    expected_manifest_hash = manifest.get("manifest_sha256")
    actual_manifest_hash = canonical_sha256(manifest, exclude=("manifest_sha256", "manifest_file_sha256"))
    if expected_manifest_hash != actual_manifest_hash:
        raise RuntimeError("Split manifest canonical SHA256 mismatch")
    if manifest["train_npz_sha256"] != sha256_file(train_path):
        raise RuntimeError("official train.npz changed after the internal split manifest was frozen")
    if any(manifest["scene_overlap_counts"].values()):
        raise RuntimeError("Manifest contains overlapping scene_id sets")
    return manifest


class TrainArchiveDataset(Dataset):
    """Load only trajectory input/target arrays from official train.npz."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as archive:
            self.scene_ids = archive["scene_id"].astype(str)
            self.target_ids = archive["target_id"].astype(str)
            self.target_obs = torch.from_numpy(archive["target_obs"].astype(np.float32))
            self.target_abs_obs = torch.from_numpy(archive["target_abs_obs"].astype(np.float32))
            self.future_gt = torch.from_numpy(archive["future_gt"].astype(np.float32))
            self.image_size = torch.from_numpy(archive["image_size"].astype(np.float32))
            self.scene_feat = torch.from_numpy(archive["scene_feat"].astype(np.float32))
            self.obs_end_frame = archive["obs_end_frame"].copy()

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target_obs": self.target_obs[index],
            "target_abs_obs": self.target_abs_obs[index],
            "future_gt": self.future_gt[index],
            "image_size": self.image_size[index],
            "scene_feat": self.scene_feat[index],
        }


class VideoSubsetDataset(Dataset):
    """Index-only view over the train archive, restricted to a manifest video set."""

    def __init__(self, base: TrainArchiveDataset, indices: np.ndarray, allowed_scene_ids: list[str]):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.allowed_scene_ids = set(allowed_scene_ids)
        self.scene_ids = base.scene_ids[self.indices]
        unexpected = set(self.scene_ids.tolist()) - self.allowed_scene_ids
        if unexpected:
            raise RuntimeError(f"Subset contains out-of-split videos: {sorted(unexpected)[:5]}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.base[int(self.indices[index])]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(
    dataset: Dataset,
    shuffle: bool,
    batch_size: int,
    seed: int | None = None,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: SceneTrajectoryTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    errors_norm: list[torch.Tensor] = []
    errors_pixel: list[torch.Tensor] = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            # This experiment always uses the zero-scene baseline.
            scene = torch.zeros_like(batch["scene_feat"].to(device))
            future = batch["future_gt"].to(device)
            prediction = model(target, scene)
            loss = nn.functional.smooth_l1_loss(prediction, future)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MODEL_CONFIG["gradient_clip_norm"])
                optimizer.step()
            total_loss += float(loss.detach().item()) * len(future)
            pred_cpu = prediction.detach().cpu()
            gt_cpu = future.detach().cpu()
            scale = batch["image_size"]
            errors_norm.append(torch.linalg.vector_norm(pred_cpu - gt_cpu, dim=-1))
            errors_pixel.append(torch.linalg.vector_norm((pred_cpu - gt_cpu) * scale[:, None, :], dim=-1))
    if not errors_pixel:
        raise RuntimeError("Cannot evaluate an empty split")
    normalized = torch.cat(errors_norm)
    pixel = torch.cat(errors_pixel)
    return {
        "loss": total_loss / len(loader.dataset),
        "trajectory_ade_normalized": float(normalized.mean()),
        "trajectory_fde_normalized": float(normalized[:, -1].mean()),
        "trajectory_ade_pixel": float(pixel.mean()),
        "trajectory_fde_pixel": float(pixel[:, -1].mean()),
    }


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINT_ROOT / f"trajectory_zero_scene_internal_seed{seed}.pt"


def output_dir(seed: int) -> Path:
    return RESULT_ROOT / f"trajectory_seed{seed}"


def train_all(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.result_root, args.train_path)
    completion_path = args.result_root / "phase_b_complete.json"
    if completion_path.exists():
        raise FileExistsError("PHASE B is already marked complete; refusing to retrain/overwrite")
    paths = [checkpoint_path(seed) for seed in SEEDS]
    if any(path.exists() for path in paths):
        raise FileExistsError("Internal replication checkpoint path already exists; refusing to overwrite")
    if any((output_dir(seed) / "metrics.json").exists() for seed in SEEDS):
        raise FileExistsError("Internal trajectory metrics already exist; refusing to overwrite")

    base = TrainArchiveDataset(args.train_path)
    train_split = manifest["splits"]["internal_train"]
    val_split = manifest["splits"]["internal_val"]
    train_indices = indices_for_split(base.scene_ids, train_split["scene_ids"])
    val_indices = indices_for_split(base.scene_ids, val_split["scene_ids"])
    holdout_ids = set(manifest["splits"]["internal_holdout"]["scene_ids"])
    train_set = VideoSubsetDataset(base, train_indices, train_split["scene_ids"])
    val_set = VideoSubsetDataset(base, val_indices, val_split["scene_ids"])
    if set(train_set.scene_ids) & set(val_set.scene_ids) or set(train_set.scene_ids) & holdout_ids or set(val_set.scene_ids) & holdout_ids:
        raise RuntimeError("PHASE B loaders are not video-disjoint")
    train_loader_size = len(train_set)
    val_loader_size = len(val_set)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "phase": "PHASE B starting",
        "device": str(device),
        "seeds": list(SEEDS),
        "internal_train_samples": train_loader_size,
        "internal_val_samples": val_loader_size,
        "holdout_loader_created": False,
        "holdout_metrics_accessed": False,
        "config": MODEL_CONFIG,
    }, ensure_ascii=False), flush=True)

    for seed in SEEDS:
        set_seed(seed)
        train_loader = _loader(train_set, shuffle=True, batch_size=MODEL_CONFIG["batch_size"], seed=seed)
        val_loader = _loader(val_set, shuffle=False, batch_size=MODEL_CONFIG["batch_size"])
        model = SceneTrajectoryTransformer(
            input_dim=MODEL_CONFIG["input_dim"],
            scene_dim=int(base.scene_feat.shape[-1]),
            d_model=MODEL_CONFIG["d_model"],
            nhead=MODEL_CONFIG["nhead"],
            num_layers=MODEL_CONFIG["num_layers"],
            pred_len=MODEL_CONFIG["pred_len"],
            dropout=MODEL_CONFIG["dropout"],
            max_obs_len=MODEL_CONFIG["max_obs_len"],
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=MODEL_CONFIG["learning_rate"],
            weight_decay=MODEL_CONFIG["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
        best_ade = float("inf")
        best_epoch = 0
        history: list[dict[str, Any]] = []
        path = checkpoint_path(seed)
        args_metadata = {
            **MODEL_CONFIG,
            "seed": seed,
            "scene_mode": SCENE_MODE,
            "split_manifest_sha256": manifest["manifest_sha256"],
            "selection_split": "internal_val",
            "selection_metric": "lowest internal_val trajectory_ade_pixel",
            "initialization": "fresh random initialization; no checkpoint loaded",
        }
        for epoch in range(1, MODEL_CONFIG["epochs"] + 1):
            train_metrics = run_epoch(model, train_loader, device, optimizer)
            with torch.no_grad():
                val_metrics = run_epoch(model, val_loader, device)
            scheduler.step(val_metrics["trajectory_ade_pixel"])
            row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "internal_train": train_metrics,
                "internal_val": val_metrics,
            }
            history.append(row)
            print(json.dumps({"phase": "PHASE B", "seed": seed, **row}, ensure_ascii=False), flush=True)
            if val_metrics["trajectory_ade_pixel"] < best_ade:
                best_ade = val_metrics["trajectory_ade_pixel"]
                best_epoch = epoch
                temporary_checkpoint = path.with_name(path.name + ".tmp")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "args": args_metadata,
                        "scene_mode": SCENE_MODE,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_internal_val": val_metrics,
                    },
                    temporary_checkpoint,
                )
                os.replace(temporary_checkpoint, path)

        if best_epoch == 0 or not path.is_file():
            raise RuntimeError(f"No validation-selected checkpoint produced for seed {seed}")
        if select_best_epoch(history) != best_epoch:
            raise RuntimeError("Saved checkpoint epoch disagrees with internal_val-only selection")
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        with torch.no_grad():
            best_val_metrics = run_epoch(model, val_loader, device)
        run_result = {
            "phase": "PHASE B; internal holdout not evaluated",
            "seed": seed,
            "scene_mode": SCENE_MODE,
            "initialization": "fresh random initialization; no checkpoint loaded",
            "split_manifest_sha256": manifest["manifest_sha256"],
            "internal_train_sample_count": train_loader_size,
            "internal_val_sample_count": val_loader_size,
            "internal_holdout_sample_count": manifest["splits"]["internal_holdout"]["sample_count"],
            "checkpoint": str(path),
            "checkpoint_sha256": sha256_file(path),
            "best_epoch": best_epoch,
            "best_internal_val": best_val_metrics,
            "checkpoint_selection": "lowest internal_val pixel ADE; internal_holdout excluded",
            "history": history,
        }
        write_json(output_dir(seed) / "metrics.json", run_result)

    checkpoints = {
        str(seed): {
            "path": str(checkpoint_path(seed)),
            "sha256": sha256_file(checkpoint_path(seed)),
        }
        for seed in SEEDS
    }
    write_json(completion_path, {
        "phase": "PHASE B complete",
        "completed_at_utc": utc_now(),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_selection_split": "internal_val only",
        "holdout_evaluated": False,
        "checkpoints": checkpoints,
    })


def _load_frozen_models(
    base: TrainArchiveDataset,
    device: torch.device,
    expected_manifest_sha256: str,
) -> tuple[list[SceneTrajectoryTransformer], dict[str, str]]:
    models: list[SceneTrajectoryTransformer] = []
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = checkpoint_path(seed)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("scene_mode") != SCENE_MODE or int(payload.get("seed", -1)) != seed:
            raise RuntimeError(f"Unexpected checkpoint metadata: {path}")
        if payload.get("args", {}).get("split_manifest_sha256") != expected_manifest_sha256:
            raise RuntimeError(f"Checkpoint is not linked to the frozen split manifest: {path}")
        model = SceneTrajectoryTransformer(
            input_dim=MODEL_CONFIG["input_dim"],
            scene_dim=int(base.scene_feat.shape[-1]),
            d_model=MODEL_CONFIG["d_model"],
            nhead=MODEL_CONFIG["nhead"],
            num_layers=MODEL_CONFIG["num_layers"],
            pred_len=MODEL_CONFIG["pred_len"],
            dropout=MODEL_CONFIG["dropout"],
            max_obs_len=MODEL_CONFIG["max_obs_len"],
        )
        model.load_state_dict(payload["model"], strict=True)
        model.to(device).eval()
        models.append(model)
        hashes[str(seed)] = sha256_file(path)
    return models, hashes


def infer_subset(
    split_name: str,
    base: TrainArchiveDataset,
    indices: np.ndarray,
    scene_ids: list[str],
    models: list[SceneTrajectoryTransformer],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, Any]:
    subset = VideoSubsetDataset(base, indices, scene_ids)
    loader = _loader(subset, shuffle=False, batch_size=batch_size)
    prediction_batches: list[list[np.ndarray]] = [[], [], []]
    future_batches: list[np.ndarray] = []
    size_batches: list[np.ndarray] = []
    abs_obs_batches: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            scene_zero = torch.zeros_like(batch["scene_feat"].to(device))
            future_batches.append(batch["future_gt"].numpy())
            size_batches.append(batch["image_size"].numpy())
            abs_obs_batches.append(batch["target_abs_obs"].numpy())
            for model_index, model in enumerate(models):
                prediction_batches[model_index].append(model(target, scene_zero).cpu().numpy())
    predictions = np.stack([np.concatenate(items, axis=0) for items in prediction_batches], axis=0)
    future = np.concatenate(future_batches, axis=0)
    image_size = np.concatenate(size_batches, axis=0)
    abs_obs = np.concatenate(abs_obs_batches, axis=0)
    if predictions.shape[1] != len(subset) or future.shape[0] != len(subset):
        raise RuntimeError(f"{split_name}: shared inference loader count mismatch")
    diagnostic = prediction_diagnostics(predictions, future, image_size)
    observed_centers_px = abs_obs[:, :, :2] * image_size[:, None, :]
    motion = np.linalg.norm(observed_centers_px[:, -1] - observed_centers_px[:, 0], axis=-1)
    gt_endpoint_displacement = np.linalg.norm(future[:, -1] * image_size, axis=-1)
    return {
        "split": split_name,
        "sample_count": len(subset),
        "sample_order": ordered_sample_metadata_sha256(
            subset.scene_ids, base.target_ids[subset.indices], base.obs_end_frame[subset.indices]
        ),
        "scene_ids": subset.scene_ids.copy(),
        "target_ids": base.target_ids[subset.indices].copy(),
        "image_size": image_size,
        "motion": motion,
        "future_endpoint_displacement_diagnostic_only": gt_endpoint_displacement,
        "predictions_normalized": predictions,
        "future_normalized": future,
        "diagnostics": diagnostic,
        "ade": diagnostic["sample_errors"]["ade_pixel"],
        "fde": diagnostic["sample_errors"]["fde_pixel"],
        "u_mean": diagnostic["scores"][PRIMARY_SCORE],
        "u_mean_normalized": diagnostic["scores_normalized"][PRIMARY_SCORE],
        "ade_normalized": diagnostic["sample_errors_normalized"]["ade_normalized"],
        "fde_normalized": diagnostic["sample_errors_normalized"]["fde_normalized"],
    }


def _fit_internal_val_adjustment(motion: np.ndarray, uncertainty: np.ndarray) -> dict[str, Any]:
    if np.any(motion < 0) or np.any(uncertainty < 0):
        raise ValueError("Motion and u_mean magnitudes must be non-negative")
    x = np.log1p(motion)
    y = np.log1p(uncertainty)
    b2, b1, b0 = np.polyfit(x, y, deg=2)
    return {
        "b0": float(b0),
        "b1": float(b1),
        "b2": float(b2),
        "fit_split": "internal_val",
        "target": "log1p(u_mean)",
        "predictor": "log1p(observed_motion_magnitude_pixel)",
        "degree": 2,
        "fit_sample_count": int(len(motion)),
        "ADE_or_FDE_used": False,
        "future_gt_used": False,
        "holdout_used": False,
    }


def _split_score_metrics(
    result: dict[str, Any],
    adjusted: np.ndarray,
    ade_threshold: float,
    fde_threshold: float,
) -> dict[str, Any]:
    ade, fde = result["ade"], result["fde"]
    high_ade = ade >= ade_threshold
    high_fde = fde >= fde_threshold
    return {
        name: score_metrics(score, ade, fde, high_ade, high_fde)
        for name, score in (
            ("raw_u_mean", result["u_mean"]),
            ("motion_only", result["motion"]),
            ("adjusted_u_mean", adjusted),
        )
    }


def _reliability_summary(
    result: dict[str, Any],
    adjusted: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    metrics = _split_score_metrics(
        result, adjusted,
        thresholds["high_ade_pixel_threshold"],
        thresholds["high_fde_pixel_threshold"],
    )
    return {
        "split": result["split"],
        "sample_count": result["sample_count"],
        "individual_model_performance": result["diagnostics"]["individual_model_performance"],
        "ensemble_mean_performance": result["diagnostics"]["ensemble_mean_performance"],
        "fixed_high_error_thresholds_pixel": thresholds,
        "metrics": metrics,
        "partial_spearman_raw_u_given_motion": {
            "u_mean_vs_ade": partial_spearman(result["u_mean"], result["ade"], result["motion"]),
            "u_mean_vs_fde": partial_spearman(result["u_mean"], result["fde"], result["motion"]),
        },
        "raw_and_adjusted_u_vs_motion_spearman": {
            "raw_u_mean": _simple_corr(result["u_mean"], result["motion"]),
            "adjusted_u_mean": _simple_corr(adjusted, result["motion"]),
        },
        "normalized_space_sanity": {
            "u_mean_normalized_vs_ade_normalized": _simple_corr(result["u_mean_normalized"], result["ade_normalized"]),
            "u_mean_normalized_vs_fde_normalized": _simple_corr(result["u_mean_normalized"], result["fde_normalized"]),
            "mean_u_mean_normalized": float(np.mean(result["u_mean_normalized"])),
            "mean_ade_normalized": float(np.mean(result["ade_normalized"])),
            "mean_fde_normalized": float(np.mean(result["fde_normalized"])),
        },
    }


def _simple_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(x) != len(y) or len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"rho": None, "p_value": None}
    result = spearmanr(x, y)
    value = float(result.statistic)
    pvalue = float(result.pvalue)
    return {
        "rho": value if np.isfinite(value) else None,
        "p_value": pvalue if np.isfinite(pvalue) else None,
    }


def _fit_val_cutpoints(val_result: dict[str, Any], adjusted_val: np.ndarray) -> dict[str, Any]:
    motion = val_result["motion"]
    tertiles = np.quantile(motion, [1 / 3, 2 / 3])
    deciles = np.quantile(motion, np.linspace(0, 1, 11))
    return {
        "fit_split": "internal_val",
        "slow_medium_fast_q33_q67_pixel": [float(value) for value in tertiles],
        "motion_decile_pixel_boundaries": [float(value) for value in deciles],
        "adjusted_u_tertile_q33_q67": [float(value) for value in np.quantile(adjusted_val, [1 / 3, 2 / 3])],
    }


def freeze_protocol(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.result_root, args.train_path)
    phase_b = read_json(args.result_root / "phase_b_complete.json")
    if phase_b.get("holdout_evaluated") is not False:
        raise RuntimeError("PHASE B marker indicates holdout access; refusing protocol freeze")
    protocol_path = args.result_root / "protocol_frozen.json"
    if protocol_path.exists():
        raise FileExistsError("PHASE C protocol already frozen; refusing to refit or alter it")
    if (args.result_root / "holdout_access_record.json").exists():
        raise RuntimeError("Holdout access marker already exists before PHASE C")

    base = TrainArchiveDataset(args.train_path)
    val_info = manifest["splits"]["internal_val"]
    val_indices = indices_for_split(base.scene_ids, val_info["scene_ids"])
    val_set = VideoSubsetDataset(base, val_indices, val_info["scene_ids"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models, hashes = _load_frozen_models(base, device, manifest["manifest_sha256"])
    val_result = infer_subset("internal_val", base, val_indices, val_info["scene_ids"], models, device)
    if set(val_result["scene_ids"].tolist()) != set(val_info["scene_ids"]):
        raise RuntimeError("PHASE C inference contains scenes outside internal_val")

    thresholds = freeze_val_thresholds(val_result["ade"], val_result["fde"])
    adjustment = _fit_internal_val_adjustment(val_result["motion"], val_result["u_mean"])
    adjusted_val = apply_motion_adjustment(val_result["motion"], val_result["u_mean"], adjustment)
    internal_val_report = _reliability_summary(val_result, adjusted_val, thresholds)
    cutpoints = _fit_val_cutpoints(val_result, adjusted_val)
    internal_val_report["validation_only_cutpoints"] = cutpoints
    internal_val_report["score_selection"] = "fixed by protocol: raw_u_mean, motion_only, adjusted_u_mean; no score reselected"
    internal_val_report["primary_score"] = PRIMARY_SCORE
    write_json(args.result_root / "internal_val_reliability.json", internal_val_report)
    write_json(args.result_root / "motion_adjustment.json", adjustment)

    criteria = {
        "version": "internal replication criteria v1",
        "decision_source_split": "internal_holdout only",
        "replicated_conditions": {
            "partial_spearman_u_ade_given_motion_at_least_0_15": "holdout partial rho >= 0.15",
            "adjusted_u_ade_spearman_at_least_0_15": "holdout adjusted-U vs ADE rho >= 0.15",
            "adjusted_video_cluster_ci_lower_above_zero": "video-cluster bootstrap adjusted-U rho 95% lower bound > 0",
            "adjusted_stratified_20pct_ade_at_least_5pct_better_than_within_motion_random": "adjusted-U motion-decile stratified 20% ADE <= 0.95 * 500-permutation within-motion random ADE mean",
            "at_least_two_motion_tertiles_have_raw_or_adjusted_rho_above_0_10": "among slow/medium/fast, at least two strata have raw or adjusted U vs ADE rho > 0.10",
        },
        "decision_rule": "REPLICATED if at least 4/5 conditions pass and adjusted video-cluster CI lower bound > 0; PARTIAL if 2-3 pass or cluster CI is too wide/crosses zero; NOT_REPLICATED if <=1 pass",
    }
    frozen = {
        "phase": "PHASE C; protocol frozen before holdout evaluation",
        "frozen_at_utc": utc_now(),
        "split_manifest_sha256": manifest["manifest_sha256"],
        "train_npz_sha256": manifest["train_npz_sha256"],
        "checkpoint_sha256": hashes,
        "model_seeds": list(SEEDS),
        "model_config": MODEL_CONFIG,
        "scene_mode": SCENE_MODE,
        "primary_score": PRIMARY_SCORE,
        "primary_score_definition": "mean across 15 future steps of the three models' average 2D pixel distance from their per-step ensemble mean",
        "secondary_scores_fixed": ["motion_only", "adjusted_u_mean"],
        "high_error_thresholds": thresholds,
        "motion_adjustment": adjustment,
        "motion_stratification_cutpoints": cutpoints,
        "permutation": {"within_each_motion_decile": True, "repetitions": 500, "seed": BOOTSTRAP_SEED},
        "video_cluster_bootstrap": {"cluster": "scene_id", "repetitions": 2000, "seed": BOOTSTRAP_SEED},
        "track_cluster_bootstrap": {"cluster": "(scene_id,target_id)", "repetitions": 1000, "seed": BOOTSTRAP_SEED},
        "risk_coverage_levels": list(RISK_COVERAGES),
        "success_criteria": criteria,
        "official_validation_test_used": False,
        "holdout_evaluated_after_protocol_frozen": False,
        "internal_val_reliability_summary": internal_val_report,
    }
    frozen["protocol_sha256"] = canonical_sha256(frozen)
    write_json(protocol_path, frozen)
    actual_file_hash = sha256_file(protocol_path)
    (args.result_root / "protocol_frozen.sha256").write_text(
        f"canonical_content_sha256={frozen['protocol_sha256']}\nfile_sha256={actual_file_hash}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "phase": "PHASE C frozen",
        "protocol_sha256": frozen["protocol_sha256"],
        "thresholds": thresholds,
        "motion_adjustment": adjustment,
        "internal_val_metrics": internal_val_report["metrics"],
        "holdout_evaluated": False,
    }, ensure_ascii=False, indent=2), flush=True)


def load_and_verify_frozen_protocol(result_root: Path, train_path: Path) -> dict[str, Any]:
    protocol = read_json(result_root / "protocol_frozen.json")
    expected = protocol.get("protocol_sha256")
    actual = canonical_sha256(protocol, exclude=("protocol_sha256",))
    if expected != actual:
        raise RuntimeError("Frozen protocol canonical SHA256 mismatch")
    if protocol.get("official_validation_test_used") is not False:
        raise RuntimeError("Protocol records official val/test access; holdout evaluation is blocked")
    if protocol.get("holdout_evaluated_after_protocol_frozen") is not False:
        raise RuntimeError("Unexpected pre-existing holdout evaluation state")
    manifest = load_manifest(result_root, train_path)
    if protocol["split_manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("Frozen protocol does not match the split manifest")
    return protocol


def validate_recovery_hashes(
    protocol_sha256: str,
    manifest_sha256: str,
    train_npz_sha256: str,
    checkpoint_sha256: dict[str, str],
) -> None:
    """Require the exact protocol, split, data archive, and model artifacts frozen earlier."""
    if protocol_sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("Frozen protocol SHA256 differs from the authorized recovery protocol")
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Split manifest SHA256 differs from the frozen manifest")
    if train_npz_sha256 != EXPECTED_TRAIN_NPZ_SHA256:
        raise RuntimeError("train.npz SHA256 differs from the frozen archive")
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Checkpoint SHA256 values differ from the three frozen models")


def holdout_output_presence(result_root: Path) -> dict[str, bool]:
    return {
        "internal_holdout_reliability_exists": (result_root / "internal_holdout_reliability.json").is_file(),
        "decision_exists": (result_root / "decision.json").is_file(),
        "cluster_bootstrap_exists": (result_root / "cluster_bootstrap.json").is_file(),
        "risk_coverage_exists": (result_root / "risk_coverage.json").is_file(),
    }


def validate_holdout_access_state(
    access_record: dict[str, Any] | None,
    final_outputs_present: dict[str, bool],
    recovery_requested: bool,
) -> None:
    """Enforce normal one-shot execution and the narrowly scoped crash-recovery path."""
    if not recovery_requested:
        if access_record is not None or any(final_outputs_present.values()):
            raise RuntimeError("Holdout was already started; use explicit --recover-holdout only for an incomplete run")
        return

    if access_record is None:
        raise RuntimeError("Crash recovery requires the original holdout_access_record.json")
    phase = str(access_record.get("phase", "")).lower()
    status = str(access_record.get("status", "")).lower()
    if status == "completed" or "completed" in phase:
        raise RuntimeError("Completed holdout evaluation is immutable; recovery is permanently refused")
    started = (
        status in ("started", "running")
        or "access started" in phase
        or "evaluation started" in phase
        or "evaluation running" in phase
    )
    if not started:
        raise RuntimeError("Access record does not prove that an interrupted holdout evaluation had started")
    if final_outputs_present.get("decision_exists", False):
        if all(final_outputs_present.values()):
            return  # Caller may finalize the audit record without rereading holdout.
        raise RuntimeError("decision.json exists with incomplete outputs; refusing a second holdout evaluation")


def frozen_recovery_parameters(protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return the already-frozen thresholds, adjustment, and motion cutpoints verbatim."""
    return (
        protocol["high_error_thresholds"],
        protocol["motion_adjustment"],
        protocol["motion_stratification_cutpoints"],
    )


def verify_frozen_holdout_metadata(train_path: Path, manifest: dict[str, Any]) -> None:
    """Verify the manifest's fixed 21-video/3277-sample holdout using IDs only."""
    holdout = manifest["splits"]["internal_holdout"]
    if holdout["video_count"] != 21 or holdout["sample_count"] != 3277:
        raise RuntimeError("Frozen internal_holdout must remain exactly 21 videos / 3277 samples")
    expected_ids = set(holdout["scene_ids"])
    with np.load(train_path, allow_pickle=False) as archive:
        scene_ids = archive["scene_id"].astype(str)
    actual_ids = set(scene_ids.tolist())
    sample_count = int(np.isin(scene_ids, list(expected_ids)).sum())
    if actual_ids.intersection(expected_ids) != expected_ids or sample_count != 3277:
        raise RuntimeError("Frozen holdout IDs/sample count no longer match train.npz metadata")


def begin_holdout_recovery(
    result_root: Path,
    access_record: dict[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
    train_npz_sha256: str,
    checkpoint_sha256: dict[str, str],
    final_outputs_present: dict[str, bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist an auditable recovery attempt before re-entering frozen PHASE D."""
    now = utc_now()
    audit_path = result_root / "holdout_recovery_audit.json"
    reason = "Explicit user authorization to re-execute the exact frozen PHASE D after interrupted computation."
    if audit_path.exists():
        audit = read_json(audit_path)
        for key, value in (
            ("protocol_sha256", protocol_sha256),
            ("manifest_sha256", manifest_sha256),
            ("train_npz_sha256", train_npz_sha256),
            ("checkpoint_sha256", checkpoint_sha256),
        ):
            if audit.get(key) != value:
                raise RuntimeError(f"Existing recovery audit {key} does not match current frozen artifacts")
    else:
        audit = {
            "existing_access_record": access_record,
            "protocol_sha256": protocol_sha256,
            "manifest_sha256": manifest_sha256,
            "train_npz_sha256": train_npz_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "final_outputs_present_before_recovery": final_outputs_present,
            "recovery_authorized": True,
            "recovery_authorization_reason": reason,
            "recovery_reason": "interrupted computation after holdout access marker was written",
            "protocol_changed": False,
            "checkpoint_changed": False,
            "split_changed": False,
            "threshold_changed": False,
            "adjustment_changed": False,
            "score_changed": False,
            "decision_rule_changed": False,
            "recovery_attempts": [],
        }
    attempt = {
        "attempt": len(audit["recovery_attempts"]) + 1,
        "started_at_utc": now,
        "status": "running",
        "final_outputs_present_before_attempt": final_outputs_present,
    }
    audit["recovery_attempts"].append(attempt)
    audit["status"] = "running"
    audit["recovery_authorized"] = True
    audit["last_recovery_started_at_utc"] = now
    write_json(audit_path, audit)

    updated_access = dict(access_record)
    updated_access.setdefault("initial_started_at_utc", access_record.get("started_at_utc"))
    updated_access.update({
        "phase": "PHASE D holdout evaluation running after crash recovery",
        "status": "running",
        "recovered_after_interruption": True,
        "recovery_started_at_utc": now,
        "protocol_sha256": protocol_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "holdout_evaluated_after_protocol_frozen": True,
    })
    write_json(result_root / "holdout_access_record.json", updated_access)
    return audit, updated_access


def finish_holdout_recovery(
    result_root: Path,
    audit: dict[str, Any],
    access_record: dict[str, Any],
    error: str | None = None,
) -> None:
    """Persist either a completed attempt or an interrupted state that can be resumed."""
    attempt = audit["recovery_attempts"][-1]
    if error is None:
        now = utc_now()
        attempt.update({"status": "completed", "completed_at_utc": now})
        audit.update({"status": "completed", "completed_at_utc": now})
        access_record.update({
            "phase": "PHASE D holdout evaluation completed after crash recovery",
            "status": "completed",
            "recovered_after_interruption": True,
            "completed_at_utc": now,
            "rerun_forbidden": True,
        })
    else:
        attempt.update({"status": "interrupted", "error": error, "interrupted_at_utc": utc_now()})
        audit.update({"status": "interrupted", "last_error": error})
        access_record.update({"status": "running", "last_recovery_error": error})
    write_json(result_root / "holdout_recovery_audit.json", audit)
    write_json(result_root / "holdout_access_record.json", access_record)


def _stratified_metrics(
    result: dict[str, Any],
    adjusted: np.ndarray,
    thresholds: dict[str, float],
    cutpoints: dict[str, Any],
) -> dict[str, Any]:
    q33, q67 = cutpoints["slow_medium_fast_q33_q67_pixel"]
    motion = result["motion"]
    strata = (
        ("slow", motion <= q33),
        ("medium", (motion > q33) & (motion <= q67)),
        ("fast", motion > q67),
    )
    tertiles = []
    for name, mask in strata:
        tertiles.append({
            "stratum": name,
            "sample_count": int(mask.sum()),
            "raw_u_vs_ade": _simple_corr(result["u_mean"][mask], result["ade"][mask]),
            "adjusted_u_vs_ade": _simple_corr(adjusted[mask], result["ade"][mask]),
            "motion_vs_ade": _simple_corr(motion[mask], result["ade"][mask]),
            "raw_u_vs_fde": _simple_corr(result["u_mean"][mask], result["fde"][mask]),
            "adjusted_u_vs_fde": _simple_corr(adjusted[mask], result["fde"][mask]),
        })
    boundaries = np.asarray(cutpoints["motion_decile_pixel_boundaries"], dtype=np.float64)
    decile_ids = assign_quantile_strata(motion, boundaries)
    deciles = []
    for index in range(10):
        mask = decile_ids == index
        deciles.append({
            "decile": index + 1,
            "lower_motion_pixel": float(boundaries[index]),
            "upper_motion_pixel": float(boundaries[index + 1]),
            "sample_count": int(mask.sum()),
            "raw_u_vs_ade": _simple_corr(result["u_mean"][mask], result["ade"][mask]),
            "adjusted_u_vs_ade": _simple_corr(adjusted[mask], result["ade"][mask]),
        })
    return {
        "cutpoints_fit_split": "internal_val",
        "slow_medium_fast_cutpoints_pixel": {"q33": q33, "q67": q67},
        "slow_medium_fast": tertiles,
        "motion_decile_boundaries_pixel": boundaries.tolist(),
        "motion_deciles": deciles,
        "test_own_quantiles_used": False,
    }


def _decision_from_holdout(
    holdout_report: dict[str, Any],
    stratified_risk: dict[str, Any],
    permutation: dict[str, Any],
    cluster: dict[str, Any],
) -> dict[str, Any]:
    partial_ade = holdout_report["partial_spearman_raw_u_given_motion"]["u_mean_vs_ade"]
    adjusted_rho = holdout_report["metrics"]["adjusted_u_mean"]["spearman_vs_ade"]["rho"]
    adjusted_ci_lower = cluster["metrics"]["adjusted_u_mean"]["spearman_vs_ade"]["lower_95"]
    adjusted_20 = stratified_risk["adjusted_u_mean"][ -1 ]
    permutation_20 = next(
        row for row in permutation["rows"] if row["nominal_coverage"] == 0.2
    )
    random_ade = permutation_20["randomized_within_motion_mean_sample_std"]["ade"]["mean"]
    relative_gain = (random_ade - adjusted_20["ade"]) / random_ade if random_ade else None
    tertiles = holdout_report["motion_stratified_metrics"]["slow_medium_fast"]
    strata_signal_count = sum(
        1
        for item in tertiles
        if (
            (item["raw_u_vs_ade"]["rho"] is not None and item["raw_u_vs_ade"]["rho"] > 0.10)
            or (item["adjusted_u_vs_ade"]["rho"] is not None and item["adjusted_u_vs_ade"]["rho"] > 0.10)
        )
    )
    conditions = {
        "partial_spearman_u_ade_given_motion_ge_0_15": partial_ade >= 0.15,
        "adjusted_u_ade_spearman_ge_0_15": adjusted_rho is not None and adjusted_rho >= 0.15,
        "video_cluster_adjusted_rho_ci_lower_gt_0": adjusted_ci_lower is not None and adjusted_ci_lower > 0,
        "adjusted_stratified_20pct_ade_at_least_5pct_better_than_permutation_mean": relative_gain is not None and relative_gain >= 0.05,
        "at_least_two_motion_tertiles_raw_or_adjusted_rho_gt_0_10": strata_signal_count >= 2,
    }
    passed = int(sum(conditions.values()))
    if passed >= 4 and conditions["video_cluster_adjusted_rho_ci_lower_gt_0"]:
        decision = "REPLICATED"
    elif passed <= 1:
        decision = "NOT_REPLICATED"
    else:
        decision = "PARTIAL"
    return {
        "decision": decision,
        "decision_source_split": "internal_holdout only",
        "official_validation_test_used_in_decision": False,
        "conditions": conditions,
        "conditions_passed": passed,
        "conditions_total": len(conditions),
        "inputs": {
            "partial_spearman_u_ade_given_motion": partial_ade,
            "partial_spearman_u_fde_given_motion": holdout_report["partial_spearman_raw_u_given_motion"]["u_mean_vs_fde"],
            "adjusted_u_vs_ade_rho": adjusted_rho,
            "adjusted_video_cluster_rho_ci_lower": adjusted_ci_lower,
            "adjusted_stratified_20pct_ade": adjusted_20["ade"],
            "within_motion_random_20pct_ade_mean": random_ade,
            "adjusted_stratified_20pct_relative_ade_gain": relative_gain,
            "tertiles_with_raw_or_adjusted_rho_gt_0_10": strata_signal_count,
        },
        "rule": "REPLICATED requires at least 4/5 frozen criteria and adjusted video-cluster rho CI lower bound > 0; PARTIAL for 2-3 criteria or insufficient cluster certainty; NOT_REPLICATED for <=1 criterion.",
        "holdout_evaluated_after_protocol_frozen": True,
    }


def evaluate_holdout_once(args: argparse.Namespace, recover_holdout: bool = False) -> None:
    manifest = load_manifest(args.result_root, args.train_path)
    protocol = load_and_verify_frozen_protocol(args.result_root, args.train_path)
    phase_b_path = args.result_root / "phase_b_complete.json"
    if not phase_b_path.is_file():
        raise RuntimeError("Cannot evaluate holdout before all three training runs complete")
    phase_b = read_json(phase_b_path)

    current_checkpoint_hashes = {str(seed): sha256_file(checkpoint_path(seed)) for seed in SEEDS}
    train_hash = sha256_file(args.train_path)
    validate_recovery_hashes(
        protocol["protocol_sha256"], manifest["manifest_sha256"], train_hash, current_checkpoint_hashes
    )
    if current_checkpoint_hashes != protocol["checkpoint_sha256"] or current_checkpoint_hashes != {
        seed: value["sha256"] for seed, value in phase_b["checkpoints"].items()
    }:
        raise RuntimeError("A checkpoint differs from PHASE B/PHASE C; holdout evaluation is blocked")
    if protocol["train_npz_sha256"] != train_hash:
        raise RuntimeError("train.npz changed after protocol freeze")
    verify_frozen_holdout_metadata(args.train_path, manifest)

    access_path = args.result_root / "holdout_access_record.json"
    access_record = read_json(access_path) if access_path.exists() else None
    final_outputs_present = holdout_output_presence(args.result_root)
    validate_holdout_access_state(access_record, final_outputs_present, recover_holdout)

    # If a prior process atomically wrote every final artifact and crashed before
    # updating the access record, finalize the audit without reading holdout again.
    if recover_holdout and final_outputs_present["decision_exists"]:
        required = all(final_outputs_present.values())
        report = read_json(args.result_root / "internal_holdout_reliability.json") if required else {}
        decision = read_json(args.result_root / "decision.json")
        if not required or report.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError("Final decision/output state is incomplete or tied to another frozen protocol")
        if decision.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError("Existing decision does not match the frozen protocol")
        now = utc_now()
        access_record.update({
            "phase": "PHASE D holdout evaluation completed after crash recovery",
            "status": "completed",
            "recovered_after_interruption": True,
            "completed_at_utc": now,
            "rerun_forbidden": True,
        })
        write_json(access_path, access_record)
        audit_path = args.result_root / "holdout_recovery_audit.json"
        if audit_path.exists():
            audit = read_json(audit_path)
            audit.update({"status": "completed", "completed_at_utc": now, "finalized_without_rerun": True})
            write_json(audit_path, audit)
        print(json.dumps({"phase": "PHASE D recovery", "status": "already-computed outputs finalized without holdout reread"}, indent=2))
        return

    recovery_audit = None
    if recover_holdout:
        if access_record is None:
            raise RuntimeError("Crash recovery requires the original holdout access record")
        recovery_audit, access_record = begin_holdout_recovery(
            args.result_root,
            access_record,
            protocol["protocol_sha256"],
            manifest["manifest_sha256"],
            train_hash,
            current_checkpoint_hashes,
            final_outputs_present,
        )
    else:
        access_record = {
            "phase": "PHASE D holdout access started",
            "status": "started",
            "started_at_utc": utc_now(),
            "protocol_sha256": protocol["protocol_sha256"],
            "checkpoint_sha256": current_checkpoint_hashes,
            "holdout_evaluated_after_protocol_frozen": True,
            "rerun_forbidden": True,
        }
        write_json(access_path, access_record)

    print(json.dumps({
        "phase": "PHASE D",
        "status": "crash recovery started" if recover_holdout else "holdout access started once",
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }), flush=True)

    try:
        base = TrainArchiveDataset(args.train_path)
        holdout_info = manifest["splits"]["internal_holdout"]
        holdout_indices = indices_for_split(base.scene_ids, holdout_info["scene_ids"])
        if len(holdout_indices) != 3277 or set(base.scene_ids[holdout_indices].tolist()) != set(holdout_info["scene_ids"]):
            raise RuntimeError("Holdout rows/videos differ from the frozen manifest")
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        models, hashes = _load_frozen_models(base, device, manifest["manifest_sha256"])
        if hashes != current_checkpoint_hashes:
            raise RuntimeError("Checkpoint hashes changed while preparing the holdout models")
        result = infer_subset(
            "internal_holdout", base, holdout_indices, holdout_info["scene_ids"], models, device
        )
        thresholds, adjustment, val_cutpoints = frozen_recovery_parameters(protocol)
        adjusted = apply_motion_adjustment(result["motion"], result["u_mean"], adjustment)
        report = _reliability_summary(result, adjusted, thresholds)
        report["motion_stratified_metrics"] = _stratified_metrics(result, adjusted, thresholds, val_cutpoints)
        report["gt_future_endpoint_displacement_diagnostic_only"] = {
            "raw_u_mean": _simple_corr(result["u_mean"], result["future_endpoint_displacement_diagnostic_only"]),
            "adjusted_u_mean": _simple_corr(adjusted, result["future_endpoint_displacement_diagnostic_only"]),
            "note": "post-hoc only; not used in score, adjustment, threshold, or decision",
        }

        global_risk = global_risk_curves(
            {"raw_u_mean": result["u_mean"], "adjusted_u_mean": adjusted, "motion_only": result["motion"]},
            result["ade"], result["fde"], seed=protocol["permutation"]["seed"],
        )
        motion_deciles = assign_quantile_strata(
            result["motion"], np.asarray(val_cutpoints["motion_decile_pixel_boundaries"])
        )
        stratified = {
            "raw_u_mean": stratified_risk_curve(result["u_mean"], motion_deciles, result["ade"], result["fde"]),
            "adjusted_u_mean": stratified_risk_curve(adjusted, motion_deciles, result["ade"], result["fde"]),
        }
        permutation = within_motion_permutation_test(
            result["u_mean"], adjusted, motion_deciles, result["ade"], result["fde"],
            repetitions=protocol["permutation"]["repetitions"], seed=protocol["permutation"]["seed"],
        )
        video_bootstrap = cluster_bootstrap_metrics(
            result["scene_ids"],
            {"raw_u_mean": result["u_mean"], "adjusted_u_mean": adjusted, "motion_only": result["motion"]},
            result["ade"], result["ade"] >= thresholds["high_ade_pixel_threshold"],
            repetitions=protocol["video_cluster_bootstrap"]["repetitions"],
            seed=protocol["video_cluster_bootstrap"]["seed"],
        )
        video_bootstrap["cluster_level"] = "scene_id video"
        track_ids = np.char.add(np.char.add(result["scene_ids"].astype(str), "::"), result["target_ids"].astype(str))
        track_bootstrap = cluster_bootstrap_metrics(
            track_ids,
            {"raw_u_mean": result["u_mean"], "adjusted_u_mean": adjusted},
            result["ade"], result["ade"] >= thresholds["high_ade_pixel_threshold"],
            repetitions=protocol["track_cluster_bootstrap"]["repetitions"],
            seed=protocol["track_cluster_bootstrap"]["seed"],
        )
        track_bootstrap["cluster_level"] = "(scene_id,target_id) pedestrian track; secondary only"

        report["sample_order"] = result["sample_order"]
        report["primary_score"] = protocol["primary_score"]
        report["split_manifest_sha256"] = manifest["manifest_sha256"]
        report["protocol_sha256"] = protocol["protocol_sha256"]
        report["holdout_evaluated_after_protocol_frozen"] = True
        report["official_validation_test_used"] = False
        report["high_error_threshold_source"] = "internal_val frozen pixel 80th percentile"
        report["motion_adjustment_source"] = "internal_val frozen coefficients; no refit on holdout"
        report["shared_ordered_inference"] = True
        report["ensemble_risk_coverage_global"] = global_risk
        report["ensemble_risk_coverage_motion_stratified"] = {
            "strata": "internal_val motion deciles frozen before holdout",
            "curves": stratified,
            "within_motion_permutation": permutation,
        }
        report["normalized_space_sanity"] = {
            "u_mean_normalized_vs_ade_normalized": _simple_corr(result["u_mean_normalized"], result["ade_normalized"]),
            "u_mean_normalized_vs_fde_normalized": _simple_corr(result["u_mean_normalized"], result["fde_normalized"]),
            "mean_u_mean_normalized": float(np.mean(result["u_mean_normalized"])),
            "mean_ade_normalized": float(np.mean(result["ade_normalized"])),
            "mean_fde_normalized": float(np.mean(result["fde_normalized"])),
        }
        decision = _decision_from_holdout(report, stratified, permutation, video_bootstrap)
        decision["protocol_sha256"] = protocol["protocol_sha256"]
        report["decision"] = decision

        # Every file is atomically replaced; decision.json is the final completion sentinel.
        write_json(args.result_root / "internal_holdout_reliability.json", report)
        write_json(args.result_root / "risk_coverage.json", {
            "global": global_risk,
            "motion_stratified": report["ensemble_risk_coverage_motion_stratified"],
            "fixed_motion_deciles_from_internal_val": val_cutpoints["motion_decile_pixel_boundaries"],
        })
        write_json(args.result_root / "cluster_bootstrap.json", {
            "video_level_primary": video_bootstrap,
            "track_level_secondary": track_bootstrap,
            "high_ade_threshold_pixel": thresholds["high_ade_pixel_threshold"],
        })
        for seed in SEEDS:
            metrics_path = output_dir(seed) / "metrics.json"
            metrics = read_json(metrics_path)
            per_model = report["individual_model_performance"][str(seed)]
            metrics["internal_holdout"] = {
                "trajectory_ade_pixel": per_model["ade_pixel"],
                "trajectory_fde_pixel": per_model["fde_pixel"],
                "trajectory_ade_normalized": per_model["ade_normalized"],
                "trajectory_fde_normalized": per_model["fde_normalized"],
                "evaluated_after_all_checkpoints_and_protocol_frozen": True,
            }
            write_json(metrics_path, metrics)
        write_json(args.result_root / "decision.json", decision)

        if recover_holdout:
            finish_holdout_recovery(args.result_root, recovery_audit, access_record)
        else:
            access_record.update({
                "phase": "PHASE D holdout evaluation completed once",
                "completed_at_utc": utc_now(),
                "status": "completed",
                "rerun_forbidden": True,
            })
            write_json(access_path, access_record)
        print(json.dumps({
            "phase": "PHASE D complete",
            "decision": decision["decision"],
            "holdout_ensemble": report["ensemble_mean_performance"],
            "partial_spearman": report["partial_spearman_raw_u_given_motion"],
            "sample_order_sha256": result["sample_order"]["sha256"],
            "results": str(args.result_root / "internal_holdout_reliability.json"),
        }, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if recover_holdout:
            finish_holdout_recovery(args.result_root, recovery_audit, access_record, error=error)
        else:
            access_record.update({"status": "running", "last_error": error})
            write_json(access_path, access_record)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "freeze", "evaluate-holdout"))
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--recover-holdout", action="store_true", help="resume only an interrupted frozen PHASE D")
    args = parser.parse_args()
    if args.recover_holdout and args.phase != "evaluate-holdout":
        parser.error("--recover-holdout is only valid with evaluate-holdout")
    if args.phase == "prepare":
        manifest = prepare_manifest(args.train_path, args.result_root)
        print(json.dumps({
            "phase": "PHASE A complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "video_counts": {key: item["video_count"] for key, item in manifest["splits"].items()},
            "sample_counts": {key: item["sample_count"] for key, item in manifest["splits"].items()},
            "overlaps": manifest["scene_overlap_counts"],
            "official_val_test_accessed": False,
        }, ensure_ascii=False, indent=2), flush=True)
    elif args.phase == "train":
        train_all(args)
    elif args.phase == "freeze":
        freeze_protocol(args)
    else:
        evaluate_holdout_once(args, recover_holdout=args.recover_holdout)


if __name__ == "__main__":
    main()
