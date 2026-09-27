#!/usr/bin/env python3
"""Train video-cross-fitted trajectory models and build intention feature caches.

The test split is deliberately loaded without its intent labels. Its features and
trajectory-only audit are permitted before the intention protocol is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.trajectory_transformer import SceneTrajectoryTransformer

DATA_ROOT = PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15"
OUTPUT_ROOT = PROJECT_ROOT / "results/reliability_gated_intent_15x15"
CACHE_ROOT = OUTPUT_ROOT / "cache"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
FULL_SEEDS = (42, 123, 2024)
FOLD_SEED = 424242
TRAJECTORY_CONFIG = {
    "input_dim": 8,
    "d_model": 128,
    "num_layers": 3,
    "nhead": 4,
    "obs_len": 15,
    "pred_len": 15,
    "dropout": 0.1,
    "epochs": 20,
    "batch_size": 512,
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "selection": "lowest official validation pixel ADE",
    "scene_mode": "zero",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_scene_ids(scene_ids: list[str]) -> str:
    return hashlib.sha256(canonical_json(scene_ids)).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


class FeatureSplit(Dataset):
    """Read only trajectory inputs/targets and identity; never reads intent labels."""

    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as raw:
            required = (
                "target_obs", "target_abs_obs", "future_gt", "scene_id", "target_id",
                "obs_end_frame", "image_size", "scene_feat",
            )
            missing = [key for key in required if key not in raw.files]
            if missing:
                raise KeyError(f"{path} missing required fields: {missing}")
            self.target_obs = raw["target_obs"].astype(np.float32, copy=True)
            self.target_abs_obs = raw["target_abs_obs"].astype(np.float32, copy=True)
            self.future_gt = raw["future_gt"].astype(np.float32, copy=True)
            self.scene_ids = raw["scene_id"].astype(str)
            self.target_ids = raw["target_id"].astype(str)
            self.obs_end_frame = raw["obs_end_frame"].astype(np.int64, copy=True)
            self.image_size = raw["image_size"].astype(np.float32, copy=True)
            self.scene_dim = int(raw["scene_feat"].shape[-1])
        if self.target_obs.shape[1:] != (15, 4) or self.target_abs_obs.shape[1:] != (15, 4):
            raise ValueError(f"Expected 15x4 observed tensors in {path}")
        if self.future_gt.shape[1:] != (15, 2):
            raise ValueError(f"Expected 15x2 future_gt in {path}")

    def __len__(self) -> int:
        return len(self.target_obs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "target_obs": torch.from_numpy(self.target_obs[index]),
            "target_abs_obs": torch.from_numpy(self.target_abs_obs[index]),
            "future_gt": torch.from_numpy(self.future_gt[index]),
            "image_size": torch.from_numpy(self.image_size[index]),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(scene_ids: np.ndarray, seed: int = FOLD_SEED) -> list[list[str]]:
    unique = sorted(set(np.asarray(scene_ids).astype(str).tolist()))
    rng = np.random.RandomState(seed)
    shuffled = np.asarray(unique, dtype=object)
    rng.shuffle(shuffled)
    return [list(map(str, fold.tolist())) for fold in np.array_split(shuffled, 3)]


def make_manifest(train: FeatureSplit, train_path: Path, val_path: Path) -> dict[str, Any]:
    folds = make_folds(train.scene_ids)
    train_ids = set(np.unique(train.scene_ids).tolist())
    fold_sets = [set(fold) for fold in folds]
    if set.union(*fold_sets) != train_ids or any(fold_sets[i] & fold_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Video folds do not form a disjoint exhaustive partition")
    payload: dict[str, Any] = {
        "protocol": "3-fold video-level trajectory cross-fitting",
        "group_key": "scene_id",
        "random_seed": FOLD_SEED,
        "train_npz_sha256": sha256_file(train_path),
        "validation_npz_sha256": sha256_file(val_path),
        "train_sample_count": len(train),
        "train_video_count": len(train_ids),
        "folds": [
            {
                "fold": index,
                "video_count": len(fold),
                "sample_count": int(np.isin(train.scene_ids, fold).sum()),
                "scene_ids": fold,
                "scene_ids_sha256": sha256_scene_ids(fold),
            }
            for index, fold in enumerate(folds)
        ],
        "trajectory_config": TRAJECTORY_CONFIG,
        "trajectory_seeds": list(FULL_SEEDS),
        "heldout_fold_used_for_checkpoint_selection": False,
        "checkpoint_selection_split": "official_val",
        "official_test_used": False,
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def trajectory_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
    scene = torch.zeros((target.shape[0], int(batch.get("scene_dim", 512))), device=device)
    return target, scene


def evaluate_trajectory(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int = 512) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    predictions: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            pred = model(target, torch.zeros((len(target), model.scene_encoder[0].in_features), device=device))
            predictions.append(pred.cpu().numpy())
            errors.append(batch["future_gt"].numpy())
    pred = np.concatenate(predictions)
    gt = np.concatenate(errors)
    if isinstance(dataset, torch.utils.data.Subset):
        scales = dataset.dataset.image_size[np.asarray(dataset.indices)]
    else:
        scales = dataset.image_size
    per_horizon_pixel = np.linalg.norm((pred - gt) * scales[:, None, :], axis=-1)
    return {
        "prediction": pred,
        "ade_pixel": float(per_horizon_pixel.mean()),
        "fde_pixel": float(per_horizon_pixel[:, -1].mean()),
    }


def train_crossfit_checkpoint(
    fold_index: int,
    seed: int,
    train_set: FeatureSplit,
    val_set: FeatureSplit,
    train_indices: np.ndarray,
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = CHECKPOINT_ROOT / f"crossfit_traj_fold{fold_index}_seed{seed}.pt"
    expected = {
        "task_tag": "reliability_gated_intent_15x15_crossfit_v1",
        "fold": fold_index,
        "seed": seed,
        "crossfit_manifest_sha256": manifest["manifest_sha256"],
    }
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise FileExistsError(f"Existing checkpoint has different protocol metadata: {checkpoint_path}")
        return payload

    seed_everything(seed)
    train_subset = torch.utils.data.Subset(train_set, train_indices.tolist())
    train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False)
    model = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=train_set.scene_dim,
        d_model=128,
        nhead=4,
        num_layers=3,
        pred_len=15,
        dropout=0.1,
        max_obs_len=15,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_val_ade = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, 21):
        model.train()
        loss_total = 0.0
        for batch in train_loader:
            target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
            future = batch["future_gt"].to(device)
            scene = torch.zeros((len(target), train_set.scene_dim), device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(target, scene)
            loss = nn.functional.smooth_l1_loss(pred, future)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach()) * len(target)

        model.eval()
        val_errors: list[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
                future = batch["future_gt"].numpy()
                scene = torch.zeros((len(target), train_set.scene_dim), device=device)
                pred = model(target, scene).cpu().numpy()
                val_errors.append(np.linalg.norm((pred - future) * batch["image_size"].numpy()[:, None, :], axis=-1))
        val_ade = float(np.concatenate(val_errors).mean())
        scheduler.step(val_ade)
        history.append({"epoch": epoch, "train_loss": loss_total / len(train_subset), "val_ade_pixel": val_ade})
        print(json.dumps({"fold": fold_index, "seed": seed, "epoch": epoch, "val_ade_pixel": val_ade}, ensure_ascii=False), flush=True)
        if val_ade < best_val_ade:
            best_val_ade, best_epoch = val_ade, epoch
            payload = {
                **expected,
                "model": model.state_dict(),
                "args": {**TRAJECTORY_CONFIG, "seed": seed, "fold": fold_index},
                "scene_mode": "zero",
                "best_epoch": best_epoch,
                "best_val_ade_pixel": best_val_ade,
                "train_scene_ids_sha256": sha256_scene_ids(sorted(np.unique(train_set.scene_ids[train_indices]).tolist())),
                "heldout_scene_ids_sha256": manifest["folds"][fold_index]["scene_ids_sha256"],
                "validation_npz_sha256": manifest["validation_npz_sha256"],
                "history": history.copy(),
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path)

    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_trajectory_checkpoint(path: Path, seed: int, scene_dim: int, device: torch.device, *, crossfit: bool = False, fold: int | None = None, manifest_sha: str | None = None) -> tuple[SceneTrajectoryTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("scene_mode", payload.get("args", {}).get("scene_mode")) != "zero":
        raise ValueError(f"Expected a zero-scene trajectory checkpoint: {path}")
    if int(payload.get("args", {}).get("seed", -1)) != seed:
        raise ValueError(f"Seed metadata mismatch: {path}")
    if crossfit:
        expected = {
            "task_tag": "reliability_gated_intent_15x15_crossfit_v1",
            "fold": fold,
            "seed": seed,
            "crossfit_manifest_sha256": manifest_sha,
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Cross-fit metadata mismatch: {path}")
    model = SceneTrajectoryTransformer(
        input_dim=8, scene_dim=scene_dim, d_model=128, nhead=4, num_layers=3,
        pred_len=15, dropout=0.1, max_obs_len=15,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model, payload


def predict_subset(model: nn.Module, dataset: FeatureSplit, indices: np.ndarray, device: torch.device) -> tuple[np.ndarray, float, float]:
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    result = evaluate_trajectory(model, subset, device)
    return result["prediction"], result["ade_pixel"], result["fde_pixel"]


def ensemble_features(predictions: np.ndarray, dataset: FeatureSplit, indices: np.ndarray) -> dict[str, np.ndarray]:
    """Convert ordered [3,N,15,2] predictions into the permitted feature cache."""
    if predictions.shape != (3, len(indices), 15, 2):
        raise ValueError(f"Unexpected ensemble prediction shape: {predictions.shape}")
    mean_prediction = predictions.mean(axis=0)
    sizes = dataset.image_size[indices]
    disagreement = np.linalg.norm((predictions - mean_prediction[None]) * sizes[None, :, None, :], axis=-1).mean(axis=0).mean(axis=-1)
    abs_center_pixels = dataset.target_abs_obs[indices, :, :2] * sizes[:, None, :]
    motion = np.linalg.norm(abs_center_pixels[:, -1] - abs_center_pixels[:, 0], axis=-1)
    return {
        "sample_index": indices.astype(np.int64),
        "scene_id": dataset.scene_ids[indices],
        "target_id": dataset.target_ids[indices],
        "obs_end_frame": dataset.obs_end_frame[indices],
        "target_obs": np.concatenate([dataset.target_obs[indices], dataset.target_abs_obs[indices]], axis=-1).astype(np.float32),
        "future_pred_mean": mean_prediction.astype(np.float32),
        "u_mean_pixel": disagreement.astype(np.float32),
        "observed_motion_pixel": motion.astype(np.float32),
        "image_size": dataset.image_size[indices].astype(np.int32),
    }


def trajectory_audit(prediction: np.ndarray, gt: np.ndarray, image_size: np.ndarray) -> dict[str, float]:
    pixel_error = np.linalg.norm((prediction - gt) * image_size[:, None, :], axis=-1)
    pred_displacement = np.linalg.norm(prediction * image_size[:, None, :], axis=-1)
    return {
        "ensemble_ade_pixel": float(pixel_error.mean()),
        "ensemble_fde_pixel": float(pixel_error[:, -1].mean()),
        "future_prediction_displacement_mean_pixel": float(pred_displacement.mean()),
        "future_prediction_endpoint_displacement_mean_pixel": float(np.linalg.norm(prediction[:, -1] * image_size, axis=-1).mean()),
        "sample_count": int(len(prediction)),
    }


def add_stats(audit: dict[str, Any], features: dict[str, np.ndarray]) -> None:
    for key, values in (("u_mean_pixel", features["u_mean_pixel"]), ("observed_motion_pixel", features["observed_motion_pixel"])):
        audit[key] = {
            "mean": float(np.mean(values)), "std": float(np.std(values)), "median": float(np.median(values)),
        }


def main() -> None:
    global OUTPUT_ROOT, CACHE_ROOT, CHECKPOINT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("crossfit", "features", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_root
    CACHE_ROOT = OUTPUT_ROOT / "cache"
    CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    train_path, val_path, test_path = (args.data_root / f"{name}.npz" for name in ("train", "val", "test"))
    for path in (train_path, val_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    train_set, val_set = FeatureSplit(train_path), FeatureSplit(val_path)
    with np.load(train_path, allow_pickle=False) as raw:
        train_labels = raw["intent_label"].astype(np.int64, copy=True)
        train_crossing_labels = raw["crossing_label"].astype(np.int64, copy=True)
    with np.load(val_path, allow_pickle=False) as raw:
        val_labels = raw["intent_label"].astype(np.int64, copy=True)
        val_crossing_labels = raw["crossing_label"].astype(np.int64, copy=True)
    if not all(set(np.unique(labels)).issubset({0, 1}) for labels in (train_labels, train_crossing_labels, val_labels, val_crossing_labels)):
        raise ValueError("This protocol requires clean intent labels 0/1 only")
    manifest = make_manifest(train_set, train_path, val_path)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_ROOT / "crossfit_manifest.json", manifest)
    if args.stage in ("crossfit", "all"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_reports: list[dict[str, Any]] = []
        fold_id_arrays = [np.asarray(fold["scene_ids"]) for fold in manifest["folds"]]
        for fold_index, heldout_ids in enumerate(fold_id_arrays):
            heldout_mask = np.isin(train_set.scene_ids, heldout_ids)
            heldout_indices = np.flatnonzero(heldout_mask)
            fit_indices = np.flatnonzero(~heldout_mask)
            if set(train_set.scene_ids[fit_indices]) & set(train_set.scene_ids[heldout_indices]):
                raise RuntimeError("Leakage in cross-fit scene partition")
            for seed in FULL_SEEDS:
                payload = train_crossfit_checkpoint(fold_index, seed, train_set, val_set, fit_indices, manifest, device)
                fold_reports.append({
                    "fold": fold_index, "seed": seed, "heldout_video_count": len(heldout_ids),
                    "heldout_sample_count": len(heldout_indices), "best_epoch": int(payload["best_epoch"]),
                    "best_val_ade_pixel": float(payload["best_val_ade_pixel"]),
                    "checkpoint": str((CHECKPOINT_ROOT / f"crossfit_traj_fold{fold_index}_seed{seed}.pt").relative_to(PROJECT_ROOT)),
                    "checkpoint_sha256": sha256_file(CHECKPOINT_ROOT / f"crossfit_traj_fold{fold_index}_seed{seed}.pt"),
                    "train_scene_ids_sha256": payload["train_scene_ids_sha256"],
                    "heldout_scene_ids_sha256": payload["heldout_scene_ids_sha256"],
                })
        write_json(OUTPUT_ROOT / "trajectory_crossfit_checkpoints.json", {"models": fold_reports})

    if args.stage in ("features", "all"):
        if not (OUTPUT_ROOT / "trajectory_crossfit_checkpoints.json").is_file():
            raise FileNotFoundError("Run --stage crossfit first")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_id_arrays = [np.asarray(fold["scene_ids"]) for fold in manifest["folds"]]
        oof_prediction = np.empty((3, len(train_set), 15, 2), dtype=np.float32)
        crossfit_heldout_audits: list[dict[str, Any]] = []
        for fold_index, heldout_ids in enumerate(fold_id_arrays):
            heldout_indices = np.flatnonzero(np.isin(train_set.scene_ids, heldout_ids))
            for seed_index, seed in enumerate(FULL_SEEDS):
                path = CHECKPOINT_ROOT / f"crossfit_traj_fold{fold_index}_seed{seed}.pt"
                model, payload = load_trajectory_checkpoint(path, seed, train_set.scene_dim, device, crossfit=True, fold=fold_index, manifest_sha=manifest["manifest_sha256"])
                prediction, ade, fde = predict_subset(model, train_set, heldout_indices, device)
                oof_prediction[seed_index, heldout_indices] = prediction
                crossfit_heldout_audits.append({"fold": fold_index, "seed": seed, "heldout_ade_pixel": ade, "heldout_fde_pixel": fde, "best_epoch": int(payload["best_epoch"])})
        checkpoint_rows = fold_reports_from_json(OUTPUT_ROOT)
        write_json(OUTPUT_ROOT / "trajectory_crossfit_summary.json", {
            "crossfit_manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_selection_split": "official_val",
            "heldout_fold_used_for_selection": False,
            "official_test_used_for_selection": False,
            "models": [
                {
                    **row,
                    "heldout_ade_pixel": next(item["heldout_ade_pixel"] for item in crossfit_heldout_audits if item["fold"] == row["fold"] and item["seed"] == row["seed"]),
                    "heldout_fde_pixel": next(item["heldout_fde_pixel"] for item in crossfit_heldout_audits if item["fold"] == row["fold"] and item["seed"] == row["seed"]),
                }
                for row in checkpoint_rows
            ],
        })
        train_indices = np.arange(len(train_set))
        train_features = ensemble_features(oof_prediction, train_set, train_indices)
        train_features["intent_label"] = train_labels
        train_features["crossing_label"] = train_crossing_labels
        train_features["fold_index"] = np.array([next(i for i, ids in enumerate(fold_id_arrays) if scene in set(ids)) for scene in train_set.scene_ids], dtype=np.int8)
        train_prediction_mean = oof_prediction.mean(axis=0)
        train_audit = trajectory_audit(train_prediction_mean, train_set.future_gt, train_set.image_size)
        add_stats(train_audit, train_features)
        np.savez_compressed(CACHE_ROOT / "train_oof_features.npz", **train_features)

        full_paths = [PROJECT_ROOT / f"checkpoints/trajectory_transformer_zero_scene_15x15_seed{seed}.pt" for seed in FULL_SEEDS]
        full_hashes: dict[str, str] = {}
        full_models: list[SceneTrajectoryTransformer] = []
        for seed, path in zip(FULL_SEEDS, full_paths):
            model, payload = load_trajectory_checkpoint(path, seed, val_set.scene_dim, device)
            full_models.append(model)
            full_hashes[str(seed)] = sha256_file(path)
            expected_args = payload.get("args", {})
            if int(expected_args.get("d_model", -1)) != 128 or int(expected_args.get("num_layers", -1)) != 3 or int(expected_args.get("epochs", -1)) != 20:
                raise ValueError(f"Unexpected full-train checkpoint configuration: {path}")

        def full_ensemble(dataset: FeatureSplit, labels: np.ndarray | None, cache_name: str, *, keep_labels: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
            indices = np.arange(len(dataset))
            predictions: list[np.ndarray] = []
            for model in full_models:
                result = evaluate_trajectory(model, dataset, device)
                predictions.append(result["prediction"])
            model_predictions = np.stack(predictions)
            features = ensemble_features(model_predictions, dataset, indices)
            if keep_labels:
                if labels is None:
                    raise RuntimeError("Labels must be provided when keep_labels=True")
                features["intent_label"] = labels
            audit = trajectory_audit(model_predictions.mean(axis=0), dataset.future_gt, dataset.image_size)
            add_stats(audit, features)
            np.savez_compressed(CACHE_ROOT / cache_name, **features)
            return features, audit

        val_features, val_audit = full_ensemble(val_set, val_labels, "val_features.npz", keep_labels=True)
        # Keep the descriptive crossing field distinct from the intention target in the cache.
        with np.load(CACHE_ROOT / "val_features.npz", allow_pickle=False) as current:
            val_saved = {key: current[key].copy() for key in current.files}
        val_saved["crossing_label"] = val_crossing_labels
        np.savez_compressed(CACHE_ROOT / "val_features.npz", **val_saved)
        # Intent labels are purposefully absent from FeatureSplit and the cache at this stage.
        test_set = FeatureSplit(test_path)
        _, test_audit = full_ensemble(test_set, None, "test_features_unlabeled.npz", keep_labels=False)

        cache_hashes = {path.name: sha256_file(path) for path in sorted(CACHE_ROOT.glob("*.npz"))}
        cache_manifest = {
            "protocol": "OOF train; frozen full-train ensemble for val/test",
            "crossfit_manifest_sha256": manifest["manifest_sha256"],
            "train_npz_sha256": sha256_file(train_path),
            "validation_npz_sha256": sha256_file(val_path),
            "test_npz_sha256": sha256_file(test_path),
            "full_train_zero_scene_checkpoint_sha256": full_hashes,
            "trajectory_crossfit_checkpoint_sha256": {
                row["checkpoint"]: row["checkpoint_sha256"] for row in fold_reports_from_json(OUTPUT_ROOT)
            },
            "trajectory_crossfit_summary_sha256": sha256_file(OUTPUT_ROOT / "trajectory_crossfit_summary.json"),
            "feature_cache_sha256": cache_hashes,
            "feature_fields_pre_freeze": ["sample_index", "scene_id", "target_id", "obs_end_frame", "target_obs", "future_pred_mean", "u_mean_pixel", "observed_motion_pixel", "image_size"],
            "intent_and_crossing_labels_saved": {"train_oof": True, "val": True, "test_before_freeze": False},
            "future_gt_saved": False,
            "test_intent_label_read_or_saved": False,
            "oof_train_sample_count": len(train_set),
            "val_sample_count": len(val_set),
            "test_sample_count": len(test_set),
        }
        cache_manifest["manifest_sha256"] = hashlib.sha256(canonical_json(cache_manifest)).hexdigest()
        write_json(OUTPUT_ROOT / "feature_cache_manifest.json", cache_manifest)
        write_json(OUTPUT_ROOT / "feature_distribution_audit.json", {
            "train_oof": train_audit,
            "val": val_audit,
            "test": test_audit,
            "test_intent_labels_used": False,
            "fold_heldout_trajectory_audits": crossfit_heldout_audits,
            "full_train_zero_scene_checkpoints": full_hashes,
            "trajectory_crossfit_checkpoints": fold_reports_from_json(OUTPUT_ROOT),
        })
        print(json.dumps({"stage": "features_complete", "cache_manifest_sha256": cache_manifest["manifest_sha256"], "train_oof": train_audit, "val": val_audit, "test": test_audit}, ensure_ascii=False, indent=2), flush=True)


def fold_reports_from_json(root: Path) -> list[dict[str, Any]]:
    path = root / "trajectory_crossfit_checkpoints.json"
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["models"]


if __name__ == "__main__":
    main()
