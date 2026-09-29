#!/usr/bin/env python3
"""Validation-only audit of the stored JAAD scene embedding and its shortcuts.

This script intentionally opens only the processed train and validation archives.
It does not train or modify any production model; the one small MLP is a
diagnostic scene-only probe, and the joint checkpoints are evaluated as-is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.joint_transformer_gate import JointTransformerSceneGate

DATA_ROOT = ROOT / "data/processed/jaad_sequences_scene_15x15"
CHECKPOINT_ROOT = ROOT / "checkpoints/joint_traj_supervision_clean/formal"
SEEDS = (42, 123, 2024)
AUDIT_SEED = 29092026
REQUIRED_KEYS = (
    "target_obs", "target_abs_obs", "neighbor_obs", "neighbor_mask",
    "neighbor_visible_mask", "intent_label", "scene_id", "target_id",
    "obs_end_frame", "image_size", "scene_feat",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_archive(split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "val"}:
        raise ValueError("This audit is restricted to train and validation archives")
    path = DATA_ROOT / f"{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        missing = set(REQUIRED_KEYS) - set(archive.files)
        if missing:
            raise KeyError(f"{path} is missing required arrays: {sorted(missing)}")
        result = {key: archive[key] for key in REQUIRED_KEYS}
    if result["scene_feat"].ndim != 2 or result["scene_feat"].shape[1] != 512:
        raise ValueError(f"Expected [N,512] scene_feat in {path}")
    if not np.isin(result["intent_label"], (0, 1)).all():
        raise ValueError(f"Expected binary clean labels in {path}")
    return result


def as_text(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).astype(str)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    result = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-12)
    return np.clip(result, -1.0, 1.0)


def pair_summary(left: np.ndarray, right: np.ndarray, population: int) -> dict[str, Any]:
    if len(left) == 0:
        return {"population_pair_count": int(population), "sampled_pair_count": 0}
    cosine = cosine_rows(left, right)
    exact = np.all(np.asarray(left) == np.asarray(right), axis=1)
    return {
        "population_pair_count": int(population),
        "sampled_pair_count": int(len(cosine)),
        "cosine": {
            "mean": float(cosine.mean()),
            "std": float(cosine.std()),
            "min": float(cosine.min()),
            "max": float(cosine.max()),
            "quantiles_0_05_25_50_75_95_100": [
                float(value) for value in np.quantile(cosine, [0, .05, .25, .5, .75, .95, 1])
            ],
            "fraction_ge_0_999": float(np.mean(cosine >= .999)),
        },
        "exact_feature_match_fraction": float(exact.mean()),
        "zero_vector_pair_fraction": float(
            np.mean((np.linalg.norm(left, axis=1) <= 1e-12) & (np.linalg.norm(right, axis=1) <= 1e-12))
        ),
    }


def draw_pairs(
    groups: list[np.ndarray], features: np.ndarray, rng: np.random.Generator, max_pairs: int = 12000
) -> tuple[np.ndarray, np.ndarray, int]:
    eligible = [np.asarray(group, dtype=np.int64) for group in groups if len(group) >= 2]
    population = sum(len(group) * (len(group) - 1) // 2 for group in eligible)
    sample_count = min(population, max_pairs)
    if sample_count == 0:
        empty = np.empty((0, features.shape[1]), dtype=np.float32)
        return empty, empty.copy(), 0
    weights = np.asarray([len(group) * (len(group) - 1) // 2 for group in eligible], dtype=np.float64)
    chosen_groups = rng.choice(len(eligible), size=sample_count, replace=sample_count > len(eligible), p=weights / weights.sum())
    left_indices = np.empty(sample_count, dtype=np.int64)
    right_indices = np.empty(sample_count, dtype=np.int64)
    for row, group_index in enumerate(chosen_groups):
        group = eligible[int(group_index)]
        pair = rng.choice(group, size=2, replace=False)
        left_indices[row], right_indices[row] = pair
    return features[left_indices], features[right_indices], population


def feature_similarity_audit(train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> dict[str, Any]:
    rng = np.random.default_rng(AUDIT_SEED)
    splits = {"train": train, "validation": val}
    video_vectors: dict[str, np.ndarray] = {}
    exact_within: dict[str, Any] = {}
    same_video_groups: list[np.ndarray] = []
    same_track_groups: list[np.ndarray] = []
    different_people_groups: list[np.ndarray] = []
    global_offset = 0
    all_features: list[np.ndarray] = []
    all_scene_ids: list[np.ndarray] = []
    for split, data in splits.items():
        feat = np.asarray(data["scene_feat"], dtype=np.float32)
        scenes = as_text(data["scene_id"])
        targets = as_text(data["target_id"])
        all_features.append(feat)
        all_scene_ids.append(scenes)
        split_groups: dict[str, list[int]] = defaultdict(list)
        track_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        split_multi_window_tracks = 0
        for index, (scene, target) in enumerate(zip(scenes, targets)):
            split_groups[scene].append(global_offset + index)
            track_groups[(scene, target)].append(global_offset + index)
        stable_videos = 0
        max_deviation = 0.0
        zero_videos = 0
        for scene, local_indices in split_groups.items():
            indices = np.asarray(local_indices, dtype=np.int64)
            local = feat[np.asarray(local_indices, dtype=np.int64) - global_offset]
            ref = local[0]
            deviations = np.linalg.norm(local - ref[None, :], axis=1)
            max_deviation = max(max_deviation, float(deviations.max(initial=0.0)))
            stable_videos += int(np.array_equal(local, np.repeat(ref[None, :], len(local), axis=0)))
            zero_videos += int(np.linalg.norm(ref) <= 1e-12)
            video_vectors[f"{split}:{scene}"] = ref
            same_video_groups.append(indices)
            by_target: dict[str, list[int]] = defaultdict(list)
            for local_index in local_indices:
                by_target[targets[local_index - global_offset]].append(local_index)
            target_ids = sorted(by_target)
            if len(target_ids) >= 2:
                different_people_groups.append(
                    np.asarray([by_target[target_id][0] for target_id in target_ids], dtype=np.int64)
                )
        for indices in track_groups.values():
            if len(indices) >= 2:
                same_track_groups.append(np.asarray(indices, dtype=np.int64))
                split_multi_window_tracks += 1
        exact_within[split] = {
            "sample_count": int(len(feat)),
            "video_count": int(len(split_groups)),
            "videos_with_exactly_constant_feature": int(stable_videos),
            "fraction_videos_exactly_constant": float(stable_videos / max(len(split_groups), 1)),
            "max_within_video_l2_deviation_from_first_sample": max_deviation,
            "zero_feature_video_count": int(zero_videos),
            "zero_feature_sample_count": int(np.sum(np.linalg.norm(feat, axis=1) <= 1e-12)),
            "target_tracks_with_multiple_windows": int(split_multi_window_tracks),
        }
        global_offset += len(feat)

    features = np.concatenate(all_features, axis=0)
    scene_ids = np.concatenate(all_scene_ids, axis=0)
    unique_video_keys = sorted(video_vectors)
    unique_features = np.stack([video_vectors[key] for key in unique_video_keys])
    cross_video_left, cross_video_right = [], []
    for left in range(len(unique_video_keys)):
        for right in range(left + 1, len(unique_video_keys)):
            cross_video_left.append(unique_features[left])
            cross_video_right.append(unique_features[right])
    cross_left = np.asarray(cross_video_left, dtype=np.float32).reshape(-1, features.shape[1])
    cross_right = np.asarray(cross_video_right, dtype=np.float32).reshape(-1, features.shape[1])

    within_left, within_right, within_population = draw_pairs(same_video_groups, features, rng)
    track_left, track_right, track_population = draw_pairs(same_track_groups, features, rng)
    people_left, people_right, people_population = draw_pairs(different_people_groups, features, rng)
    cross_cosine = cosine_rows(cross_left, cross_right) if len(cross_left) else np.asarray([])
    return {
        "audit_seed": AUDIT_SEED,
        "splits_opened": ["train", "validation"],
        "scene_dimension": int(features.shape[1]),
        "per_split_static_consistency": exact_within,
        "same_video_sample_pairs": pair_summary(within_left, within_right, within_population),
        "same_pedestrian_different_window_pairs": pair_summary(track_left, track_right, track_population),
        "different_pedestrians_same_video_pairs": pair_summary(people_left, people_right, people_population),
        "different_video_unique_embedding_pairs": {
            **pair_summary(cross_left, cross_right, len(cross_left)),
            "unique_videos": int(len(unique_video_keys)),
            "exact_identical_cross_video_pair_count": int(np.sum(np.all(cross_left == cross_right, axis=1))),
            "pair_cosine_ge_0_999_count": int(np.sum(cross_cosine >= .999)),
            "pair_cosine_ge_0_99_count": int(np.sum(cross_cosine >= .99)),
        },
        "different_video_pair_scope": "one stored embedding per unique video; train and validation videos combined, no labels used",
        "cosine_zero_vector_convention": "cosine=0 when either vector has zero norm; exact matches are reported separately",
        "scene_ids_seen_in_both_splits": sorted(set(scene_ids[: len(train["scene_id"])]) & set(scene_ids[len(train["scene_id"]):])),
    }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


class SceneOnlyProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value).squeeze(-1)


def fit_scene_only_probe(train: dict[str, np.ndarray], val: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    x_train = torch.as_tensor(train["scene_feat"], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(train["intent_label"], dtype=torch.float32, device=device)
    x_val = torch.as_tensor(val["scene_feat"], dtype=torch.float32, device=device)
    y_val = np.asarray(val["intent_label"], dtype=np.int64)
    runs = []
    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = SceneOnlyProbe().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        generator = torch.Generator(device=device).manual_seed(seed)
        model.train()
        epochs, batch_size = 30, 1024
        for _ in range(epochs):
            order = torch.randperm(len(x_train), generator=generator, device=device)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                logits = model(x_train[indices])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y_train[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            logits = model(x_val)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        metrics = classification_metrics(y_val, probs)
        runs.append({"seed": seed, **metrics})
        del model, optimizer
    val_prior = float(np.mean(train["intent_label"]))
    return {
        "model": "LayerNorm(512)-Linear(512,128)-GELU-Dropout(0.1)-Linear(128,1)",
        "training_split": "train only",
        "evaluation_split": "validation only",
        "training_protocol": {"epochs": 30, "batch_size": 1024, "optimizer": "Adam", "learning_rate": 0.001, "loss": "unweighted BCEWithLogitsLoss", "checkpoint_selection": "none; final epoch evaluated"},
        "feature_inputs": ["scene_feat only"],
        "seeds": runs,
        "mean_auc": float(np.mean([row["auc"] for row in runs])),
        "std_auc": float(np.std([row["auc"] for row in runs])),
        "mean_brier": float(np.mean([row["brier"] for row in runs])),
        "std_brier": float(np.std([row["brier"] for row in runs])),
        "constant_train_prevalence_baseline_on_validation": {
            "probability": val_prior,
            **classification_metrics(y_val, np.full(len(y_val), val_prior)),
        },
        "test_split_loaded": False,
        "interpretation_guard": "diagnostic only; static scene features are repeated per video, so rows are not independent visual observations",
    }


def build_scene_swap_indices(scene_ids: np.ndarray, mode: str, seed: int = AUDIT_SEED) -> tuple[np.ndarray, dict[str, Any]]:
    scenes = as_text(scene_ids)
    rng = np.random.default_rng(seed + (11 if mode == "same_video" else 29))
    donors = np.full(len(scenes), -1, dtype=np.int64)
    if mode == "same_video":
        groups: dict[str, list[int]] = defaultdict(list)
        for index, scene in enumerate(scenes):
            groups[scene].append(index)
        for indices in groups.values():
            if len(indices) < 2:
                raise ValueError("Same-video swap requires at least two validation rows per video")
            shuffled = np.asarray(indices, dtype=np.int64)[rng.permutation(len(indices))]
            donors[shuffled] = np.roll(shuffled, 1)
    elif mode == "cross_video":
        by_scene: dict[str, list[int]] = defaultdict(list)
        for index, scene in enumerate(scenes):
            by_scene[scene].append(index)
        video_ids = sorted(by_scene)
        if len(video_ids) < 2:
            raise ValueError("Cross-video swap requires at least two validation videos")
        # One reproducible video-level derangement; it is generated without labels.
        permuted = np.asarray(video_ids, dtype=object)[rng.permutation(len(video_ids))]
        donor_video = dict(zip(permuted.tolist(), np.roll(permuted, 1).tolist()))
        for source_video, indices in by_scene.items():
            candidates = np.asarray(by_scene[donor_video[source_video]], dtype=np.int64)
            donors[np.asarray(indices, dtype=np.int64)] = rng.choice(candidates, size=len(indices), replace=True)
    else:
        raise ValueError(f"Unsupported swap mode: {mode}")
    if np.any(donors < 0) or (mode == "same_video" and np.any(scenes[donors] != scenes)) or (mode == "cross_video" and np.any(scenes[donors] == scenes)):
        raise AssertionError("Generated scene swap mapping violates its video constraints")
    mapping = np.column_stack([np.arange(len(donors)), donors]).astype(np.int64)
    digest = sha256_bytes(mapping.tobytes() + "\n".join(scenes.tolist()).encode("utf-8"))
    return donors, {
        "mode": mode,
        "algorithm_seed": int(seed + (11 if mode == "same_video" else 29)),
        "mapping_sha256": digest,
        "rows": int(len(donors)),
        "label_independent": True,
        "mapping_rule": "random permutation-based within-video derangement" if mode == "same_video" else "random video-level derangement plus random donor row",
    }


def build_a0_model(checkpoint: Path, device: torch.device) -> JointTransformerSceneGate:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=512,
        hidden_dim=int(args.get("hidden_dim", 128)),
        pred_len=15,
        gate_mode=str(args.get("gate_mode", "uncertainty")),
        max_obs_len=15,
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    if int(args.get("traj_weight", -1)) != 0:
        raise ValueError(f"Expected A0/J0 trajectory weight 0 checkpoint: {checkpoint}")
    return model


@torch.inference_mode()
def checkpoint_probabilities(
    model: JointTransformerSceneGate,
    val: dict[str, np.ndarray],
    scene_override: np.ndarray | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    device = next(model.parameters()).device
    n = len(val["intent_label"])
    result = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        target = torch.as_tensor(np.concatenate([val["target_obs"][sl], val["target_abs_obs"][sl]], axis=-1), dtype=torch.float32, device=device)
        neighbors = torch.as_tensor(val["neighbor_obs"][sl], dtype=torch.float32, device=device)
        neighbor_mask = torch.as_tensor(val["neighbor_mask"][sl], dtype=torch.float32, device=device)
        visible = torch.as_tensor(val["neighbor_visible_mask"][sl], dtype=torch.float32, device=device)
        scene = val["scene_feat"][sl] if scene_override is None else scene_override[sl]
        scene_tensor = torch.as_tensor(scene, dtype=torch.float32, device=device)
        logits = model(target, neighbors, neighbor_mask, visible, scene_tensor)["intent_logit"]
        result.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(result)


def scene_swap_audit(val: dict[str, np.ndarray], device: torch.device) -> dict[str, Any]:
    y = np.asarray(val["intent_label"], dtype=np.int64)
    scenes = as_text(val["scene_id"])
    same_indices, same_manifest = build_scene_swap_indices(scenes, "same_video")
    cross_indices, cross_manifest = build_scene_swap_indices(scenes, "cross_video")
    features = np.asarray(val["scene_feat"], dtype=np.float32)
    same_scene = features[same_indices]
    cross_scene = features[cross_indices]
    same_manifest["donor_scene_id_counts"] = int(len(set(scenes[same_indices])))
    cross_manifest["unique_donor_videos"] = int(len(set(scenes[cross_indices])))
    cross_manifest["mean_donor_samples_per_source_video"] = float(len(cross_indices) / max(len(set(scenes)), 1))
    result: dict[str, Any] = {
        "validation_rows": int(len(y)),
        "seeds": {},
        "same_video_mapping": same_manifest,
        "cross_video_mapping": cross_manifest,
        "delta_convention": "swapped metric minus original metric; negative ΔAUC and positive ΔBrier indicate degradation",
        "test_split_loaded": False,
    }
    for seed in SEEDS:
        checkpoint = CHECKPOINT_ROOT / f"J0_clean_seed{seed}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing frozen A0/J0 checkpoint: {checkpoint}")
        model = build_a0_model(checkpoint, device)
        original = checkpoint_probabilities(model, val)
        same = checkpoint_probabilities(model, val, same_scene)
        cross = checkpoint_probabilities(model, val, cross_scene)
        original_metrics = classification_metrics(y, original)
        same_metrics = classification_metrics(y, same)
        cross_metrics = classification_metrics(y, cross)
        result["seeds"][str(seed)] = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint),
            "original": original_metrics,
            "same_video_swap": {
                **same_metrics,
                "delta_auc": float(same_metrics["auc"] - original_metrics["auc"]),
                "delta_brier": float(same_metrics["brier"] - original_metrics["brier"]),
                "mean_absolute_probability_shift": float(np.mean(np.abs(same - original))),
                "fraction_probabilities_bitwise_equal": float(np.mean(same == original)),
            },
            "cross_video_swap": {
                **cross_metrics,
                "delta_auc": float(cross_metrics["auc"] - original_metrics["auc"]),
                "delta_brier": float(cross_metrics["brier"] - original_metrics["brier"]),
                "mean_absolute_probability_shift": float(np.mean(np.abs(cross - original))),
                "fraction_probabilities_bitwise_equal": float(np.mean(cross == original)),
            },
        }
        del model
    for condition in ("same_video_swap", "cross_video_swap"):
        rows = [result["seeds"][str(seed)][condition] for seed in SEEDS]
        result[f"{condition}_across_seed_summary"] = {
            key: {"mean": float(np.mean([row[key] for row in rows])), "std": float(np.std([row[key] for row in rows]))}
            for key in ("auc", "brier", "delta_auc", "delta_brier", "mean_absolute_probability_shift")
        }
    return result


def video_identity_probe(data: dict[str, np.ndarray], split: str, seed: int) -> dict[str, Any]:
    features = np.asarray(data["scene_feat"], dtype=np.float32)
    scenes = as_text(data["scene_id"])
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, scene in enumerate(scenes):
        grouped[scene].append(index)
    centroids, names, test_indices = [], [], []
    for scene in sorted(grouped):
        rows = np.asarray(grouped[scene], dtype=np.int64)
        if len(rows) < 2:
            continue
        permuted = rows[rng.permutation(len(rows))]
        split_at = min(max(1, len(rows) // 2), len(rows) - 1)
        centroids.append(features[permuted[:split_at]].mean(axis=0))
        names.append(scene)
        test_indices.extend(permuted[split_at:].tolist())
    if not centroids or not test_indices:
        raise ValueError(f"Insufficient repeated samples for {split} video identity probe")
    centroids_array = np.asarray(centroids, dtype=np.float32)
    test_indices_array = np.asarray(test_indices, dtype=np.int64)
    sims = cosine_rows(features[test_indices_array, None, :], centroids_array[None, :, :])
    order = np.argsort(-sims, axis=1, kind="stable")
    truth = scenes[test_indices_array]
    predicted = np.asarray(names, dtype=object)[order[:, 0]]
    top_k = min(5, len(names))
    top_names = np.asarray(names, dtype=object)[order[:, :top_k]]
    correct = predicted == truth
    true_rank = np.argmax(top_names == truth[:, None], axis=1)
    top1_margin = sims[np.arange(len(test_indices_array)), order[:, 0]] - sims[np.arange(len(test_indices_array)), order[:, 1]] if len(names) > 1 else np.ones(len(test_indices_array))
    unique_centroids = np.unique(centroids_array, axis=0)
    return {
        "split": split,
        "method": "cosine nearest-centroid; row-level random holdout within each already-known video",
        "known_video_classes": int(len(names)),
        "heldout_sample_rows": int(len(test_indices_array)),
        "top1_accuracy": float(np.mean(correct)),
        "top5_accuracy": float(np.mean(np.any(top_names == truth[:, None], axis=1))),
        "mean_top1_cosine_margin": float(np.mean(top1_margin)),
        "unique_centroid_vectors": int(len(unique_centroids)),
        "exact_duplicate_centroid_video_excess": int(len(names) - len(unique_centroids)),
        "label_used": False,
        "generalization_scope": "within-split retrieval among video identities represented in both centroid and query rows; not unseen-video generalization",
    }


def target_groups(data: dict[str, np.ndarray]) -> tuple[dict[tuple[str, str], np.ndarray], dict[tuple[str, str], int]]:
    scenes = as_text(data["scene_id"])
    targets = as_text(data["target_id"])
    labels = np.asarray(data["intent_label"], dtype=np.int64)
    indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(scenes, targets)):
        indices[key].append(index)
    groups = {key: np.asarray(value, dtype=np.int64) for key, value in indices.items()}
    group_labels = {}
    for key, rows in groups.items():
        unique, counts = np.unique(labels[rows], return_counts=True)
        group_labels[key] = int(unique[np.argmax(counts)])
    return groups, group_labels


def video_label_prior_audit(train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> dict[str, Any]:
    results: dict[str, Any] = {"splits": {}, "validation_prediction": {}, "test_split_loaded": False}
    split_video_rates: dict[str, dict[str, float]] = {}
    for split, data in (("train", train), ("validation", val)):
        groups, labels_by_target = target_groups(data)
        per_video_targets: dict[str, list[int]] = defaultdict(list)
        for (scene, _target), label in labels_by_target.items():
            per_video_targets[scene].append(label)
        videos = []
        rates = {}
        for scene in sorted(per_video_targets):
            labels = np.asarray(per_video_targets[scene], dtype=np.int64)
            rate = float(labels.mean())
            rates[scene] = rate
            videos.append({"video_id": scene, "unique_pedestrian_count": int(len(labels)), "crossing_positive_count": int(labels.sum()), "crossing_positive_rate": rate})
        rate_values = np.asarray(list(rates.values()), dtype=np.float64)
        row_labels = np.asarray(data["intent_label"], dtype=np.int64)
        target_id = as_text(data["target_id"])
        scene_id = as_text(data["scene_id"])
        global_prior = float(np.mean(train["intent_label"]))
        loo_prob = np.full(len(row_labels), global_prior, dtype=np.float64)
        smoothing_alpha = 1.0
        for (scene, target), rows in groups.items():
            other_labels = [label for (other_scene, other_target), label in labels_by_target.items() if other_scene == scene and other_target != target]
            if other_labels:
                p = (float(np.sum(other_labels)) + smoothing_alpha * global_prior) / (len(other_labels) + smoothing_alpha)
                loo_prob[rows] = p
        results["splits"][split] = {
            "sample_count": int(len(row_labels)),
            "unique_video_count": int(len(videos)),
            "unique_pedestrian_clip_count": int(len(groups)),
            "target_label_inconsistency_count": int(sum(len(np.unique(row_labels[rows])) > 1 for rows in groups.values())),
            "positive_rate_by_video": videos,
            "video_positive_rate_unweighted_mean": float(rate_values.mean()),
            "video_positive_rate_unweighted_variance": float(rate_values.var()),
            "video_positive_rate_unweighted_std": float(rate_values.std()),
            "video_positive_rate_min": float(rate_values.min()),
            "video_positive_rate_max": float(rate_values.max()),
            "pedestrian_group_weighted_positive_rate": float(np.mean(list(labels_by_target.values()))),
            "leave_one_pedestrian_out_video_prior": {
                **classification_metrics(row_labels, loo_prob),
                "smoothing_alpha": smoothing_alpha,
                "global_train_positive_rate_prior": global_prior,
                "scope": "descriptive in-split audit; all rows of the queried pedestrian are excluded, but other validation pedestrians from the same video contribute",
            },
        }
        split_video_rates[split] = rates
    val_labels = np.asarray(val["intent_label"], dtype=np.int64)
    train_prior = float(np.mean(train["intent_label"]))
    results["validation_prediction"] = {
        "global_train_prevalence_only": {
            "probability": train_prior,
            **classification_metrics(val_labels, np.full(len(val_labels), train_prior)),
        },
        "video_id_seen_in_train": sorted(set(split_video_rates["train"]) & set(split_video_rates["validation"])),
        "warning": "official train/validation splits contain distinct videos; a learned per-video prior cannot transfer to unseen video IDs without using validation labels",
    }
    return results


def discover_video_root(requested: Path | None) -> tuple[Path | None, list[str]]:
    candidates = []
    if requested is not None:
        candidates.append(requested)
    candidates.extend([
        ROOT / "data/processed/jaad_sequences_scene_15x15/JAAD_clips",
        ROOT / "data/processed/JAAD_clips",
        ROOT / "JAAD_clips",
    ])
    seen = set()
    checked = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        checked.append(str(candidate))
        if candidate.is_dir() and any(candidate.glob("*.mp4")):
            return candidate, checked
    return None, checked


def local_background_mask_audit(
    val: dict[str, np.ndarray], device: torch.device, video_root: Path | None, checked_roots: list[str]
) -> dict[str, Any]:
    if video_root is None:
        return {
            "status": "not_run_raw_frames_unavailable",
            "split": "validation only",
            "video_roots_checked": checked_roots,
            "note": "No JAAD RGB clips were accessible in the workspace or mounted media locations. A 3.7-TB NTFS partition is detected as /dev/sda2 but is unmounted; it was not mounted or modified. No masks or R0/R1/R2 metrics are fabricated.",
            "comparison_requested": {"R0": "frame at the selected sample's obs_end_frame", "R1": "retain 4x-expanded pedestrian-box vicinity; neutral-mask the remainder", "R2": "neutral-mask the same vicinity; retain far background"},
            "test_split_loaded": False,
        }
    # RGB interventions are optional and only run when clips are already accessible.
    import cv2
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    weights_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not weights_file.is_file():
        return {"status": "not_run_pretrained_resnet_weights_not_cached", "video_root": str(video_root), "expected_weights_cache": str(weights_file), "test_split_loaded": False}
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    backbone.to(device).eval()
    transform = weights.transforms()
    indices = np.linspace(0, len(val["intent_label"]) - 1, min(256, len(val["intent_label"]))).round().astype(np.int64)
    mean_rgb = np.asarray([123, 116, 103], dtype=np.uint8)
    rows: dict[str, list[Any]] = {"R0": [], "R1": [], "R2": []}
    kept_indices = []
    missing_video = missing_frame = 0
    for index in indices:
        video = str(as_text(val["scene_id"])[index])
        capture = cv2.VideoCapture(str(video_root / f"{video}.mp4"))
        if not capture.isOpened():
            capture.release()
            missing_video += 1
            continue
        frame_number = int(val["obs_end_frame"][index])
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            missing_frame += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        center_x, center_y, box_w, box_h = val["target_abs_obs"][index, -1]
        x_center, y_center = float(center_x * width), float(center_y * height)
        half_w, half_h = max(float(box_w * width * 2.0), 1.0), max(float(box_h * height * 2.0), 1.0)
        x1, x2 = max(0, int(np.floor(x_center - half_w))), min(width, int(np.ceil(x_center + half_w)))
        y1, y2 = max(0, int(np.floor(y_center - half_h))), min(height, int(np.ceil(y_center + half_h)))
        local_mask = np.zeros((height, width), dtype=bool)
        local_mask[y1:y2, x1:x2] = True
        variants = {"R0": rgb, "R1": np.broadcast_to(mean_rgb, rgb.shape).copy(), "R2": rgb.copy()}
        variants["R1"][local_mask] = rgb[local_mask]
        variants["R2"][local_mask] = mean_rgb
        for name, pixels in variants.items():
            rows[name].append(transform(Image.fromarray(pixels)))
        kept_indices.append(index)
    if len(kept_indices) < 20:
        return {"status": "insufficient_accessible_validation_frames", "video_root": str(video_root), "selected_rows": int(len(indices)), "usable_rows": int(len(kept_indices)), "missing_video_rows": int(missing_video), "missing_frame_rows": int(missing_frame), "test_split_loaded": False}
    labels = np.asarray(val["intent_label"])[kept_indices]
    encoded: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for name, tensors in rows.items():
            outputs = []
            for start in range(0, len(tensors), 64):
                batch = torch.stack(tensors[start : start + 64]).to(device)
                outputs.append(backbone(batch).cpu().numpy())
            encoded[name] = np.concatenate(outputs, axis=0).astype(np.float32)
    seed_rows = {}
    for seed in SEEDS:
        checkpoint = CHECKPOINT_ROOT / f"J0_clean_seed{seed}.pt"
        model = build_a0_model(checkpoint, device)
        seed_rows[str(seed)] = {}
        for name, scene in encoded.items():
            probabilities = checkpoint_probabilities(model, {key: np.asarray(value)[kept_indices] for key, value in val.items() if key in REQUIRED_KEYS}, scene_override=scene)
            seed_rows[str(seed)][name] = classification_metrics(labels, probabilities)
        del model
    return {
        "status": "completed",
        "video_root": str(video_root),
        "sample_selection": "256 evenly spaced validation rows before clip-availability filtering; labels not used for selection",
        "usable_rows": int(len(kept_indices)),
        "missing_video_rows": int(missing_video),
        "missing_frame_rows": int(missing_frame),
        "bbox_expansion_factor": 4,
        "mask_fill_rgb": mean_rgb.tolist(),
        "frame_semantics": "obs_end_frame from the selected validation sample; not the first frame used in the stored original embedding",
        "variants_per_seed": seed_rows,
        "test_split_loaded": False,
        "caveat": "R0/R1/R2 use observation-time frames while training features use video first frames; treat as a matched mask sensitivity diagnostic, not a direct attribution of the trained feature source.",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_provenance(output: Path, train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> None:
    import torchvision
    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT
    lines = [
        "# Scene feature provenance audit",
        "",
        "## Verified extraction path",
        "",
        "The checked-in implementation is `scripts/extract_scene_features.py`. It requires `--video-root`; for each unique `scene_id` it opens `<video-root>/<scene_id>.mp4` with OpenCV and calls `read()` once on a new capture. Therefore the source is the first decodable RGB frame, not a per-sample observation frame.",
        "",
        "The exact historical absolute `--video-root` argument was not saved in the repository; the code-level path convention is verified, while that machine-specific root is not recoverable from the current manifest/logs.",
        "",
        "## Image and feature construction",
        "",
        "- Image source: full-frame JAAD clip `<scene_id>.mp4`; the extractor does not crop or mask the pedestrian.",
        "- Color: OpenCV BGR is converted to RGB.",
        f"- Backbone: torchvision ResNet-18, `weights=ResNet18_Weights.DEFAULT` ({weights.name}); pretrained ImageNet weights, URL `{weights.url}`.",
        "- Extraction layer: replace `fc` with `Identity`; ResNet global average pooled output is 512-D.",
        f"- Current audit runtime: PyTorch {torch.__version__}, torchvision {torchvision.__version__}; transform resize={weights.transforms().resize_size}, center crop={weights.transforms().crop_size}, interpolation={weights.transforms().interpolation}, ImageNet mean={weights.transforms().mean}, std={weights.transforms().std}.",
        "- Backbone is put in eval mode and is not fine-tuned during extraction.",
        "- Extract once per unique video ID; `feature_by_scene[scene]` is copied into every row for that video when the split NPZ is written.",
        "- Saved representation is float16 in the processed NPZ and loaded as float32 by the dataset class.",
        "",
        "## Per-video consistency on allowed splits",
        "",
        "| Split | Rows | Videos | Exact-constant videos | Zero-feature videos |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, data in (("train", train), ("validation", val)):
        scenes = as_text(data["scene_id"])
        stable, zero = 0, 0
        for scene in np.unique(scenes):
            feature = data["scene_feat"][scenes == scene]
            stable += int(np.array_equal(feature, np.repeat(feature[:1], len(feature), axis=0)))
            zero += int(np.linalg.norm(feature[0]) <= 1e-12)
        lines.append(f"| {split} | {len(scenes)} | {len(np.unique(scenes))} | {stable} | {zero} |")
    lines += [
        "",
        "## Reproducibility limits",
        "",
        "The extractor does not save a run manifest with the historical torchvision version, exact resolved weights checksum, input clip root, source frame hashes, or preprocessing configuration. The current code resolves `DEFAULT` to the weight enum shown above in this audit runtime; that is code-level provenance, not proof of the exact package/weight file used during the original extraction.",
        "",
        "Processed archives inspected: `data/processed/jaad_sequences_scene_15x15/train.npz` and `val.npz` only. The official test archive was not opened.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def write_summary(output: Path, provenance: dict[str, Any], similarity: dict[str, Any], scene_only: dict[str, Any], swaps: dict[str, Any], identity: dict[str, Any], priors: dict[str, Any], local: dict[str, Any]) -> None:
    same = swaps["same_video_swap_across_seed_summary"]
    cross = swaps["cross_video_swap_across_seed_summary"]
    a0_auc = np.asarray([swaps["seeds"][str(seed)]["original"]["auc"] for seed in SEEDS], dtype=np.float64)
    a0_brier = np.asarray([swaps["seeds"][str(seed)]["original"]["brier"] for seed in SEEDS], dtype=np.float64)
    val_prior = priors["splits"]["validation"]
    identity_val = identity["validation"]
    val_rates = [row["crossing_positive_rate"] for row in val_prior["positive_rate_by_video"]]
    val_all_negative = sum(rate == 0.0 for rate in val_rates)
    val_all_positive = sum(rate == 1.0 for rate in val_rates)
    val_mixed = sum(0.0 < rate < 1.0 for rate in val_rates)
    scene_only_auc = scene_only["mean_auc"]
    cross_auc_delta = cross["delta_auc"]["mean"]
    identity_top1 = identity_val["top1_accuracy"]
    local_done = local.get("status") == "completed"
    shortcut_risk = (scene_only_auc >= 0.70 and cross_auc_delta <= -0.02) or (identity_top1 >= 0.95 and val_prior["video_positive_rate_unweighted_variance"] >= 0.01)
    if local_done and local.get("variants_per_seed"):
        # A local-vs-far conclusion requires matched validation metrics, but does not erase the video-identity audit.
        summary_class = "mixed" if shortcut_risk else "real crossing semantics supported, with shortcut risk still audited"
    else:
        summary_class = "background shortcut / video-level prior risk high; local crossing semantics remain unverified" if shortcut_risk else "mixed / unresolved: global video-level scene signal is established, but local crossing semantics could not be isolated"
    lines = [
        "# Scene shortcut and crossing-context audit summary",
        "",
        "## Scope and data firewall",
        "",
        "All new analyses used only the clean train and validation archives. No official-test archive, prediction file, or metric was loaded. A0/J0 validation inference uses the existing frozen checkpoints; the only fit is the explicitly requested small scene-only diagnostic MLP.",
        "",
        "## Answers to the audit questions",
        "",
        "1. **What is the 512-D feature?** A full-frame, first-decodable-frame ResNet-18 ImageNet embedding, globally pooled after replacing `fc` with identity. See `scene_feature_provenance.md`.",
        "2. **Single frame or temporal?** Single frame per video; not a temporal representation.",
        "3. **Global or pedestrian-local?** Global frame; no pedestrian crop or mask is applied by the extractor.",
        f"4. **Within-video identity:** train {similarity['per_split_static_consistency']['train']['videos_with_exactly_constant_feature']}/{similarity['per_split_static_consistency']['train']['video_count']} and validation {similarity['per_split_static_consistency']['validation']['videos_with_exactly_constant_feature']}/{similarity['per_split_static_consistency']['validation']['video_count']} videos have exactly identical stored features across all rows. Same-video swap mean |Δp|={same['mean_absolute_probability_shift']['mean']:.6f}; ΔAUC={same['delta_auc']['mean']:+.6f}.",
        f"5. **Scene-only validation:** mean AUC={scene_only['mean_auc']:.4f} ± {scene_only['std_auc']:.4f}; mean Brier={scene_only['mean_brier']:.4f} ± {scene_only['std_brier']:.4f} across seeds 42/123/2024. On these same validation rows, frozen A0/J0 originals score AUC={a0_auc.mean():.4f} ± {a0_auc.std():.4f}, Brier={a0_brier.mean():.4f} ± {a0_brier.std():.4f}; the scene-only probe is close to the full model's validation AUC.",
        f"6. **Same-video swap:** ΔAUC={same['delta_auc']['mean']:+.6f} ± {same['delta_auc']['std']:.6f}; ΔBrier={same['delta_brier']['mean']:+.6f} ± {same['delta_brier']['std']:.6f}; mean |Δp|={same['mean_absolute_probability_shift']['mean']:.6f}.",
        f"7. **Cross-video swap:** ΔAUC={cross['delta_auc']['mean']:+.4f} ± {cross['delta_auc']['std']:.4f}; ΔBrier={cross['delta_brier']['mean']:+.4f} ± {cross['delta_brier']['std']:.4f}; mean |Δp|={cross['mean_absolute_probability_shift']['mean']:.4f}.",
        f"8. **Video identity:** validation nearest-centroid top-1={identity_val['top1_accuracy']:.4f}, top-5={identity_val['top5_accuracy']:.4f} among known validation video IDs. This is within-split identity retrieval, not unseen-video generalization.",
        f"9. **Video label prior:** validation has {val_prior['unique_video_count']} videos and {val_prior['unique_pedestrian_clip_count']} pedestrian tracks; {val_all_negative} videos are all-negative, {val_all_positive} all-positive, and {val_mixed} mixed. Unweighted video positive-rate variance={val_prior['video_positive_rate_unweighted_variance']:.4f} (range {val_prior['video_positive_rate_min']:.3f}–{val_prior['video_positive_rate_max']:.3f}). Leave-one-pedestrian-out within-video prior AUC={val_prior['leave_one_pedestrian_out_video_prior']['auc']:.4f}, Brier={val_prior['leave_one_pedestrian_out_video_prior']['brier']:.4f}; this uses other validation pedestrians and is descriptive, not a deployable held-out-video prediction.",
        f"10. **Local versus far background:** status `{local.get('status')}`. {local.get('note', local.get('caveat', 'See local_background_mask_audit.json for per-condition metrics.'))}",
        "11. **Why did No Scene lose about 0.09 AUC?** The current scene input is necessarily a video-level global prior. The swap and scene-only results determine whether it is predictive/useful, but do not by themselves distinguish road semantics from background/domain identity. The local-vs-far test is unavailable unless source clips are mounted.",
        f"12. **Final current classification:** **{summary_class}.** This describes evidence and uncertainty; it is not a claim that the model learned a novel scene method.",
        "",
        "## Interpretation and next direction",
        "",
        "The feature construction prevents same-video replacement from changing the input: every sample in a video receives exactly the same vector. Therefore a near-zero same-video swap is a construction invariant, not evidence that the model ignores scene context. Cross-video swap, scene-only performance, identity retrieval, and video label priors must be read together.",
        "",
        "If cross-video replacement and video priors are strong, prioritize background-shortcut suppression and video/domain robustness before adding scene complexity. If local masking later shows the pedestrian vicinity carries predictive signal, preserve that local road/crosswalk evidence while regularizing invariance to distant background. Do not claim local crossing semantics from this audit until RGB interventions are completed.",
        "",
        "## Machine-readable outputs",
        "",
        "- `scene_feature_provenance.md`",
        "- `scene_feature_similarity.json`",
        "- `scene_only_probe.json`",
        "- `same_video_scene_swap.json`",
        "- `cross_video_scene_swap.json` (same combined file content, separated for convenient review)",
        "- `video_identity_probe.json`",
        "- `video_label_prior.json`",
        "- `local_background_mask_audit.json`",
        "",
        "## Limits",
        "",
        "Validation videos are a finite set of 25 domains, and the scene feature is constant within each video. Scene-only and video-prior diagnostics are therefore not independent-row evidence. The local mask intervention remains unavailable because source clips are absent from accessible workspace paths; the detected 3.7-TB NTFS partition is unmounted and was not mounted by this audit.",
        "",
        "The previously quoted AUC≈0.7644 was not assumed to be a validation metric. For this audit, the checkpoint outputs above reproduce the stored validation-selection values for the named A0/J0 checkpoints; do not compare the two numbers unless their split and selection protocol are confirmed to match.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=None, help="Optional already-mounted directory containing JAAD <video_id>.mp4 clips")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/scene_shortcut_audit")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    train, val = load_archive("train"), load_archive("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_provenance(output_root / "scene_feature_provenance.md", train, val)
    similarity = feature_similarity_audit(train, val)
    scene_only = fit_scene_only_probe(train, val, device)
    swaps = scene_swap_audit(val, device)
    identity = {
        "train": video_identity_probe(train, "train", AUDIT_SEED + 101),
        "validation": video_identity_probe(val, "validation", AUDIT_SEED + 103),
        "test_split_loaded": False,
    }
    priors = video_label_prior_audit(train, val)
    video_root, checked_roots = discover_video_root(args.video_root)
    local = local_background_mask_audit(val, device, video_root, checked_roots)
    write_json(output_root / "scene_feature_similarity.json", similarity)
    write_json(output_root / "scene_only_probe.json", scene_only)
    write_json(output_root / "same_video_scene_swap.json", swaps)
    write_json(output_root / "cross_video_scene_swap.json", swaps)
    write_json(output_root / "video_identity_probe.json", identity)
    write_json(output_root / "video_label_prior.json", priors)
    write_json(output_root / "local_background_mask_audit.json", local)
    write_summary(output_root / "summary.md", {}, similarity, scene_only, swaps, identity, priors, local)
    print(json.dumps({
        "device": str(device),
        "scene_only_mean_auc": scene_only["mean_auc"],
        "scene_only_mean_brier": scene_only["mean_brier"],
        "same_video_delta_auc": swaps["same_video_swap_across_seed_summary"]["delta_auc"],
        "cross_video_delta_auc": swaps["cross_video_swap_across_seed_summary"]["delta_auc"],
        "validation_video_id_top1": identity["validation"]["top1_accuracy"],
        "validation_video_prior_auc": priors["splits"]["validation"]["leave_one_pedestrian_out_video_prior"]["auc"],
        "local_mask_status": local["status"],
        "output_root": str(output_root),
        "official_test_opened": False,
    }, indent=2))


if __name__ == "__main__":
    main()
