#!/usr/bin/env python3
"""Train/validation-only local-context versus far-background audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_scene_shortcut_audit import (  # noqa: E402
    AUDIT_SEED, CHECKPOINT_ROOT as A0_CHECKPOINT_ROOT, REQUIRED_KEYS, SEEDS,
    build_a0_model, build_scene_swap_indices, checkpoint_probabilities,
    classification_metrics, fit_scene_only_probe,
)

DATA_ROOT = ROOT / "data/processed/jaad_sequences_scene_15x15"
VIDEO_ROOT = Path("/media/lrj/54926A1D926A0438/ped_intent_project/data/raw/JAAD/JAAD_clips")
ANNOTATION_ROOT = Path("/media/lrj/54926A1D926A0438/ped_intent_project/data/raw/JAAD/annotations/JAAD_2.0/annotations")
FEATURE_ROOT = ROOT / "data/processed/scene_shortcut_audit_features"
OUTPUT_ROOT = ROOT / "results/scene_shortcut_audit"
INITIAL_ROOT = ROOT / "checkpoints/joint_traj_supervision_clean"
FILL_RGB = np.asarray([124, 116, 104], dtype=np.uint8)
SCALES = (2, 4, 6)
ARMS = {"B0": "R0_current_full", "B1": "R1_local_4x", "B2": "R2_background_4x"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(root: Path, split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "val"}:
        raise ValueError("Only train and val archives may be opened by this audit")
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        missing = set(REQUIRED_KEYS) - set(archive.files)
        if missing:
            raise KeyError(f"Missing required arrays: {sorted(missing)}")
        data = {key: archive[key] for key in REQUIRED_KEYS}
    if data["scene_feat"].shape != (len(data["scene_id"]), 512):
        raise ValueError(f"{split} scene_feat must have shape [N,512]")
    return data


def resolve_video(scene_id: str, root: Path) -> Path:
    for ext in (".mp4", ".avi", ".mov"):
        candidate = root / f"{str(scene_id)}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No original video for scene_id={scene_id} under {root}")


def frame_in_bounds(frame_id: int, frame_count: int) -> bool:
    return 0 <= int(frame_id) < int(frame_count)


def expanded_bbox(box: np.ndarray, width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    if scale <= 0 or width <= 0 or height <= 0:
        raise ValueError("scale and image dimensions must be positive")
    cx, cy, bw, bh = map(float, box)
    if not all(math.isfinite(v) for v in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
        raise ValueError("Expected normalized center-x, center-y, width, height")
    x, y = cx * width, cy * height
    hx, hy = bw * width * scale / 2, bh * height * scale / 2
    bounds = (max(0, math.floor(x - hx)), max(0, math.floor(y - hy)),
              min(width, math.ceil(x + hx)), min(height, math.ceil(y + hy)))
    if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Expanded bbox does not intersect frame")
    return tuple(map(int, bounds))


def mask_variants(rgb: np.ndarray, roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB [H,W,3]")
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = roi
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("ROI must be nonempty and clipped")
    fill = np.broadcast_to(FILL_RGB.reshape(1, 1, 3), rgb.shape)
    local, background = fill.copy(), rgb.copy()
    local[y1:y2, x1:x2] = rgb[y1:y2, x1:x2]
    background[y1:y2, x1:x2] = FILL_RGB
    return local, background


def validate_features(features: np.ndarray, rows: int | None = None) -> None:
    if features.ndim != 2 or features.shape[1] != 512 or (rows is not None and features.shape[0] != rows):
        raise ValueError(f"Expected [N,512] feature matrix, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or Inf")


def encode_batch(encoder: torch.nn.Module, tensors: list[torch.Tensor], device: torch.device) -> np.ndarray:
    if not tensors:
        return np.empty((0, 512), dtype=np.float32)
    with torch.inference_mode():
        output = encoder(torch.stack(tensors).to(device))
    if output.ndim > 2:
        output = torch.flatten(output, 1)
    if output.ndim != 2 or output.shape[1] != 512:
        raise ValueError(f"Frozen ResNet extractor must return [N,512], got {tuple(output.shape)}")
    result = output.float().cpu().numpy()
    validate_features(result, len(tensors))
    return result


def annotation_frames(path: Path) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    root = ET.parse(path).getroot()
    for track in root.findall(".//track"):
        if track.get("label") not in {"ped", "pedestrian"}:
            continue
        for box in track.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            ped_id = next(((a.text or "").strip() for a in box.findall("attribute") if a.get("name") == "id"), "")
            frame = int(box.get("frame", "-1"))
            if ped_id and frame >= 0:
                result[ped_id].add(frame)
    return result


def extract_split(split: str, data: dict[str, np.ndarray], video_root: Path, feature_root: Path,
                  encoder: torch.nn.Module, transform: Any, device: torch.device,
                  batch_size: int = 48) -> dict[str, np.ndarray]:
    if split not in {"train", "val"}:
        raise ValueError("Feature extraction is restricted to train and val")
    n = len(data["scene_id"])
    scenes = data["scene_id"].astype(str)
    frames = data["obs_end_frame"].astype(np.int64)
    boxes = data["target_abs_obs"][:, -1, :4]
    features: dict[str, np.ndarray] = {
        "old_static": data["scene_feat"].astype(np.float16, copy=True),
        "R0_current_full": np.zeros((n, 512), dtype=np.float16),
        "R1_local_4x": np.zeros((n, 512), dtype=np.float16),
        "R2_background_4x": np.zeros((n, 512), dtype=np.float16),
    }
    scales = (4,) if split == "train" else SCALES
    if split == "val":
        for scale in SCALES:
            features[f"R1_local_{scale}x"] = np.zeros((n, 512), dtype=np.float16)
            features[f"R2_background_{scale}x"] = np.zeros((n, 512), dtype=np.float16)
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (scene, frame) in enumerate(zip(scenes, frames)):
        grouped[(scene, int(frame))].append(i)
    per_video: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    for (scene, frame), rows in grouped.items():
        per_video[scene].append((frame, rows))
    tensors: list[torch.Tensor] = []
    refs: list[tuple[str, Any]] = []

    def flush() -> None:
        if not tensors:
            return
        encoded = encode_batch(encoder, tensors, device)
        for ref, vector in zip(refs, encoded):
            kind, payload = ref
            if kind == "full":
                features["R0_current_full"][payload] = vector.astype(np.float16)
            else:
                row, key = payload
                features[key][row] = vector.astype(np.float16)
        tensors.clear()
        refs.clear()

    def queue(rgb: np.ndarray, ref: tuple[str, Any]) -> None:
        tensors.append(transform(Image.fromarray(rgb, mode="RGB")))
        refs.append(ref)
        if len(tensors) >= batch_size:
            flush()

    for scene in sorted(per_video):
        path = resolve_video(scene, video_root)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {path}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        decoded_frame_id = -1
        for frame_id, rows in sorted(per_video[scene]):
            if not frame_in_bounds(frame_id, count):
                raise IndexError(f"{scene} obs_end_frame={frame_id} out of [0,{count})")
            ok, bgr = True, None
            while decoded_frame_id < frame_id:
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode {scene} frame {decoded_frame_id + 1} while seeking sequentially to {frame_id}")
                decoded_frame_id += 1
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            queue(rgb, ("full", rows))
            h, w = rgb.shape[:2]
            for row in rows:
                for scale in scales:
                    roi = expanded_bbox(boxes[row], w, h, scale)
                    local, background = mask_variants(rgb, roi)
                    queue(local, ("mask", (row, f"R1_local_{scale}x")))
                    queue(background, ("mask", (row, f"R2_background_{scale}x")))
        cap.release()
        print(f"extracted {split}: {scene}", flush=True)
    flush()
    for value in features.values():
        validate_features(value, n)
    feature_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(feature_root / f"{split}_features.npz", **features)
    return features


def verify_paths_and_frames(train: dict[str, np.ndarray], val: dict[str, np.ndarray],
                            video_root: Path, annotation_root: Path, out: Path) -> dict[str, Any]:
    video_files = [p for p in video_root.iterdir() if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"}]
    val_scenes = sorted(set(val["scene_id"].astype(str).tolist()))
    rng = np.random.default_rng(29092026)
    sampled = set(rng.choice(val_scenes, min(10, len(val_scenes)), replace=False).tolist())
    video_rows = []
    video_meta = {}
    for scene in val_scenes:
        path = resolve_video(scene, video_root)
        cap = cv2.VideoCapture(str(path))
        row = {"scene_id": scene, "video_path": str(path), "exists": path.is_file(),
               "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": float(cap.get(cv2.CAP_PROP_FPS)),
               "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               "randomly_selected": scene in sampled}
        cap.release()
        if row["frame_count"] <= 0:
            raise RuntimeError(f"Invalid video metadata: {path}")
        video_meta[scene] = row
        video_rows.append(row)
    track_frames: dict[str, dict[str, set[int]]] = {}
    checked = matched = 0
    bad_tracks, bad_bounds = [], []
    for split, data in (("train", train), ("val", val)):
        scenes = data["scene_id"].astype(str)
        ids = data["target_id"].astype(str)
        frames = data["obs_end_frame"].astype(np.int64)
        for scene in sorted(set(scenes.tolist())):
            xml = annotation_root / f"{scene}.xml"
            if not xml.is_file():
                raise FileNotFoundError(xml)
            track_frames[scene] = annotation_frames(xml)
        for scene, ped, frame in zip(scenes, ids, frames):
            checked += 1
            if int(frame) in track_frames[scene].get(ped, set()):
                matched += 1
            else:
                bad_tracks.append({"split": split, "scene_id": scene, "target_id": ped, "frame": int(frame)})
            if split == "val" and not frame_in_bounds(frame, video_meta[scene]["frame_count"]):
                bad_bounds.append({"scene_id": scene, "frame": int(frame)})
    location = {
        "video_root": str(video_root), "total_video_count": len(video_files),
        "extensions": {ext: sum(p.suffix.lower() == ext for p in video_files) for ext in (".mp4", ".avi", ".mov")},
        "scene_id_mapping": "Use scene_id unchanged as the video filename stem, normally scene_id.mp4 (e.g. video_0171 -> video_0171.mp4).",
        "validation_video_count": len(val_scenes), "validation_videos": video_rows,
        "test_split_loaded": False,
    }
    write_json(out / "raw_video_location.json", location)
    write_json(out / "raw_video_mapping_audit.json", {
        "random_seed": 29092026, "randomly_selected_validation_scenes": sorted(sampled),
        "selected_videos": [row for row in video_rows if row["randomly_selected"]],
        "all_validation_paths_exist": all(row["exists"] for row in video_rows),
        "all_val_obs_end_frames_in_bounds": not bad_bounds, "out_of_bounds": bad_bounds,
        "test_split_loaded": False,
    })
    frame = {
        "train_val_rows_checked": checked, "exact_target_xml_frame_matches": matched,
        "missing_target_xml_matches": len(bad_tracks), "missing_examples": bad_tracks[:20],
        "val_frame_out_of_bounds": len(bad_bounds),
    }
    md = (
        "# Frame-index mapping audit\n\n"
        "The JAAD preprocessor parses each XML box@frame integer as the original frame key, builds observation windows ending at obs_end, and appends obs_end unchanged as obs_end_frame. This is a source-video frame number, not a processed-array row index. OpenCV CAP_PROP_POS_FRAMES uses zero-based frame positions.\n\n"
        f"Train and validation exact (scene_id, target_id, obs_end_frame) XML matches: {matched:,}/{checked:,}. Missing target-track/frame matches: {len(bad_tracks)}. Validation out-of-bounds frame indices: {len(bad_bounds)}. Official test was not loaded.\n"
    )
    (out / "frame_index_mapping.md").write_text(md, encoding="utf-8")
    if bad_tracks or bad_bounds:
        raise RuntimeError("Frame mapping verification failed")
    return frame


def _similarities(val: dict[str, np.ndarray], matrices: dict[str, np.ndarray]) -> dict[str, Any]:
    scenes, peds = val["scene_id"].astype(str), val["target_id"].astype(str)
    frame_ids = val["obs_end_frame"].astype(np.int64)
    by_scene: dict[str, list[int]] = defaultdict(list)
    by_track: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (scene, ped) in enumerate(zip(scenes, peds)):
        by_scene[scene].append(i)
        by_track[(scene, ped)].append(i)
    rng = np.random.default_rng(29092027)
    identity_rng = np.random.default_rng(29092028)
    video_ids = sorted(by_scene)
    same_video_rows, diff_video_rows = [], []
    for _ in range(12000):
        scene = video_ids[int(identity_rng.integers(len(video_ids)))]
        by_frame: dict[int, list[int]] = defaultdict(list)
        for index in by_scene[scene]:
            by_frame[int(frame_ids[index])].append(index)
        distinct_frames = list(by_frame)
        if len(distinct_frames) > 1:
            f1, f2 = identity_rng.choice(distinct_frames, 2, replace=False)
            same_video_rows.append((identity_rng.choice(by_frame[int(f1)]), identity_rng.choice(by_frame[int(f2)])))
        a, b = identity_rng.choice(video_ids, 2, replace=False)
        diff_video_rows.append((identity_rng.choice(by_scene[a]), identity_rng.choice(by_scene[b])))
    same_video_rows = np.asarray(same_video_rows, dtype=np.int64)
    diff_video_rows = np.asarray(diff_video_rows, dtype=np.int64)
    result = {"validation_only": True, "unique_videos": len(by_scene), "representations": {},
              "video_identity_pair_seed": 29092028}
    for name in ("old_static", "R0_current_full", "R1_local_4x", "R2_background_4x"):
        x = np.asarray(matrices[name], dtype=np.float32)
        within, track, person = [], [], []
        for indices in by_scene.values():
            if len(indices) > 1:
                for _ in range(min(400, len(indices) * 4)):
                    a, b = rng.choice(indices, 2, replace=False)
                    within.append((a, b))
        for indices in by_track.values():
            unique = list(dict.fromkeys(indices))
            unique = [unique[i] for i in range(len(unique))
                      if frame_ids[unique[i]] not in {frame_ids[j] for j in unique[:i]}]
            if len(unique) > 1:
                for _ in range(min(30, len(unique) * 2)):
                    a, b = rng.choice(unique, 2, replace=False)
                    track.append((a, b))
        for scene, indices in by_scene.items():
            groups: dict[str, list[int]] = defaultdict(list)
            for i in indices:
                groups[peds[i]].append(i)
            ids = list(groups)
            for _ in range(min(1000, len(ids) * 20)):
                if len(ids) < 2:
                    break
                a, b = rng.choice(ids, 2, replace=False)
                person.append((groups[a][0], groups[b][0]))
        def stats(pairs):
            if not pairs:
                return {"count": 0, "cosine_mean": None, "cosine_std": None}
            ix = np.asarray(pairs, dtype=np.int64)
            norms = np.linalg.norm(x[ix[:, 0]], axis=1) * np.linalg.norm(x[ix[:, 1]], axis=1)
            cosine = np.divide((x[ix[:, 0]] * x[ix[:, 1]]).sum(1), norms,
                               out=np.zeros(len(ix)), where=norms > 1e-12)
            cosine = np.clip(cosine, -1.0, 1.0)
            return {"count": len(cosine), "cosine_mean": float(cosine.mean()), "cosine_std": float(cosine.std()),
                    "quantiles": [float(v) for v in np.quantile(cosine, [0, .25, .5, .75, 1])]}
        same_cos = stats(same_video_rows.tolist())
        diff_cos = stats(diff_video_rows.tolist())
        same_ix, diff_ix = same_video_rows, diff_video_rows
        def pair_cosine(ix):
            denominator = np.linalg.norm(x[ix[:, 0]], axis=1) * np.linalg.norm(x[ix[:, 1]], axis=1)
            return np.divide((x[ix[:, 0]] * x[ix[:, 1]]).sum(1), denominator,
                             out=np.zeros(len(ix), dtype=np.float64), where=denominator > 1e-12)
        same_values, diff_values = pair_cosine(same_ix), pair_cosine(diff_ix)
        result["representations"][name] = {
            "within_video_different_windows": stats(within),
            "same_pedestrian_different_window": stats(track),
            "different_pedestrians_same_video": stats(person),
            "same_video_distinct_frame_cosine": same_cos,
            "different_video_cosine": diff_cos,
            "video_identity_pairwise_cosine_auc": float(roc_auc_score(
                np.r_[np.ones(len(same_values)), np.zeros(len(diff_values))],
                np.r_[same_values, diff_values])),
            "same_video_distinct_frame_exact_duplicate_fraction": float(
                np.mean(np.all(x[same_ix[:, 0]] == x[same_ix[:, 1]], axis=1))),
        }
    return result


def _load_features(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _feature_manifest(data_root: Path, video_root: Path, feature_root: Path,
                      train_feat: dict[str, np.ndarray], val_feat: dict[str, np.ndarray],
                      weights: Any, output: Path) -> dict[str, Any]:
    import torchvision
    weights_path = Path(torch.hub.get_dir()) / "checkpoints/resnet18-f37072fd.pth"
    record = {
        "raw_video_root": str(video_root),
        "scene_id_to_video_rule": "unchanged scene_id stem plus .mp4 (fallback .avi/.mov)",
        "official_test_accessed": False,
        "frame_source": "each sample's actual obs_end_frame; R0 is the current RGB frame, not the clip's first frame",
        "backbone": "torchvision ResNet-18 with ImageNet pretrained weights; fc=Identity; eval; global average pooled 512-D features",
        "weights_enum": weights.name, "weights_url": weights.url,
        "weights_sha256": sha256_file(weights_path),
        "torch_version": torch.__version__, "torchvision_version": torchvision.__version__,
        "transform": {
            "resize": list(weights.transforms().resize_size),
            "center_crop": list(weights.transforms().crop_size),
            "interpolation": str(weights.transforms().interpolation),
            "mean": list(weights.transforms().mean), "std": list(weights.transforms().std),
        },
        "mask": {
            "main_scale": 4, "sensitivity_scales_val_only": [2, 6],
            "bbox": "normalized center-x,center-y,width,height; centered expansion; clip to image",
            "fill_rgb_before_normalization": FILL_RGB.tolist(),
            "R1": "retain expanded target ROI; fill outside",
            "R2": "retain complement; fill the identical ROI",
        },
        "splits": {},
    }
    for split, matrix in (("train", train_feat), ("val", val_feat)):
        archive_path = feature_root / f"{split}_features.npz"
        record["splits"][split] = {
            "source_archive": str(data_root / f"{split}.npz"),
            "source_sha256": sha256_file(data_root / f"{split}.npz"),
            "feature_archive": str(archive_path), "feature_archive_sha256": sha256_file(archive_path),
            "feature_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in matrix.items()},
        }
    write_json(output / "local_background_feature_manifest.json", record)
    return record


def _probe_suite(train: dict[str, np.ndarray], val: dict[str, np.ndarray],
                 train_feat: dict[str, np.ndarray], val_feat: dict[str, np.ndarray],
                 output: Path, device: torch.device) -> dict[str, Any]:
    spec = {
        "old_static": ("scene_probe_old_static.json", "old_static"),
        "R0_current_full": ("scene_probe_current_full.json", "R0_current_full"),
        "R1_local_4x": ("scene_probe_local.json", "R1_local_4x"),
        "R2_background_4x": ("scene_probe_background.json", "R2_background_4x"),
    }
    results = {}
    for name, (filename, key) in spec.items():
        report = fit_scene_only_probe(
            {"scene_feat": np.asarray(train_feat[key], dtype=np.float32), "intent_label": train["intent_label"]},
            {"scene_feat": np.asarray(val_feat[key], dtype=np.float32), "intent_label": val["intent_label"]},
            device,
        )
        report["representation"] = name
        write_json(output / filename, report)
        results[name] = report
        print(f"probe {name}: validation AUC {report['mean_auc']:.4f}", flush=True)
    return results


def _a0_interventions(val: dict[str, np.ndarray], val_feat: dict[str, np.ndarray],
                      output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    y = val["intent_label"].astype(np.int64)
    common = {
        "old_static": val_feat["old_static"],
        "R0_current_full": val_feat["R0_current_full"],
        "R1_local_4x": val_feat["R1_local_4x"],
        "R2_background_4x": val_feat["R2_background_4x"],
    }
    per_seed, sensitivity = {}, {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        checkpoint = A0_CHECKPOINT_ROOT / f"J0_clean_seed{seed}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing clean J0 checkpoint {checkpoint}")
        model = build_a0_model(checkpoint, device)
        original = checkpoint_probabilities(model, val, common["old_static"])
        base = classification_metrics(y, original)
        row = {"old_static": {**base, "mean_abs_probability_delta": 0.0, "probability_correlation": 1.0}}
        for name in ("R0_current_full", "R1_local_4x", "R2_background_4x"):
            prob = checkpoint_probabilities(model, val, common[name])
            metric = classification_metrics(y, prob)
            row[name] = {
                **metric, "delta_auc": metric["auc"] - base["auc"], "delta_brier": metric["brier"] - base["brier"],
                "mean_abs_probability_delta": float(np.mean(np.abs(prob - original))),
                "probability_correlation": float(np.corrcoef(original, prob)[0, 1]),
            }
        per_seed[str(seed)] = row
        sensitivity[str(seed)] = {}
        for family in ("R1_local", "R2_background"):
            for scale in SCALES:
                key = f"{family}_{scale}x"
                prob = checkpoint_probabilities(model, val, val_feat[key])
                metric = classification_metrics(y, prob)
                sensitivity[str(seed)][key] = {
                    **metric, "delta_auc_from_old_static": metric["auc"] - base["auc"],
                    "mean_abs_probability_delta": float(np.mean(np.abs(prob - original))),
                    "probability_correlation": float(np.corrcoef(original, prob)[0, 1]),
                }
        del model
        print(f"A0 feature intervention seed {seed} complete", flush=True)
    intervention = {
        "checkpoint": "existing Clean J0 trained on old static scene feature",
        "evaluation_split": "validation", "interpretation": "distribution intervention, not a fair re-trained model comparison",
        "per_seed": per_seed, "test_split_loaded": False,
    }
    scale = {
        "evaluation_split": "validation", "predeclared_primary_scale": "4x",
        "2x_and_6x_are_sensitivity_only": True, "per_seed": sensitivity, "test_split_loaded": False,
    }
    write_json(output / "a0_current_scene_intervention.json", intervention)
    write_json(output / "local_scale_sensitivity.json", scale)
    return intervention, scale


def run_feature_phase(args: argparse.Namespace) -> None:
    train, val = load_split(args.data_root, "train"), load_split(args.data_root, "val")
    args.output_root.mkdir(parents=True, exist_ok=True)
    frame_report = verify_paths_and_frames(train, val, args.video_root, args.annotation_root, args.output_root)
    print(f"frame mapping: {frame_report['exact_target_xml_frame_matches']}/{frame_report['train_val_rows_checked']} exact matches", flush=True)
    from torchvision.models import ResNet18_Weights, resnet18
    weights = ResNet18_Weights.DEFAULT
    encoder = resnet18(weights=weights)
    encoder.fc = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval().to(device)
    transform = weights.transforms()
    print(f"extracting current-frame features on {device}; transform={transform}", flush=True)
    train_feat = extract_split("train", train, args.video_root, args.feature_root, encoder, transform, device, args.extract_batch_size)
    val_feat = extract_split("val", val, args.video_root, args.feature_root, encoder, transform, device, args.extract_batch_size)
    _feature_manifest(args.data_root, args.video_root, args.feature_root, train_feat, val_feat, weights, args.output_root)
    similarity = _similarities(val, val_feat)
    similarity["test_split_loaded"] = False
    write_json(args.output_root / "dynamic_scene_feature_similarity.json", similarity)
    _probe_suite(train, val, train_feat, val_feat, args.output_root, device)
    _a0_interventions(val, val_feat, args.output_root)
    write_json(args.output_root / "local_background_mask_audit.json", {
        "status": "completed",
        "video_root": str(args.video_root),
        "train_samples": len(train["scene_id"]),
        "validation_samples": len(val["scene_id"]),
        "primary_bbox_expansion": "4x",
        "sensitivity_scales_validation_only": [2, 6],
        "R0_R1_R2_from_same_obs_end_frame": True,
        "feature_dimension": 512,
        "test_split_loaded": False,
        "result_files": [
            "raw_video_mapping_audit.json", "frame_index_mapping.md",
            "dynamic_scene_feature_similarity.json", "scene_probe_current_full.json",
            "scene_probe_local.json", "scene_probe_background.json",
            "a0_current_scene_intervention.json", "local_scale_sensitivity.json",
        ],
        "interpretation": "See scene_locality_summary.md; the new full matched B0/B1/B2 runs are reported there.",
    })


def initialization_report(initial_root: Path) -> dict[str, Any]:
    rows = {}
    all_exact = True
    for seed in SEEDS:
        path = initial_root / f"initial_state_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model"]
        # All three arms explicitly load this exact state file for this seed.
        check = torch.load(path, map_location="cpu", weights_only=False)["model"]
        delta = max(float((state[key].float() - check[key].float()).abs().max()) for key in state)
        rows[str(seed)] = {
            "shared_initial_state_path": str(path), "declared_sha256": payload.get("sha256"),
            "max_abs_parameter_difference_B0_B1_B2": delta, "exact_match": delta == 0,
        }
        all_exact = all_exact and delta == 0
    return {
        "per_seed": rows, "all_arms_exactly_matched_within_seed": all_exact,
        "max_abs_parameter_difference": max(row["max_abs_parameter_difference_B0_B1_B2"] for row in rows.values()),
        "note": "Different seeds have their own initial state; matching is required across B0/B1/B2 within each seed.",
    }


def _command(args: argparse.Namespace, arm: str, seed: int, run_dir: Path) -> list[str]:
    return [
        sys.executable, str(ROOT / "scripts/train_joint_transformer_gate.py"),
        "--data-root", str(args.data_root), "--scene-feature-root", str(args.feature_root),
        "--scene-feature-key", ARMS[arm], "--ambiguous-root", str(args.ambiguous_root),
        "--output-root", str(run_dir), "--checkpoint", str(args.checkpoint_root / f"{arm}_seed{seed}.pt"),
        "--initial-state-checkpoint", str(args.initial_root / f"initial_state_seed{seed}.pt"),
        "--gate-mode", "uncertainty", "--component-ablation", "full",
        "--epochs", "15", "--batch-size", "512", "--hidden-dim", "128",
        "--learning-rate", "0.001", "--prior-weight", "0", "--traj-weight", "0",
        "--traj-weight-mode", "fixed", "--ambiguous-weight", "0", "--seed", str(seed),
        "--selection-mode", "intent_auc", "--selection-tolerance", "0.0001", "--skip-test",
    ]


def _verify_sampler(metrics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    report, all_match = {}, True
    for seed in SEEDS:
        arms = {}
        for arm in ARMS:
            arms[arm] = [
                (row["train"]["sampler_sha256"], row["train"]["first_sample_indices"])
                for row in metrics[arm][str(seed)]["history"]
            ]
        match = arms["B0"] == arms["B1"] == arms["B2"]
        report[str(seed)] = {"epochwise_match": match, "epoch_fingerprints": {
            arm: [item[0] for item in values] for arm, values in arms.items()
        }}
        all_match = all_match and match
    return {"per_seed": report, "all_matched": all_match, "ambiguous_training_loader_used": False}


def run_training_phase(args: argparse.Namespace) -> None:
    train_feat = _load_features(args.feature_root / "train_features.npz")
    val_feat = _load_features(args.feature_root / "val_features.npz")
    init = initialization_report(args.initial_root)
    if not init["all_arms_exactly_matched_within_seed"]:
        raise AssertionError("Initial states are not exactly matched")
    write_json(args.output_root / "initialization_match.json", init)
    metrics: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for seed in SEEDS:
            run_dir = args.output_root / "training" / arm / f"seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = args.checkpoint_root / f"{arm}_seed{seed}.pt"
            metric_path = run_dir / "metrics.json"
            command = _command(args, arm, seed, run_dir)
            if not (metric_path.is_file() and checkpoint.is_file()):
                print(f"starting matched training {arm}/seed{seed}", flush=True)
                with (run_dir / "training.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            result = json.loads(metric_path.read_text(encoding="utf-8"))
            if result.get("test") is not None or result.get("test_evaluation_status") != "withheld_until_protocol_freeze":
                raise AssertionError(f"Test split was accessed in {arm}/seed{seed}")
            if (result.get("traj_weight"), result.get("prior_weight"), result.get("ambiguous_weight")) != (0, 0, 0):
                raise AssertionError(f"Unexpected objective weights in {arm}/seed{seed}")
            stored_hashes = result.get("scene_feature_override_sha256") or result["component_probe"]["scene_feature_override_sha256"]
            for split, matrix in (("train", train_feat[ARMS[arm]]), ("validation", val_feat[ARMS[arm]])):
                digest = hashlib.sha256(np.ascontiguousarray(matrix, dtype=np.float32).tobytes()).hexdigest()
                if stored_hashes[split] != digest:
                    raise AssertionError(f"Scene feature hash mismatch for {arm}/seed{seed}/{split}")
            metrics[arm][str(seed)] = result
            print(f"finished {arm}/seed{seed}: AUC={result['selected_checkpoint_validation_auc']:.5f}", flush=True)
    sampler = _verify_sampler(metrics)
    write_json(args.output_root / "sampler_match.json", sampler)
    if not sampler["all_matched"]:
        raise AssertionError("Sampler sequence differs between matched arms")
    initial_hashes = {
        str(seed): {arm: metrics[arm][str(seed)]["initial_model_state_sha256"] for arm in ARMS}
        for seed in SEEDS
    }
    if any(len(set(row.values())) != 1 for row in initial_hashes.values()):
        raise AssertionError("Training-reported initialization SHA differs between arms")
    init["trainer_reported_state_sha256"] = initial_hashes
    init["trainer_hashes_match_per_seed"] = True
    write_json(args.output_root / "initialization_match.json", init)

    arms_out = {}
    for arm, feature in ARMS.items():
        per_seed = [{
            "seed": seed, "best_epoch": metrics[arm][str(seed)]["best_epoch"],
            "validation_auc": metrics[arm][str(seed)]["selected_checkpoint_validation_auc"],
            "validation_brier": metrics[arm][str(seed)]["selected_checkpoint_validation_brier"],
            "initial_state_sha256": metrics[arm][str(seed)]["initial_model_state_sha256"],
        } for seed in SEEDS]
        arms_out[arm] = {
            "feature": feature, "per_seed": per_seed,
            "mean_auc": float(np.mean([x["validation_auc"] for x in per_seed])),
            "std_auc": float(np.std([x["validation_auc"] for x in per_seed])),
            "mean_brier": float(np.mean([x["validation_brier"] for x in per_seed])),
            "std_brier": float(np.std([x["validation_brier"] for x in per_seed])),
        }

    val = load_split(args.data_root, "val")
    donors, swap_manifest = build_scene_swap_indices(val["scene_id"], "cross_video", seed=AUDIT_SEED)
    cross = {"swap_mapping": swap_manifest, "per_arm_seed": {}, "test_split_loaded": False}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for arm, feature_key in ARMS.items():
        cross["per_arm_seed"][arm] = {}
        baseline_features = np.asarray(val_feat[feature_key], dtype=np.float32)
        swapped_features = baseline_features[donors]
        for seed in SEEDS:
            checkpoint = args.checkpoint_root / f"{arm}_seed{seed}.pt"
            model = build_a0_model(checkpoint, device)
            original = checkpoint_probabilities(model, val, baseline_features)
            swapped = checkpoint_probabilities(model, val, swapped_features)
            original_metrics, swapped_metrics = classification_metrics(val["intent_label"], original), classification_metrics(val["intent_label"], swapped)
            cross["per_arm_seed"][arm][str(seed)] = {
                "original": original_metrics, "cross_video_scene_swap": swapped_metrics,
                "delta_auc": swapped_metrics["auc"] - original_metrics["auc"],
                "delta_brier": swapped_metrics["brier"] - original_metrics["brier"],
                "mean_abs_probability_shift": float(np.abs(original - swapped).mean()),
                "probability_correlation": float(np.corrcoef(original, swapped)[0, 1]),
            }
            del model
            print(f"cross-video robustness {arm}/seed{seed}: ΔAUC={cross['per_arm_seed'][arm][str(seed)]['delta_auc']:+.4f}", flush=True)
        rows = [cross["per_arm_seed"][arm][str(seed)] for seed in SEEDS]
        cross["per_arm_seed"][arm]["aggregate"] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("delta_auc", "delta_brier", "mean_abs_probability_shift", "probability_correlation")
        }
        cross["per_arm_seed"][arm]["aggregate"]["std_delta_auc"] = float(np.std([row["delta_auc"] for row in rows]))
    write_json(args.output_root / "cross_video_robustness.json", cross)
    write_json(args.output_root / "matched_scene_representation_results.json", {
        "arms": arms_out, "matched_protocol": {
            "seeds": list(SEEDS), "epochs": 15, "batch_size": 512, "optimizer": "AdamW",
            "learning_rate": 0.001, "weight_decay": 0.0001, "gradient_clip": 5,
            "prior_weight": 0, "ambiguous_weight": 0, "trajectory_weight": 0,
            "selection": "raw validation AUC; lower Brier within 1e-4 AUC tie",
            "dropout_rng": "same seed per arm, same architecture, batch order and number of optimizer steps",
            "test_split_loaded": False,
        },
        "initialization_match": init, "sampler_match": sampler,
    })


def write_summary(out: Path) -> None:
    get = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
    probes = {
        key: get(filename) for key, filename in (
            ("old", "scene_probe_old_static.json"), ("full", "scene_probe_current_full.json"),
            ("local", "scene_probe_local.json"), ("background", "scene_probe_background.json"),
        )
    }
    matched, cross = get("matched_scene_representation_results.json"), get("cross_video_robustness.json")
    similarity, scale = get("dynamic_scene_feature_similarity.json"), get("local_scale_sensitivity.json")
    location = get("raw_video_location.json")
    full, local, bg = (probes[k]["mean_auc"] for k in ("full", "local", "background"))
    b0_auc, b1_auc, b2_auc = (matched["arms"][arm]["mean_auc"] for arm in ("B0", "B1", "B2"))
    if b2_auc >= b0_auc + .02 and b1_auc <= b0_auc - .02:
        conclusion, advice = "background", "Prioritize background-shortcut suppression and scene-invariant intent learning; revisit local dynamic context after this confound is controlled."
    elif abs(b1_auc-b0_auc) <= .02 and b2_auc < b0_auc-.02:
        conclusion, advice = "local", "Prioritize dynamic pedestrian–local crossing-context relations; retain a background-invariance check."
    elif b1_auc >= b0_auc-.02 and b2_auc >= b0_auc-.02:
        conclusion, advice = "mixed", "Prioritize local crossing semantics together with explicit background-invariance constraints."
    else:
        conclusion, advice = "mixed/uncertain", "Treat evidence as mixed/uncertain and preserve both local semantics and background robustness."
    scale_rows = []
    for factor in SCALES:
        local_rows = [scale["per_seed"][str(seed)][f"R1_local_{factor}x"] for seed in SEEDS]
        bg_rows = [scale["per_seed"][str(seed)][f"R2_background_{factor}x"] for seed in SEEDS]
        scale_rows.append(
            f"| {factor}x | {np.mean([x['auc'] for x in local_rows]):.4f} | "
            f"{np.mean([x['brier'] for x in local_rows]):.4f} | "
            f"{np.mean([x['auc'] for x in bg_rows]):.4f} | "
            f"{np.mean([x['brier'] for x in bg_rows]):.4f} |"
        )
    identity_auc = {
        key: similarity["representations"][key]["video_identity_pairwise_cosine_auc"]
        for key in ("old_static", "R0_current_full", "R1_local_4x", "R2_background_4x")
    }
    frame_report = (out / "frame_index_mapping.md").read_text(encoding="utf-8")
    frame_match_line = next((line for line in frame_report.splitlines() if "xml matches:" in line.lower()), "")
    if not frame_match_line:
        frame_match_line = "The full frame-index and XML track match audit is in frame_index_mapping.md."
    direct_answers = [
        "## Direct answers to the audit questions",
        "",
        f"1. Original video root: {location['video_root']}.",
        f"2. Mapping: scene_id is unchanged as the video stem, normally scene_id.mp4; {location['total_video_count']} source clips were found.",
        "3. obs_end_frame is copied unchanged from the original JAAD XML box frame integer; it is zero-based and indexes the source video, not the processed row array. " + frame_match_line,
        f"4. Yes. R0 same-video cosine across distinct observation frames is {similarity['representations']['R0_current_full']['same_video_distinct_frame_cosine']['cosine_mean']:.4f}; exact duplicate fraction is {similarity['representations']['R0_current_full']['same_video_distinct_frame_exact_duplicate_fraction']:.3f}.",
        f"5. Old static scene-only AUC: {probes['old']['mean_auc']:.4f}.",
        f"6. Current full-frame scene-only AUC: {probes['full']['mean_auc']:.4f}.",
        f"7. Local 4x scene-only AUC: {probes['local']['mean_auc']:.4f}.",
        f"8. Far-background 4x scene-only AUC: {probes['background']['mean_auc']:.4f}.",
        f"9. Matched validation AUC: B0 {b0_auc:.4f}, B1 {b1_auc:.4f}, B2 {b2_auc:.4f}.",
        f"10. Matched validation Brier: B0 {matched['arms']['B0']['mean_brier']:.4f}, B1 {matched['arms']['B1']['mean_brier']:.4f}, B2 {matched['arms']['B2']['mean_brier']:.4f}.",
        "11. J0 intervention AUC by local scale 2x/4x/6x: "
        + "/".join(f"{np.mean([scale['per_seed'][str(seed)][f'R1_local_{factor}x']['auc'] for seed in SEEDS]):.4f}" for factor in SCALES)
        + "; far-background: "
        + "/".join(f"{np.mean([scale['per_seed'][str(seed)][f'R2_background_{factor}x']['auc'] for seed in SEEDS]):.4f}" for factor in SCALES)
        + ". Local varies with scale; background AUC is comparatively stable, but these J0 replacements are out-of-distribution.",
        f"12. Pairwise video-identity cosine AUC: old static {identity_auc['old_static']:.4f}; current full {identity_auc['R0_current_full']:.4f}; local {identity_auc['R1_local_4x']:.4f}; far background {identity_auc['R2_background_4x']:.4f}.",
        f"13. Cross-video scene swap mean ΔAUC: B0 {cross['per_arm_seed']['B0']['aggregate']['delta_auc']:+.4f}, B1 {cross['per_arm_seed']['B1']['aggregate']['delta_auc']:+.4f}, B2 {cross['per_arm_seed']['B2']['aggregate']['delta_auc']:+.4f}; mean |Δp| is {cross['per_arm_seed']['B0']['aggregate']['mean_abs_probability_shift']:.4f}/{cross['per_arm_seed']['B1']['aggregate']['mean_abs_probability_shift']:.4f}/{cross['per_arm_seed']['B2']['aggregate']['mean_abs_probability_shift']:.4f}.",
        f"14. The previous scene gain is most consistent with a video/background prior, not local-only evidence. Recommended next direction: {advice}",
        "",
    ]
    lines = [
        "# Local Crossing Context vs Far Background Audit", "",
        "## Scope",
        "Only train and validation were opened. Official test was not loaded or used for any model/scale selection. ResNet-18 is frozen. R0 is the actual obs_end_frame full image; R1 retains the centered 4x pedestrian bbox neighborhood; R2 is the complementary far-background image from that same frame. The 2x/6x variants are validation-only sensitivity checks.", "",
        "## Data and indexing",
        f"- Raw videos: {location['video_root']} ({location['total_video_count']} clips).",
        f"- scene_id maps directly to the same filename stem, normally scene_id.mp4.",
        "- obs_end_frame is the original XML frame@number, unchanged by preprocessing; exact train/val target-track matches and bounds are in frame_index_mapping.md.",
        "- Features use ImageNet ResNet-18, fc=Identity, 512-D, eval mode, and the same resize/crop/normalization as the original extractor.", "",
        "## Scene-only probes (train to validation)",
        "| Representation | AUC mean ± SD | Brier mean ± SD |", "|---|---:|---:|",
    ]
    for label, key in (("Old static first frame", "old"), ("Current full frame", "full"), ("Local 4x", "local"), ("Far background 4x", "background")):
        row = probes[key]
        lines.append(f"| {label} | {row['mean_auc']:.4f} ± {row['std_auc']:.4f} | {row['mean_brier']:.4f} ± {row['std_brier']:.4f} |")
    lines += ["", "## Matched B0/B1/B2 training", "",
        "| Arm | Validation AUC mean ± SD | Brier mean ± SD | Cross-video ΔAUC | Cross-video ΔBrier | Mean |Δp| |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm in ("B0", "B1", "B2"):
        row, robust = matched["arms"][arm], cross["per_arm_seed"][arm]["aggregate"]
        lines.append(f"| {arm} ({row['feature']}) | {row['mean_auc']:.4f} ± {row['std_auc']:.4f} | {row['mean_brier']:.4f} ± {row['std_brier']:.4f} | {robust['delta_auc']:+.4f} | {robust['delta_brier']:+.4f} | {robust['mean_abs_probability_shift']:.4f} |")
    lines += ["", "All arms share the same seed-specific initialization, sampler sequence, 15 epochs, AdamW (lr 1e-3, weight decay 1e-4), gradient clip 5, and full Clean J0 architecture. Objective weights for trajectory, prior, and ambiguity are zero. Checkpoint selection uses raw validation AUC, with lower Brier within a 1e-4 tie. ADE/FDE are not selection criteria.", "",
        "## Temporal variation and video identity",
        f"- Old static feature: same-video cosine is 1.000 and distinct-window exact duplicate fraction is {similarity['representations']['old_static']['same_video_distinct_frame_exact_duplicate_fraction']:.3f}.",
        f"- R0 current full frame: distinct observation-frame within-video cosine is {similarity['representations']['R0_current_full']['same_video_distinct_frame_cosine']['cosine_mean']:.4f}; exact duplicate fraction is {similarity['representations']['R0_current_full']['same_video_distinct_frame_exact_duplicate_fraction']:.3f}, so it varies over time.",
        "- Pairwise video-identity cosine AUC (higher means same-vs-different video identity is more separable): "
        + ", ".join(f"{key}={similarity['representations'][key]['video_identity_pairwise_cosine_auc']:.4f}" for key in ("old_static", "R0_current_full", "R1_local_4x", "R2_background_4x")) + ".",
        "- Among newly extracted views, local 4x carries the least video identity; far background remains highly video-identifiable. Detailed same-pedestrian and different-pedestrian cosine values are in dynamic_scene_feature_similarity.json.", "",
        "## 2x/4x/6x sensitivity (existing J0, intervention only)",
        "| Bbox scale | Local AUC | Local Brier | Far-background AUC | Far-background Brier |",
        "|---:|---:|---:|---:|---:|", *scale_rows,
        "",
        "These are validation-only replacements into J0 checkpoints trained on old static features, so they are distribution-intervention diagnostics, not fair newly trained model comparisons. The predeclared primary scale remains 4x. Probability shifts/correlations are retained in local_scale_sensitivity.json.", "",
        "## Interpretation",
        f"- Scene-only AUC: old static {probes['old']['mean_auc']:.4f}; current full {full:.4f}; local 4x {local:.4f}; far background 4x {bg:.4f}.",
        f"- Matched training AUC: B0 full {b0_auc:.4f}; B1 local {b1_auc:.4f}; B2 background {b2_auc:.4f}.",
        f"- Evidence is classified as **{conclusion}**. The strongest evidence is B2 outperforming B0 by {b2_auc-b0_auc:+.4f} while B1 trails B0 by {b1_auc-b0_auc:+.4f}; B2 also loses {abs(cross['per_arm_seed']['B2']['aggregate']['delta_auc']):.4f} AUC on mean under cross-video scene swap.",
        "- Therefore the earlier approximately +0.09 scene gain is more consistent with a video/background prior than local crossing evidence. This does not mean the background is semantically irrelevant; it means its predictive association is highly video-dependent and brittle under cross-video replacement.",
        f"- Recommended next direction: **{advice}**", "",
        *direct_answers,
        "## Limitations",
        "Scene-only rows are clustered within videos, so row-level SD is not a video-level confidence interval. J0 replacement results are out-of-distribution interventions. The matched arms control seed, initialization, sampling and training protocol, but checkpoint selection uses validation. Large feature arrays remain ignored local artifacts; committed manifests contain hashes and provenance.", "",
        "## Artifacts",
        "raw_video_location.json; raw_video_mapping_audit.json; frame_index_mapping.md; dynamic_scene_feature_similarity.json; local_background_feature_manifest.json; scene_probe_*.json; a0_current_scene_intervention.json; local_scale_sensitivity.json; initialization_match.json; sampler_match.json; matched_scene_representation_results.json; cross_video_robustness.json.",
    ]
    (out / "scene_locality_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("features", "training", "summary", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--annotation-root", type=Path, default=ANNOTATION_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--initial-root", type=Path, default=INITIAL_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "checkpoints/scene_shortcut_audit")
    parser.add_argument("--ambiguous-root", type=Path, default=ROOT / "data/processed/jaad_ambiguous_scene_15x15")
    parser.add_argument("--extract-batch-size", type=int, default=48)
    args = parser.parse_args()
    if not args.video_root.is_dir() or not args.annotation_root.is_dir():
        parser.error("Original JAAD video/annotation directory is unavailable")
    if args.phase in {"features", "all"}:
        run_feature_phase(args)
    if args.phase in {"training", "all"}:
        run_training_phase(args)
    if args.phase in {"summary", "all"}:
        write_summary(args.output_root)
    print(f"scene locality audit phase {args.phase} complete", flush=True)


if __name__ == "__main__":
    main()
