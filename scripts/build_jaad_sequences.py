#!/usr/bin/env python3
"""Build trajectory/intention samples from the downloaded JAAD 2.0 data.

The generated samples use pixel-coordinate normalization relative to the last
observed target position. Labels are taken from the annotation attributes,
never from the future trajectory input supplied to the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np


def as_int(value: str | None, default: int = -1) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def attr_value(box: ET.Element, name: str) -> str:
    for attribute in box.findall("attribute"):
        if attribute.get("name") == name:
            return (attribute.text or "").strip()
    return ""


def read_split(split_dir: Path, split: str) -> set[str]:
    path = split_dir / "default" / f"{split}.txt"
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def read_attributes(path: Path) -> dict[tuple[str, str], dict[str, int | str]]:
    records: dict[tuple[str, str], dict[str, int | str]] = {}
    for xml_path in sorted(path.glob("*_attributes.xml")):
        video = xml_path.stem.replace("_attributes", "")
        root = ET.parse(xml_path).getroot()
        for pedestrian in root.findall(".//pedestrian"):
            ped_id = pedestrian.get("id")
            if not ped_id:
                continue
            crossing = as_int(pedestrian.get("crossing"))
            records[(video, ped_id)] = {
                "crossing": crossing,
                "intent": 1 if crossing == 1 else 0,
                "crossing_point": as_int(pedestrian.get("crossing_point")),
                "decision_point": as_int(pedestrian.get("decision_point")),
            }
    return records


def read_tracks(path: Path) -> tuple[dict[str, list[dict]], dict[str, tuple[int, int]]]:
    tracks_by_video: dict[str, list[dict]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for xml_path in sorted(path.glob("video_*.xml")):
        video = xml_path.stem
        root = ET.parse(xml_path).getroot()
        size = root.find(".//original_size")
        if size is not None:
            sizes[video] = (
                as_int(size.get("width"), 1920),
                as_int(size.get("height"), 1080),
            )
        video_tracks = []
        for track in root.findall(".//track"):
            if track.get("label") not in {"ped", "pedestrian"}:
                continue
            boxes: dict[int, tuple[float, float]] = {}
            box_sizes: dict[int, tuple[float, float]] = {}
            ped_id = ""
            for box in track.findall("box"):
                if as_int(box.get("outside"), 0) == 1:
                    continue
                frame = as_int(box.get("frame"))
                if frame < 0:
                    continue
                ped_id = ped_id or attr_value(box, "id")
                xtl = as_float(box.get("xtl"))
                ytl = as_float(box.get("ytl"))
                xbr = as_float(box.get("xbr"))
                ybr = as_float(box.get("ybr"))
                boxes[frame] = ((xtl + xbr) / 2.0, (ytl + ybr) / 2.0)
                box_sizes[frame] = (max(0.0, xbr - xtl), max(0.0, ybr - ytl))
            if ped_id and boxes:
                video_tracks.append({"ped_id": ped_id, "boxes": boxes, "box_sizes": box_sizes})
        tracks_by_video[video] = video_tracks
    return tracks_by_video, sizes


def continuous_positions(
    boxes: dict[int, tuple[float, float]], frames: list[int]
) -> np.ndarray | None:
    if any(frame not in boxes for frame in frames):
        return None
    return np.asarray([boxes[frame] for frame in frames], dtype=np.float32)


def continuous_sizes(
    sizes: dict[int, tuple[float, float]], frames: list[int]
) -> np.ndarray | None:
    if any(frame not in sizes for frame in frames):
        return None
    return np.asarray([sizes[frame] for frame in frames], dtype=np.float32)


def features(positions: np.ndarray, origin: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([width, height], dtype=np.float32)
    relative = (positions - origin[None, :]) / scale[None, :]
    velocity = np.zeros_like(relative)
    if len(relative) > 1:
        velocity[1:] = np.diff(positions, axis=0) / scale[None, :]
    return np.concatenate([relative, velocity], axis=1).astype(np.float32)


def absolute_features(
    positions: np.ndarray,
    box_sizes: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    scale = np.asarray([width, height], dtype=np.float32)
    return np.concatenate([positions / scale[None, :], box_sizes / scale[None, :]], axis=1).astype(
        np.float32
    )


def build_split(
    split: str,
    split_videos: set[str],
    tracks_by_video: dict[str, list[dict]],
    sizes: dict[str, tuple[int, int]],
    attributes: dict[tuple[str, str], dict[str, int | str]],
    obs_len: int,
    pred_len: int,
    max_neighbors: int,
    min_visible_ratio: float,
    neighbor_radius_px: float,
    frame_step: int,
    label_policy: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, int]]:
    target_obs = []
    target_abs_obs = []
    future_gt = []
    neighbor_obs = []
    neighbor_mask = []
    neighbor_visible_mask = []
    intent_labels = []
    scene_ids = []
    target_ids = []
    obs_end_frames = []
    origins = []
    image_sizes = []
    metadata: list[dict[str, object]] = []
    counts = defaultdict(int)

    required_obs = math.ceil(obs_len * min_visible_ratio)
    total_span = (obs_len - 1 + pred_len) * frame_step

    for video in sorted(split_videos):
        tracks = tracks_by_video.get(video, [])
        width, height = sizes.get(video, (1920, 1080))
        behavior_tracks = [
            track
            for track in tracks
            if (video, track["ped_id"]) in attributes
        ]
        for target in behavior_tracks:
            attrs = attributes[(video, target["ped_id"])]
            crossing = int(attrs["crossing"])
            if label_policy == "clean" and crossing not in {0, 1}:
                counts["skipped_ambiguous_label"] += 1
                continue
            if label_policy == "unknown_only" and crossing != -1:
                continue
            boxes = target["boxes"]
            frame_start = min(boxes)
            frame_end = max(boxes)
            for obs_end in range(frame_start + (obs_len - 1) * frame_step,
                                 frame_end - pred_len * frame_step + 1):
                obs_frames = [
                    obs_end - (obs_len - 1 - index) * frame_step
                    for index in range(obs_len)
                ]
                future_frames = [obs_end + index * frame_step for index in range(1, pred_len + 1)]
                target_obs_pos = continuous_positions(boxes, obs_frames)
                target_obs_sizes = continuous_sizes(target["box_sizes"], obs_frames)
                future_pos = continuous_positions(boxes, future_frames)
                if target_obs_pos is None or target_obs_sizes is None or future_pos is None:
                    counts["skipped_target_missing_frame"] += 1
                    continue

                decision_point = int(attrs["decision_point"])
                crossing_point = int(attrs["crossing_point"])
                cutoff = decision_point if decision_point >= 0 else crossing_point
                if cutoff >= 0 and obs_end >= cutoff:
                    counts["skipped_after_behavior_point"] += 1
                    continue

                origin = target_obs_pos[-1]
                target_feat = features(target_obs_pos, origin, width, height)
                target_abs_feat = absolute_features(target_obs_pos, target_obs_sizes, width, height)
                future_feat = (future_pos - origin[None, :]) / np.asarray(
                    [width, height], dtype=np.float32
                )

                candidates = []
                for neighbor in tracks:
                    if neighbor["ped_id"] == target["ped_id"]:
                        continue
                    visible_frames = [frame for frame in obs_frames if frame in neighbor["boxes"]]
                    if len(visible_frames) < required_obs or obs_end not in neighbor["boxes"]:
                        continue
                    neighbor_last = np.asarray(neighbor["boxes"][obs_end], dtype=np.float32)
                    distance = float(np.linalg.norm(neighbor_last - origin))
                    if distance <= neighbor_radius_px:
                        candidates.append((distance, neighbor))
                candidates.sort(key=lambda item: item[0])
                candidates = candidates[:max_neighbors]

                n_features = np.zeros((max_neighbors, obs_len, 4), dtype=np.float32)
                n_mask = np.zeros((max_neighbors,), dtype=np.float32)
                n_visible = np.zeros((max_neighbors, obs_len), dtype=np.float32)
                for n_index, (_, neighbor) in enumerate(candidates):
                    positions = np.zeros((obs_len, 2), dtype=np.float32)
                    visible = np.zeros((obs_len,), dtype=np.float32)
                    for t_index, frame in enumerate(obs_frames):
                        if frame in neighbor["boxes"]:
                            positions[t_index] = neighbor["boxes"][frame]
                            visible[t_index] = 1.0
                    first_valid = np.flatnonzero(visible)
                    if len(first_valid):
                        first_position = positions[first_valid[0]]
                        positions[visible == 0] = first_position
                    n_features[n_index] = features(positions, origin, width, height)
                    n_mask[n_index] = 1.0
                    n_visible[n_index] = visible

                target_obs.append(target_feat)
                target_abs_obs.append(target_abs_feat)
                future_gt.append(future_feat)
                neighbor_obs.append(n_features)
                neighbor_mask.append(n_mask)
                neighbor_visible_mask.append(n_visible)
                intent_label = -1 if crossing == -1 else int(attrs["intent"])
                intent_labels.append(intent_label)
                scene_ids.append(video)
                target_ids.append(target["ped_id"])
                obs_end_frames.append(obs_end)
                origins.append(origin)
                image_sizes.append((width, height))
                metadata.append(
                    {
                        "split": split,
                        "video_id": video,
                        "target_id": target["ped_id"],
                        "obs_end_frame": obs_end,
                        "intent_label": intent_label,
                        "crossing": crossing,
                        "decision_point": decision_point,
                        "crossing_point": crossing_point,
                        "num_neighbors": len(candidates),
                    }
                )
                counts["samples"] += 1
                counts[f"intent_{intent_label}"] += 1

    arrays = {
        "target_obs": np.asarray(target_obs, dtype=np.float32),
        "target_abs_obs": np.asarray(target_abs_obs, dtype=np.float32),
        "future_gt": np.asarray(future_gt, dtype=np.float32),
        "neighbor_obs": np.asarray(neighbor_obs, dtype=np.float32),
        "neighbor_mask": np.asarray(neighbor_mask, dtype=np.float32),
        "neighbor_visible_mask": np.asarray(neighbor_visible_mask, dtype=np.float32),
        "intent_label": np.asarray(intent_labels, dtype=np.int64),
        "crossing_label": np.asarray(
            [int(row["crossing"]) for row in metadata], dtype=np.int64
        ),
        "scene_id": np.asarray(scene_ids),
        "target_id": np.asarray(target_ids),
        "obs_end_frame": np.asarray(obs_end_frames, dtype=np.int64),
        "origin_xy": np.asarray(origins, dtype=np.float32),
        "image_size": np.asarray(image_sizes, dtype=np.int64),
    }
    return arrays, metadata, dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lrj/ped_intent_project/data/processed/jaad_sequences"),
    )
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--min-visible-ratio", type=float, default=0.8)
    parser.add_argument("--neighbor-radius-px", type=float, default=300.0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--label-policy",
        choices=("clean", "merge_unknown", "unknown_only"),
        default="clean",
        help="clean=use crossing 0/1 only; merge_unknown=legacy -1 to 0; unknown_only=export crossing -1 samples",
    )
    args = parser.parse_args()

    annotation_root = args.data_root / "annotations" / "JAAD_2.0"
    attributes = read_attributes(annotation_root / "annotations_attributes")
    tracks_by_video, sizes = read_tracks(annotation_root / "annotations")
    split_dir = annotation_root / "split_ids"
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_summary = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "config": {
            "obs_len": args.obs_len,
            "pred_len": args.pred_len,
            "max_neighbors": args.max_neighbors,
            "min_visible_ratio": args.min_visible_ratio,
            "neighbor_radius_px": args.neighbor_radius_px,
            "frame_step": args.frame_step,
            "split_policy": "official JAAD default video split",
            "label_policy": args.label_policy,
            "intent_mapping": (
                "crossing=1 -> 1; crossing=0 -> 0; crossing=-1 excluded"
                if args.label_policy == "clean"
                else "crossing=-1 -> -1" if args.label_policy == "unknown_only"
                else "crossing=1 -> 1; crossing=0/-1 -> 0"
            ),
        },
        "splits": {},
    }

    for split in ("train", "val", "test"):
        split_videos = read_split(split_dir, split)
        arrays, metadata, counts = build_split(
            split,
            split_videos,
            tracks_by_video,
            sizes,
            attributes,
            args.obs_len,
            args.pred_len,
            args.max_neighbors,
            args.min_visible_ratio,
            args.neighbor_radius_px,
            args.frame_step,
            args.label_policy,
        )
        np.savez_compressed(args.output_root / f"{split}.npz", **arrays)
        with (args.output_root / f"{split}_metadata.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            fieldnames = list(metadata[0]) if metadata else []
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(metadata)
        run_summary["splits"][split] = {
            "videos": len(split_videos),
            "counts": counts,
            "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        }

    summary_path = args.output_root / "build_summary.json"
    summary_path.write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
