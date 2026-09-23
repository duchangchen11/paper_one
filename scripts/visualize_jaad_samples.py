#!/usr/bin/env python3
"""Draw target tracks from generated JAAD samples on raw video frames."""

from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def as_int(value: str | None, default: int = -1) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def attr_value(box: ET.Element, name: str) -> str:
    for attribute in box.findall("attribute"):
        if attribute.get("name") == name:
            return (attribute.text or "").strip()
    return ""


def load_tracks(annotation_path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    root = ET.parse(annotation_path).getroot()
    tracks = {}
    for track in root.findall(".//track"):
        boxes = {}
        ped_id = ""
        for box in track.findall("box"):
            if as_int(box.get("outside"), 0) == 1:
                continue
            ped_id = ped_id or attr_value(box, "id")
            frame = as_int(box.get("frame"))
            if frame < 0:
                continue
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            boxes[frame] = ((xtl + xbr) / 2, (ytl + ybr) / 2)
        if ped_id and boxes:
            tracks[ped_id] = boxes
    return tracks


def draw_polyline(image: np.ndarray, points: list[tuple[float, float]], color: tuple[int, int, int]) -> None:
    if len(points) >= 2:
        array = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [array], False, color, 3, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 4, color, -1, cv2.LINE_AA)


def select_indices(labels: np.ndarray, count: int) -> list[int]:
    selected = []
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) == 0:
            continue
        take = min(count // 2, len(indices))
        positions = np.linspace(0, len(indices) - 1, take, dtype=int)
        selected.extend(indices[positions].tolist())
    return [int(index) for index in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.sequence_root / f"{args.split}.npz")
    labels = data["intent_label"]
    indices = select_indices(labels, args.num_samples)
    annotation_root = args.data_root / "annotations" / "JAAD_2.0" / "annotations"
    video_root = args.data_root / "JAAD_clips"

    written = 0
    cache: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
    for output_index, sample_index in enumerate(indices):
        video = str(data["scene_id"][sample_index])
        target_id = str(data["target_id"][sample_index])
        frame_id = int(data["obs_end_frame"][sample_index])
        if video not in cache:
            cache[video] = load_tracks(annotation_root / f"{video}.xml")
        tracks = cache[video]

        capture = cv2.VideoCapture(str(video_root / f"{video}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
        capture.release()
        if not ok:
            continue

        for ped_id, boxes in tracks.items():
            if ped_id == target_id:
                continue
            if frame_id in boxes:
                x, y = boxes[frame_id]
                cv2.circle(image, (int(x), int(y)), 10, (150, 150, 150), 2, cv2.LINE_AA)

        target_boxes = tracks.get(target_id, {})
        obs_frames = range(frame_id - 7, frame_id + 1)
        future_frames = range(frame_id + 1, frame_id + 13)
        history = [target_boxes[frame] for frame in obs_frames if frame in target_boxes]
        future = [target_boxes[frame] for frame in future_frames if frame in target_boxes]
        draw_polyline(image, history, (255, 0, 0))
        draw_polyline(image, future, (0, 220, 0))
        if frame_id in target_boxes:
            x, y = target_boxes[frame_id]
            cv2.rectangle(image, (int(x) - 25, int(y) - 50), (int(x) + 25, int(y) + 50), (0, 0, 255), 3)

        label = int(labels[sample_index])
        text = f"{video} | {target_id} | frame={frame_id} | intent={label}"
        cv2.putText(image, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        output_path = args.output_dir / f"{output_index:03d}_{video}_{frame_id}_intent{label}.jpg"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1

    print(f"requested={len(indices)} written={written} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
