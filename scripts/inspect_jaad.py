#!/usr/bin/env python3
"""Audit the downloaded JAAD 2.0 annotations before sequence generation.

This script only reads the raw JAAD files. It does not extract frames or
modify the downloaded dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


VIDEO_RE = re.compile(r"video_(\d+)")


def as_int(value: str | None, default: int = -1) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def video_name(path: Path) -> str:
    match = VIDEO_RE.search(path.stem)
    if not match:
        raise ValueError(f"Cannot infer video id from {path}")
    return f"video_{int(match.group(1)):04d}"


def parse_attributes(attributes_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    records: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(attributes_dir.glob("*_attributes.xml")):
        video = video_name(path)
        root = ET.parse(path).getroot()
        for pedestrian in root.findall(".//pedestrian"):
            ped_id = pedestrian.get("id")
            if not ped_id:
                continue
            crossing = as_int(pedestrian.get("crossing"))
            records[(video, ped_id)] = {
                "video_id": video,
                "ped_id": ped_id,
                "crossing": crossing,
                "mapped_intent": 1 if crossing == 1 else 0,
                "crossing_point": as_int(pedestrian.get("crossing_point")),
                "decision_point": as_int(pedestrian.get("decision_point")),
                "intersection": pedestrian.get("intersection", ""),
                "motion_direction": pedestrian.get("motion_direction", ""),
                "num_lanes": as_int(pedestrian.get("num_lanes"), 0),
            }
    return records


def box_attribute(box: ET.Element, name: str) -> str:
    for attribute in box.findall("attribute"):
        if attribute.get("name") == name:
            return (attribute.text or "").strip()
    return ""


def parse_tracks(annotation_dir: Path) -> list[dict[str, object]]:
    tracks: list[dict[str, object]] = []
    for path in sorted(annotation_dir.glob("video_*.xml")):
        video = video_name(path)
        root = ET.parse(path).getroot()
        for track in root.findall(".//track"):
            # JAAD uses ``ped`` for generic pedestrians and
            # ``pedestrian`` for pedestrians with behavior attributes.
            if track.get("label") != "pedestrian":
                continue
            boxes = track.findall("box")
            if not boxes:
                continue
            ped_id = box_attribute(boxes[0], "id")
            if not ped_id or not ped_id.endswith("b"):
                continue
            frames = [as_int(box.get("frame"), -1) for box in boxes]
            frames = [frame for frame in frames if frame >= 0]
            tracks.append(
                {
                    "video_id": video,
                    "ped_id": ped_id,
                    "num_boxes": len(boxes),
                    "frame_start": min(frames) if frames else -1,
                    "frame_end": max(frames) if frames else -1,
                }
            )
    return tracks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/lrj/ped_intent_project/results/data_audit"),
    )
    args = parser.parse_args()

    annotation_root = args.data_root / "annotations" / "JAAD_2.0"
    attributes_dir = annotation_root / "annotations_attributes"
    annotation_dir = annotation_root / "annotations"
    clips_dir = args.data_root / "JAAD_clips"

    for required in (attributes_dir, annotation_dir, clips_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required JAAD path does not exist: {required}")

    attributes = parse_attributes(attributes_dir)
    tracks = parse_tracks(annotation_dir)

    rows: list[dict[str, object]] = []
    missing_attributes = []
    for track in tracks:
        key = (str(track["video_id"]), str(track["ped_id"]))
        record = attributes.get(key)
        if record is None:
            missing_attributes.append(key)
            record = {
                "crossing": -1,
                "mapped_intent": 0,
                "crossing_point": -1,
                "decision_point": -1,
                "intersection": "",
                "motion_direction": "",
                "num_lanes": 0,
            }
        video = str(track["video_id"])
        row = {**track, **record}
        row["video_exists"] = (clips_dir / f"{video}.mp4").exists()
        rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "jaad_target_records.csv"
    fieldnames = [
        "video_id",
        "ped_id",
        "num_boxes",
        "frame_start",
        "frame_end",
        "crossing",
        "mapped_intent",
        "crossing_point",
        "decision_point",
        "intersection",
        "motion_direction",
        "num_lanes",
        "video_exists",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    crossing_counts = Counter(str(row["crossing"]) for row in rows)
    intersection_counts = Counter(str(row["intersection"]) for row in rows)
    summary = {
        "data_root": str(args.data_root),
        "attribute_records": len(attributes),
        "behavior_tracks": len(tracks),
        "tracks_missing_attributes": len(missing_attributes),
        "crossing_counts": dict(sorted(crossing_counts.items())),
        "mapped_intent_counts": {
            "non_crossing_or_no_clear_intent": sum(
                int(row["mapped_intent"]) == 0 for row in rows
            ),
            "crossing": sum(int(row["mapped_intent"]) == 1 for row in rows),
        },
        "intersection_counts": dict(sorted(intersection_counts.items())),
        "tracks_with_video": sum(bool(row["video_exists"]) for row in rows),
        "tracks_without_video": sum(not bool(row["video_exists"]) for row in rows),
        "track_length_frames": {
            "min": min((int(row["num_boxes"]) for row in rows), default=0),
            "max": max((int(row["num_boxes"]) for row in rows), default=0),
            "mean": round(
                sum(int(row["num_boxes"]) for row in rows) / len(rows), 2
            )
            if rows
            else 0.0,
        },
        "records_with_decision_point": sum(
            int(row["decision_point"]) >= 0 for row in rows
        ),
        "records_with_crossing_point": sum(
            int(row["crossing_point"]) >= 0 for row in rows
        ),
        "csv_path": str(csv_path),
    }
    summary_path = args.out_dir / "jaad_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
