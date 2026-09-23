#!/usr/bin/env python3
"""Visualize model predictions over original JAAD video frames."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.scene_social_gate import SceneSocialGate
from src.models.uncertainty_social_gate import UncertaintySocialGate


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
    tracks: dict[str, dict[int, tuple[float, float]]] = {}
    for track in root.findall(".//track"):
        if track.get("label") not in {"ped", "pedestrian"}:
            continue
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
            boxes[frame] = ((xtl + xbr) / 2.0, (ytl + ybr) / 2.0)
        if ped_id and boxes:
            tracks[ped_id] = boxes
    return tracks


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 4,
) -> None:
    if len(points) >= 2:
        array = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [array], False, color, thickness, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1, cv2.LINE_AA)


def select_indices(labels: np.ndarray, count: int) -> list[int]:
    selected: list[int] = []
    per_class = max(1, count // 2)
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        if len(indices) == 0:
            continue
        take = min(per_class, len(indices))
        positions = np.linspace(0, len(indices) - 1, take, dtype=int)
        selected.extend(indices[positions].tolist())
    return [int(index) for index in selected[:count]]


def entropy(prob: float) -> float:
    prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(-prob * np.log(prob) - (1.0 - prob) * np.log(1.0 - prob))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sequence-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-model",
        action="store_true",
        help="Load SceneSocialGate and use the stored scene_feat vectors.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = JAADSequenceDataset(args.sequence_root / f"{args.split}.npz")
    loader_indices = select_indices(dataset.intent_label.numpy().astype(np.int64), args.num_samples)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    target_features = saved_args.get("target_features")
    if target_features is None:
        # The scene-model training script uses relative_abs features but older
        # checkpoints did not store this argument explicitly.
        target_input_dim = int(checkpoint["model"]["target_encoder.weight_ih_l0"].shape[1])
        target_features = "relative_abs" if target_input_dim == 8 else "relative"
    gate_mode = saved_args.get("gate_mode", "uncertainty")
    hidden_dim = int(saved_args.get("hidden_dim", 128))
    input_dim = 8 if target_features == "relative_abs" else 4
    if args.scene_model:
        scene_dim = int(dataset.scene_feat.shape[-1])
        if scene_dim <= 0:
            raise ValueError("--scene-model requires scene_feat in the sequence npz")
        model = SceneSocialGate(
            input_dim=input_dim,
            scene_dim=scene_dim,
            hidden_dim=hidden_dim,
            pred_len=dataset.future_gt.shape[1],
            gate_mode=gate_mode,
        ).to(device)
    else:
        model = UncertaintySocialGate(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pred_len=dataset.future_gt.shape[1],
            gate_mode=gate_mode,
        ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    sequence_data = np.load(args.sequence_root / f"{args.split}.npz", allow_pickle=False)

    annotation_root = args.data_root / "annotations" / "JAAD_2.0" / "annotations"
    video_root = args.data_root / "JAAD_clips"
    track_cache: dict[str, dict[str, dict[int, tuple[float, float]]]] = {}
    written = 0

    for output_index, sample_index in enumerate(loader_indices):
        batch = dataset[sample_index]
        target_obs = batch["target_obs"]
        if target_features == "relative_abs":
            target_obs = torch.cat([target_obs, batch["target_abs_obs"]], dim=-1)
        target_obs = target_obs.unsqueeze(0).to(device)
        neighbor_obs = batch["neighbor_obs"].unsqueeze(0).to(device)
        neighbor_mask = batch["neighbor_mask"].unsqueeze(0).to(device)
        visible_mask = batch["neighbor_visible_mask"].unsqueeze(0).to(device)
        scene_feat = batch["scene_feat"].unsqueeze(0).to(device)
        with torch.no_grad():
            if args.scene_model:
                output = model(
                    target_obs,
                    neighbor_obs,
                    neighbor_mask,
                    visible_mask,
                    scene_feat,
                )
            else:
                output = model(target_obs, neighbor_obs, neighbor_mask, visible_mask)

        prob = float(torch.sigmoid(output["intent_logit"])[0].cpu())
        ent = float(output["entropy"][0].cpu())
        gate = float(output["gate"][0].cpu())
        pred_norm = output["future_pred"][0].cpu().numpy()
        gt_norm = batch["future_gt"].numpy()
        origin = batch["target_obs"].new_tensor(sequence_data["origin_xy"][sample_index]).numpy()
        image_size = sequence_data["image_size"][sample_index]
        width, height = int(image_size[0]), int(image_size[1])
        scale = np.asarray([width, height], dtype=np.float32)
        pred_pixels = origin[None, :] + pred_norm * scale[None, :]
        gt_pixels = origin[None, :] + gt_norm * scale[None, :]
        ade = float(np.linalg.norm(pred_norm - gt_norm, axis=-1).mean())
        fde = float(np.linalg.norm(pred_norm[-1] - gt_norm[-1]))

        video = str(sequence_data["scene_id"][sample_index])
        target_id = str(sequence_data["target_id"][sample_index])
        frame_id = int(sequence_data["obs_end_frame"][sample_index])
        label = int(batch["intent_label"].item())
        if video not in track_cache:
            track_cache[video] = load_tracks(annotation_root / f"{video}.xml")
        tracks = track_cache[video]
        capture = cv2.VideoCapture(str(video_root / f"{video}.mp4"))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = capture.read()
        capture.release()
        if not ok:
            continue

        for ped_id, boxes in tracks.items():
            if ped_id == target_id or frame_id not in boxes:
                continue
            x, y = boxes[frame_id]
            cv2.circle(image, (int(x), int(y)), 10, (150, 150, 150), 2, cv2.LINE_AA)

        target_boxes = tracks.get(target_id, {})
        history = np.asarray(
            [target_boxes[frame] for frame in range(frame_id - 7, frame_id + 1) if frame in target_boxes],
            dtype=np.float32,
        )
        draw_polyline(image, history, (255, 0, 0))
        draw_polyline(image, gt_pixels, (0, 220, 0))
        draw_polyline(image, pred_pixels, (0, 140, 255))
        if frame_id in target_boxes:
            x, y = target_boxes[frame_id]
            cv2.rectangle(image, (int(x) - 28, int(y) - 55), (int(x) + 28, int(y) + 55), (0, 0, 255), 3)

        lines = [
            f"{video} | {target_id} | frame={frame_id} | gt_intent={label} | scene_model={args.scene_model}",
            f"p_cross={prob:.3f} | pred_intent={int(prob >= 0.5)} | entropy={ent:.3f} | gate={gate:.3f}",
            f"ADE_norm={ade:.4f} | FDE_norm={fde:.4f}",
            "blue=history | green=ground truth future | orange=model prediction | gray=other pedestrians",
        ]
        for line_index, text in enumerate(lines):
            y = 42 + line_index * 34
            cv2.putText(image, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        output_path = args.output_dir / f"{output_index:03d}_{video}_{frame_id}_gt{label}.jpg"
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        written += 1

    print(f"requested={len(loader_indices)} written={written} output_dir={args.output_dir} device={device}")


if __name__ == "__main__":
    main()
