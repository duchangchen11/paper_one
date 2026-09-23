"""Extract frozen video-level ResNet scene embeddings for processed JAAD samples."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


def extract_split(
    input_path: Path,
    output_path: Path,
    video_root: Path,
    backbone,
    transform,
    device: torch.device,
    batch_size: int,
) -> int:
    data = np.load(input_path, allow_pickle=False)
    scenes = data["scene_id"].astype(str)
    unique_scenes = sorted(set(scenes.tolist()))
    feature_by_scene: dict[str, np.ndarray] = {}
    missing = 0
    image_batch, scene_batch = [], []

    def flush() -> None:
        nonlocal missing
        if not image_batch:
            return
        with torch.no_grad():
            tensor = torch.stack(image_batch).to(device)
            embeddings = backbone(tensor).cpu().numpy().astype(np.float16)
        for scene, embedding in zip(scene_batch, embeddings):
            feature_by_scene[scene] = embedding
        image_batch.clear()
        scene_batch.clear()

    for scene in unique_scenes:
        capture = cv2.VideoCapture(str(video_root / f"{scene}.mp4"))
        ok, frame = capture.read()
        capture.release()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_batch.append(transform(Image.fromarray(rgb)))
            scene_batch.append(scene)
            if len(image_batch) >= batch_size:
                flush()
        else:
            missing += 1
            feature_by_scene[scene] = np.zeros((512,), dtype=np.float16)
    flush()

    scene_feat = np.stack(
        [feature_by_scene[scene] for scene in scenes],
        axis=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: data[key] for key in data.files}
    arrays["scene_feat"] = scene_feat
    np.savez_compressed(output_path, **arrays)
    print(f"{input_path.name}: samples={len(scenes)} unique_scenes={len(unique_scenes)} missing={missing}")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    transform = weights.transforms()
    total_missing = 0
    for split in ("train", "val", "test"):
        total_missing += extract_split(
            args.input_root / f"{split}.npz",
            args.output_root / f"{split}.npz",
            args.video_root,
            backbone,
            transform,
            device,
            args.batch_size,
        )
    print(f"device={device} total_missing={total_missing} output_root={args.output_root}")


if __name__ == "__main__":
    main()
