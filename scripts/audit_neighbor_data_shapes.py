#!/usr/bin/env python3
"""Audit processed JAAD tensor shapes and social-neighbor statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARRAY_KEYS = (
    "target_obs",
    "target_abs_obs",
    "neighbor_obs",
    "neighbor_mask",
    "neighbor_visible_mask",
    "future_gt",
    "scene_feat",
    "intent_label",
)


def audit_split(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in ARRAY_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path} is missing required arrays: {missing}")
        labels = data["intent_label"].astype(np.int64)
        neighbor_mask = data["neighbor_mask"]
        neighbors_per_sample = neighbor_mask.sum(axis=1)
        return {
            "sample_count": int(len(labels)),
            "positive_count": int((labels == 1).sum()),
            "negative_count": int((labels == 0).sum()),
            "other_label_count": int(((labels != 0) & (labels != 1)).sum()),
            "average_valid_neighbors": float(neighbors_per_sample.mean()),
            "no_neighbor_sample_ratio": float((neighbors_per_sample == 0).mean()),
            "shapes": {key: list(data[key].shape) for key in ARRAY_KEYS},
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "data_root": str(args.data_root),
        "neighbor_obs_layout": "[batch, neighbor, time, feature]",
        "splits": {
            split: audit_split(args.data_root / f"{split}.npz")
            for split in ("train", "val", "test")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
