#!/usr/bin/env python3
"""Validate generated JAAD sequence files."""

from pathlib import Path

import numpy as np


def main() -> None:
    root = Path("/home/lrj/ped_intent_project/data/processed/jaad_sequences")
    scene_sets = {}
    for split in ("train", "val", "test"):
        data = np.load(root / f"{split}.npz")
        scene_sets[split] = set(data["scene_id"].astype(str).tolist())
        print(f"[{split}]")
        for key in (
            "target_obs",
            "target_abs_obs",
            "future_gt",
            "neighbor_obs",
            "neighbor_mask",
            "neighbor_visible_mask",
            "intent_label",
            "crossing_label",
        ):
            array = data[key]
            print(
                f"  {key}: shape={array.shape}, "
                f"finite={bool(np.isfinite(array).all())}"
            )
        valid_labels = data["intent_label"][data["intent_label"] >= 0]
        counts = np.bincount(valid_labels, minlength=2)
        neighbor_counts = data["neighbor_mask"].sum(axis=1)
        print(f"  intent_counts={counts.tolist()}")
        print(
            f"  raw_crossing_counts={dict(zip(*np.unique(data['crossing_label'], return_counts=True)))}"
        )
        print(
            "  neighbor_count="
            f"min:{neighbor_counts.min():.0f}, "
            f"max:{neighbor_counts.max():.0f}, "
            f"mean:{neighbor_counts.mean():.2f}, "
            f"zero_ratio:{(neighbor_counts == 0).mean():.3f}"
        )
        print(
            f"  target_obs_range=({data['target_obs'].min():.4f}, "
            f"{data['target_obs'].max():.4f})"
        )
        print(
            f"  future_gt_range=({data['future_gt'].min():.4f}, "
            f"{data['future_gt'].max():.4f})"
        )
        print(
            f"  target_abs_obs_range=({data['target_abs_obs'].min():.4f}, "
            f"{data['target_abs_obs'].max():.4f})"
        )

    print("scene_overlap_train_val=", sorted(scene_sets["train"] & scene_sets["val"]))
    print("scene_overlap_train_test=", sorted(scene_sets["train"] & scene_sets["test"]))
    print("scene_overlap_val_test=", sorted(scene_sets["val"] & scene_sets["test"]))


if __name__ == "__main__":
    main()
