#!/usr/bin/env python3
"""Prove matched-seed P1/P2 trajectory paths equal the standalone checkpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trajectory_preserving_utils import (
    TrajectoryIntentDataset,
    backbone_sha256,
    load_seed_backbone,
    trajectory_metrics,
)
from src.models.trajectory_transformer import SceneTrajectoryTransformer

SEEDS = (42, 123, 2024)
SAMPLE_COUNT = 1024
MAX_ABS_TOLERANCE = 1e-6
METRIC_TOLERANCE = 1e-5


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = ROOT / "results/trajectory_preserving_joint"
    val_set = TrajectoryIntentDataset(
        ROOT / "data/processed/jaad_sequences_scene_15x15/val.npz"
    )
    if len(val_set) < SAMPLE_COUNT:
        raise RuntimeError(f"Need at least {SAMPLE_COUNT} validation samples; got {len(val_set)}")
    batch = {
        key: getattr(val_set, key)[:SAMPLE_COUNT].to(device)
        for key in ("target", "scene_feat", "future_gt", "image_size")
    }
    per_seed = {}
    loading_reports = []
    all_pass = True

    for seed in SEEDS:
        source_path = ROOT / "checkpoints" / f"trajectory_transformer_scene_15x15_seed{seed}.pt"
        source_payload = torch.load(source_path, map_location="cpu", weights_only=False)
        args = source_payload["args"]
        standalone = SceneTrajectoryTransformer(
            input_dim=int(batch["target"].shape[-1]),
            scene_dim=int(batch["scene_feat"].shape[-1]),
            d_model=int(args["d_model"]),
            num_layers=int(args["num_layers"]),
            pred_len=int(batch["future_gt"].shape[1]),
            max_obs_len=int(batch["target"].shape[1]),
        ).to(device)
        standalone.load_state_dict(source_payload["model"], strict=True)
        standalone.eval()

        with torch.no_grad():
            original_prediction = standalone(batch["target"], batch["scene_feat"])
        original_array = original_prediction.detach().cpu().numpy()
        original_metrics = trajectory_metrics(
            original_array,
            batch["future_gt"].cpu().numpy(),
            batch["image_size"].cpu().numpy(),
        )
        arm_results = {}
        for method, intent_input in (("P1_target_only", "target"), ("P2_target_scene", "target_scene")):
            preserving, loading_report, checkpoint_path, checkpoint_sha = load_seed_backbone(
                seed,
                intent_input,
                device=device,
                input_dim=int(batch["target"].shape[-1]),
                scene_dim=int(batch["scene_feat"].shape[-1]),
                observed_length=int(batch["target"].shape[1]),
                prediction_length=int(batch["future_gt"].shape[1]),
            )
            preserving.eval()
            with torch.no_grad():
                candidate_prediction = preserving(batch["target"], batch["scene_feat"])["future_pred"]
            candidate_array = candidate_prediction.detach().cpu().numpy()
            candidate_metrics = trajectory_metrics(
                candidate_array,
                batch["future_gt"].cpu().numpy(),
                batch["image_size"].cpu().numpy(),
            )
            difference = np.abs(original_array - candidate_array)
            max_abs = float(difference.max(initial=0.0))
            mean_abs = float(difference.mean())
            ade_difference = candidate_metrics["ade_pixel"] - original_metrics["ade_pixel"]
            fde_difference = candidate_metrics["fde_pixel"] - original_metrics["fde_pixel"]
            passed = (
                max_abs < MAX_ABS_TOLERANCE
                and abs(ade_difference) < METRIC_TOLERANCE
                and abs(fde_difference) < METRIC_TOLERANCE
            )
            arm_results[method] = {
                "max_abs_difference": max_abs,
                "mean_abs_difference": mean_abs,
                "trajectory_only_ade_pixel": original_metrics["ade_pixel"],
                "preserving_model_ade_pixel": candidate_metrics["ade_pixel"],
                "ade_difference_pixel": ade_difference,
                "trajectory_only_fde_pixel": original_metrics["fde_pixel"],
                "preserving_model_fde_pixel": candidate_metrics["fde_pixel"],
                "fde_difference_pixel": fde_difference,
                "pass": passed,
            }
            loading_reports.append(
                {
                    "method": method,
                    "seed": seed,
                    "source_checkpoint": str(checkpoint_path.relative_to(ROOT)),
                    "source_checkpoint_sha256": checkpoint_sha,
                    "backbone_sha256_after_load": backbone_sha256(preserving),
                    **loading_report,
                }
            )
            all_pass = all_pass and passed
        per_seed[str(seed)] = {"samples": SAMPLE_COUNT, "arms": arm_results}

    result = {
        "samples_per_seed": SAMPLE_COUNT,
        "validation_only": True,
        "max_abs_difference_threshold": MAX_ABS_TOLERANCE,
        "ade_fde_difference_threshold_pixel": METRIC_TOLERANCE,
        "device": str(device),
        "per_seed": per_seed,
        "pass": all_pass,
    }
    (output / "equivalence_test.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "weight_loading_report.json").write_text(
        json.dumps(
            {
                "principle": "All pretrained backbone state tensors must load exactly; new intention-only tensors are explicitly initialized for the new task.",
                "reports": loading_reports,
                "all_backbone_loads_complete": all(row["complete"] for row in loading_reports),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all_pass:
        raise SystemExit("Initialization equivalence failed; do not train P1/P2")


if __name__ == "__main__":
    main()
