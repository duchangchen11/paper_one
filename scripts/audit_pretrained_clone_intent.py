#!/usr/bin/env python3
"""Pre-training gates for clone parity, branch isolation, and trajectory preservation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrained_clone_intent_utils import (
    RESULTS_ROOT,
    build_pretrained_clone,
    encoder_state_sha256,
    head_state_sha256,
    load_config,
    model_sha256,
    state_sha256,
    trajectory_checkpoint_path,
    trajectory_model_sha256,
    write_json,
)
from scripts.trajectory_preserving_utils import (
    SEEDS,
    TrajectoryIntentDataset,
    set_seed,
    sha256_file,
    trajectory_metrics,
)
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def build_reference(seed: int, device: torch.device) -> SceneTrajectoryTransformer:
    config = load_config()["model"]["intention_branch"]
    path = trajectory_checkpoint_path(seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload["args"]["seed"]) != seed:
        raise RuntimeError(f"Reference trajectory checkpoint seed mismatch for {seed}")
    with torch.random.fork_rng(devices=[]):
        model = SceneTrajectoryTransformer(
            input_dim=8,
            scene_dim=512,
            d_model=int(config["hidden_dimension"]),
            nhead=int(config["heads"]),
            num_layers=int(config["layers"]),
            pred_len=15,
            dropout=float(config["dropout"]),
            max_obs_len=15,
        )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def trajectory_outputs(
    model: SceneTrajectoryTransformer,
    target: torch.Tensor,
    scene: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(target), 256):
            outputs.append(model(target[start : start + 256].to(device), scene[start : start + 256].to(device)).cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    config = load_config()
    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    train_set = TrajectoryIntentDataset(data_root / "train.npz")
    val_set = TrajectoryIntentDataset(data_root / "val.npz")
    n_validation = min(max(1024, int(config["trajectory_preservation"]["validation_samples_minimum"])), len(val_set))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clone_runs: dict[str, Any] = {}
    independence_runs: dict[str, Any] = {}
    trajectory_runs: dict[str, Any] = {}
    m0_initialization_matches: dict[str, bool] = {}

    for seed in SEEDS:
        model, init_report = build_pretrained_clone(seed, device="cpu")
        trajectory_branch = model.trajectory_branch
        intention_branch = model.intention_branch
        clone_runs[str(seed)] = init_report

        m0_init_path = RESULTS_ROOT.parent / "intention_scratch_matched" / f"seed{seed}/initialization_report.json"
        m0_initialization = json.loads(m0_init_path.read_text(encoding="utf-8"))
        m0_initialization_matches[str(seed)] = (
            init_report["random_m0_equivalent_initial_state_sha256_before_clone"]
            == m0_initialization["model_sha256_before_training"]
        )

        shared_names = init_report["parameter_independence"]["same_parameter_object_names"]
        storage_names = init_report["parameter_independence"]["same_storage_names"]
        if shared_names or storage_names:
            raise RuntimeError(f"M1 branches unexpectedly share parameters/storage for seed {seed}")

        # One smoke optimizer step, on intention parameters only.
        set_seed(seed + 99)
        optimizer = torch.optim.AdamW(
            intention_branch.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        trajectory_before = trajectory_model_sha256(trajectory_branch)
        intention_encoder_before = encoder_state_sha256(intention_branch)
        intention_all_before = model_sha256(intention_branch)
        trajectory_ids = {id(parameter) for parameter in trajectory_branch.parameters()}
        optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        train_batch = train_set.target[:8]
        train_labels = train_set.intent_label[:8]
        if len(train_batch) != 8:
            raise RuntimeError("M1 smoke step requires at least eight train samples")
        intention_branch.train()
        logits = intention_branch(train_batch)["intent_logit"]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(intention_branch.parameters(), float(config["training"]["gradient_clip_norm"]))
        optimizer.step()
        trajectory_after = trajectory_model_sha256(trajectory_branch)
        intention_encoder_after = encoder_state_sha256(intention_branch)
        intention_all_after = model_sha256(intention_branch)
        independence_row = {
            "initial_value_equal": init_report["parameter_independence"]["initial_value_equal"],
            "same_parameter_object": init_report["parameter_independence"]["same_parameter_object"],
            "same_storage": init_report["parameter_independence"]["same_storage"],
            "optimizer_contains_trajectory_parameter": bool(optimizer_ids & trajectory_ids),
            "optimizer_parameter_count": len(optimizer_ids),
            "intention_changed_after_step": intention_all_after != intention_all_before,
            "intent_encoder_changed_after_step": intention_encoder_after != intention_encoder_before,
            "trajectory_changed_after_step": trajectory_after != trajectory_before,
            "trajectory_hash_before": trajectory_before,
            "trajectory_hash_after": trajectory_after,
            "smoke_loss": float(loss.detach()),
        }
        independence_row["test_pass"] = (
            independence_row["initial_value_equal"]
            and independence_row["same_parameter_object"] is False
            and independence_row["same_storage"] is False
            and independence_row["optimizer_contains_trajectory_parameter"] is False
            and independence_row["intention_changed_after_step"]
            and independence_row["intent_encoder_changed_after_step"]
            and independence_row["trajectory_changed_after_step"] is False
        )
        independence_runs[str(seed)] = independence_row
        if not independence_row["test_pass"]:
            raise RuntimeError(f"M1 parameter independence smoke test failed for seed {seed}: {independence_row}")

        # Compare the independent original trajectory-only model and the
        # separately loaded M1 frozen trajectory branch on >=1024 val samples.
        reference = build_reference(seed, device)
        model.to(device).eval()
        target = val_set.target[:n_validation]
        scene = val_set.scene_feat[:n_validation]
        gt = val_set.future_gt[:n_validation].numpy()
        sizes = val_set.image_size[:n_validation].numpy()
        reference_future = trajectory_outputs(reference, target, scene, device=device).numpy()
        m1_future = trajectory_outputs(model.trajectory_branch, target, scene, device=device).numpy()
        reference_metrics = trajectory_metrics(reference_future, gt, sizes)
        m1_metrics = trajectory_metrics(m1_future, gt, sizes)
        max_abs_future_diff = float(abs(reference_future - m1_future).max())
        row = {
            "seed": seed,
            "trajectory_checkpoint_sha256": sha256_file(trajectory_checkpoint_path(seed)),
            "trajectory_backbone_sha256": trajectory_model_sha256(model.trajectory_branch),
            "validation_samples": n_validation,
            "max_abs_future_prediction_difference": max_abs_future_diff,
            "reference_validation_ade_pixel": reference_metrics["ade_pixel"],
            "m1_validation_ade_pixel": m1_metrics["ade_pixel"],
            "validation_ade_difference_pixel": m1_metrics["ade_pixel"] - reference_metrics["ade_pixel"],
            "reference_validation_fde_pixel": reference_metrics["fde_pixel"],
            "m1_validation_fde_pixel": m1_metrics["fde_pixel"],
            "validation_fde_difference_pixel": m1_metrics["fde_pixel"] - reference_metrics["fde_pixel"],
        }
        tol = config["trajectory_preservation"]
        row["pass"] = (
            max_abs_future_diff < tol["max_abs_future_prediction_difference"]
            and abs(row["validation_ade_difference_pixel"]) < tol["max_abs_ade_difference_pixel"]
            and abs(row["validation_fde_difference_pixel"]) < tol["max_abs_fde_difference_pixel"]
            and trajectory_model_sha256(model.trajectory_branch) == init_report["trajectory_branch_sha256"]
        )
        trajectory_runs[str(seed)] = row
        if not row["pass"]:
            raise RuntimeError(f"M1 trajectory equivalence gate failed for seed {seed}: {row}")

    clone_payload = {
        "experiment": config["experiment"],
        "all_seeds_clone_pass": all(row["clone_report"]["clone_pass"] for row in clone_runs.values()),
        "max_abs_tensor_diff": max(row["clone_report"]["max_abs_tensor_diff"] for row in clone_runs.values()),
        "m0_random_initialization_hash_matches": m0_initialization_matches,
        "all_m0_head_initializations_and_seeded_initial_models_match": all(m0_initialization_matches.values()),
        "seeds": clone_runs,
        "test_split_loaded": False,
    }
    independence_payload = {
        "all_seeds_test_pass": all(row["test_pass"] for row in independence_runs.values()),
        "seeds": independence_runs,
        "test_split_loaded": False,
    }
    trajectory_payload = {
        "comparison": "standalone trajectory checkpoint model vs independent frozen M1 trajectory branch",
        "minimum_validation_samples": 1024,
        "all_seeds_pass": all(row["pass"] for row in trajectory_runs.values()),
        "seeds": trajectory_runs,
        "test_split_loaded": False,
    }
    write_json(RESULTS_ROOT / "pretrained_clone_report.json", clone_payload)
    write_json(RESULTS_ROOT / "parameter_independence_test.json", independence_payload)
    write_json(RESULTS_ROOT / "trajectory_equivalence.json", trajectory_payload)

    failed = (
        not clone_payload["all_seeds_clone_pass"]
        or not clone_payload["all_m0_head_initializations_and_seeded_initial_models_match"]
        or not independence_payload["all_seeds_test_pass"]
        or not trajectory_payload["all_seeds_pass"]
    )
    print(
        json.dumps(
            {
                "clone_pass": clone_payload["all_seeds_clone_pass"],
                "clone_max_abs_tensor_diff": clone_payload["max_abs_tensor_diff"],
                "m0_init_match": m0_initialization_matches,
                "parameter_independence_pass": independence_payload["all_seeds_test_pass"],
                "trajectory_equivalence_pass": trajectory_payload["all_seeds_pass"],
                "validation_samples": n_validation,
                "test_split_loaded": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if failed:
        raise SystemExit("A pre-training M1 gate failed; formal training is forbidden")


if __name__ == "__main__":
    main()
