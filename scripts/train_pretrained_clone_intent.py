#!/usr/bin/env python3
"""Train M1's cloned pretrained intention encoder without touching test data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrained_clone_intent_utils import (
    CHECKPOINT_ROOT,
    RESULTS_ROOT,
    build_pretrained_clone,
    encoder_drift,
    encoder_state_sha256,
    head_state_sha256,
    load_config,
    model_sha256,
    trajectory_checkpoint_path,
    trajectory_model_sha256,
    write_json,
)
from scripts.trajectory_preserving_utils import (
    SEEDS,
    TrajectoryIntentDataset,
    intention_metrics,
    probabilities_from_logits,
    sha256_file,
    trajectory_metrics,
)
from scripts.reliability_gated_intent_utils import choose_balanced_accuracy_threshold, fit_temperature


def evaluate_validation(model, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_future: list[np.ndarray] = []
    all_ground_truth: list[np.ndarray] = []
    all_sizes: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            intent_output = model.forward_intention(target)
            future = model.forward_trajectory(target, batch["scene_feat"].to(device))
            all_logits.append(intent_output["intent_logit"].cpu().numpy())
            all_labels.append(batch["intent_label"].numpy())
            all_future.append(future.cpu().numpy())
            all_ground_truth.append(batch["future_gt"].numpy())
            all_sizes.append(batch["image_size"].numpy())
    logits = np.concatenate(all_logits).astype(np.float64)
    labels = np.concatenate(all_labels).astype(np.int64)
    prediction = np.concatenate(all_future).astype(np.float32)
    ground_truth = np.concatenate(all_ground_truth).astype(np.float32)
    image_size = np.concatenate(all_sizes).astype(np.float32)
    return (
        intention_metrics(labels, logits),
        trajectory_metrics(prediction, ground_truth, image_size),
        logits,
        labels,
        prediction,
        ground_truth,
        image_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    seed = args.seed
    config = load_config()
    gate_dir = RESULTS_ROOT
    clone_gate = json.loads((gate_dir / "pretrained_clone_report.json").read_text(encoding="utf-8"))
    independence_gate = json.loads((gate_dir / "parameter_independence_test.json").read_text(encoding="utf-8"))
    trajectory_gate = json.loads((gate_dir / "trajectory_equivalence.json").read_text(encoding="utf-8"))
    if not clone_gate.get("all_seeds_clone_pass") or not clone_gate.get("all_m0_head_initializations_and_seeded_initial_models_match"):
        raise RuntimeError("Pretraining clone/M0 initialization audit must pass before M1 training")
    if not independence_gate.get("all_seeds_test_pass") or not trajectory_gate.get("all_seeds_pass"):
        raise RuntimeError("Parameter-independence and trajectory-preservation gates must pass before training")
    if not clone_gate["seeds"][str(seed)]["clone_report"]["clone_pass"]:
        raise RuntimeError(f"Pretrained clone gate is missing for seed {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    train_set = TrajectoryIntentDataset(data_root / "train.npz")
    val_set = TrajectoryIntentDataset(data_root / "val.npz")
    train_config = config["training"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    labels = train_set.intent_label.numpy().astype(np.int64)
    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())
    if min(positive_count, negative_count) <= 0:
        raise RuntimeError("Both intention classes must be present in M1 training split")
    class_weights = {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive": len(labels) / (2.0 * positive_count),
        "negative": len(labels) / (2.0 * negative_count),
        "sampling": "natural shuffle=True; inverse-frequency weighted BCE; no WeightedRandomSampler",
    }

    # build_pretrained_clone seeds and constructs the M0-shaped intention
    # branch first, clones only the encoder, and preserves the RNG stream.
    model, initialization = build_pretrained_clone(seed, device=device)
    m0_init_report = json.loads(
        (RESULTS_ROOT.parent / "intention_scratch_matched" / f"seed{seed}/initialization_report.json").read_text(encoding="utf-8")
    )
    if initialization["random_m0_equivalent_initial_state_sha256_before_clone"] != m0_init_report["model_sha256_before_training"]:
        raise RuntimeError(f"M1 random pre-clone state does not match the recorded M0 seed {seed} initialization")
    if head_state_sha256(model.intention_branch) != initialization["intention_head_initialization_sha256"]:
        raise RuntimeError("M1 intention-head initialization unexpectedly changed during pretrained clone")
    initial_intention_encoder_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.intention_branch.state_dict().items()
        if name == "position_embedding"
        or name.startswith(("input_projection.", "temporal_encoder."))
    }
    trajectory_hash_before = trajectory_model_sha256(model.trajectory_branch)
    model.trajectory_branch.eval()
    if any(parameter.requires_grad for parameter in model.trajectory_branch.parameters()):
        raise RuntimeError("M1 trajectory branch must remain frozen")
    if not all(parameter.requires_grad for parameter in model.intention_branch.parameters()):
        raise RuntimeError("All M1 intention branch parameters must be trainable")

    # Match M0's configured weighted BCE exactly; the trajectory branch has no
    # loss and none of its Parameters may enter this optimizer.
    optimizer = torch.optim.AdamW(
        model.intention_branch.parameters(),
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trajectory_parameter_ids = {id(parameter) for parameter in model.trajectory_branch.parameters()}
    optimizer_parameter_ids = optimizer_ids
    if optimizer_parameter_ids & trajectory_parameter_ids:
        raise RuntimeError("Frozen trajectory parameters entered the M1 optimizer")

    reference = json.loads(
        (ROOT / "results/trajectory_preserving_joint/trajectory_reference.json").read_text(encoding="utf-8")
    )["per_seed"][str(seed)]["validation"]
    _, baseline_trajectory_metrics, _, _, _, _, _ = evaluate_validation(model, val_loader, device)
    if (
        abs(baseline_trajectory_metrics["ade_pixel"] - float(reference["trajectory_ade_pixel"])) >= 0.05
        or abs(baseline_trajectory_metrics["fde_pixel"] - float(reference["trajectory_fde_pixel"])) >= 0.05
    ):
        raise RuntimeError(f"M1 trajectory validation baseline differs from original checkpoint reference, seed {seed}")

    run_dir = RESULTS_ROOT / f"seed{seed}"
    checkpoint_path = CHECKPOINT_ROOT / f"M1_pretrained_clone_seed{seed}.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "initialization_report.json",
        {
            **initialization,
            "m0_recorded_initial_model_hash": m0_init_report["model_sha256_before_training"],
            "m0_head_initialization_matched": True,
            "pretrained_checkpoint_loaded_for_intention_encoder": True,
            "pretrained_checkpoint_loaded_for_trajectory_branch": True,
            "trainable_parameter_count": sum(p.numel() for p in model.intention_branch.parameters()),
            "trajectory_parameter_count_frozen": sum(p.numel() for p in model.trajectory_branch.parameters()),
            "optimizer_contains_trajectory_parameter": False,
            "test_accessed": False,
        },
    )

    tolerance = float(train_config["selection_tolerance"])
    best_auc = -float("inf")
    best_brier = float("inf")
    best_epoch = 0
    history: list[dict[str, Any]] = []
    drift_history: list[dict[str, Any]] = []
    parameter_hash_rows: list[dict[str, Any]] = [
        {"epoch": 0, "trajectory_sha256": trajectory_hash_before, "unchanged": True}
    ]

    for epoch in range(1, int(train_config["maximum_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            y = batch["intent_label"].to(device)
            logits = model.forward_intention(target)["intent_logit"]
            per_sample = nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none")
            weights = torch.where(
                y > 0.5,
                torch.as_tensor(class_weights["positive"], dtype=y.dtype, device=device),
                torch.as_tensor(class_weights["negative"], dtype=y.dtype, device=device),
            )
            loss = (per_sample * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.intention_branch.parameters(), float(train_config["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            sample_count += len(y)

        trajectory_hash_after = trajectory_model_sha256(model.trajectory_branch)
        hash_unchanged = trajectory_hash_after == trajectory_hash_before
        parameter_hash_rows.append(
            {"epoch": epoch, "trajectory_sha256": trajectory_hash_after, "unchanged": hash_unchanged}
        )
        if not hash_unchanged or model.trajectory_branch.training:
            write_json(run_dir / "parameter_hashes.json", {"seed": seed, "history": parameter_hash_rows, "unchanged_every_epoch": False})
            raise RuntimeError(f"Frozen trajectory branch changed or entered train mode at epoch {epoch}, seed {seed}")

        raw_metrics, traj_metrics, val_logits, val_labels, _, _, _ = evaluate_validation(model, val_loader, device)
        ade_delta = traj_metrics["ade_pixel"] - float(reference["trajectory_ade_pixel"])
        fde_delta = traj_metrics["fde_pixel"] - float(reference["trajectory_fde_pixel"])
        if abs(ade_delta) >= 0.05 or abs(fde_delta) >= 0.05:
            raise RuntimeError(f"Trajectory validation drift exceeded 0.05 px at epoch {epoch}, seed {seed}")

        drift = encoder_drift(initial_intention_encoder_state, model.intention_branch.state_dict())
        drift_row = {"epoch": epoch, **drift}
        drift_history.append(drift_row)
        epoch_row = {
            "epoch": epoch,
            "train_weighted_bce": loss_sum / sample_count,
            "validation_raw": raw_metrics,
            "trajectory_validation": traj_metrics,
            "trajectory_ade_difference_vs_original_pixel": ade_delta,
            "trajectory_fde_difference_vs_original_pixel": fde_delta,
            "trajectory_parameter_sha256": trajectory_hash_after,
            "trajectory_parameter_hash_unchanged": True,
            "encoder_drift": drift,
        }
        history.append(epoch_row)
        write_json(run_dir / "validation_history.json", {"seed": seed, "method": "M1_pretrained_clone_trainable", "epochs": history})
        write_json(run_dir / "parameter_hashes.json", {"seed": seed, "history": parameter_hash_rows, "unchanged_every_epoch": True})
        write_json(
            run_dir / "encoder_drift.json",
            {
                "seed": seed,
                "reference": "corresponding pretrained target encoder immediately after exact clone, before M1 optimization",
                "history": drift_history,
                "selected_checkpoint_epoch": None,
            },
        )

        auc = raw_metrics["roc_auc"]
        brier = raw_metrics["brier"]
        select = auc > best_auc + tolerance
        if abs(auc - best_auc) <= tolerance and brier < best_brier:
            select = True
        print(
            json.dumps(
                {
                    "method": "M1_pretrained_clone_trainable",
                    "seed": seed,
                    "epoch": epoch,
                    "train_weighted_bce": epoch_row["train_weighted_bce"],
                    "val_auc": auc,
                    "val_brier": brier,
                    "val_ade_pixel": traj_metrics["ade_pixel"],
                    "val_fde_pixel": traj_metrics["fde_pixel"],
                    "trajectory_hash_unchanged": True,
                    "encoder_relative_l2_drift": drift["overall"]["relative_l2_parameter_change"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if select:
            best_auc, best_brier, best_epoch = auc, brier, epoch
            torch.save(
                {
                    "intention_model": model.intention_branch.state_dict(),
                    "seed": seed,
                    "method": "M1_pretrained_clone_trainable",
                    "pretrained_trajectory_checkpoint": str(trajectory_checkpoint_path(seed).relative_to(ROOT)),
                    "pretrained_trajectory_checkpoint_sha256": initialization["trajectory_checkpoint_sha256"],
                    "pretrained_encoder_sha256_before_training": initialization[
                        "intention_encoder_sha256_after_clone"
                    ],
                    "trajectory_branch_sha256": trajectory_hash_before,
                    "initial_m0_equivalent_state_sha256": initialization["random_m0_equivalent_initial_state_sha256_before_clone"],
                    "initialization_report": initialization,
                    "config": config,
                    "selected_epoch": epoch,
                    "selection_auc": auc,
                    "selection_brier": brier,
                },
                checkpoint_path,
            )

    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.intention_branch.load_state_dict(selected["intention_model"], strict=True)
    selected_raw, selected_trajectory, selected_logits, selected_labels, _, _, _ = evaluate_validation(
        model, val_loader, device
    )
    temperature = fit_temperature(selected_logits, selected_labels)
    val_probability = probabilities_from_logits(selected_logits, temperature)
    threshold = choose_balanced_accuracy_threshold(val_probability, selected_labels)
    selected_calibrated = intention_metrics(
        selected_labels, selected_logits, temperature=temperature, threshold=threshold
    )
    if trajectory_model_sha256(model.trajectory_branch) != trajectory_hash_before:
        raise RuntimeError("M1 selected checkpoint procedure changed the frozen trajectory branch")
    selected_drift = encoder_drift(initial_intention_encoder_state, model.intention_branch.state_dict())
    drift_history[-1]["is_selected_epoch"] = False
    for row in drift_history:
        row["is_selected_epoch"] = row["epoch"] == best_epoch
    write_json(
        run_dir / "encoder_drift.json",
        {
            "seed": seed,
            "reference": "corresponding pretrained target encoder immediately after exact clone, before M1 optimization",
            "history": drift_history,
            "selected_checkpoint_epoch": best_epoch,
            "selected_checkpoint_drift": selected_drift,
        },
    )
    metrics = {
        "method": "M1_pretrained_clone_trainable",
        "seed": seed,
        "input": config["input"],
        "pretrained_trajectory_checkpoint": str(trajectory_checkpoint_path(seed).relative_to(ROOT)),
        "pretrained_trajectory_checkpoint_sha256": initialization["trajectory_checkpoint_sha256"],
        "pretrained_encoder_sha256_before_training": initialization["intention_encoder_sha256_after_clone"],
        "trajectory_branch_sha256_before_training": trajectory_hash_before,
        "trajectory_branch_sha256_after_training": trajectory_model_sha256(model.trajectory_branch),
        "trajectory_branch_frozen": True,
        "trajectory_hash_unchanged_every_epoch": all(row["unchanged"] for row in parameter_hash_rows),
        "trainable_parameter_names": [name for name, p in model.intention_branch.named_parameters() if p.requires_grad],
        "optimizer_parameter_count": len(optimizer_ids),
        "optimizer_contains_trajectory_parameter": False,
        "best_epoch": best_epoch,
        "checkpoint_selection": train_config["checkpoint_selection"],
        "selected_validation_raw_metrics": selected_raw,
        "selected_validation_trajectory_metrics": selected_trajectory,
        "selected_validation_calibration": {
            "temperature": temperature,
            "temperature_fit_split": "validation",
            "threshold": threshold,
            "threshold_fit_split": "validation balanced accuracy",
            "metrics": selected_calibrated,
        },
        "training": {
            **train_config,
            "class_weights": class_weights,
            "data_splits_loaded": ["train", "val"],
            "test_accessed": False,
            "trajectory_loss_used": False,
            "optimizer": "AdamW intention branch only",
        },
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "history": history,
        "test": None,
        "test_evaluation_status": "withheld_until_protocol_freeze",
    }
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "metrics_validation.json",
        {
            "seed": seed,
            "best_epoch": best_epoch,
            "selected_validation_raw_metrics": selected_raw,
            "selected_validation_trajectory_metrics": selected_trajectory,
            "selected_validation_calibration": metrics["selected_validation_calibration"],
            "all_epochs": history,
            "test": None,
        },
    )
    print(
        json.dumps(
            {
                "method": "M1_pretrained_clone_trainable",
                "seed": seed,
                "best_epoch": best_epoch,
                "validation_raw": selected_raw,
                "validation_calibrated": selected_calibrated,
                "trajectory_validation": selected_trajectory,
                "test": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
