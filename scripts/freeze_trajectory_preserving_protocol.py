#!/usr/bin/env python3
"""Freeze the validation-selected P1/P2 protocol before any test access."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trajectory_preserving_utils import SEEDS, sha256_file

RESULTS = ROOT / "results/trajectory_preserving_joint"
METHODS = ("P1_target_only", "P2_target_scene")
SOURCE_FILES = (
    "configs/trajectory_preserving_joint.json",
    "src/models/trajectory_transformer.py",
    "src/models/trajectory_preserving_joint.py",
    "scripts/trajectory_preserving_utils.py",
    "scripts/train_trajectory_preserving_joint.py",
    "scripts/evaluate_trajectory_preserving_joint_test.py",
    "scripts/freeze_trajectory_preserving_protocol.py",
    "scripts/summarize_trajectory_preserving_joint.py",
    "scripts/audit_trajectory_preserving_reference.py",
    "scripts/run_trajectory_preserving_equivalence.py",
    "scripts/reliability_gated_intent_utils.py",
)
COMPARATOR_FILES = (
    "results/reliability_gated_intent_15x15/test_metrics.json",
    "results/joint_loss_balance/summary.md",
    "results/trajectory_transformer_scene_15x15_seed42/metrics.json",
    "results/trajectory_transformer_scene_15x15_seed123/metrics.json",
    "results/trajectory_transformer_scene_15x15_seed2024/metrics.json",
    "results/joint_loss_balance/lambda100/seed42/metrics.json",
    "results/joint_loss_balance/lambda100/seed123/metrics.json",
    "results/joint_loss_balance/lambda100/seed2024/metrics.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = load_json(ROOT / "configs/trajectory_preserving_joint.json")
    equivalence = load_json(RESULTS / "equivalence_test.json")
    loading = load_json(RESULTS / "weight_loading_report.json")
    if not equivalence.get("pass") or not loading.get("all_backbone_loads_complete"):
        raise RuntimeError("Initialization equivalence and complete checkpoint loading are required")
    if equivalence.get("validation_only") is not True or equivalence.get("samples_per_seed", 0) < 1024:
        raise RuntimeError("Equivalence gate must use at least 1,024 validation samples per seed")

    reference = load_json(RESULTS / "trajectory_reference.json")
    selected_checkpoints: dict[str, dict[str, Any]] = {}
    run_checks: list[dict[str, Any]] = []
    for method in METHODS:
        selected_checkpoints[method] = {}
        for seed in SEEDS:
            run_dir = RESULTS / method / f"seed{seed}"
            metrics_path = run_dir / "metrics.json"
            history_path = run_dir / "validation_history.json"
            hashes_path = run_dir / "parameter_hashes.json"
            checkpoint_path = ROOT / "checkpoints/trajectory_preserving_joint" / f"{method}_seed{seed}.pt"
            for path in (metrics_path, history_path, hashes_path, checkpoint_path):
                if not path.is_file():
                    raise FileNotFoundError(f"Required completed validation artifact missing: {path}")
            if (run_dir / "test_access_record.json").exists() or (run_dir / "test_predictions.npz").exists():
                raise RuntimeError(f"A test artifact exists before freeze: {method} seed {seed}")
            metrics = load_json(metrics_path)
            history = load_json(history_path)
            hashes = load_json(hashes_path)
            expected_epochs = int(config["training"]["maximum_epochs"])
            if metrics.get("test") is not None or metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
                raise RuntimeError(f"Test data appears to have been accessed before freeze: {method} seed {seed}")
            if len(metrics.get("history", [])) != expected_epochs or len(history.get("epochs", [])) != expected_epochs:
                raise RuntimeError(f"Incomplete validation training history: {method} seed {seed}")
            if not metrics.get("trajectory_backbone_frozen") or not hashes.get("unchanged_every_epoch"):
                raise RuntimeError(f"Frozen backbone verification failed: {method} seed {seed}")
            if not hashes.get("final_hash_unchanged"):
                raise RuntimeError(f"Frozen backbone final hash mismatch: {method} seed {seed}")
            expected_hash_observations = expected_epochs + 1
            hash_history = hashes.get("hash_history", [])
            if len(hash_history) != expected_hash_observations or any(
                row.get("sha256") != hashes.get("hash_before") for row in hash_history
            ):
                raise RuntimeError(f"Not every epoch has a matching frozen hash: {method} seed {seed}")
            reference_validation = reference["per_seed"][str(seed)]["validation"]
            for epoch_row in metrics["history"]:
                if (
                    abs(float(epoch_row["validation"]["ade_pixel"]) - float(reference_validation["trajectory_ade_pixel"])) >= 0.05
                    or abs(float(epoch_row["validation"]["fde_pixel"]) - float(reference_validation["trajectory_fde_pixel"])) >= 0.05
                ):
                    raise RuntimeError(f"Validation trajectory drift exceeded tolerance: {method} seed {seed}")
            reference_hash = reference["per_seed"][str(seed)]["checkpoint_sha256"]
            if metrics["trajectory_checkpoint_sha256"] != reference_hash:
                raise RuntimeError(f"Wrong seed-matched trajectory checkpoint: {method} seed {seed}")
            if metrics["trajectory_backbone_sha256_before_training"] != metrics["trajectory_backbone_sha256_after_training"]:
                raise RuntimeError(f"Backbone changed during training: {method} seed {seed}")
            checkpoint_sha = sha256_file(checkpoint_path)
            selected_checkpoints[method][str(seed)] = {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "checkpoint_sha256": checkpoint_sha,
                "best_epoch": int(metrics["best_epoch"]),
                "validation_auc": float(metrics["selected_validation_raw_metrics"]["roc_auc"]),
                "validation_brier": float(metrics["selected_validation_raw_metrics"]["brier"]),
                "trajectory_backbone_sha256": metrics["trajectory_backbone_sha256_after_training"],
                "trajectory_checkpoint_path": metrics["trajectory_checkpoint"],
                "trajectory_checkpoint_sha256": metrics["trajectory_checkpoint_sha256"],
            }
            run_checks.append(
                {
                    "method": method,
                    "seed": seed,
                    "epochs_completed": len(metrics["history"]),
                    "best_epoch": int(metrics["best_epoch"]),
                    "backbone_hash_unchanged_every_epoch": bool(hashes["unchanged_every_epoch"]),
                    "test_metric_absent_before_freeze": metrics["test"] is None,
                    "checkpoint_sha256": checkpoint_sha,
                }
            )

    source_sha = {path: sha256_file(ROOT / path) for path in SOURCE_FILES}
    comparator_sha = {path: sha256_file(ROOT / path) for path in COMPARATOR_FILES}
    input_sha = {
        "data/processed/jaad_sequences_scene_15x15/train.npz": sha256_file(
            ROOT / "data/processed/jaad_sequences_scene_15x15/train.npz"
        ),
        "data/processed/jaad_sequences_scene_15x15/val.npz": sha256_file(
            ROOT / "data/processed/jaad_sequences_scene_15x15/val.npz"
        ),
        "results/trajectory_preserving_joint/trajectory_reference.json": sha256_file(
            RESULTS / "trajectory_reference.json"
        ),
        "results/trajectory_preserving_joint/equivalence_test.json": sha256_file(
            RESULTS / "equivalence_test.json"
        ),
        "results/trajectory_preserving_joint/weight_loading_report.json": sha256_file(
            RESULTS / "weight_loading_report.json"
        ),
    }
    for seed in SEEDS:
        path = reference["per_seed"][str(seed)]["checkpoint"]
        input_sha[path] = sha256_file(ROOT / path)

    protocol = {
        "protocol_id": config["protocol_id"],
        "experiment": config["experiment"],
        "frozen": True,
        "frozen_before_test_access": True,
        "test_access_before_freeze": False,
        "test_archive_hash_recorded": False,
        "base_commit": config["base_commit"],
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "model_and_input_definition": {
            "arms": config["arms"],
            "trajectory_backbone": config["trajectory_backbone"],
            "intention_head": config["intention_head"],
            "forbidden_components": config["forbidden"],
        },
        "full_experiment_config": config,
        "training_config": config["training"],
        "checkpoint_selection_and_calibration": {
            "selection": config["training"]["checkpoint_selection"],
            "temperature_fit": "validation split only",
            "threshold_fit": "validation balanced accuracy only",
        },
        "equivalence_gate": equivalence,
        "weight_loading_complete": loading["all_backbone_loads_complete"],
        "selected_checkpoints": selected_checkpoints,
        "validation_run_checks": run_checks,
        "input_sha256": input_sha,
        "source_sha256": source_sha,
        "historical_comparator_sha256": comparator_sha,
    }
    protocol_path = RESULTS / "protocol_frozen.json"
    checksum_path = RESULTS / "protocol_frozen.sha256"
    protocol_bytes = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    protocol_path.write_bytes(protocol_bytes)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    checksum_path.write_text(f"{protocol_hash}  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({"protocol": str(protocol_path), "sha256": protocol_hash, "test_archive_opened": False, "runs_frozen": len(run_checks)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
