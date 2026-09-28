#!/usr/bin/env python3
"""Freeze M0's validation-selected checkpoints before first M0 test access."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_trajectory_preserving_joint_test import verify_frozen_protocol as verify_p1_protocol
from scripts.trajectory_preserving_utils import SEEDS, sha256_file
from scripts.train_intention_scratch_matched import state_sha256

RESULTS = ROOT / "results/intention_scratch_matched"
METHOD = "M0_scratch_matched"
SOURCE_FILES = (
    "configs/intention_scratch_matched.json",
    "src/models/intention_scratch_transformer.py",
    "src/models/trajectory_transformer.py",
    "src/models/trajectory_preserving_joint.py",
    "scripts/train_intention_scratch_matched.py",
    "scripts/audit_intention_scratch_architecture.py",
    "scripts/analyze_intention_scratch_representations.py",
    "scripts/freeze_intention_scratch_protocol.py",
    "scripts/evaluate_intention_scratch_test.py",
    "scripts/analyze_intention_scratch_matched.py",
    "scripts/trajectory_preserving_utils.py",
    "scripts/reliability_gated_intent_utils.py",
    "scripts/evaluate_trajectory_preserving_joint_test.py",
    "tests/test_intention_scratch_matched.py",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    config = load_json(ROOT / "configs/intention_scratch_matched.json")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != config["base_commit"]:
        raise RuntimeError(f"Base commit mismatch: protocol={config['base_commit']} HEAD={head}")

    architecture = load_json(RESULTS / "architecture_equivalence.json")
    if not architecture.get("matched"):
        raise RuntimeError("P1/M0 architecture equivalence did not pass")
    representations = load_json(RESULTS / "representation_analysis.json")
    if representations.get("no_test_split_loaded") is not True:
        raise RuntimeError("Representation analysis is not certified validation-only")
    previous_protocol_path = ROOT / "results/trajectory_preserving_joint/protocol_frozen.json"
    previous_checksum_path = ROOT / "results/trajectory_preserving_joint/protocol_frozen.sha256"
    previous_protocol, previous_protocol_sha = verify_p1_protocol(previous_protocol_path, previous_checksum_path)
    if not previous_protocol.get("frozen"):
        raise RuntimeError("The P1 comparator does not have a frozen prior protocol")
    p1_config = load_json(ROOT / "configs/trajectory_preserving_joint.json")

    initialization_records: dict[str, Any] = {}
    selected_checkpoints: dict[str, Any] = {}
    validation_runs: dict[str, Any] = {}
    p1_baseline_artifacts: dict[str, Any] = {}
    artifact_sha: dict[str, str] = {}
    for seed in SEEDS:
        run_dir = RESULTS / f"seed{seed}"
        checkpoint_path = ROOT / "checkpoints/intention_scratch_matched" / f"M0_scratch_seed{seed}.pt"
        metrics_path = run_dir / "metrics.json"
        validation_path = run_dir / "metrics_validation.json"
        history_path = run_dir / "validation_history.json"
        init_path = run_dir / "initialization_report.json"
        for path in (checkpoint_path, metrics_path, validation_path, history_path, init_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required M0 validation artifact missing: {path}")
        if (run_dir / "test_predictions.npz").exists() or (run_dir / "test_access_record.json").exists():
            raise RuntimeError(f"M0 test artifact exists before protocol freeze for seed {seed}")

        metrics = load_json(metrics_path)
        validation_metrics = load_json(validation_path)
        history = load_json(history_path)
        initialization = load_json(init_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if metrics.get("test") is not None or validation_metrics.get("test") is not None:
            raise RuntimeError(f"M0 test metrics must remain absent before freeze, seed {seed}")
        if metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
            raise RuntimeError(f"M0 test-access status is invalid, seed {seed}")
        if metrics.get("training", {}).get("test_accessed") is not False:
            raise RuntimeError(f"M0 training must record no test access, seed {seed}")
        if initialization.get("pretrained_checkpoint_loaded") is not False:
            raise RuntimeError(f"M0 was not recorded as scratch initialized, seed {seed}")
        if checkpoint.get("pretrained_checkpoint_loaded") is not False or checkpoint.get("method") != METHOD:
            raise RuntimeError(f"M0 checkpoint metadata is invalid, seed {seed}")
        if int(checkpoint.get("seed", -1)) != seed or metrics.get("seed") != seed:
            raise RuntimeError(f"M0 seed mismatch in saved artifacts for seed {seed}")
        if len(history.get("epochs", [])) != int(config["training"]["maximum_epochs"]):
            raise RuntimeError(f"M0 validation history is incomplete for seed {seed}")
        if len(metrics.get("history", [])) != int(config["training"]["maximum_epochs"]):
            raise RuntimeError(f"M0 metrics history is incomplete for seed {seed}")
        selected_model_state_sha = state_sha256(checkpoint["model"])
        if selected_model_state_sha == initialization.get("model_sha256_before_training"):
            # Training should have changed at least some scratch parameters.
            raise RuntimeError(f"M0 selected model equals its initial random state, seed {seed}")
        if checkpoint.get("initialization_sha256") != initialization.get("model_sha256_before_training"):
            raise RuntimeError(f"M0 initial model hash does not match its checkpoint metadata, seed {seed}")
        if checkpoint.get("selected_epoch") != metrics.get("best_epoch"):
            raise RuntimeError(f"Selected M0 epoch mismatch, seed {seed}")

        initialization_records[str(seed)] = initialization
        selected_checkpoints[str(seed)] = {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "initialization_sha256": initialization["model_sha256_before_training"],
            "selected_model_state_sha256": selected_model_state_sha,
            "selected_epoch": int(metrics["best_epoch"]),
            "validation_auc": float(metrics["selected_validation_raw_metrics"]["roc_auc"]),
            "validation_brier": float(metrics["selected_validation_raw_metrics"]["brier"]),
            "parameter_count_total": int(metrics["parameter_count_total"]),
            "pretrained_checkpoint_loaded": False,
        }
        validation_runs[str(seed)] = {
            "metrics_path": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256_file(metrics_path),
            "history_sha256": sha256_file(history_path),
            "validation_metrics_sha256": sha256_file(validation_path),
            "initialization_report_sha256": sha256_file(init_path),
            "class_weights": metrics["training"]["class_weights"],
            "epochs_completed": len(history["epochs"]),
            "test_metrics_absent": metrics["test"] is None,
            "test_accessed": False,
        }

        p1_dir = ROOT / "results/trajectory_preserving_joint/P1_target_only" / f"seed{seed}"
        p1_metrics_path = p1_dir / "metrics.json"
        p1_predictions_path = p1_dir / "test_predictions.npz"
        p1_metrics = load_json(p1_metrics_path)
        p1_selected = previous_protocol["selected_checkpoints"]["P1_target_only"][str(seed)]
        if p1_metrics.get("test") is None or not p1_metrics.get("trajectory_backbone_frozen"):
            raise RuntimeError(f"P1 comparator result is missing or unfrozen for seed {seed}")
        p1_training = p1_metrics.get("training", {})
        m0_training = metrics.get("training", {})
        for key in ("maximum_epochs", "batch_size", "optimizer", "learning_rate", "weight_decay", "gradient_clip_norm", "selection_tolerance"):
            if p1_training.get(key) != m0_training.get(key):
                raise RuntimeError(f"P1/M0 training setting mismatch for {key}, seed {seed}")
        p1_weights = p1_training.get("class_weights", {})
        m0_weights = m0_training.get("class_weights", {})
        if p1_weights.get("positive_count") != m0_weights.get("positive_count") or p1_weights.get("negative_count") != m0_weights.get("negative_count"):
            raise RuntimeError(f"P1/M0 class counts differ for seed {seed}")
        for weight_name in ("positive", "negative"):
            if abs(float(p1_weights[weight_name]) - float(m0_weights[weight_name])) > 1e-12:
                raise RuntimeError(f"P1/M0 class weight mismatch for {weight_name}, seed {seed}")
        if p1_training.get("sampling") != m0_training.get("sampling"):
            raise RuntimeError(f"P1/M0 sampling strategy differs for seed {seed}")
        if p1_config["training"]["maximum_epochs"] != config["training"]["maximum_epochs"]:
            raise RuntimeError("P1/M0 epoch counts differ from their registered configs")
        if p1_training.get("checkpoint_selection") != m0_training.get("checkpoint_selection"):
            raise RuntimeError(f"P1/M0 checkpoint selection differs for seed {seed}")
        if sha256_file(ROOT / p1_selected["path"]) != p1_selected["checkpoint_sha256"]:
            raise RuntimeError(f"Frozen P1 checkpoint hash mismatch for seed {seed}")
        p1_baseline_artifacts[str(seed)] = {
            "metrics_path": str(p1_metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256_file(p1_metrics_path),
            "predictions_path": str(p1_predictions_path.relative_to(ROOT)),
            "predictions_sha256": sha256_file(p1_predictions_path),
            "checkpoint_path": p1_selected["path"],
            "checkpoint_sha256": p1_selected["checkpoint_sha256"],
        }

    initialization_report = {
        "method": METHOD,
        "pretrained_checkpoint_loaded": False,
        "seeds": initialization_records,
        "all_seeds_randomly_initialized": all(
            row["pretrained_checkpoint_loaded"] is False for row in initialization_records.values()
        ),
    }
    init_report_path = RESULTS / "initialization_report.json"
    init_report_path.write_text(json.dumps(initialization_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    artifact_sha.update(
        {
            "data/processed/jaad_sequences_scene_15x15/train.npz": sha256_file(data_root / "train.npz"),
            "data/processed/jaad_sequences_scene_15x15/val.npz": sha256_file(data_root / "val.npz"),
            "results/intention_scratch_matched/architecture_equivalence.json": sha256_file(RESULTS / "architecture_equivalence.json"),
            "results/intention_scratch_matched/representation_analysis.json": sha256_file(RESULTS / "representation_analysis.json"),
            "results/intention_scratch_matched/initialization_report.json": sha256_file(init_report_path),
            "results/trajectory_preserving_joint/protocol_frozen.json": sha256_file(previous_protocol_path),
            "results/trajectory_preserving_joint/protocol_frozen.sha256": sha256_file(previous_checksum_path),
            "results/reliability_gated_intent_15x15/test_metrics.json": sha256_file(
                ROOT / "results/reliability_gated_intent_15x15/test_metrics.json"
            ),
        }
    )
    for seed in SEEDS:
        reference = previous_protocol["selected_checkpoints"]["P1_target_only"][str(seed)]
        artifact_sha[reference["trajectory_checkpoint_path"]] = sha256_file(ROOT / reference["trajectory_checkpoint_path"])
        for relative_path in (
            p1_baseline_artifacts[str(seed)]["metrics_path"],
            p1_baseline_artifacts[str(seed)]["predictions_path"],
            selected_checkpoints[str(seed)]["path"],
            validation_runs[str(seed)]["metrics_path"],
        ):
            artifact_sha[relative_path] = sha256_file(ROOT / relative_path)

    protocol = {
        "protocol_id": config["protocol_id"],
        "experiment": config["experiment"],
        "frozen": True,
        "frozen_before_m0_test_access": True,
        "m0_test_accessed_before_freeze": False,
        "test_archive_hash_recorded_before_freeze": False,
        "p1_test_predictions_reused_from_separate_prior_frozen_experiment": True,
        "base_commit": head,
        "seeds": list(SEEDS),
        "method": METHOD,
        "input": config["input"],
        "model": config["model"],
        "training": config["training"],
        "paired_analysis": config["paired_analysis"],
        "transfer_decision_rule": config["paired_analysis"]["transfer_decision_rule"],
        "architecture_sha256": architecture["architecture_sha256"],
        "validation_representation_analysis_sha256": sha256_file(RESULTS / "representation_analysis.json"),
        "selected_checkpoints": selected_checkpoints,
        "validation_runs": validation_runs,
        "p1_prior_frozen_protocol_sha256": previous_protocol_sha,
        "p1_baseline_artifacts": p1_baseline_artifacts,
        "data_and_frozen_input_sha256": artifact_sha,
        "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCE_FILES},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        },
    }
    protocol_path = RESULTS / "protocol_frozen.json"
    checksum_path = RESULTS / "protocol_frozen.sha256"
    encoded = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    protocol_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    checksum_path.write_text(f"{digest}  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({"protocol": str(protocol_path), "sha256": digest, "m0_test_archive_opened": False, "seeds_frozen": list(SEEDS), "p1_prior_protocol_sha256": previous_protocol_sha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
