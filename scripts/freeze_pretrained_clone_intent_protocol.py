#!/usr/bin/env python3
"""Freeze M1 choices and immutable inputs before the one-time test evaluation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrained_clone_intent_utils import RESULTS_ROOT, load_config, write_json
from scripts.trajectory_preserving_utils import SEEDS, sha256_file

PROTOCOL_PATH = RESULTS_ROOT / "protocol_frozen.json"
PROTOCOL_SHA_PATH = RESULTS_ROOT / "protocol_frozen.sha256"

SOURCE_FILES = (
    "configs/pretrained_clone_intent.json",
    "scripts/audit_pretrained_clone_intent.py",
    "scripts/train_pretrained_clone_intent.py",
    "scripts/pretrained_clone_intent_utils.py",
    "scripts/representation_shift_pretrained_clone_intent.py",
    "scripts/freeze_pretrained_clone_intent_protocol.py",
    "scripts/evaluate_pretrained_clone_intent_test.py",
    "scripts/analyze_pretrained_clone_intent.py",
    "src/models/pretrained_clone_intent.py",
    "src/models/intention_scratch_transformer.py",
    "src/models/trajectory_transformer.py",
    "scripts/trajectory_preserving_utils.py",
    "scripts/reliability_gated_intent_utils.py",
    "tests/test_pretrained_clone_intent.py",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if PROTOCOL_PATH.exists() or PROTOCOL_SHA_PATH.exists():
        raise RuntimeError("M1 test protocol is already frozen; refusing to replace it")
    config = load_config()
    base_commit = config["base_commit"]
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != base_commit:
        raise RuntimeError(f"Expected base commit {base_commit}, found {head}; do not freeze a different baseline")

    gate_paths = {
        "clone": RESULTS_ROOT / "pretrained_clone_report.json",
        "independence": RESULTS_ROOT / "parameter_independence_test.json",
        "trajectory_equivalence": RESULTS_ROOT / "trajectory_equivalence.json",
        "representation_shift": RESULTS_ROOT / "representation_shift.json",
    }
    gates = {name: read_json(path) for name, path in gate_paths.items()}
    if not gates["clone"].get("all_seeds_clone_pass") or not gates["clone"].get("all_m0_head_initializations_and_seeded_initial_models_match"):
        raise RuntimeError("Clone/M0 initialization gate did not pass for every seed")
    if not gates["independence"].get("all_seeds_test_pass"):
        raise RuntimeError("Parameter independence gate did not pass for every seed")
    if not gates["trajectory_equivalence"].get("all_seeds_pass"):
        raise RuntimeError("Pre-training trajectory equivalence gate did not pass for every seed")
    if gates["representation_shift"].get("split") != "validation" or gates["representation_shift"].get("selection_use") is not False:
        raise RuntimeError("Representation diagnostic must be validation-only and excluded from selection")

    selected: dict[str, Any] = {}
    encoder_drift: dict[str, Any] = {"reference": "trajectory-pretrained target encoder before intention fine-tuning", "per_seed": {}}
    for seed in SEEDS:
        run_dir = RESULTS_ROOT / f"seed{seed}"
        metrics = read_json(run_dir / "metrics.json")
        validation = read_json(run_dir / "metrics_validation.json")
        hashes = read_json(run_dir / "parameter_hashes.json")
        drift = read_json(run_dir / "encoder_drift.json")
        checkpoint_path = ROOT / metrics["checkpoint"]
        if metrics.get("test") is not None or metrics["training"].get("test_accessed") is not False:
            raise RuntimeError(f"Seed {seed} indicates test access before protocol freeze")
        if len(metrics.get("history", [])) != int(config["training"]["maximum_epochs"]):
            raise RuntimeError(f"Seed {seed} has incomplete training history")
        if len(validation.get("all_epochs", [])) != int(config["training"]["maximum_epochs"]):
            raise RuntimeError(f"Seed {seed} validation history is incomplete")
        if hashes.get("unchanged_every_epoch") is not True or not all(row.get("unchanged") for row in hashes["history"]):
            raise RuntimeError(f"Seed {seed} trajectory parameter hash changed during training")
        if not metrics.get("trajectory_branch_frozen") or not metrics.get("trajectory_hash_unchanged_every_epoch"):
            raise RuntimeError(f"Seed {seed} trajectory freeze gate failed")
        if metrics.get("optimizer_contains_trajectory_parameter") is not False:
            raise RuntimeError(f"Seed {seed} trajectory parameter entered intention optimizer")
        if int(metrics["best_epoch"]) != int(validation["best_epoch"]):
            raise RuntimeError(f"Seed {seed} selected epoch mismatch between training outputs")
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != metrics["checkpoint_sha256"]:
            raise RuntimeError(f"Seed {seed} selected checkpoint is absent or has a hash mismatch")
        access_paths = [run_dir / "test_access_record.json", run_dir / "test_predictions.npz"]
        if any(path.exists() for path in access_paths):
            raise RuntimeError(f"Seed {seed} already has official M1 test access artifacts")
        selected[str(seed)] = {
            "checkpoint": metrics["checkpoint"],
            "checkpoint_sha256": metrics["checkpoint_sha256"],
            "selected_epoch": metrics["best_epoch"],
            "validation_raw_metrics": metrics["selected_validation_raw_metrics"],
            "validation_calibration": metrics["selected_validation_calibration"],
            "validation_metrics_sha256": sha256_file(run_dir / "metrics_validation.json"),
            "validation_history_sha256": sha256_file(run_dir / "validation_history.json"),
            "parameter_hashes_sha256": sha256_file(run_dir / "parameter_hashes.json"),
            "encoder_drift_sha256": sha256_file(run_dir / "encoder_drift.json"),
        }
        encoder_drift["per_seed"][str(seed)] = {
            "selected_epoch": drift["selected_checkpoint_epoch"],
            "selected_checkpoint_drift": drift["selected_checkpoint_drift"],
            "all_epochs": drift["history"],
        }

    # Hash the previously frozen M0/P1 result packages; do not inspect the
    # current raw test archive while creating this protocol.
    baseline_artifacts: dict[str, dict[str, str]] = {"M0_scratch_matched": {}, "P1_target_only": {}}
    reference_test_sha: set[str] = set()
    for seed in SEEDS:
        for method, relative in (
            ("M0_scratch_matched", f"results/intention_scratch_matched/seed{seed}"),
            ("P1_target_only", f"results/trajectory_preserving_joint/P1_target_only/seed{seed}"),
        ):
            run = ROOT / relative
            metrics_path, predictions_path = run / "metrics.json", run / "test_predictions.npz"
            if not metrics_path.is_file() or not predictions_path.is_file():
                raise RuntimeError(f"Missing pre-existing {method} official test artifacts for seed {seed}")
            prior_metrics = read_json(metrics_path)
            if not isinstance(prior_metrics.get("test"), dict):
                raise RuntimeError(f"No official test metrics present for {method}, seed {seed}")
            data_hash = prior_metrics["test"].get("test_npz_sha256_after_freeze") or prior_metrics["test"].get("test_npz_sha256_after_protocol_freeze")
            if data_hash:
                reference_test_sha.add(data_hash)
            baseline_artifacts[method][str(seed)] = {
                "metrics_path": str(metrics_path.relative_to(ROOT)),
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_path": str(predictions_path.relative_to(ROOT)),
                "predictions_sha256": sha256_file(predictions_path),
            }
    if len(reference_test_sha) > 1:
        raise RuntimeError("Existing M0/P1 test reports refer to different held-out test archives")

    source_sha = {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}
    data_sha = {
        relative: sha256_file(ROOT / relative)
        for relative in (config["training"]["train_split"], config["training"]["validation_split"])
    }
    checkpoint_sha = {
        str(seed): sha256_file(ROOT / config["seed_to_trajectory_checkpoint"][str(seed)])
        for seed in SEEDS
    }
    protocol = {
        "protocol_id": config["protocol_id"],
        "frozen": True,
        "base_commit": base_commit,
        "working_tree_policy": "source/config/test selection are frozen by source_sha256 and artifact hashes",
        "test_access_before_freeze": False,
        "test_not_accessed_before_freeze": True,
        "test_archive_sha256_reference_from_prior_reports": next(iter(reference_test_sha), None),
        "seeds": list(SEEDS),
        "architecture_and_input": config["model"],
        "input_protocol": config["input"],
        "training_protocol": config["training"],
        "trajectory_preservation_protocol": config["trajectory_preservation"],
        "paired_analysis_protocol": config["paired_analysis"],
        "selected_checkpoints": selected,
        "trajectory_checkpoints": {
            str(seed): {
                "path": config["seed_to_trajectory_checkpoint"][str(seed)],
                "sha256": checkpoint_sha[str(seed)],
            }
            for seed in SEEDS
        },
        "training_validation_data_sha256": data_sha,
        "immutable_baseline_artifacts": baseline_artifacts,
        "pretest_audit_artifacts_sha256": {
            name: sha256_file(path) for name, path in gate_paths.items()
        },
        "representation_shift_sha256": sha256_file(RESULTS_ROOT / "representation_shift.json"),
        "source_sha256": source_sha,
        "environment": {
            "python": sys.version,
            "torch": __import__("torch").__version__,
            "numpy": __import__("numpy").__version__,
        },
    }
    write_json(PROTOCOL_PATH, protocol)
    protocol_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    PROTOCOL_SHA_PATH.write_text(f"{protocol_hash}  protocol_frozen.json\n", encoding="utf-8")
    write_json(RESULTS_ROOT / "encoder_drift.json", encoder_drift)
    print(json.dumps({"protocol": str(PROTOCOL_PATH), "sha256": protocol_hash, "frozen": True}, indent=2))


if __name__ == "__main__":
    main()
