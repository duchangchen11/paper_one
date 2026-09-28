#!/usr/bin/env python3
"""Freeze all pre-test artifacts and hashes for the J0/J100 attribution study."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_utils import (
    CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, load_config, normalized_training_contract,
    sha256_file, write_json,
)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def hash_record(paths: list[Path]) -> dict[str, str]:
    return {relative(path): sha256_file(path) for path in paths}


def best_composite_epoch(history: list[dict[str, Any]]) -> int:
    scores = [row["val"]["intent_auc"] + 0.1 * row["val"]["intent_f1"] - 0.01 * row["val"]["trajectory_ade_pixel"] for row in history]
    return int(np.argmax(scores) + 1)


def main() -> None:
    config = load_config()
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if base_commit != config["base_commit"]:
        raise RuntimeError(f"Base commit changed: expected {config['base_commit']}, found {base_commit}")

    manifest_path = RESULTS_ROOT / "j0_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "all_three_validation_runs_complete":
        raise RuntimeError("All three J0 training runs must finish before protocol freeze")

    config_audit_path = RESULTS_ROOT / "j0_vs_j100_config_audit.json"
    gradient_audit_path = RESULTS_ROOT / "j0_gradient_audit.json"
    config_audit = json.loads(config_audit_path.read_text(encoding="utf-8"))
    gradient_audit = json.loads(gradient_audit_path.read_text(encoding="utf-8"))
    if not config_audit.get("pass") or not gradient_audit.get("all_seeds_pass"):
        raise RuntimeError("Pre-training configuration or gradient audit did not pass")
    if gradient_audit.get("test_split_loaded") is not False:
        raise RuntimeError("Test split was marked as loaded before freeze")

    selection_path = RESULTS_ROOT / "checkpoint_selection_sensitivity.json"
    representation_path = RESULTS_ROOT / "representation_drift.json"
    gradients_path = RESULTS_ROOT / "gradient_diagnostics.json"
    pytest_xml_path = RESULTS_ROOT / "pytest_results.xml"
    for path in (selection_path, representation_path, gradients_path, pytest_xml_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing pre-test diagnostic: {path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    representation = json.loads(representation_path.read_text(encoding="utf-8"))
    gradients = json.loads(gradients_path.read_text(encoding="utf-8"))
    if selection.get("test_split_loaded") is not False or representation.get("selection_use") is not False or gradients.get("selection_use") is not False:
        raise RuntimeError("A validation diagnostic has an invalid test/selection-use declaration")
    pytest_root = ET.parse(pytest_xml_path).getroot()
    suites = [pytest_root] if pytest_root.tag == "testsuite" else list(pytest_root.findall("testsuite"))
    test_counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    test_counts["passed"] = test_counts["tests"] - test_counts["failures"] - test_counts["errors"] - test_counts["skipped"]
    if test_counts["failures"] or test_counts["errors"]:
        raise RuntimeError(f"Pre-freeze full pytest suite has failures/errors: {test_counts}")

    source_paths = [
        ROOT / "configs/joint_traj_supervision_attribution.json",
        ROOT / "scripts/joint_traj_supervision_utils.py",
        ROOT / "scripts/audit_joint_traj_supervision.py",
        ROOT / "scripts/run_joint_traj_supervision_zero.py",
        ROOT / "scripts/analyze_joint_traj_supervision_validation.py",
        ROOT / "scripts/freeze_joint_traj_supervision_protocol.py",
        ROOT / "scripts/evaluate_joint_traj_supervision_test.py",
        ROOT / "scripts/analyze_joint_traj_supervision_attribution.py",
        ROOT / "scripts/train_joint_transformer_gate.py",
        ROOT / "src/models/joint_transformer_gate.py",
        ROOT / "tests/test_joint_traj_supervision_attribution.py",
    ]
    data_paths = [
        ROOT / config["training"]["data_root"] / "train.npz",
        ROOT / config["training"]["data_root"] / "val.npz",
        ROOT / config["training"]["ambiguous_root"] / "train.npz",
    ]
    if not all(path.is_file() for path in source_paths + data_paths):
        raise FileNotFoundError("One or more source/data files needed for freeze are missing")

    per_seed: dict[str, Any] = {}
    expected_test_hashes: dict[str, str] = {}
    artifact_paths: list[Path] = [manifest_path, config_audit_path, gradient_audit_path, selection_path, representation_path, gradients_path, pytest_xml_path]
    for seed in SEEDS:
        run_row = manifest["seeds"][str(seed)]
        if run_row.get("status") != "completed_no_test":
            raise RuntimeError(f"J0 seed{seed} did not complete without test evaluation")
        command = run_row.get("command", [])
        if "--skip-test" not in command or any("test.npz" in str(value) for value in command):
            raise RuntimeError(f"J0 seed{seed} command did not explicitly withhold test data")
        j0_metrics_path = RESULTS_ROOT / "j0" / f"seed{seed}/metrics.json"
        j0_validation_history_path = RESULTS_ROOT / "j0" / f"seed{seed}/validation_history.json"
        j0_gradient_path = RESULTS_ROOT / "j0" / f"seed{seed}/gradient_history.json"
        j0_log_path = RESULTS_ROOT / "j0" / f"seed{seed}/training.log"
        j0_checkpoint_path = CHECKPOINT_ROOT / f"j0_seed{seed}.pt"
        j100_metrics_path = ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json"
        j100_gradient_path = ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/gradient_history.json"
        j100_checkpoint_path = ROOT / f"checkpoints/joint_loss_balance/lambda100_seed{seed}.pt"
        m0_metrics_path = ROOT / f"results/intention_scratch_matched/seed{seed}/metrics.json"
        p1_metrics_path = ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/metrics.json"
        p1_predictions_path = ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/test_predictions.npz"
        t0_metrics_path = ROOT / f"results/trajectory_transformer_scene_15x15_seed{seed}/metrics.json"
        required = [j0_metrics_path, j0_gradient_path, j0_log_path, j0_checkpoint_path, j100_metrics_path, j100_gradient_path, j100_checkpoint_path, m0_metrics_path, p1_metrics_path, p1_predictions_path, t0_metrics_path]
        if not all(path.is_file() for path in required):
            missing = [relative(path) for path in required if not path.is_file()]
            raise FileNotFoundError(f"Missing frozen comparator or J0 artifact(s): {missing}")

        j0_metrics = json.loads(j0_metrics_path.read_text(encoding="utf-8"))
        j100_metrics = json.loads(j100_metrics_path.read_text(encoding="utf-8"))
        m0_metrics = json.loads(m0_metrics_path.read_text(encoding="utf-8"))
        p1_metrics = json.loads(p1_metrics_path.read_text(encoding="utf-8"))
        if not j0_validation_history_path.exists():
            write_json(j0_validation_history_path, j0_metrics["history"])
        j0_validation_history = json.loads(j0_validation_history_path.read_text(encoding="utf-8"))
        if j0_metrics.get("test") is not None or j0_metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
            raise RuntimeError(f"J0 seed{seed} contains a pre-freeze test result")
        if len(j0_metrics.get("history", [])) != 15 or len(j100_metrics.get("history", [])) != 15:
            raise RuntimeError(f"Incomplete training history for seed {seed}")
        if j0_validation_history != j0_metrics["history"]:
            raise RuntimeError(f"J0 seed{seed} validation history serialization differs from its metrics artifact")
        historic_hashes = {
            str(m0_metrics.get("test", {}).get("test_npz_sha256_after_protocol_freeze", "")),
            str(p1_metrics.get("test", {}).get("test_npz_sha256_after_freeze", "")),
        }
        historic_hashes.discard("")
        if len(historic_hashes) != 1:
            raise RuntimeError(f"Historical M0/P1 test archive references disagree or are missing for seed {seed}")
        expected_test_hashes[str(seed)] = next(iter(historic_hashes))
        if best_composite_epoch(j0_metrics["history"]) != int(j0_metrics["best_epoch"]):
            raise RuntimeError(f"J0 seed{seed} selected checkpoint does not match the frozen validation score")
        if best_composite_epoch(j100_metrics["history"]) != int(j100_metrics["best_epoch"]):
            raise RuntimeError(f"J100 seed{seed} selected checkpoint does not match the historical validation score")
        for path in (j0_metrics_path, j0_validation_history_path, j0_gradient_path, j0_log_path, j0_checkpoint_path, j100_metrics_path, j100_gradient_path, j100_checkpoint_path, m0_metrics_path, p1_metrics_path, p1_predictions_path, t0_metrics_path):
            artifact_paths.append(path)
        per_seed[str(seed)] = {
            "J0": {
                "metrics": relative(j0_metrics_path), "gradient_history": relative(j0_gradient_path),
                "validation_history": relative(j0_validation_history_path),
                "checkpoint": relative(j0_checkpoint_path), "best_epoch": int(j0_metrics["best_epoch"]),
                "initial_model_state_sha256": j0_metrics["initial_model_state_sha256"],
                "trajectory_metrics_at_selected_epoch": {
                    "validation_ADE_pixel": j0_metrics["history"][int(j0_metrics["best_epoch"]) - 1]["val"]["trajectory_ade_pixel"],
                    "validation_FDE_pixel": j0_metrics["history"][int(j0_metrics["best_epoch"]) - 1]["val"]["trajectory_fde_pixel"],
                },
            },
            "J100": {
                "metrics": relative(j100_metrics_path), "gradient_history": relative(j100_gradient_path),
                "checkpoint": relative(j100_checkpoint_path), "best_epoch": int(j100_metrics["best_epoch"]),
                "historical_test_metrics_reused_without_retraining": j100_metrics["test"],
                "initial_model_hash_note": "not persisted in historical J100 metrics; same seed/model-construction sequence is source-audited and reconstructed in validation diagnostics",
            },
            "M0_metrics": relative(m0_metrics_path),
            "P1_metrics": relative(p1_metrics_path),
            "P1_test_prediction_reference": relative(p1_predictions_path),
            "T0_trajectory_only_metrics": relative(t0_metrics_path),
        }

    # No test.npz path is opened, hashed, stat'ed, or loaded here. Only already
    # materialized historical summaries/predictions are fingerprinted.
    protocol = {
        "protocol_id": config["protocol_id"],
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": base_commit,
        "definition": config["primary_comparison"],
        "training_contract_source": "actual J100 checkpoint args and metrics, audited per seed",
        "training_contracts_J100": {seed: config_audit["per_seed_J100_frozen_artifacts"][seed]["training_contract"] for seed in map(str, SEEDS)},
        "training_contracts_J0": config_audit["per_seed_J0_contract_from_J100"],
        "only_training_difference": "traj_weight: 0.0 vs 100.0",
        "selection_rule": "validation intent_auc + 0.1*intent_f1 - 0.01*trajectory_ade_pixel",
        "test_access_before_freeze": {
            "test_dataset_loaded_or_evaluated_by_this_protocol": False,
            "test_archive_opened_or_newly_hashed_by_this_protocol": False,
            "historical_J100_test_metrics_reused": True,
            "historical_P1_prediction_artifacts_fingerprinted_without_loading": True,
            "planned_test_archive_path": str(Path(config["training"]["data_root"]) / "test.npz"),
            "expected_archive_sha256_reused_from_frozen_M0_P1_records": expected_test_hashes,
        },
        "official_test_plan": {
            "official_test_evaluation_count": 1,
            "evaluate_selected_J0_and_frozen_J100_checkpoints_on_same_samples": True,
            "primary_endpoint": "intention ROC-AUC; paired delta J100 minus J0",
            "bootstrap_unit": "scene_id",
            "bootstrap_repetitions_per_seed": 2000,
            "bootstrap_seed": 9124,
            "video_id_join": "exact unique (scene_id,target_id,obs_end_frame) match to frozen P1 test_predictions.npz",
            "calibration": "none for J0/J100; raw sigmoid probabilities and the historical 0.5 threshold",
        },
        "pytest": {
            "command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q --junitxml=results/joint_traj_supervision_attribution/pytest_results.xml",
            "counts": test_counts,
            "junit_xml": relative(pytest_xml_path),
        },
        "per_seed_artifacts": per_seed,
        "sha256": {
            "config_sources_and_code": hash_record(source_paths),
            "training_and_validation_data_only": hash_record(data_paths),
            "pretest_and_comparator_artifacts": hash_record(artifact_paths),
        },
        "test_archive_sha256": None,
        "status": "frozen_before_first_access_to_test_archive",
    }
    protocol_path = RESULTS_ROOT / "protocol_frozen.json"
    protocol_hash_path = RESULTS_ROOT / "protocol_frozen.sha256"
    if protocol_path.exists() or protocol_hash_path.exists():
        raise FileExistsError("Protocol freeze already exists; refusing to overwrite")
    write_json(protocol_path, protocol)
    digest = sha256_file(protocol_path)
    protocol_hash_path.write_text(f"{digest}  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({"protocol": relative(protocol_path), "sha256": digest, "test_dataset_loaded_or_evaluated": False, "status": protocol["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
