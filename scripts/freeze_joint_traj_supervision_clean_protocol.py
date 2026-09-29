#!/usr/bin/env python3
"""Freeze all clean-study choices/artifacts before its first test access."""

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

from scripts.joint_traj_supervision_clean_utils import (
    ARMS, CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, load_config,
    sampler_sequences_match, sha256_file, sha256_state, write_json,
)
from scripts.train_joint_transformer_gate import intent_auc_selection_decision


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes(relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: sha256_file(ROOT / relative) for relative in relative_paths}


def main() -> None:
    config = load_config()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != config["base_commit"]:
        raise RuntimeError(f"Base commit changed before freeze: config={config['base_commit']} HEAD={head}")
    if (RESULTS_ROOT / "protocol_frozen.json").exists() or (RESULTS_ROOT / "protocol_frozen.sha256").exists():
        raise FileExistsError("Clean protocol is already frozen; do not overwrite it")
    if (RESULTS_ROOT / "official_test_access_record.json").exists():
        raise RuntimeError("Clean test access marker exists before freeze")

    config_audit = load_json(RESULTS_ROOT / "clean_j0_vs_j100_config_audit.json")
    gradient_audit = load_json(RESULTS_ROOT / "gradient_sanity_audit.json")
    smoke = load_json(RESULTS_ROOT / "smoke_test.json")
    manifest = load_json(RESULTS_ROOT / "formal_run_manifest.json")
    init_match = load_json(RESULTS_ROOT / "initialization_match.json")
    if not config_audit.get("pass") or not config_audit.get("only_traj_weight_differs"):
        raise RuntimeError("Clean config audit did not pass")
    if not gradient_audit.get("all_seeds_pass") or smoke.get("pass") is not True:
        raise RuntimeError("Gradient sanity or smoke test did not pass")
    if manifest.get("status") != "all_six_formal_runs_complete_no_test" or manifest.get("test_accessed") is not False:
        raise RuntimeError("Six formal runs did not complete under test withholding")
    if init_match.get("all_completed_pairs_exact_match") is not True:
        raise RuntimeError("Matched initialization audit did not pass")

    selected: dict[str, dict[str, Any]] = {arm: {} for arm in ARMS}
    frozen: dict[str, str] = {}
    for relative in (
        "configs/joint_traj_supervision_clean.json",
        "src/models/joint_transformer_gate.py",
        "src/data/jaad_sequence_dataset.py",
        "scripts/train_joint_transformer_gate.py",
        "scripts/joint_traj_supervision_clean_utils.py",
        "scripts/run_joint_traj_supervision_clean.py",
        "scripts/analyze_joint_traj_supervision_clean_diagnostics.py",
        "scripts/freeze_joint_traj_supervision_clean_protocol.py",
        "scripts/evaluate_joint_traj_supervision_clean_test.py",
        "scripts/summarize_joint_traj_supervision_clean.py",
        "scripts/reliability_gated_intent_utils.py",
        "tests/test_joint_traj_supervision_clean.py",
    ):
        frozen[relative] = sha256_file(ROOT / relative)

    pretest_files = [
        "data/processed/jaad_sequences_scene_15x15/train.npz",
        "data/processed/jaad_sequences_scene_15x15/val.npz",
        "data/processed/jaad_ambiguous_scene_15x15/train.npz",
        "results/joint_traj_supervision_clean/clean_j0_vs_j100_config_audit.json",
        "results/joint_traj_supervision_clean/gradient_sanity_audit.json",
        "results/joint_traj_supervision_clean/smoke_test.json",
        "results/joint_traj_supervision_clean/formal_run_manifest.json",
        "results/joint_traj_supervision_clean/initialization_match.json",
        "results/joint_traj_supervision_clean/formal_run_manifest.json",
    ]
    for relative in pretest_files:
        frozen[relative] = sha256_file(ROOT / relative)

    for seed in SEEDS:
        seed_key = str(seed)
        init_path = CHECKPOINT_ROOT / f"initial_state_seed{seed}.pt"
        init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        init_sha = sha256_state(init_payload["model"])
        if init_sha != init_payload["sha256"]:
            raise RuntimeError(f"Initial state hash mismatch for seed {seed}")
        init_row = init_match["per_seed"][seed_key]
        if init_row.get("exact_match") is not True or init_row.get("max_abs_parameter_difference") != 0.0:
            raise RuntimeError(f"Formal arms do not have exact shared initialization for seed {seed}")
        frozen[str(init_path.relative_to(ROOT))] = sha256_file(init_path)

        histories: dict[str, list[dict[str, Any]]] = {}
        for arm, weight in ARMS.items():
            run_dir = RESULTS_ROOT / arm / f"seed{seed}"
            checkpoint_path = CHECKPOINT_ROOT / "formal" / f"{arm}_seed{seed}.pt"
            for path in (
                run_dir / "metrics.json", run_dir / "validation_history.json",
                run_dir / "gradient_history.json", run_dir / "training.log", checkpoint_path,
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing pre-freeze run artifact: {path}")
            if (run_dir / "test_predictions.npz").exists() or (run_dir / "official_test_metrics.json").exists():
                raise RuntimeError(f"Clean test result exists before protocol freeze: {run_dir}")
            metrics = load_json(run_dir / "metrics.json")
            history = load_json(run_dir / "validation_history.json")
            gradient_history = load_json(run_dir / "gradient_history.json")
            manifest_row = manifest.get(f"{arm}/seed{seed}", {})
            epochs = int(config["training"]["epochs"])
            if metrics.get("test") is not None or metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
                raise RuntimeError(f"Test was not withheld for {arm}/seed{seed}")
            if metrics.get("selection_mode") != "intent_auc" or metrics.get("scheduler_monitor_metric") != "intent_auc":
                raise RuntimeError(f"AUC-only selection/scheduler is missing for {arm}/seed{seed}")
            if len(metrics.get("history", [])) != epochs or len(history) != epochs or len(gradient_history.get("epochs", [])) != epochs:
                raise RuntimeError(f"Incomplete 15-epoch record for {arm}/seed{seed}")
            if int(metrics["seed"]) != seed or float(metrics["traj_weight"]) != weight:
                raise RuntimeError(f"Arm/seed metadata mismatch for {arm}/seed{seed}")
            if metrics["initial_model_state_sha256"] != init_sha:
                raise RuntimeError(f"Initial state changed for {arm}/seed{seed}")
            for row in history:
                if row["scheduler_monitor_metric"] != "intent_auc" or row["selection"]["selection_used_trajectory_metric"]:
                    raise RuntimeError(f"Forbidden scheduler/selection metric used for {arm}/seed{seed}")
                if row["selection"]["primary_metric"] != "raw_validation_intent_auc":
                    raise RuntimeError(f"Primary checkpoint metric is not AUC for {arm}/seed{seed}")
            # Replay the registered AUC/Brier checkpoint rule without reading test.
            selected_epoch = None
            best_auc = selected_auc = selected_brier = None
            for row in history:
                val = row["val"]
                choose, _, best_auc = intent_auc_selection_decision(
                    float(val["intent_auc"]), float(val["intent_brier"]), best_auc,
                    selected_auc, selected_brier, float(config["selection"]["tolerance"]),
                )
                if choose:
                    selected_epoch = int(row["epoch"])
                    selected_auc = float(val["intent_auc"])
                    selected_brier = float(val["intent_brier"])
            if selected_epoch != int(metrics["best_epoch"]):
                raise RuntimeError(f"AUC/Brier selection replay disagrees for {arm}/seed{seed}")
            if manifest_row.get("status") != "completed_no_test":
                raise RuntimeError(f"Formal run manifest incomplete for {arm}/seed{seed}")
            if arm == "J0_clean" and any(float(row["trajectory_gradient_norm_weighted"]) != 0.0 for row in gradient_history["epochs"]):
                raise RuntimeError(f"J0 trajectory gradient is not zero for seed {seed}")
            if arm == "J100_clean" and any(float(row["trajectory_gradient_norm_weighted"]) <= 0.0 for row in gradient_history["epochs"]):
                raise RuntimeError(f"J100 trajectory gradient is absent for seed {seed}")
            histories[arm] = metrics["history"]
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state_sha = sha256_state(checkpoint["model"])
            key = str(checkpoint_path.relative_to(ROOT))
            frozen[key] = sha256_file(checkpoint_path)
            for artifact in ("metrics.json", "validation_history.json", "gradient_history.json", "training.log"):
                artifact_path = run_dir / artifact
                frozen[str(artifact_path.relative_to(ROOT))] = sha256_file(artifact_path)
            selected[arm][seed_key] = {
                "checkpoint_path": key,
                "checkpoint_sha256": frozen[key],
                "selected_model_state_sha256": state_sha,
                "initial_state_sha256": init_sha,
                "selected_epoch": int(metrics["best_epoch"]),
                "selected_validation_auc": float(metrics["selected_checkpoint_validation_auc"]),
                "selected_validation_brier": float(metrics["selected_checkpoint_validation_brier"]),
                "trajectory_weight": weight,
                "test_metrics_absent_before_freeze": True,
            }
        if not sampler_sequences_match(histories["J0_clean"], histories["J100_clean"]):
            raise RuntimeError(f"Matched main-sampler sequences differ for seed {seed}")

    # Reuse only an already frozen historical fingerprint; do not open/hash the
    # current test archive here. This records the expected immutable JAAD test file.
    prior_protocol_path = ROOT / "results/joint_traj_supervision_attribution/protocol_frozen.json"
    prior_protocol = load_json(prior_protocol_path)
    expected_hashes = set(prior_protocol["test_access_before_freeze"]["expected_archive_sha256_reused_from_frozen_M0_P1_records"].values())
    if len(expected_hashes) != 1:
        raise RuntimeError("Prior frozen protocol does not provide one consistent expected test archive hash")
    prior_protocol_sha_path = prior_protocol_path.with_name("protocol_frozen.sha256")
    frozen[str(prior_protocol_path.relative_to(ROOT))] = sha256_file(prior_protocol_path)
    frozen[str(prior_protocol_sha_path.relative_to(ROOT))] = sha256_file(prior_protocol_sha_path)
    for seed in SEEDS:
        reference = ROOT / f"results/trajectory_preserving_joint/P1_target_only/seed{seed}/test_predictions.npz"
        if not reference.is_file():
            raise FileNotFoundError(f"Prior frozen P1 sample-ID reference is missing: {reference}")
        frozen[str(reference.relative_to(ROOT))] = sha256_file(reference)

    protocol = {
        "protocol_id": config["protocol_id"],
        "experiment": config["experiment"],
        "status": "frozen_before_first_access_to_test_archive",
        "base_commit": head,
        "working_tree_changes_are_frozen_by_sha256": True,
        "seeds": list(SEEDS),
        "arms": config["arms"],
        "training": config["training"],
        "scheduler": config["scheduler"],
        "selection": config["selection"],
        "initialization": config["initialization"],
        "primary_endpoint": config["primary_endpoint"],
        "secondary_endpoints": config["secondary_endpoints"],
        "bootstrap": config["bootstrap"],
        "trajectory_metrics_role": config["trajectory_metrics_role"],
        "historical_reference_policy": config["historical_reference"],
        "pre_registered_support_rules": config["support_rules"],
        "test_access_before_freeze": {
            "clean_test_archive_opened_or_hashed": False,
            "clean_test_dataset_loaded_or_evaluated": False,
            "clean_test_metrics_or_predictions_examined": False,
            "test_access_count_for_this_protocol": 0,
            "planned_test_archive_path": "data/processed/jaad_sequences_scene_15x15/test.npz",
            "expected_archive_sha256_from_prior_frozen_protocol": next(iter(expected_hashes)),
            "historical_pre_freeze_metadata_note": "The earlier historical J100 metrics JSON was read only to verify that a test result field existed; no historical metric value was used in this protocol or training. Current clean test archive and clean test metrics remained untouched.",
        },
        "formal_run_manifest_sha256": sha256_file(RESULTS_ROOT / "formal_run_manifest.json"),
        "config_audit_sha256": sha256_file(RESULTS_ROOT / "clean_j0_vs_j100_config_audit.json"),
        "gradient_sanity_audit_sha256": sha256_file(RESULTS_ROOT / "gradient_sanity_audit.json"),
        "smoke_test_sha256": sha256_file(RESULTS_ROOT / "smoke_test.json"),
        "initialization_match_sha256": sha256_file(RESULTS_ROOT / "initialization_match.json"),
        "selected_checkpoints": selected,
        "paired_sampling_fingerprints_match": True,
        "test_access_count": 0,
        "sha256": {"frozen_pretest_artifacts": dict(sorted(frozen.items()))},
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "device_available": "cuda" if torch.cuda.is_available() else "cpu",
        },
    }
    protocol_path = RESULTS_ROOT / "protocol_frozen.json"
    checksum_path = RESULTS_ROOT / "protocol_frozen.sha256"
    encoded = (json.dumps(protocol, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    protocol_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    checksum_path.write_text(f"{digest}  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({
        "protocol": str(protocol_path.relative_to(ROOT)),
        "protocol_sha256": digest,
        "test_not_accessed_before_freeze": True,
        "selected_checkpoints": {arm: {seed: record["selected_epoch"] for seed, record in rows.items()} for arm, rows in selected.items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
