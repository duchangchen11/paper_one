#!/usr/bin/env python3
"""Validate DGB validation-only outputs, then freeze the protocol before test use."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "ad05ef51711d74f423cecd0b7e8d89debb0a3ae4"
SEEDS = (42, 123, 2024)
SOURCE_FILES = (
    "scripts/train_joint_transformer_gate.py",
    "src/models/joint_transformer_gate.py",
    "scripts/evaluate_joint_dynamic_balance_test.py",
    "scripts/summarize_joint_dynamic_balance.py",
    "scripts/freeze_joint_dynamic_balance_protocol.py",
    "configs/joint_dynamic_balance.json",
    "tests/test_joint_dynamic_balance.py",
)
DATA_FILES = (
    "data/processed/jaad_sequences_scene_15x15/train.npz",
    "data/processed/jaad_sequences_scene_15x15/val.npz",
    "data/processed/jaad_sequences_scene_15x15/test.npz",
    "data/processed/jaad_ambiguous_scene_15x15/train.npz",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if current_commit != BASE_COMMIT:
        raise RuntimeError(f"Expected baseline {BASE_COMMIT}, got {current_commit}")
    if subprocess.check_output(
        ["git", "diff", "--name-only", "--", "src/models/joint_transformer_gate.py"],
        cwd=ROOT,
        text=True,
    ).strip():
        raise RuntimeError("Model architecture source was modified; refusing to freeze")

    smoke_report = json.loads(
        (ROOT / "results/joint_dynamic_balance/smoke_test/report.json").read_text(encoding="utf-8")
    )
    if smoke_report.get("status") != "passed" or not all(smoke_report.get("checks", {}).values()):
        raise RuntimeError("DGB two-epoch smoke checks did not all pass")

    seed_review = {}
    for seed in SEEDS:
        run_dir = ROOT / "results/joint_dynamic_balance/dgb20" / f"seed{seed}"
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        gradients = json.loads((run_dir / "gradient_history.json").read_text(encoding="utf-8"))
        checkpoint = ROOT / "checkpoints/joint_dynamic_balance" / f"dgb20_seed{seed}.pt"
        if metrics.get("test") is not None or metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
            raise RuntimeError(f"Seed {seed} has a test result before protocol freeze")
        if metrics.get("seed") != seed or len(metrics.get("history", [])) != 15:
            raise RuntimeError(f"Seed {seed} is incomplete or has unexpected training settings")
        if not checkpoint.is_file():
            raise RuntimeError(f"Missing validation-selected checkpoint: {checkpoint}")
        best_score = max(
            row["val"]["intent_auc"]
            + 0.1 * row["val"]["intent_f1"]
            - 0.01 * row["val"]["trajectory_ade_pixel"]
            for row in metrics["history"]
        )
        selected = next(row for row in metrics["history"] if row["epoch"] == metrics["best_epoch"])
        selected_score = (
            selected["val"]["intent_auc"]
            + 0.1 * selected["val"]["intent_f1"]
            - 0.01 * selected["val"]["trajectory_ade_pixel"]
        )
        if not np.isclose(best_score, selected_score, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"Seed {seed} best checkpoint does not follow the frozen selection rule")
        if len(gradients["epochs"]) != 15:
            raise RuntimeError(f"Seed {seed} gradient history is incomplete")
        updates = [
            update
            for epoch in gradients["epochs"]
            for update in epoch["update_records"]
            if update.get("controller_update")
        ]
        if len(updates) != 56 or any(update.get("update_status") == "invalid_gradient_norm" for update in updates):
            raise RuntimeError(f"Seed {seed} controller update audit failed")
        lambdas = [
            float(value)
            for epoch in gradients["epochs"]
            for value in epoch["lambda_per_training_batch"]
        ]
        if not all(np.isfinite(lambdas)) or min(lambdas) < 10.0 or max(lambdas) > 300.0:
            raise RuntimeError(f"Seed {seed} lambda trace is invalid")
        seed_review[str(seed)] = {
            "best_epoch": int(metrics["best_epoch"]),
            "validation_at_best_checkpoint": selected["val"],
            "lambda_min": float(min(lambdas)),
            "lambda_max": float(max(lambdas)),
            "lambda_mean": float(np.mean(lambdas)),
            "controller_updates_applied": len(updates),
            "controller_bound_hits": int(
                sum(bool(update.get("hit_lambda_min")) or bool(update.get("hit_lambda_max")) for update in updates)
            ),
            "checkpoint_sha256": sha256_file(checkpoint),
        }

    manifest_path = ROOT / "results/joint_dynamic_balance/shared_parameter_names.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["scalar_count"] != 830530 or any(
        name.startswith(("intent_head.", "traj_head.")) for name in manifest["parameter_names"]
    ):
        raise RuntimeError("Shared parameter scope differs from the audited scope")

    protocol = {
        "frozen": True,
        "protocol_name": "DGB-20 validation-selected joint intention/trajectory experiment",
        "frozen_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_commit": BASE_COMMIT,
        "source_state": "Implementation frozen as an uncommitted working-tree patch based on base_commit; source SHA256 values identify the exact code used.",
        "source_sha256": {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES},
        "data_sha256": {relative: sha256_file(ROOT / relative) for relative in DATA_FILES},
        "seeds": list(SEEDS),
        "method": {
            "name": "Dynamic Gradient Balance DGB-20",
            "trajectory_weight_mode": "dynamic_gradient",
            "initial_lambda": 100.0,
            "target_weighted_shared_gradient_ratio": 20.0,
            "log_space_ema_beta": 0.9,
            "lambda_bounds": [10.0, 300.0],
            "warmup_epochs": 1,
            "update_interval_training_batches": 10,
            "gradient_epsilon": 1e-12,
            "update_timing": "sample on each tenth batch; after a valid update the new lambda starts on the following batch",
            "gradient_scope_manifest": "results/joint_dynamic_balance/shared_parameter_names.json",
        },
        "fixed_training_settings": {
            "data_root": "data/processed/jaad_sequences_scene_15x15",
            "ambiguous_root": "data/processed/jaad_ambiguous_scene_15x15",
            "epochs": 15,
            "batch_size": 512,
            "hidden_dim": 128,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "prior_weight": 0.5,
            "ambiguous_weight": 0.2,
            "gate_mode": "uncertainty",
            "architecture_changed": False,
            "JAAD_splits_changed": False,
            "test_split_used_for_training_or_selection": False,
        },
        "checkpoint_selection_rule": "validation AUC + 0.1 * validation F1 - 0.01 * validation ADE_pixel",
        "validation_only_review": seed_review,
        "smoke_test": smoke_report,
        "fixed100_reference_artifacts": {
            str(seed): {
                "metrics": f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json",
                "metrics_sha256": sha256_file(
                    ROOT / "results/joint_loss_balance/lambda100" / f"seed{seed}" / "metrics.json"
                ),
                "gradient_history": f"results/joint_loss_balance/lambda100/seed{seed}/gradient_history.json",
                "gradient_history_sha256": sha256_file(
                    ROOT / "results/joint_loss_balance/lambda100" / f"seed{seed}" / "gradient_history.json"
                ),
            }
            for seed in SEEDS
        },
        "test_access_audit": {
            "before_freeze": "A one-time schema preflight read the processed test archive to list field names and array shapes; no labels, feature values, predictions, metrics, or aggregates were inspected or used for a decision. The training script then withheld the test split for all three formal runs.",
            "after_freeze": "Official test metrics and per-sample predictions may now be generated once from the frozen best checkpoints; no DGB changes may follow from test results.",
        },
        "unit_tests": {"command": "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q", "passed": 94, "failed": 0},
        "no_sweep_or_post_test_tuning": True,
    }
    protocol_path = ROOT / "results/joint_dynamic_balance/protocol_frozen.json"
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum = sha256_file(protocol_path)
    checksum_path = ROOT / "results/joint_dynamic_balance/protocol_frozen.sha256"
    checksum_path.write_text(f"{checksum}  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({"protocol": str(protocol_path), "sha256": checksum, "validation_review": seed_review}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
