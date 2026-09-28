#!/usr/bin/env python3
"""Run J0 sequentially using only the per-seed settings extracted from J100."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.joint_traj_supervision_utils import CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, load_config, validate_metrics_payload, write_json


def load_contract(seed: int) -> dict[str, Any]:
    audit = json.loads((RESULTS_ROOT / "j0_vs_j100_config_audit.json").read_text(encoding="utf-8"))
    if audit.get("pass") is not True:
        raise RuntimeError("J0/J100 config audit has not passed")
    return audit["per_seed_J0_contract_from_J100"][str(seed)]


def command_for_seed(seed: int) -> list[str]:
    contract = load_contract(seed)
    output = Path("results/joint_traj_supervision_attribution/j0") / f"seed{seed}"
    checkpoint = Path("checkpoints/joint_traj_supervision_attribution") / f"j0_seed{seed}.pt"
    return [
        sys.executable, "scripts/train_joint_transformer_gate.py",
        "--data-root", contract["data_root"], "--ambiguous-root", contract["ambiguous_root"],
        "--output-root", str(output), "--checkpoint", str(checkpoint),
        "--gate-mode", contract["gate_mode"], "--epochs", str(contract["epochs"]),
        "--batch-size", str(contract["batch_size"]), "--hidden-dim", str(contract["hidden_dim"]),
        "--learning-rate", str(contract["learning_rate"]), "--prior-weight", str(contract["prior_weight"]),
        "--traj-weight", "0", "--traj-weight-mode", "fixed",
        "--ambiguous-weight", str(contract["ambiguous_weight"]), "--seed", str(seed), "--skip-test",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute the three frozen J0 runs")
    args = parser.parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": {}, "test_split_loaded": False, "status": "planned"}
    for seed in SEEDS:
        run_dir = RESULTS_ROOT / "j0" / f"seed{seed}"
        checkpoint = CHECKPOINT_ROOT / f"j0_seed{seed}.pt"
        manifest["seeds"][str(seed)] = {
            "command": command_for_seed(seed),
            "output_root": str(run_dir.relative_to(ROOT)),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "status": "pending" if args.run else "planned",
        }
        if args.run and (run_dir.exists() or checkpoint.exists()):
            raise FileExistsError(f"Refusing to overwrite prior J0 artifacts for seed {seed}")
    write_json(RESULTS_ROOT / "j0_run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if not args.run:
        return

    for seed in SEEDS:
        row = manifest["seeds"][str(seed)]
        row["status"] = "running"
        write_json(RESULTS_ROOT / "j0_run_manifest.json", manifest)
        (ROOT / row["output_root"]).mkdir(parents=True, exist_ok=False)
        log_path = ROOT / row["output_root"] / "training.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.run(row["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        if process.returncode != 0:
            row["status"] = "failed"
            row["return_code"] = process.returncode
            write_json(RESULTS_ROOT / "j0_run_manifest.json", manifest)
            raise RuntimeError(f"J0 seed {seed} failed; inspect {log_path}")
        metrics_path = ROOT / row["output_root"] / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_metrics_payload(metrics, seed=seed, traj_weight=0.0, epoch_count=int(load_contract(seed)["epochs"]))
        write_json(ROOT / row["output_root"] / "validation_history.json", metrics["history"])
        if metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
            raise RuntimeError(f"J0 seed {seed} did not preserve the test holdout")
        row["status"] = "completed_no_test"
        row["best_epoch"] = int(metrics["best_epoch"])
        print(json.dumps({"seed": seed, "status": row["status"], "best_epoch": row["best_epoch"]}), flush=True)
        write_json(RESULTS_ROOT / "j0_run_manifest.json", manifest)
    manifest["status"] = "all_three_validation_runs_complete"
    write_json(RESULTS_ROOT / "j0_run_manifest.json", manifest)


if __name__ == "__main__":
    main()
