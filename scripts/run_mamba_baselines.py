#!/usr/bin/env python3
"""Run the authorized baseline stage, including at most one two-layer MT check."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.mamba_jaad_utils import CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS


def run_logged(command: list[str], log_name: str) -> None:
    log_dir = CHECKPOINT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / log_name
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing local log: {path}")
    with path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_method(method: str, script: str, layers: int = 3) -> None:
    directory = method if layers == 3 else f"{method}_layers{layers}"
    for seed in SEEDS:
        path = RESULTS_ROOT / directory / f"seed{seed}" / "metrics_validation.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("test_accessed") is not False or not payload.get("all_losses_and_gradients_finite"):
                raise RuntimeError(f"Invalid completed-run metadata: {path}")
            print(f"Preserving already completed run: {directory}/seed{seed}", flush=True)
            continue
        command = [sys.executable, str(ROOT / "scripts" / script), "--seed", str(seed), "--num-layers", str(layers)]
        run_logged(command, f"{directory}_seed{seed}.log")


def summarize() -> dict:
    subprocess.run([sys.executable, str(ROOT / "scripts/summarize_mamba_baselines.py")], cwd=ROOT, check=True)
    return json.loads((RESULTS_ROOT / "comparison.json").read_text(encoding="utf-8"))


def main() -> None:
    run_method("trajectory_transformer_target", "train_transformer_target_trajectory.py")
    run_method("trajectory_mamba", "train_mamba_trajectory.py")
    run_method("intention_mamba", "train_mamba_intention.py")
    result = summarize()
    if result["needs_layer2_sweep"]:
        print("Three-layer MT misses the prespecified gate; running the single permitted two-layer MT check.", flush=True)
        run_method("trajectory_mamba", "train_mamba_trajectory.py", layers=2)
        summarize()
    print(f"Baseline stage complete: {RESULTS_ROOT / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
