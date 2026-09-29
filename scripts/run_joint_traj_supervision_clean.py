#!/usr/bin/env python3
"""Prepare, smoke-test, and run the matched clean J0/J100 experiment."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_joint_model_gradients import balanced_indices, stack_batch
from scripts.joint_traj_supervision_clean_utils import (
    ARMS, CHECKPOINT_ROOT, RESULTS_ROOT, SEEDS, SHARED_PREFIXES, config_path,
    gradient_norm, load_config, normalized_clean_contract, only_traj_weight_differs,
    sampler_sequences_match, sha256_file, sha256_state, write_json,
)
from scripts.train_joint_transformer_gate import (
    set_seed, shared_named_parameters,
)
from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.joint_transformer_gate import JointTransformerSceneGate


def args_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else vars(value)


def contract_for_arm(config_audit: dict[str, Any], seed: int, arm: str) -> dict[str, Any]:
    return config_audit["per_seed"][str(seed)][f"{arm}_contract"]


def model_from_dataset(dataset: JAADSequenceDataset, hidden_dim: int, gate_mode: str) -> JointTransformerSceneGate:
    return JointTransformerSceneGate(
        input_dim=8, scene_dim=int(dataset.scene_feat.shape[-1]), hidden_dim=hidden_dim,
        pred_len=int(dataset.future_gt.shape[1]), gate_mode=gate_mode,
        max_obs_len=int(dataset.target_obs.shape[1]),
    )


def historical_args(seed: int) -> dict[str, Any]:
    path = ROOT / f"checkpoints/joint_loss_balance/lambda100_seed{seed}.pt"
    metrics_path = ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json"
    if not path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Historical J100 source artifacts are missing for seed {seed}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    source = args_dict(checkpoint["args"])
    if int(source["seed"]) != seed or float(source["traj_weight"]) != 100.0 or metrics.get("test") is None:
        raise RuntimeError(f"Historical J100 source metadata is inconsistent for seed {seed}")
    return source


def create_initial_states(config: dict[str, Any]) -> dict[str, str]:
    data_root = ROOT / config["training"]["data_root"]
    train_set = JAADSequenceDataset(data_root / "train.npz")
    state_hashes: dict[str, str] = {}
    for seed in SEEDS:
        path = CHECKPOINT_ROOT / f"initial_state_seed{seed}.pt"
        set_seed(seed)
        model = model_from_dataset(train_set, int(config["training"]["hidden_dim"]), config["training"]["gate_mode"])
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        digest = sha256_state(state)
        if path.exists():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            if int(saved.get("seed", -1)) != seed or saved.get("sha256") != digest or sha256_state(saved["model"]) != digest:
                raise RuntimeError(f"Existing initial state differs from deterministic seed-{seed} initialization")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": state, "seed": seed, "sha256": digest, "base_commit": config["base_commit"]}, path)
        state_hashes[str(seed)] = digest
    return state_hashes


def audit_config(config: dict[str, Any], initial_hashes: dict[str, str]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    all_pass = True
    for seed in SEEDS:
        source = historical_args(seed)
        expected_values = {
            "epochs": 15, "batch_size": 512, "hidden_dim": 128,
            "learning_rate": 1e-3, "prior_weight": 0.5,
            "ambiguous_weight": 0.2, "gate_mode": "uncertainty",
        }
        mismatch = {key: {"actual": source.get(key), "expected": value} for key, value in expected_values.items() if source.get(key) != value}
        initial_path = str((CHECKPOINT_ROOT / f"initial_state_seed{seed}.pt").relative_to(ROOT))
        contracts = {
            arm: normalized_clean_contract(source, seed=seed, traj_weight=weight, initial_state_path=initial_path)
            for arm, weight in ARMS.items()
        }
        identical = only_traj_weight_differs(contracts["J0_clean"], contracts["J100_clean"])
        row_pass = not mismatch and identical
        all_pass &= row_pass
        per_seed[str(seed)] = {
            "historical_J100_metrics": f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json",
            "historical_J100_checkpoint": f"checkpoints/joint_loss_balance/lambda100_seed{seed}.pt",
            "historical_checkpoint_sha256": sha256_file(ROOT / f"checkpoints/joint_loss_balance/lambda100_seed{seed}.pt"),
            "historical_metrics_sha256": sha256_file(ROOT / f"results/joint_loss_balance/lambda100/seed{seed}/metrics.json"),
            "initial_state_checkpoint": initial_path,
            "initial_state_sha256": initial_hashes[str(seed)],
            "J0_clean_contract": contracts["J0_clean"],
            "J100_clean_contract": contracts["J100_clean"],
            "only_traj_weight_differs": identical,
            "historical_hyperparameter_mismatches": mismatch,
            "pass": row_pass,
        }
    payload = {
        "experiment": config["protocol_id"],
        "base_commit": config["base_commit"],
        "common_clean_selection": config["selection"],
        "common_clean_scheduler": config["scheduler"],
        "only_traj_weight_differs": all(row["only_traj_weight_differs"] for row in per_seed.values()),
        "all_expected_historical_hyperparameters_match": all(not row["historical_hyperparameter_mismatches"] for row in per_seed.values()),
        "per_seed": per_seed,
        "pass": all_pass,
        "test_accessed": False,
    }
    write_json(RESULTS_ROOT / "clean_j0_vs_j100_config_audit.json", payload)
    if not all_pass:
        raise RuntimeError("Clean J0/J100 config audit failed; do not smoke-test or train")
    return payload


def gradient_sanity_for_seed(seed: int, train_set: JAADSequenceDataset, ambiguous_set: JAADSequenceDataset, config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["training"]
    initial_path = CHECKPOINT_ROOT / f"initial_state_seed{seed}.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    model = model_from_dataset(train_set, int(cfg["hidden_dim"]), cfg["gate_mode"])
    model.load_state_dict(initial["model"], strict=True)
    model.train()
    indices = balanced_indices(train_set.intent_label, int(cfg["batch_size"]), seed + 6_100_001)
    amb_gen = torch.Generator(device="cpu").manual_seed(seed + 6_200_001)
    amb_idx = torch.randperm(len(ambiguous_set), generator=amb_gen)[: min(int(cfg["batch_size"]), len(ambiguous_set))]
    main_batch = stack_batch(train_set, indices)
    amb_batch = stack_batch(ambiguous_set, amb_idx)
    target = torch.cat([main_batch["target_obs"], main_batch["target_abs_obs"]], dim=-1)
    output = model(target, main_batch["neighbor_obs"], main_batch["neighbor_mask"], main_batch["neighbor_visible_mask"], main_batch["scene_feat"])
    label = main_batch["intent_label"]
    main_bce = nn.functional.binary_cross_entropy_with_logits(output["intent_logit"], label)
    prior_bce = nn.functional.binary_cross_entropy_with_logits(output["prior_logit"], label)
    amb_target = torch.cat([amb_batch["target_obs"], amb_batch["target_abs_obs"]], dim=-1)
    amb_output = model(amb_target, amb_batch["neighbor_obs"], amb_batch["neighbor_mask"], amb_batch["neighbor_visible_mask"], amb_batch["scene_feat"])
    ambiguity = 0.5 * (amb_output["prior_logit"].square().mean() + amb_output["intent_logit"].square().mean())
    intent_loss = main_bce + float(cfg["prior_weight"]) * prior_bce + float(cfg["ambiguous_weight"]) * ambiguity
    trajectory_loss = nn.functional.smooth_l1_loss(output["future_pred"], main_batch["future_gt"])
    shared = shared_named_parameters(model)
    names, params = [name for name, _ in shared], [param for _, param in shared]
    gi = torch.autograd.grad(intent_loss, params, retain_graph=True, allow_unused=True)
    gt = torch.autograd.grad(trajectory_loss, params, retain_graph=True, allow_unused=True)
    g0 = torch.autograd.grad(intent_loss + 0.0 * trajectory_loss, params, retain_graph=True, allow_unused=True)
    g100 = torch.autograd.grad(intent_loss + 100.0 * trajectory_loss, params, retain_graph=True, allow_unused=True)
    g0_traj = torch.autograd.grad(0.0 * trajectory_loss, params, retain_graph=True, allow_unused=True)
    g100_traj = torch.autograd.grad(100.0 * trajectory_loss, params, allow_unused=True)
    intent_norm, intent_vector = gradient_norm(gi, params)
    traj_norm, traj_vector = gradient_norm(gt, params)
    g0_norm, g0_vector = gradient_norm(g0, params)
    g100_norm, _ = gradient_norm(g100, params)
    j0_traj_contribution, _ = gradient_norm(g0_traj, params)
    j100_weighted_traj, weighted_vector = gradient_norm(g100_traj, params)
    cosine = float(torch.dot(intent_vector, traj_vector) / max(intent_norm * traj_norm, 1e-12))
    passed = (
        output["future_pred"].shape == (len(label), int(train_set.future_gt.shape[1]), 2)
        and intent_norm > 0.0 and traj_norm > 0.0
        and j0_traj_contribution == 0.0 and j100_weighted_traj > 0.0
        and torch.allclose(g0_vector, intent_vector, atol=1e-7, rtol=1e-5)
        and abs(j100_weighted_traj - 100.0 * traj_norm) <= max(1e-6, 1e-5 * j100_weighted_traj)
    )
    return {
        "seed": seed, "fixed_main_batch_size": len(label), "fixed_ambiguous_batch_size": len(amb_idx),
        "initial_state_sha256": initial["sha256"], "shared_parameter_count": sum(param.numel() for param in params),
        "future_pred_shape": list(output["future_pred"].shape),
        "intent_objective": float(intent_loss.detach()), "trajectory_raw_loss": float(trajectory_loss.detach()),
        "J0": {"traj_weight": 0.0, "intent_shared_grad_norm": intent_norm, "trajectory_contribution_shared_grad_norm": j0_traj_contribution, "total_shared_grad_norm": g0_norm, "trajectory_head_included_in_forward": True},
        "J100": {"traj_weight": 100.0, "trajectory_shared_grad_norm_raw": traj_norm, "trajectory_shared_grad_norm_weighted": j100_weighted_traj, "total_shared_grad_norm": g100_norm},
        "gradient_cosine_intent_vs_raw_trajectory": cosine,
        "J100_weighted_gradient_vector_norm_check": float(torch.linalg.vector_norm(weighted_vector)),
        "shared_parameter_names": names,
        "pass": bool(passed),
    }


def gradient_sanity_audit(config: dict[str, Any]) -> dict[str, Any]:
    train_set = JAADSequenceDataset(ROOT / config["training"]["data_root"] / "train.npz")
    ambiguous_set = JAADSequenceDataset(ROOT / config["training"]["ambiguous_root"] / "train.npz")
    per_seed = {str(seed): gradient_sanity_for_seed(seed, train_set, ambiguous_set, config) for seed in SEEDS}
    payload = {"per_seed": per_seed, "all_seeds_pass": all(row["pass"] for row in per_seed.values()), "test_split_loaded": False}
    write_json(RESULTS_ROOT / "gradient_sanity_audit.json", payload)
    if not payload["all_seeds_pass"]:
        raise RuntimeError("Gradient sanity audit failed; stop before smoke testing")
    return payload


def build_train_command(arm: str, seed: int, epochs: int, *, smoke: bool, contract: dict[str, Any]) -> tuple[list[str], Path, Path]:
    phase_dir = Path("results/joint_traj_supervision_clean") / ("smoke_test" if smoke else arm) / (arm if smoke else "")
    output_root = phase_dir / f"seed{seed}" if smoke else phase_dir / f"seed{seed}"
    # Formal layout is J0_clean/seedXX and J100_clean/seedXX.
    if not smoke:
        output_root = Path("results/joint_traj_supervision_clean") / arm / f"seed{seed}"
    checkpoint = Path("checkpoints/joint_traj_supervision_clean") / ("smoke_test" if smoke else "formal") / f"{arm}_seed{seed}.pt"
    initial = ROOT / contract["initial_state_checkpoint"]
    command = [
        sys.executable, "scripts/train_joint_transformer_gate.py",
        "--data-root", contract["data_root"], "--ambiguous-root", contract["ambiguous_root"],
        "--output-root", str(output_root), "--checkpoint", str(checkpoint),
        "--initial-state-checkpoint", str(initial), "--gate-mode", contract["gate_mode"],
        "--epochs", str(epochs), "--batch-size", str(contract["batch_size"]),
        "--hidden-dim", str(contract["hidden_dim"]), "--learning-rate", str(contract["learning_rate"]),
        "--prior-weight", str(contract["prior_weight"]), "--traj-weight", str(contract["traj_weight"]),
        "--traj-weight-mode", "fixed", "--ambiguous-weight", str(contract["ambiguous_weight"]),
        "--seed", str(seed), "--selection-mode", "intent_auc", "--selection-tolerance", str(contract["selection_tolerance"]),
        "--skip-test",
    ]
    return command, ROOT / output_root, ROOT / checkpoint


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def validate_run(arm: str, seed: int, output_root: Path, checkpoint_path: Path, expected_init_sha: str, epochs: int) -> dict[str, Any]:
    metrics = json.loads((output_root / "metrics.json").read_text(encoding="utf-8"))
    gradients = json.loads((output_root / "gradient_history.json").read_text(encoding="utf-8"))
    if int(metrics["seed"]) != seed or float(metrics["traj_weight"]) != ARMS[arm]:
        raise RuntimeError(f"Wrong seed or loss weight in run {arm}/seed{seed}")
    if metrics["test"] is not None or metrics["test_evaluation_status"] != "withheld_until_protocol_freeze":
        raise RuntimeError(f"Test was not withheld for {arm}/seed{seed}")
    if metrics["initial_model_state_sha256"] != expected_init_sha:
        raise RuntimeError(f"Initial model state differs from matched seed state for {arm}/seed{seed}")
    if len(metrics["history"]) != epochs or len(gradients["epochs"]) != epochs:
        raise RuntimeError(f"Incomplete epoch history for {arm}/seed{seed}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not finite_tree(metrics["history"]):
        raise RuntimeError(f"NaN/Inf in validation history for {arm}/seed{seed}")
    for epoch_row in metrics["history"]:
        selection = epoch_row["selection"]
        if epoch_row["scheduler_monitor_metric"] != "intent_auc" or not math.isclose(epoch_row["scheduler_monitor_value"], epoch_row["val"]["intent_auc"], rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError(f"Scheduler did not monitor raw validation AUC for {arm}/seed{seed}")
        if selection["primary_metric"] != "raw_validation_intent_auc" or selection["selection_used_trajectory_metric"]:
            raise RuntimeError(f"Selection used a forbidden metric for {arm}/seed{seed}")
    selected_epochs = [row["epoch"] for row in metrics["history"] if row["selection"]["selected_checkpoint"]]
    if not selected_epochs or selected_epochs[-1] != int(metrics["best_epoch"]):
        raise RuntimeError(f"Selected checkpoint epoch does not match metrics for {arm}/seed{seed}")
    for row in gradients["epochs"]:
        raw = float(row["trajectory_gradient_norm_unweighted"])
        weighted = float(row["trajectory_gradient_norm_weighted"])
        if arm == "J0_clean" and weighted != 0.0:
            raise RuntimeError(f"J0 trajectory gradient contribution is nonzero in {arm}/seed{seed}")
        if arm == "J100_clean" and (raw <= 0.0 or weighted <= 0.0):
            raise RuntimeError(f"J100 trajectory gradient is absent in {arm}/seed{seed}")
    write_json(output_root / "validation_history.json", metrics["history"])
    return {"arm": arm, "seed": seed, "status": "completed_no_test", "best_epoch": metrics["best_epoch"], "initial_model_state_sha256": metrics["initial_model_state_sha256"], "checkpoint_sha256": sha256_file(checkpoint_path), "metrics_sha256": sha256_file(output_root / "metrics.json"), "gradient_history_sha256": sha256_file(output_root / "gradient_history.json"), "validation_history_sha256": sha256_file(output_root / "validation_history.json"), "sampler_sha256_by_epoch": [row["train"]["sampler_sha256"] for row in metrics["history"]], "selection_mode": metrics["selection_mode"], "scheduler_monitor_metric": metrics["scheduler_monitor_metric"]}


def record_initialization_match(initial_hashes: dict[str, str], completed: dict[str, Any]) -> dict[str, Any]:
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed, "initial_state_sha256": initial_hashes[str(seed)], "max_abs_parameter_difference": None, "exact_match": None, "arms": {}}
        for arm in ARMS:
            key = f"{arm}/seed{seed}"
            if key in completed:
                row["arms"][arm] = {"initial_model_state_sha256": completed[key]["initial_model_state_sha256"], "run_status": completed[key]["status"]}
        hashes = [item["initial_model_state_sha256"] for item in row["arms"].values()]
        if len(hashes) == 2:
            matched = hashes[0] == hashes[1] == initial_hashes[str(seed)]
            row["exact_match"] = matched
            # Both training processes strict-loaded the same immutable tensor state;
            # equal bytewise state hashes establish zero parameter-wise difference.
            row["max_abs_parameter_difference"] = 0.0 if matched else None
        per_seed[str(seed)] = row
    payload = {"strategy": "both arms strict-load the same per-seed initial_state_seed*.pt", "per_seed": per_seed, "all_completed_pairs_exact_match": all(row["exact_match"] is True for row in per_seed.values() if row["exact_match"] is not None)}
    write_json(RESULTS_ROOT / "initialization_match.json", payload)
    return payload


def run_one(arm: str, seed: int, epochs: int, smoke: bool, contract: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    command, output_root, checkpoint_path = build_train_command(arm, seed, epochs, smoke=smoke, contract=contract)
    if output_root.exists() or checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_root} or {checkpoint_path}")
    output_root.mkdir(parents=True, exist_ok=False)
    log_path = output_root / "training.log"
    key = f"{arm}/seed{seed}"
    manifest[key] = {"status": "running", "command": command, "output_root": str(output_root.relative_to(ROOT)), "checkpoint": str(checkpoint_path.relative_to(ROOT))}
    write_json(RESULTS_ROOT / ("smoke_run_manifest.json" if smoke else "formal_run_manifest.json"), manifest)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if process.returncode != 0:
        manifest[key]["status"] = "failed"
        manifest[key]["return_code"] = process.returncode
        write_json(RESULTS_ROOT / ("smoke_run_manifest.json" if smoke else "formal_run_manifest.json"), manifest)
        raise RuntimeError(f"Training failed for {key}; inspect {log_path}")
    config = load_config()
    contract_audit = json.loads((RESULTS_ROOT / "clean_j0_vs_j100_config_audit.json").read_text(encoding="utf-8"))
    expected_sha = contract_audit["per_seed"][str(seed)]["initial_state_sha256"]
    result = validate_run(arm, seed, output_root, checkpoint_path, expected_sha, epochs)
    result["training_log_sha256"] = sha256_file(log_path)
    manifest[key].update(result)
    write_json(RESULTS_ROOT / ("smoke_run_manifest.json" if smoke else "formal_run_manifest.json"), manifest)
    return result


def run_smoke(config: dict[str, Any], initial_hashes: dict[str, str], config_audit: dict[str, Any], gradient_audit: dict[str, Any]) -> dict[str, Any]:
    smoke_manifest: dict[str, Any] = {"phase": "two_epoch_smoke", "test_accessed": False}
    results: dict[str, Any] = {}
    seed = 123
    for arm in ARMS:
        contract = contract_for_arm(config_audit, seed, arm)
        results[arm] = run_one(arm, seed, 2, True, contract, smoke_manifest)
    histories = {
        arm: json.loads((RESULTS_ROOT / "smoke_test" / arm / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))["history"]
        for arm in ARMS
    }
    exact_init = results["J0_clean"]["initial_model_state_sha256"] == results["J100_clean"]["initial_model_state_sha256"] == initial_hashes[str(seed)]
    same_sampler = sampler_sequences_match(histories["J0_clean"], histories["J100_clean"])
    finite_validation = all(finite_tree([row["val"] for row in history]) for history in histories.values())
    scheduler_and_selector = all(
        all(row["scheduler_monitor_metric"] == "intent_auc" and row["selection"]["primary_metric"] == "raw_validation_intent_auc" and not row["selection"]["selection_used_trajectory_metric"] for row in history)
        for history in histories.values()
    )
    j0_losses = json.loads((RESULTS_ROOT / "smoke_test/J0_clean/seed123/metrics.json").read_text(encoding="utf-8"))["history"]
    j100_gradient = json.loads((RESULTS_ROOT / "smoke_test/J100_clean/seed123/gradient_history.json").read_text(encoding="utf-8"))["epochs"]
    trajectory_checks = (
        all(row["train"]["trajectory_loss"] > 0.0 and row["train"]["weighted_trajectory_loss"] == 0.0 for row in j0_losses)
        and all(row["trajectory_gradient_norm_unweighted"] > 0.0 and row["trajectory_gradient_norm_weighted"] > 0.0 for row in j100_gradient)
    )
    passed = exact_init and same_sampler and finite_validation and scheduler_and_selector and trajectory_checks
    payload = {
        "seed": seed, "epochs_per_arm": 2, "J0_clean": results["J0_clean"], "J100_clean": results["J100_clean"],
        "initial_hash_exact_match": exact_init, "max_abs_initial_parameter_difference": 0.0 if exact_init else None,
        "same_main_sampler_sequences": same_sampler, "scheduler_and_selection_are_intent_auc_only": scheduler_and_selector,
        "J0_future_output_and_zero_weighted_trajectory_loss": trajectory_checks,
        "J100_trajectory_loss_and_backward_gradient_nonzero": trajectory_checks,
        "validation_finite": finite_validation, "test_accessed": False, "pass": passed,
    }
    write_json(RESULTS_ROOT / "smoke_test.json", payload)
    record_initialization_match(initial_hashes, {f"{arm}/seed{seed}": {**result, "status": "smoke_completed"} for arm, result in results.items()})
    if not passed:
        raise RuntimeError("Clean two-arm smoke test failed; do not launch formal runs")
    return payload


def run_formal(config: dict[str, Any], initial_hashes: dict[str, str], config_audit: dict[str, Any]) -> dict[str, Any]:
    smoke_path = RESULTS_ROOT / "smoke_test.json"
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("pass") is not True:
        raise RuntimeError("Formal runs require a passed J0/J100 smoke test")
    manifest: dict[str, Any] = {"phase": "formal_six_runs", "test_accessed": False, "status": "running"}
    completed: dict[str, Any] = {}
    for seed in SEEDS:
        for arm in ARMS:
            contract = contract_for_arm(config_audit, seed, arm)
            completed[f"{arm}/seed{seed}"] = run_one(arm, seed, int(contract["epochs"]), False, contract, manifest)
        history0 = json.loads((RESULTS_ROOT / "J0_clean" / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))["history"]
        history100 = json.loads((RESULTS_ROOT / "J100_clean" / f"seed{seed}/metrics.json").read_text(encoding="utf-8"))["history"]
        if not sampler_sequences_match(history0, history100):
            raise RuntimeError(f"Main sampler fingerprints differ between clean arms for seed {seed}")
    manifest["status"] = "all_six_formal_runs_complete_no_test"
    write_json(RESULTS_ROOT / "formal_run_manifest.json", manifest)
    init_match = record_initialization_match(initial_hashes, completed)
    if not init_match["all_completed_pairs_exact_match"]:
        raise RuntimeError("Formal J0/J100 initial state hashes do not match exactly")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    args = parser.parse_args()
    config = load_config()
    if config["base_commit"] != subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Repository base commit differs from the clean-study config; freeze/update protocol before training")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    initial_hashes = create_initial_states(config)
    config_audit = audit_config(config, initial_hashes)
    if args.phase == "smoke":
        gradient_audit = gradient_sanity_audit(config)
        payload = run_smoke(config, initial_hashes, config_audit, gradient_audit)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    run_formal(config, initial_hashes, config_audit)
    print(json.dumps({"status": "all_six_formal_runs_complete_no_test", "initialization_match": str(RESULTS_ROOT / "initialization_match.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
