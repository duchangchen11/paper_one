#!/usr/bin/env python3
"""Run the predeclared Clean J0 component-attribution protocol.

Phases are deliberately separate: prepare -> smoke -> train -> freeze ->
evaluate -> summarize. The evaluate phase refuses to open test.npz unless the
pre-test protocol and every frozen artifact verify successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results/joint_component_attribution"
CHECKPOINTS = ROOT / "checkpoints/joint_component_attribution"
DATA = ROOT / "data/processed/jaad_sequences_scene_15x15"
AMBIGUOUS = ROOT / "data/processed/jaad_ambiguous_scene_15x15"
BASELINE_RESULTS = ROOT / "results/joint_traj_supervision_clean/J0_clean"
BASELINE_CHECKPOINTS = ROOT / "checkpoints/joint_traj_supervision_clean/formal"
INITIAL_STATES = ROOT / "checkpoints/joint_traj_supervision_clean"
BASELINE_MANIFEST = ROOT / "results/joint_traj_supervision_clean/formal_run_manifest.json"
PROTOCOL_PATH = RESULTS / "protocol_frozen.json"
PROTOCOL_SHA_PATH = RESULTS / "protocol_frozen.sha256"
TEST_ACCESS_PATH = RESULTS / "official_test_access_record.json"
SEEDS = (42, 123, 2024)
VARIANTS = {
    "A1_no_scene": "no_scene",
    "A2_no_social": "no_social",
    "A3_no_proposal_loss": "no_proposal_loss",
    "A4_no_adaptive_gate": "no_adaptive_gate",
    "A5_no_ambiguity": "no_ambiguity",
}
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 20260929
BASE_TRAIN_ARGS = {
    "epochs": 15,
    "batch_size": 512,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "optimizer": "AdamW",
    "weight_decay": 1e-4,
    "gradient_clip": 5.0,
    "gate_mode": "uncertainty",
    "prior_weight": 0.5,
    "traj_weight": 0.0,
    "traj_weight_mode": "fixed",
    "ambiguous_weight": 0.2,
    "selection_mode": "intent_auc",
    "selection_tolerance": 1e-4,
    "scheduler": "ReduceLROnPlateau(mode=max,factor=0.5,patience=2)",
    "scheduler_monitor": "raw_validation_auc",
    "checkpoint_primary": "raw_validation_auc",
    "checkpoint_tie_break": "raw_validation_brier_within_1e-4",
    "trajectory_metrics_used_for_selection": False,
}


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_state(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    return state, payload if isinstance(payload, dict) else {}


def write_component_definitions() -> dict[str, Any]:
    definitions = {
        "study": "Clean J0 single-component intention architecture attribution",
        "baseline": "A0 Full Clean J0, traj_weight=0",
        "primary_endpoint": "official test ROC-AUC, accessed only after protocol_frozen.json and SHA256 are written",
        "seeds": list(SEEDS),
        "training_protocol": BASE_TRAIN_ARGS,
        "sampling": "class-balanced WeightedRandomSampler with replacement; same integer seed and epoch sampler fingerprint as A0",
        "initialization": "load the corresponding checkpoints/joint_traj_supervision_clean/initial_state_seed{seed}.pt for every arm",
        "variants": {
            "A1_no_scene": {
                "intervention": "replace scene_context by zeros_like before proposal_fusion, gate input, and main fusion",
                "retained": ["scene_encoder module", "parameter registration", "fusion dimensions"],
            },
            "A2_no_social": {
                "intervention": "replace social_context by zeros_like before proposal_fusion and main fusion",
                "retained": ["neighbor GRU", "parameter registration", "fusion dimensions"],
            },
            "A3_no_proposal_loss": {
                "intervention": "prior_weight=0",
                "retained": ["proposal forward", "prior entropy", "uncertainty gate", "fusion"],
            },
            "A4_no_adaptive_gate": {
                "intervention": "replace sample-dependent gate by fixed g=0.5",
                "retained": ["scene/social contexts", "both fusion branches", "all registered modules"],
                "rationale": "neutral scale for the implemented g*social_context path; avoids deleting social input with g=0 or g=1",
            },
            "A5_no_ambiguity": {
                "intervention": "ambiguous_weight=0",
                "retained": ["architecture", "ordinary training paths"],
            },
        },
        "effect_signs": {
            "primary_reported_delta": "Ablation - Full for bootstrap, as predeclared in primary endpoint section",
            "component_contribution": "Full - Ablation; positive means removing the component reduced AUC",
            "brier_delta": "Ablation - Full; positive means calibration worsened after removal",
        },
        "inference": "exploratory component attribution; five comparisons; no confirmatory significance claims",
        "forbidden_this_round": [
            "proposal branch removal", "fusion removal", "target encoder removal", "pairwise ablations",
            "trajectory auxiliary supervision", "reliability inputs", "DGB", "adapters", "PCGrad", "GradNorm", "lambda sweeps",
        ],
    }
    write_json(RESULTS / "component_definitions.json", definitions)
    return definitions


def legacy_forward(model: torch.nn.Module, target, neighbor, neighbor_mask, visible_mask, scene):
    """Reference implementation of the pre-ablation forward tensor flow."""
    obs_len = target.shape[1]
    target_encoded = model.target_projection(target) + model.position_embedding[:, :obs_len]
    target_context = model.target_encoder(target_encoded)[:, -1]
    scene_context = model.scene_encoder(scene)
    batch_size, max_neighbors, obs_len, feature_dim = neighbor.shape
    neighbor_input = neighbor.reshape(batch_size * max_neighbors, obs_len, feature_dim)
    _, neighbor_hidden = model.neighbor_encoder(neighbor_input)
    neighbor_context = neighbor_hidden[-1].view(batch_size, max_neighbors, -1)
    slot_mask = neighbor_mask.bool()
    masked_context = neighbor_context * slot_mask.unsqueeze(-1)
    denominator = slot_mask.sum(dim=1, keepdim=True).clamp_min(1).to(masked_context.dtype)
    social_context = masked_context.sum(dim=1) / denominator
    social_context = social_context * slot_mask.any(dim=1, keepdim=True).to(social_context.dtype)
    proposal_context = model.proposal_fusion(torch.cat([target_context, scene_context, social_context], dim=-1))
    prior_logit = model.proposal_head(proposal_context).squeeze(-1)
    prior_prob = torch.sigmoid(prior_logit).clamp(1e-6, 1.0 - 1e-6)
    entropy = -prior_prob * torch.log(prior_prob) - (1.0 - prior_prob) * torch.log(1.0 - prior_prob)
    gate = model.gate(torch.cat([target_context, scene_context, entropy.unsqueeze(-1)], dim=-1)).squeeze(-1)
    fused = model.fusion(torch.cat([target_context, scene_context, gate.unsqueeze(-1) * social_context], dim=-1))
    intent_logit = model.intent_head(fused).squeeze(-1)
    future_pred = model.traj_head(torch.cat([fused, target_context], dim=-1)).view(-1, model.pred_len, 2)
    return intent_logit, future_pred


def prepare_baseline_equivalence() -> dict[str, Any]:
    from src.models.joint_transformer_gate import JointTransformerSceneGate

    RESULTS.mkdir(parents=True, exist_ok=True)
    val_path = DATA / "val.npz"
    with np.load(val_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    val_order_hash = hashlib.sha256(
        canonical_bytes(
            [
                [str(scene), str(target), int(frame)]
                for scene, target, frame in zip(arrays["scene_id"], arrays["target_id"], arrays["obs_end_frame"])
            ]
        )
    ).hexdigest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_arrays: dict[str, np.ndarray] = {}
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        checkpoint_path = BASELINE_CHECKPOINTS / f"J0_clean_seed{seed}.pt"
        state, _ = load_checkpoint_state(checkpoint_path)
        model = JointTransformerSceneGate(
            input_dim=8,
            scene_dim=int(state["scene_encoder.0.weight"].shape[1]),
            hidden_dim=int(state["target_projection.weight"].shape[0]),
            pred_len=int(state["traj_head.3.weight"].shape[0] // 2),
            gate_mode="uncertainty",
            max_obs_len=int(state["position_embedding"].shape[1]),
            component_flags_all_enabled=True,
        )
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        model_intent, ref_intent = [], []
        model_future, ref_future = [], []
        with torch.inference_mode():
            for start in range(0, len(arrays["intent_label"]), 512):
                stop = min(start + 512, len(arrays["intent_label"]))
                def t(key: str) -> torch.Tensor:
                    return torch.from_numpy(np.asarray(arrays[key][start:stop], dtype=np.float32)).to(device)
                target = torch.cat([t("target_obs"), t("target_abs_obs")], dim=-1)
                inputs = (target, t("neighbor_obs"), t("neighbor_mask"), t("neighbor_visible_mask"), t("scene_feat"))
                new_output = model(*inputs)
                old_intent, old_future = legacy_forward(model, *inputs)
                model_intent.append(new_output["intent_logit"].cpu().numpy().astype(np.float32))
                ref_intent.append(old_intent.cpu().numpy().astype(np.float32))
                model_future.append(new_output["future_pred"].cpu().numpy().astype(np.float32))
                ref_future.append(old_future.cpu().numpy().astype(np.float32))
        new_intent = np.concatenate(model_intent)
        old_intent = np.concatenate(ref_intent)
        new_future = np.concatenate(model_future)
        old_future = np.concatenate(ref_future)
        intent_diff = float(np.max(np.abs(new_intent - old_intent)))
        future_diff = float(np.max(np.abs(new_future - old_future)))
        per_seed[str(seed)] = {
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "same_checkpoint_same_validation_input": True,
            "reference_forward": "literal tensor-flow transcription of pre-ablation JointTransformerSceneGate.forward",
            "max_abs_intent_logit_diff": intent_diff,
            "max_abs_future_pred_diff": future_diff,
            "intent_logits_equivalent_lt_1e-6": intent_diff < 1e-6,
            "future_predictions_equivalent_lt_1e-6": future_diff < 1e-6,
            "validation_samples": int(len(new_intent)),
            "validation_order_sha256": val_order_hash,
            "reference_intent_sha256": hashlib.sha256(old_intent.tobytes()).hexdigest(),
            "reference_future_sha256": hashlib.sha256(old_future.tobytes()).hexdigest(),
        }
        reference_arrays[f"seed{seed}_intent_logit"] = old_intent
        reference_arrays[f"seed{seed}_future_pred"] = old_future
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(
        item["intent_logits_equivalent_lt_1e-6"] and item["future_predictions_equivalent_lt_1e-6"]
        for item in per_seed.values()
    ):
        raise RuntimeError("All-enabled model failed pre-ablation forward equivalence")
    np.savez_compressed(RESULTS / "baseline_equivalence_reference.npz", **reference_arrays)
    report = {
        "status": "passed",
        "scope": "A0 Clean J0 real selected checkpoint and complete validation split; no test split accessed",
        "component_flags_all_enabled": True,
        "all_seeds_passed": True,
        "per_seed": per_seed,
        "validation_archive_sha256": sha256_file(val_path),
        "reference_outputs_path": "results/joint_component_attribution/baseline_equivalence_reference.npz",
        "reference_outputs_sha256": sha256_file(RESULTS / "baseline_equivalence_reference.npz"),
    }
    write_json(RESULTS / "baseline_equivalence.json", report)
    return report


def prepare_initialization_audit() -> dict[str, Any]:
    from scripts.train_joint_transformer_gate import state_dict_sha256

    audit: dict[str, Any] = {
        "policy": "all A1-A5 arms use the corresponding exact Clean J0 initial_state_seed*.pt; A0 reuses completed Clean J0",
        "per_seed": {},
    }
    baseline_manifest = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    for seed in SEEDS:
        path = INITIAL_STATES / f"initial_state_seed{seed}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        actual_hash = state_dict_sha256(state)
        declared_hash = payload.get("sha256") if isinstance(payload, dict) else None
        baseline = baseline_manifest[f"J0_clean/seed{seed}"]
        if actual_hash != baseline["initial_model_state_sha256"]:
            raise RuntimeError(f"Clean J0 initialization hash mismatch for seed {seed}")
        if declared_hash is not None and declared_hash != actual_hash:
            raise RuntimeError(f"Initial-state self hash mismatch for seed {seed}")
        audit["per_seed"][str(seed)] = {
            "initial_state_path": str(path.relative_to(ROOT)),
            "initial_state_file_sha256": sha256_file(path),
            "initial_model_state_sha256": actual_hash,
            "baseline_clean_j0_initial_model_state_sha256": baseline["initial_model_state_sha256"],
            "exact_match": True,
            "arms": {"A0_full_clean_j0": "existing matched baseline"} | {name: "planned" for name in VARIANTS},
        }
    write_json(RESULTS / "initialization_audit.json", audit)
    return audit


def phase_prepare() -> None:
    definitions = write_component_definitions()
    dependency_map = RESULTS / "component_dependency_map.md"
    if not dependency_map.is_file():
        raise FileNotFoundError("component_dependency_map.md must be completed before prepare")
    initialization = prepare_initialization_audit()
    equivalence = prepare_baseline_equivalence()
    print(json.dumps({
        "phase": "prepare",
        "variants": list(definitions["variants"]),
        "initialization_seeds": list(initialization["per_seed"]),
        "baseline_equivalence": equivalence["status"],
        "test_split_accessed": False,
    }, ensure_ascii=False))


def sampler_hashes_from_baseline(seed: int) -> list[str]:
    payload = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    return list(payload[f"J0_clean/seed{seed}"]["sampler_sha256_by_epoch"])


def train_command(variant: str, seed: int, epochs: int, smoke: bool) -> tuple[list[str], Path, Path]:
    slug = VARIANTS[variant]
    output = RESULTS / "smoke" / variant / f"seed{seed}" if smoke else RESULTS / variant / f"seed{seed}"
    checkpoint = CHECKPOINTS / "smoke" / f"{variant}_seed{seed}.pt" if smoke else CHECKPOINTS / variant / f"seed{seed}.pt"
    initial = INITIAL_STATES / f"initial_state_seed{seed}.pt"
    command = [
        sys.executable, "scripts/train_joint_transformer_gate.py",
        "--data-root", str(DATA.relative_to(ROOT)),
        "--ambiguous-root", str(AMBIGUOUS.relative_to(ROOT)),
        "--output-root", str(output.relative_to(ROOT)),
        "--checkpoint", str(checkpoint.relative_to(ROOT)),
        "--initial-state-checkpoint", str(initial),
        "--gate-mode", "uncertainty",
        "--component-ablation", slug,
        "--epochs", str(epochs),
        "--batch-size", "512",
        "--hidden-dim", "128",
        "--learning-rate", "0.001",
        "--prior-weight", "0.5",
        "--traj-weight", "0.0",
        "--traj-weight-mode", "fixed",
        "--ambiguous-weight", "0.2",
        "--seed", str(seed),
        "--selection-mode", "intent_auc",
        "--selection-tolerance", "0.0001",
        "--skip-test",
    ]
    return command, output, checkpoint


def run_training(variant: str, seed: int, epochs: int, smoke: bool) -> dict[str, Any]:
    command, output, checkpoint = train_command(variant, seed, epochs, smoke)
    metrics_path = output / "metrics.json"
    log_path = output / "training.log"
    if metrics_path.is_file() and checkpoint.is_file():
        existing = json.loads(metrics_path.read_text(encoding="utf-8"))
        if (
            existing.get("component_ablation") == VARIANTS[variant]
            and existing.get("seed") == seed
            and existing.get("test") is None
            and len(existing.get("history", [])) == epochs
        ):
            print(f"reuse completed {variant} seed={seed} epochs={epochs}", flush=True)
            return existing
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(f"start {variant} seed={seed} epochs={epochs}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in record and "val" in record:
                print(
                    f"{variant} seed={seed} epoch={record['epoch']}/{epochs} "
                    f"val_auc={record['val'].get('intent_auc', float('nan')):.5f} "
                    f"val_brier={record['val'].get('intent_brier', float('nan')):.5f}",
                    flush=True,
                )
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Training failed for {variant}/seed{seed}; see {log_path}")
    if not metrics_path.is_file() or not checkpoint.is_file():
        raise RuntimeError(f"Training outputs are incomplete for {variant}/seed{seed}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("test") is not None:
        raise RuntimeError(f"Training run unexpectedly evaluated test: {variant}/seed{seed}")
    return metrics


def phase_smoke() -> None:
    baseline_sampler = sampler_hashes_from_baseline(123)[:2]
    report: dict[str, Any] = {
        "status": "running",
        "seed": 123,
        "epochs": 2,
        "test_split_loaded": False,
        "sampler_reference": "A0 Clean J0 seed123 first two epoch fingerprints",
        "variants": {},
    }
    for variant in VARIANTS:
        metrics = run_training(variant, 123, epochs=2, smoke=True)
        component_probe = metrics["component_probe"]
        _, output, checkpoint = train_command(variant, 123, 2, smoke=True)
        sampler = [entry["train"]["sampler_sha256"] for entry in metrics["history"]]
        checks = {
            "two_epochs": len(metrics["history"]) == 2,
            "test_unloaded": metrics.get("test") is None and metrics.get("test_evaluation_status") == "withheld_until_protocol_freeze",
            "finite_validation_auc": all(math.isfinite(float(item["val"]["intent_auc"])) for item in metrics["history"]),
            "matched_sampler": sampler == baseline_sampler,
            "matched_initialization": metrics["initial_model_state_sha256"] == json.loads((RESULTS / "initialization_audit.json").read_text(encoding="utf-8"))["per_seed"]["123"]["initial_model_state_sha256"],
            "intent_and_future_finite": component_probe["intent_logits_finite"] and component_probe["future_predictions_finite"],
            "no_scene_tensor_zero": component_probe["scene_downstream_tensor_zero"],
            "no_social_tensor_zero": component_probe["social_downstream_tensor_zero"],
            "proposal_output_present": component_probe["proposal_output_present"],
            "no_proposal_weight_zero": component_probe["weighted_proposal_loss_zero"] == (variant == "A3_no_proposal_loss"),
            "neutral_gate_and_both_modules_present": component_probe["fixed_neutral_gate"] and component_probe["both_gate_fusion_contexts_present"],
            "no_ambiguity_weight_zero": component_probe["weighted_ambiguity_contribution_zero"] == (variant == "A5_no_ambiguity"),
            "checkpoint_saved": checkpoint.is_file(),
            "run_path": str(output.relative_to(ROOT)),
        }
        report["variants"][variant] = {
            "status": "passed" if all(value for key, value in checks.items() if key != "run_path") else "failed",
            "checks": checks,
            "sampler_sha256_by_epoch": sampler,
            "component_probe": component_probe,
        }
        if report["variants"][variant]["status"] != "passed":
            report["status"] = "failed"
            write_json(RESULTS / "smoke_test.json", report)
            raise RuntimeError(f"Smoke checks failed for {variant}: {checks}")
    report["status"] = "passed"
    report["all_test_splits_unloaded"] = True
    write_json(RESULTS / "smoke_test.json", report)
    print(json.dumps({"phase": "smoke", "status": report["status"], "runs": len(VARIANTS), "test_split_accessed": False}, ensure_ascii=False))


def phase_train() -> None:
    if not (RESULTS / "smoke_test.json").is_file():
        raise RuntimeError("Smoke phase must pass before formal training")
    smoke = json.loads((RESULTS / "smoke_test.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "passed":
        raise RuntimeError("Smoke report is not passed")
    manifest_path = RESULTS / "formal_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "status": "running", "runs": {}, "test_split_loaded": False,
    }
    for variant in VARIANTS:
        for seed in SEEDS:
            metrics = run_training(variant, seed, epochs=15, smoke=False)
            expected_initial = json.loads((RESULTS / "initialization_audit.json").read_text(encoding="utf-8"))["per_seed"][str(seed)]["initial_model_state_sha256"]
            sampler = [entry["train"]["sampler_sha256"] for entry in metrics["history"]]
            expected_sampler = sampler_hashes_from_baseline(seed)
            if len(metrics["history"]) != 15 or metrics["initial_model_state_sha256"] != expected_initial:
                raise RuntimeError(f"Formal run failed epoch/init check: {variant}/seed{seed}")
            if sampler != expected_sampler:
                raise RuntimeError(f"Sampler fingerprint mismatch vs A0: {variant}/seed{seed}")
            if metrics.get("test") is not None or metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
                raise RuntimeError(f"Formal run accessed test: {variant}/seed{seed}")
            entry = {
                "status": "completed_no_test",
                "variant": variant,
                "seed": seed,
                "command": train_command(variant, seed, 15, False)[0],
                "output_root": str((RESULTS / variant / f"seed{seed}").relative_to(ROOT)),
                "checkpoint": str((CHECKPOINTS / variant / f"seed{seed}.pt").relative_to(ROOT)),
                "initial_model_state_sha256": metrics["initial_model_state_sha256"],
                "sampler_sha256_by_epoch": sampler,
                "selected_validation_auc": metrics["selected_checkpoint_validation_auc"],
                "selected_validation_brier": metrics["selected_checkpoint_validation_brier"],
                "best_epoch": metrics["best_epoch"],
                "test_split_loaded": False,
            }
            entry["metrics_sha256"] = sha256_file(RESULTS / variant / f"seed{seed}" / "metrics.json")
            entry["checkpoint_sha256"] = sha256_file(CHECKPOINTS / variant / f"seed{seed}.pt")
            entry["training_log_sha256"] = sha256_file(RESULTS / variant / f"seed{seed}" / "training.log")
            manifest["runs"][f"{variant}/seed{seed}"] = entry
            manifest["test_split_loaded"] = False
            write_json(manifest_path, manifest)
    manifest["status"] = "all_15_formal_runs_complete_no_test"
    write_json(manifest_path, manifest)
    print(json.dumps({"phase": "train", "status": manifest["status"], "runs": len(manifest["runs"]), "test_split_accessed": False}, ensure_ascii=False))


def frozen_artifact_paths() -> list[Path]:
    paths = [
        ROOT / "src/models/joint_transformer_gate.py",
        ROOT / "scripts/train_joint_transformer_gate.py",
        ROOT / "scripts/run_joint_component_attribution.py",
        ROOT / "tests/test_joint_component_attribution.py",
        RESULTS / "component_dependency_map.md",
        RESULTS / "component_definitions.json",
        RESULTS / "initialization_audit.json",
        RESULTS / "baseline_equivalence.json",
        RESULTS / "baseline_equivalence_reference.npz",
        RESULTS / "smoke_test.json",
        RESULTS / "formal_run_manifest.json",
        BASELINE_MANIFEST,
        DATA / "train.npz",
        DATA / "val.npz",
        AMBIGUOUS / "train.npz",
    ]
    for seed in SEEDS:
        paths.extend(
            [
                INITIAL_STATES / f"initial_state_seed{seed}.pt",
                BASELINE_CHECKPOINTS / f"J0_clean_seed{seed}.pt",
            ]
        )
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = RESULTS / variant / f"seed{seed}"
            paths.extend(
                [
                    run_dir / "metrics.json",
                    run_dir / "gradient_history.json",
                    run_dir / "training.log",
                    CHECKPOINTS / variant / f"seed{seed}.pt",
                ]
            )
    return paths


def phase_freeze() -> None:
    if TEST_ACCESS_PATH.exists():
        raise RuntimeError("Test-access marker exists; refusing to freeze a post-access protocol")
    if not (RESULTS / "baseline_equivalence.json").is_file():
        raise RuntimeError("prepare phase has not completed")
    equivalence = json.loads((RESULTS / "baseline_equivalence.json").read_text(encoding="utf-8"))
    smoke = json.loads((RESULTS / "smoke_test.json").read_text(encoding="utf-8"))
    manifest_path = RESULTS / "formal_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if equivalence.get("status") != "passed" or smoke.get("status") != "passed":
        raise RuntimeError("Baseline equivalence and smoke checks must pass before freeze")
    if manifest.get("status") != "all_15_formal_runs_complete_no_test" or len(manifest.get("runs", {})) != 15:
        raise RuntimeError("All 15 formal runs must complete before freeze")

    initialization = json.loads((RESULTS / "initialization_audit.json").read_text(encoding="utf-8"))
    baseline_sampler = {str(seed): sampler_hashes_from_baseline(seed) for seed in SEEDS}
    for variant in VARIANTS:
        for seed in SEEDS:
            key = f"{variant}/seed{seed}"
            entry = manifest["runs"].get(key)
            if not entry or entry.get("status") != "completed_no_test":
                raise RuntimeError(f"Missing formal run: {key}")
            metrics_path = RESULTS / variant / f"seed{seed}" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            expected_init = initialization["per_seed"][str(seed)]["initial_model_state_sha256"]
            if metrics.get("initial_model_state_sha256") != expected_init:
                raise RuntimeError(f"Initialization mismatch in {key}")
            if metrics.get("test") is not None or metrics.get("test_evaluation_status") != "withheld_until_protocol_freeze":
                raise RuntimeError(f"Test was evaluated in {key}")
            if entry.get("sampler_sha256_by_epoch") != baseline_sampler[str(seed)]:
                raise RuntimeError(f"Sampler differed from Clean J0 in {key}")
            if len(metrics.get("history", [])) != 15:
                raise RuntimeError(f"Expected 15 training epochs in {key}")
            run_dir = RESULTS / variant / f"seed{seed}"
            for forbidden in ("test_predictions.npz", "official_test_metrics.json", "test_access_record.json"):
                if (run_dir / forbidden).exists():
                    raise RuntimeError(f"Pre-freeze test artifact exists: {run_dir / forbidden}")

    missing = [str(path.relative_to(ROOT)) for path in frozen_artifact_paths() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Cannot freeze; missing artifacts: {missing}")
    hashed = {str(path.relative_to(ROOT)): sha256_file(path) for path in frozen_artifact_paths()}
    payload = {
        "status": "frozen_before_first_access_to_test_archive",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_access_before_freeze": {
            "clean_test_archive_opened_or_hashed": False,
            "test_split_loaded_by_any_formal_run": False,
            "test_access_marker_exists": False,
            "planned_test_archive_path": str((DATA / "test.npz").relative_to(ROOT)),
        },
        "baseline": "A0 Full Clean J0; existing selected checkpoints reused, not retrained",
        "arms": {"A0": "Full Clean J0", **VARIANTS},
        "seeds": list(SEEDS),
        "training_protocol": BASE_TRAIN_ARGS,
        "matched_initialization": {seed: initialization["per_seed"][seed]["initial_model_state_sha256"] for seed in initialization["per_seed"]},
        "matched_sampler_fingerprints_by_seed": baseline_sampler,
        "primary_endpoint": "raw official-test ROC-AUC",
        "secondary_metrics": ["Brier", "F1 at 0.5", "balanced accuracy at 0.5", "accuracy at 0.5"],
        "selection": "raw validation AUC, raw validation Brier tie-break within 1e-4; scheduler monitors raw validation AUC; no trajectory metric used",
        "comparisons": [f"{variant} vs A0 Full Clean J0" for variant in VARIANTS],
        "bootstrap": {
            "unit": "scene_id cluster",
            "paired": True,
            "repetitions_per_comparison_per_seed": BOOTSTRAP_REPETITIONS,
            "random_seed": BOOTSTRAP_SEED,
            "seed_derivation": "per comparison and seed: 20260929 + variant_index*100 + seed",
            "percentile_ci": "95%",
            "seeds_pooled": False,
            "interpretation": "exploratory component attribution; five comparisons are not confirmatory multiplicity-adjusted tests",
            "reported_bootstrap_delta": "Ablation - Full; contribution is its negation, Full - Ablation",
        },
        "frozen_pretest_artifacts": hashed,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    RESULTS.mkdir(parents=True, exist_ok=True)
    PROTOCOL_PATH.write_text(raw, encoding="utf-8")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    PROTOCOL_SHA_PATH.write_text(digest + "  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({"phase": "freeze", "status": payload["status"], "sha256": digest, "frozen_artifacts": len(hashed), "test_archive_opened": False}, ensure_ascii=False))


def verify_frozen_protocol() -> tuple[dict[str, Any], str]:
    if not PROTOCOL_PATH.is_file() or not PROTOCOL_SHA_PATH.is_file():
        raise RuntimeError("Official-test evaluation requires a frozen protocol and SHA256")
    raw = PROTOCOL_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    declared = PROTOCOL_SHA_PATH.read_text(encoding="utf-8").split()[0]
    if digest != declared:
        raise RuntimeError("protocol_frozen.json SHA256 mismatch")
    protocol = json.loads(raw)
    if protocol.get("status") != "frozen_before_first_access_to_test_archive":
        raise RuntimeError("Protocol was not frozen before test access")
    if protocol.get("test_access_before_freeze", {}).get("clean_test_archive_opened_or_hashed") is not False:
        raise RuntimeError("Protocol claims test access before freeze")
    for relative, expected in protocol["frozen_pretest_artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Frozen artifact changed: {relative}")
    return protocol, digest


def load_model_for_variant(variant: str, checkpoint_path: Path) -> torch.nn.Module:
    from src.models.joint_transformer_gate import JointTransformerSceneGate

    state, _ = load_checkpoint_state(checkpoint_path)
    component = "full" if variant == "A0_full_clean_j0" else VARIANTS[variant]
    model = JointTransformerSceneGate(
        input_dim=8,
        scene_dim=int(state["scene_encoder.0.weight"].shape[1]),
        hidden_dim=int(state["target_projection.weight"].shape[0]),
        pred_len=int(state["traj_head.3.weight"].shape[0] // 2),
        gate_mode="uncertainty",
        max_obs_len=int(state["position_embedding"].shape[1]),
        component_flags_all_enabled=component == "full",
        component_ablation=None if component == "full" else component,
    )
    model.load_state_dict(state, strict=True)
    return model


def predict_test(model: torch.nn.Module, arrays: dict[str, np.ndarray], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(arrays["intent_label"]), 512):
            stop = min(start + 512, len(arrays["intent_label"]))
            def t(key: str) -> torch.Tensor:
                return torch.from_numpy(np.asarray(arrays[key][start:stop], dtype=np.float32)).to(device)
            target = torch.cat([t("target_obs"), t("target_abs_obs")], dim=-1)
            out = model(target, t("neighbor_obs"), t("neighbor_mask"), t("neighbor_visible_mask"), t("scene_feat"))
            logits.append(out["intent_logit"].cpu().numpy().astype(np.float64))
            futures.append(out["future_pred"].cpu().numpy().astype(np.float32))
    return np.concatenate(logits), np.concatenate(futures)


def binary_test_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    pred = (p >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "sample_count": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int((1 - y).sum()),
    }


def write_predictions(run_dir: Path, arrays: dict[str, np.ndarray], logits: np.ndarray, future: np.ndarray, protocol_sha: str) -> dict[str, Any]:
    labels = np.asarray(arrays["intent_label"], dtype=np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    metrics = binary_test_metrics(labels, probabilities)
    predictions_path = run_dir / "test_predictions.npz"
    np.savez_compressed(
        predictions_path,
        scene_id=np.asarray(arrays["scene_id"]).astype(str),
        target_id=np.asarray(arrays["target_id"]).astype(str),
        obs_end_frame=np.asarray(arrays["obs_end_frame"], dtype=np.int64),
        intent_label=labels,
        intent_logit=logits,
        intent_probability=probabilities,
        predicted_class=(probabilities >= 0.5).astype(np.int64),
        future_pred=future,
        future_gt=np.asarray(arrays["future_gt"], dtype=np.float32),
        image_size=np.asarray(arrays["image_size"], dtype=np.float32),
    )
    report = {
        "protocol_sha256": protocol_sha,
        "status": "official_test_evaluated_after_protocol_freeze",
        "metrics": metrics,
        "test_predictions": predictions_path.name,
        "test_predictions_sha256": sha256_file(predictions_path),
        "trajectory_metrics_interpreted": False,
        "trajectory_outputs_saved_for_forward_sanity_only": True,
    }
    write_json(run_dir / "official_test_metrics.json", report)
    return report


def prediction_fields(archive: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    files = set(archive.files)
    label_key = "intent_label" if "intent_label" in files else "label" if "label" in files else None
    probability_key = "intent_probability" if "intent_probability" in files else "probability" if "probability" in files else None
    required = {"scene_id", "target_id", "obs_end_frame"}
    if label_key is None or probability_key is None or not required.issubset(files):
        raise RuntimeError(f"Prediction artifact lacks paired sample identifiers/labels/probabilities: {sorted(files)}")
    return (
        archive[label_key].astype(np.int64).reshape(-1),
        archive[probability_key].astype(np.float64).reshape(-1),
        archive["scene_id"].astype(str).reshape(-1),
        archive["target_id"].astype(str).reshape(-1),
        archive["obs_end_frame"].astype(np.int64).reshape(-1),
    )


def align_predictions_to_test(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        labels, probabilities, scene_ids, target_ids, frames = prediction_fields(archive)
    test_scenes = np.asarray(arrays["scene_id"]).astype(str)
    test_targets = np.asarray(arrays["target_id"]).astype(str)
    test_frames = np.asarray(arrays["obs_end_frame"], dtype=np.int64)
    def row_keys(scenes: np.ndarray, targets: np.ndarray, obs_frames: np.ndarray) -> list[tuple[str, str, int]]:
        return [(str(s), str(t), int(f)) for s, t, f in zip(scenes, targets, obs_frames)]
    expected = row_keys(test_scenes, test_targets, test_frames)
    observed = row_keys(scene_ids, target_ids, frames)
    if len(set(expected)) != len(expected) or len(set(observed)) != len(observed):
        raise RuntimeError(f"Duplicate sample identity in paired test predictions: {path}")
    index = {key: i for i, key in enumerate(observed)}
    if set(index) != set(expected):
        raise RuntimeError(f"Sample identity mismatch between frozen test set and {path}")
    order = np.asarray([index[key] for key in expected], dtype=np.int64)
    aligned_labels = labels[order]
    true_labels = np.asarray(arrays["intent_label"], dtype=np.int64)
    if not np.array_equal(aligned_labels, true_labels):
        raise RuntimeError(f"Test labels differ after identity join: {path}")
    return {
        "intent_label": aligned_labels,
        "intent_probability": probabilities[order],
        "scene_id": test_scenes,
        "target_id": test_targets,
        "obs_end_frame": test_frames,
    }


def store_reused_a0_predictions(arrays: dict[str, np.ndarray], protocol_sha: str) -> None:
    for seed in SEEDS:
        source = BASELINE_RESULTS / f"seed{seed}" / "test_predictions.npz"
        aligned = align_predictions_to_test(source, arrays)
        run_dir = RESULTS / "A0_full_clean_j0" / f"seed{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        probabilities = aligned["intent_probability"]
        metrics = binary_test_metrics(aligned["intent_label"], probabilities)
        output = run_dir / "test_predictions.npz"
        np.savez_compressed(output, **aligned)
        write_json(run_dir / "official_test_metrics.json", {
            "protocol_sha256": protocol_sha,
            "status": "reused_preexisting_clean_j0_official_test_predictions_after_freeze",
            "source_path": str(source.relative_to(ROOT)),
            "source_sha256": sha256_file(source),
            "metrics": metrics,
            "test_predictions": output.name,
            "test_predictions_sha256": sha256_file(output),
            "retrained": False,
        })


def phase_evaluate() -> None:
    protocol, protocol_sha = verify_frozen_protocol()
    access_path = TEST_ACCESS_PATH
    if access_path.exists():
        prior = json.loads(access_path.read_text(encoding="utf-8"))
        if prior.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("Existing test access marker belongs to a different protocol")
    else:
        write_json(access_path, {
            "protocol_sha256": protocol_sha,
            "status": "official_test_access_started_after_protocol_freeze",
            "test_archive": str((DATA / "test.npz").relative_to(ROOT)),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "formal_runs_completed": 15,
            "protocol_frozen_first": True,
            "test_access_count": 1,
        })
    test_path = DATA / "test.npz"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    with np.load(test_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    required = {"intent_label", "scene_id", "target_id", "obs_end_frame", "target_obs", "target_abs_obs", "neighbor_obs", "neighbor_mask", "neighbor_visible_mask", "scene_feat", "future_gt", "image_size"}
    if not required.issubset(arrays):
        raise RuntimeError(f"Official test archive is missing required fields: {sorted(required - set(arrays))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    access = json.loads(access_path.read_text(encoding="utf-8"))
    access["test_archive_sha256"] = sha256_file(test_path)
    access["sample_count"] = int(len(arrays["intent_label"]))
    access["scene_count"] = int(len(np.unique(arrays["scene_id"].astype(str))))
    access["per_seed_variant_status"] = access.get("per_seed_variant_status", {})
    write_json(access_path, access)
    for variant in VARIANTS:
        for seed in SEEDS:
            run_dir = RESULTS / variant / f"seed{seed}"
            prediction_path = run_dir / "test_predictions.npz"
            if prediction_path.is_file() and (run_dir / "official_test_metrics.json").is_file():
                existing = json.loads((run_dir / "official_test_metrics.json").read_text(encoding="utf-8"))
                if existing.get("protocol_sha256") == protocol_sha:
                    access["per_seed_variant_status"][f"{variant}/seed{seed}"] = "already_evaluated_under_same_frozen_protocol"
                    continue
            checkpoint = CHECKPOINTS / variant / f"seed{seed}.pt"
            model = load_model_for_variant(variant, checkpoint)
            logits, future = predict_test(model, arrays, device)
            result = write_predictions(run_dir, arrays, logits, future, protocol_sha)
            access["per_seed_variant_status"][f"{variant}/seed{seed}"] = result["status"]
            access["last_completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(access_path, access)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"evaluated {variant} seed={seed} test_auc={result['metrics']['roc_auc']:.5f}", flush=True)
    store_reused_a0_predictions(arrays, protocol_sha)
    access["status"] = "all_15_official_test_evaluations_complete"
    access["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(access_path, access)
    print(json.dumps({"phase": "evaluate", "status": access["status"], "test_samples": access["sample_count"], "protocol_sha256": protocol_sha}, ensure_ascii=False))


def metric_from_payload(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (float, int)) and math.isfinite(float(value)):
            return float(value)
    return None


def locate_m0_metrics(seed: int) -> tuple[dict[str, Any], str]:
    candidates = [
        ROOT / f"results/intent_target_only_15x15_seed{seed}/metrics.json",
        ROOT / "results/intent_target_only_15x15" / f"seed{seed}/metrics.json",
        ROOT / "results/intent_target_only_15x15" / f"seed{seed}/official_test_metrics.json",
    ]
    candidates.extend(
        path for path in (ROOT / "results").rglob("metrics.json")
        if "intent_target_only" in str(path) and (f"seed{seed}" in str(path) or path.parent.name.endswith(f"_{seed}"))
    )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        test_metrics = payload.get("test") if isinstance(payload.get("test"), dict) else payload
        auc = metric_from_payload(test_metrics, "intent_auc", "roc_auc", "auc", "AUC")
        if auc is not None:
            return test_metrics, str(path.relative_to(ROOT))
    # If only prediction artifacts were saved, compute the metrics from those.
    prediction_candidates = [
        ROOT / f"results/intent_target_only_15x15_seed{seed}/test_predictions.npz",
        ROOT / "results/intent_target_only_15x15" / f"seed{seed}/test_predictions.npz",
    ]
    for path in prediction_candidates:
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                labels, probabilities, *_ = prediction_fields(archive)
            return binary_test_metrics(labels, probabilities), str(path.relative_to(ROOT))
    raise FileNotFoundError(f"Could not find evaluated Target-only M0 test metrics for seed {seed}")


def normalize_metric_row(payload: dict[str, Any]) -> dict[str, float]:
    mapping = {
        "roc_auc": ("roc_auc", "intent_auc", "auc", "AUC"),
        "brier": ("brier", "intent_brier"),
        "f1": ("f1", "intent_f1", "f1_positive"),
        "balanced_accuracy": ("balanced_accuracy", "intent_balanced_accuracy", "bacc"),
        "accuracy": ("accuracy", "intent_accuracy"),
    }
    result = {}
    for metric, keys in mapping.items():
        value = metric_from_payload(payload, *keys)
        result[metric] = float("nan") if value is None else value
    return result


def phase_summarize() -> None:
    from scripts.reliability_gated_intent_utils import cluster_bootstrap_paired_delta

    protocol, protocol_sha = verify_frozen_protocol()
    access = json.loads(TEST_ACCESS_PATH.read_text(encoding="utf-8"))
    if access.get("status") != "all_15_official_test_evaluations_complete" or access.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("All formal arm test evaluations must finish before summarization")

    per_seed: dict[str, dict[str, Any]] = {}
    method_rows: dict[str, list[dict[str, float]]] = {"M0_target_only": []}
    with np.load(DATA / "test.npz", allow_pickle=False) as archive:
        test_index = {
            "scene_id": archive["scene_id"].astype(str).copy(),
            "target_id": archive["target_id"].astype(str).copy(),
            "obs_end_frame": archive["obs_end_frame"].astype(np.int64).copy(),
            "intent_label": archive["intent_label"].astype(np.int64).copy(),
        }
    for seed in SEEDS:
        m0_payload, m0_source = locate_m0_metrics(seed)
        method_rows["M0_target_only"].append(normalize_metric_row(m0_payload))
        per_seed[str(seed)] = {
            "M0_target_only": {**normalize_metric_row(m0_payload), "source": m0_source},
        }

        a0_path = RESULTS / "A0_full_clean_j0" / f"seed{seed}" / "test_predictions.npz"
        a0 = align_predictions_to_test(a0_path, test_index)
        a0_metrics = binary_test_metrics(a0["intent_label"], a0["intent_probability"])
        per_seed[str(seed)]["A0_full_clean_j0"] = a0_metrics
        method_rows.setdefault("A0_full_clean_j0", []).append(a0_metrics)
        for variant in VARIANTS:
            pred_path = RESULTS / variant / f"seed{seed}" / "test_predictions.npz"
            with np.load(pred_path, allow_pickle=False) as archive:
                labels, probabilities, scenes, targets, frames = prediction_fields(archive)
            if not np.array_equal(labels, a0["intent_label"]) or not np.array_equal(scenes, a0["scene_id"]) or not np.array_equal(targets, a0["target_id"]) or not np.array_equal(frames, a0["obs_end_frame"]):
                raise RuntimeError(f"Test sample pairing mismatch in {variant}/seed{seed}")
            metrics = binary_test_metrics(labels, probabilities)
            per_seed[str(seed)][variant] = metrics
            method_rows.setdefault(variant, []).append(metrics)

    # Per-seed paired scene-cluster bootstrap; never pool the random seeds.
    bootstrap_payload: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "bootstrap_unit": "scene_id video cluster",
        "paired": True,
        "requested_repetitions": BOOTSTRAP_REPETITIONS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "seeds_pooled": False,
        "comparison_sign": "Ablation - Full; positive AUC delta means removal helped, negative means the component helped",
        "comparisons": {},
    }
    effects: dict[str, Any] = {
        "protocol_sha256": protocol_sha,
        "interpretation": "exploratory component attribution; five primary comparisons, no multiplicity-adjusted confirmatory claims",
        "per_seed": {},
        "components": {},
    }
    test_archive = np.load(DATA / "test.npz", allow_pickle=False)
    test_labels = np.asarray(test_archive["intent_label"], dtype=np.int64)
    test_scenes = np.asarray(test_archive["scene_id"]).astype(str)
    del test_archive
    for variant_index, variant in enumerate(VARIANTS):
        seed_effects = []
        bootstrap_payload["comparisons"][variant] = {}
        effects["per_seed"][variant] = {}
        for seed in SEEDS:
            a0_path = RESULTS / "A0_full_clean_j0" / f"seed{seed}" / "test_predictions.npz"
            variant_path = RESULTS / variant / f"seed{seed}" / "test_predictions.npz"
            with np.load(a0_path, allow_pickle=False) as full, np.load(variant_path, allow_pickle=False) as ablated:
                p_full = full["intent_probability"].astype(np.float64)
                p_ablation = ablated["intent_probability"].astype(np.float64)
                if not np.array_equal(full["scene_id"].astype(str), test_scenes):
                    raise RuntimeError(f"A0 test order differs from frozen test archive at seed {seed}")
            bootstrap_seed = BOOTSTRAP_SEED + variant_index * 100 + seed
            boot = cluster_bootstrap_paired_delta(
                test_labels, p_full, p_ablation, test_scenes,
                repetitions=BOOTSTRAP_REPETITIONS, seed=bootstrap_seed,
            )
            bootstrap_payload["comparisons"][variant][str(seed)] = boot
            full_metrics = per_seed[str(seed)]["A0_full_clean_j0"]
            ablated_metrics = per_seed[str(seed)][variant]
            auc_candidate_minus_full = ablated_metrics["roc_auc"] - full_metrics["roc_auc"]
            contribution = -auc_candidate_minus_full
            brier_ablation_minus_full = ablated_metrics["brier"] - full_metrics["brier"]
            auc_delta_ci = boot["delta_roc_auc"]["ci_percentile_95"]
            contribution_ci = {"lower_95": -auc_delta_ci["upper_95"], "upper_95": -auc_delta_ci["lower_95"]}
            seed_item = {
                "full_auc": full_metrics["roc_auc"],
                "ablation_auc": ablated_metrics["roc_auc"],
                "delta_auc_ablation_minus_full": auc_candidate_minus_full,
                "component_contribution_full_minus_ablation": contribution,
                "bootstrap_ci_delta_ablation_minus_full": auc_delta_ci,
                "bootstrap_ci_contribution_full_minus_ablation": contribution_ci,
                "full_brier": full_metrics["brier"],
                "ablation_brier": ablated_metrics["brier"],
                "delta_brier_ablation_minus_full": brier_ablation_minus_full,
                "bootstrap_ci_delta_brier_ablation_minus_full": boot["delta_brier"]["ci_percentile_95"],
                "bootstrap_valid_repetitions": boot["valid_repetitions"],
            }
            effects["per_seed"][variant][str(seed)] = seed_item
            seed_effects.append(seed_item)
        mean_contribution = float(np.mean([item["component_contribution_full_minus_ablation"] for item in seed_effects]))
        direction_count = sum(item["component_contribution_full_minus_ablation"] > 0 for item in seed_effects)
        ci_support_count = sum(item["bootstrap_ci_delta_ablation_minus_full"]["upper_95"] < 0 for item in seed_effects)
        opposite_brier_count = sum(item["bootstrap_ci_delta_brier_ablation_minus_full"]["upper_95"] < 0 for item in seed_effects)
        if mean_contribution < 0 and direction_count <= 1:
            grade = "harmful"
        elif mean_contribution >= 0.03 and direction_count >= 2 and ci_support_count >= 2 and opposite_brier_count < 2:
            grade = "strong"
        elif mean_contribution >= 0.015 and direction_count >= 2:
            grade = "moderate"
        else:
            grade = "weak"
        effects["components"][variant] = {
            "mean_contribution_full_minus_ablation": mean_contribution,
            "mean_delta_auc_ablation_minus_full": -mean_contribution,
            "seed_direction_full_better_count": int(direction_count),
            "seed_bootstrap_ci_support_full_better_count": int(ci_support_count),
            "seed_bootstrap_ci_support_ablation_brier_better_count": int(opposite_brier_count),
            "mean_delta_brier_ablation_minus_full": float(np.mean([item["delta_brier_ablation_minus_full"] for item in seed_effects])),
            "contribution_grade": grade,
            "seedwise": seed_effects,
        }

    def summarize_rows(rows: list[dict[str, float]]) -> dict[str, Any]:
        result = {}
        for metric in ("roc_auc", "brier", "f1", "balanced_accuracy", "accuracy"):
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[metric] = {
                "mean": float(np.mean(finite)) if len(finite) else None,
                "sample_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0 if len(finite) else None,
                "n": int(len(finite)),
            }
        return result

    method_summary = {method: summarize_rows(rows) for method, rows in method_rows.items()}
    effects["method_summary"] = method_summary
    effects["auc_gap_j0_minus_m0"] = (
        method_summary["A0_full_clean_j0"]["roc_auc"]["mean"] - method_summary["M0_target_only"]["roc_auc"]["mean"]
    )
    effects["max_single_component_mean_contribution"] = max(
        item["mean_contribution_full_minus_ablation"] for item in effects["components"].values()
    )
    write_json(RESULTS / "cluster_bootstrap.json", bootstrap_payload)
    write_json(RESULTS / "component_effects.json", effects)
    summary = make_summary_markdown(effects, bootstrap_payload, per_seed, protocol)
    (RESULTS / "summary.md").write_text(summary, encoding="utf-8")
    print(json.dumps({"phase": "summarize", "summary": str((RESULTS / "summary.md").relative_to(ROOT)), "auc_gap_j0_minus_m0": effects["auc_gap_j0_minus_m0"], "largest_contribution": effects["max_single_component_mean_contribution"]}, ensure_ascii=False))


def fmt_metric(summary: dict[str, Any], metric: str) -> str:
    item = summary[metric]
    if item["mean"] is None:
        return "n/a"
    return f"{item['mean']:.4f} ± {item['sample_std']:.4f}"


def make_summary_markdown(effects: dict[str, Any], bootstrap: dict[str, Any], per_seed: dict[str, Any], protocol: dict[str, Any]) -> str:
    methods = ["M0_target_only", "A0_full_clean_j0", *VARIANTS.keys()]
    display = {
        "M0_target_only": "M0 Target-only",
        "A0_full_clean_j0": "A0 Full Clean J0",
        "A1_no_scene": "A1 No Scene",
        "A2_no_social": "A2 No Social",
        "A3_no_proposal_loss": "A3 No Proposal Loss",
        "A4_no_adaptive_gate": "A4 No Adaptive Gate",
        "A5_no_ambiguity": "A5 No Ambiguity",
    }
    lines = [
        "# Clean J0 单组件归因结果",
        "",
        f"协议 SHA256：`{protocol_sha_from_file()}`；15 个正式消融训练已在冻结前完成。所有 test 指标均在冻结后统一计算。",
        "",
        "## 主结果",
        "",
        "三 seed 分别训练/评估，表中为均值 ± sample SD；F1、BAcc、Accuracy 均使用固定阈值 0.5。ADE/FDE 不用于比较或模型选择。",
        "",
        "| Method | AUC | Brier | F1 | BAcc | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        row = effects["method_summary"][method]
        lines.append(
            f"| {display[method]} | {fmt_metric(effects['method_summary'][method], 'roc_auc')} | "
            f"{fmt_metric(effects['method_summary'][method], 'brier')} | {fmt_metric(effects['method_summary'][method], 'f1')} | "
            f"{fmt_metric(effects['method_summary'][method], 'balanced_accuracy')} | {fmt_metric(effects['method_summary'][method], 'accuracy')} |"
        )
    lines.extend([
        "",
        "## 组件贡献与逐 seed bootstrap",
        "",
        "组件贡献定义为 `Full − Ablation`，正值表示移除后 AUC 下降；bootstrap 文件同时保存原始 `Ablation − Full`。CI 是按 scene_id 配对 cluster bootstrap，2000 次/seed；五项比较仅作探索性归因，不做确认性显著性宣称。",
        "",
        "| Removed component | Full AUC | Ablated AUC | Mean contribution Full−Ablated | Seeds Full>Ablated | Grade |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for variant in VARIANTS:
        component = effects["components"][variant]
        full_auc = effects["method_summary"]["A0_full_clean_j0"]["roc_auc"]["mean"]
        ablated_auc = effects["method_summary"][variant]["roc_auc"]["mean"]
        lines.append(
            f"| {display[variant].replace('A1 ', '').replace('A2 ', '').replace('A3 ', '').replace('A4 ', '').replace('A5 ', '')} | "
            f"{full_auc:.4f} | {ablated_auc:.4f} | {component['mean_contribution_full_minus_ablation']:+.4f} | "
            f"{component['seed_direction_full_better_count']}/3 | {component['contribution_grade']} |"
        )
    lines.extend([
        "",
        "| Removed component | Seed | ΔAUC Ablation−Full | 95% CI for Ablation−Full | ΔBrier Ablation−Full | 95% CI for ΔBrier |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in VARIANTS:
        for seed in SEEDS:
            item = effects["per_seed"][variant][str(seed)]
            auc_ci = item["bootstrap_ci_delta_ablation_minus_full"]
            brier_ci = item["bootstrap_ci_delta_brier_ablation_minus_full"]
            lines.append(
                f"| {display[variant]} | {seed} | {item['delta_auc_ablation_minus_full']:+.4f} | "
                f"[{auc_ci['lower_95']:+.4f}, {auc_ci['upper_95']:+.4f}] | {item['delta_brier_ablation_minus_full']:+.4f} | "
                f"[{brier_ci['lower_95']:+.4f}, {brier_ci['upper_95']:+.4f}] |"
            )
    gap = effects["auc_gap_j0_minus_m0"]
    largest = effects["max_single_component_mean_contribution"]
    largest_variant = max(effects["components"], key=lambda key: effects["components"][key]["mean_contribution_full_minus_ablation"])
    no_scene = effects["components"]["A1_no_scene"]
    no_social = effects["components"]["A2_no_social"]
    no_proposal = effects["components"]["A3_no_proposal_loss"]
    no_gate = effects["components"]["A4_no_adaptive_gate"]
    no_ambiguity = effects["components"]["A5_no_ambiguity"]
    all_weak = all(item["contribution_grade"] in ("weak", "harmful") for item in effects["components"].values())
    next_direction = (
        "scene refinement" if largest_variant == "A1_no_scene" and no_scene["contribution_grade"] in ("strong", "moderate")
        else "gate decomposition" if largest_variant == "A4_no_adaptive_gate" and no_gate["contribution_grade"] in ("strong", "moderate")
        else "proposal decomposition" if largest_variant == "A3_no_proposal_loss" and no_proposal["contribution_grade"] in ("strong", "moderate")
        else "interaction study" if all_weak
        else "先沿最大且方向较稳定的单组件继续；若后续重复仍不稳定，再做预注册 interaction study"
    )
    lines.extend([
        "",
        "## 核心问题回答",
        "",
        f"1. 移除后 AUC 下降最大的是 **{display[largest_variant]}**，平均贡献 `Full−Ablation={largest:+.4f}`（等级：{effects['components'][largest_variant]['contribution_grade']}）。",
        f"2. Scene 稳定贡献：{no_scene['contribution_grade']}；3 seed 中 Full AUC 更高 {no_scene['seed_direction_full_better_count']}/3，mean contribution `{no_scene['mean_contribution_full_minus_ablation']:+.4f}`。",
        f"3. Social 稳定贡献：{no_social['contribution_grade']}；3 seed 中 Full AUC 更高 {no_social['seed_direction_full_better_count']}/3，mean contribution `{no_social['mean_contribution_full_minus_ablation']:+.4f}`。",
        f"4. Proposal auxiliary loss：{no_proposal['contribution_grade']}；移除 supervision 后平均贡献 `{no_proposal['mean_contribution_full_minus_ablation']:+.4f}`。proposal forward/prior entropy/gate 均保留，因此此项只归因于 prior BCE supervision。",
        f"5. Adaptive gate 相对固定中性 `g=0.5`：{no_gate['contribution_grade']}；平均贡献 `{no_gate['mean_contribution_full_minus_ablation']:+.4f}`。",
        f"6. Ambiguity regularization：{no_ambiguity['contribution_grade']}；移除后平均贡献 `{no_ambiguity['mean_contribution_full_minus_ablation']:+.4f}`。",
        "7. 保留决策：保留达到 moderate/strong 且跨 seed 方向一致的组件；其余暂不宣称为核心创新。",
        "8. 简化决策：weak 或 harmful 的 social/proposal/gate/ambiguity 项可列为简化候选，但本轮不实际删除组合模块。",
        f"9. `M0→J0` AUC gap 为 `{gap:.4f}`；最大单项平均贡献 `{largest:+.4f}`。单一组件{('可能解释大部分差距' if largest >= 0.07 and effects['components'][largest_variant]['contribution_grade'] == 'strong' else '不足以解释全部差距')}。",
        f"10. Interaction：{'所有单项效应均弱，结果与 component interaction / architecture synergy 相容，但本轮没有直接检验，不能据此断言。' if all_weak else '存在单项贡献信号；单项消融不能识别组件间 interaction，本轮没有直接检验。'}",
        f"11. 下一步建议：**{next_direction}**。不自动启动下一阶段。",
        "",
        "## 协议与解释边界",
        "",
        f"- Frozen protocol SHA256: `{protocol_sha_from_file()}`。",
        "- A0 使用已有 Clean J0 checkpoint/test predictions；没有重新训练 A0。",
        "- A1–A5 每 seed 使用匹配的 Clean J0 initial state；正式训练 sampler fingerprints 与 A0 同 seed 完全一致。",
        "- 所有 ablation 使用 15 epochs；checkpoint 和 scheduler 只依 raw validation AUC/Brier；trajectory ADE/FDE 仅保存 forward sanity，不解释、不用于选择。",
        "- Bootstrap 按 seed 单独计算；不 pooling seeds、不对五项比较作多重比较校正。CI 跨零仅表示该 cluster-bootstrap 证据不确定，不是组件无效的证明。",
        "- 详细数字见 `component_effects.json`、`cluster_bootstrap.json`；依赖与干预定义见 `component_dependency_map.md`、`component_definitions.json`。",
    ])
    return "\n".join(lines) + "\n"


def protocol_sha_from_file() -> str:
    return PROTOCOL_SHA_PATH.read_text(encoding="utf-8").split()[0] if PROTOCOL_SHA_PATH.is_file() else "not-yet-frozen"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "train", "freeze", "evaluate", "summarize"))
    args = parser.parse_args()
    {
        "prepare": phase_prepare,
        "smoke": phase_smoke,
        "train": phase_train,
        "freeze": phase_freeze,
        "evaluate": phase_evaluate,
        "summarize": phase_summarize,
    }[args.phase]()


if __name__ == "__main__":
    main()
