#!/usr/bin/env python3
"""Build the controlled 15x15 scene-ablation summaries and protocol audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_scene_ablation_intent import (
    SequenceWithImageSize,
    collect,
    load_backbone,
)
from src.models.scene_ablation_intent import SceneAblationIntentModel


SEEDS = (42, 123, 2024)
OUT_ROOT = PROJECT_ROOT / "results/scene_ablation_15x15"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stat(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def fmt(value: dict[str, float], digits: int = 3) -> str:
    return f"{value['mean']:.{digits}f} ± {value['sample_std']:.{digits}f}"


def selected_val(metrics: dict) -> dict:
    if "val" in metrics:
        return metrics["val"]
    epoch = int(metrics["best_epoch"])
    record = next(item for item in metrics["history"] if int(item["epoch"]) == epoch)
    return record["val"]


def trajectory_summaries() -> tuple[dict, str]:
    per_seed = {}
    fields = (
        "trajectory_ade_pixel",
        "trajectory_fde_pixel",
        "trajectory_ade_normalized",
        "trajectory_fde_normalized",
    )
    for seed in SEEDS:
        real = read_json(PROJECT_ROOT / f"results/trajectory_transformer_scene_15x15_seed{seed}/metrics.json")
        zero = read_json(PROJECT_ROOT / f"results/trajectory_transformer_zero_scene_15x15_seed{seed}/metrics.json")
        if zero.get("scene_mode") != "zero":
            raise ValueError(f"Seed {seed} zero-scene result has the wrong scene mode")
        real_checkpoint = torch.load(
            PROJECT_ROOT / f"checkpoints/trajectory_transformer_scene_15x15_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
        zero_checkpoint = torch.load(
            PROJECT_ROOT / f"checkpoints/trajectory_transformer_zero_scene_15x15_seed{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
        same_parameter_shapes = real_checkpoint.keys() == zero_checkpoint.keys() and all(
            real_checkpoint[key].shape == zero_checkpoint[key].shape for key in real_checkpoint
        )
        real_parameter_count = sum(t.numel() for t in real_checkpoint.values())
        zero_parameter_count = sum(t.numel() for t in zero_checkpoint.values())
        if not same_parameter_shapes or real_parameter_count != zero_parameter_count:
            raise ValueError(f"Seed {seed} real/zero trajectory architectures are not parameter-matched")
        real_val = selected_val(real)
        zero_val = selected_val(zero)
        per_seed[str(seed)] = {
            "real_scene": {
                "validation": {key: real_val[key] for key in fields},
                "test": {key: real["test"][key] for key in fields},
                "checkpoint": f"checkpoints/trajectory_transformer_scene_15x15_seed{seed}.pt",
                "parameter_count": real_parameter_count,
            },
            "zero_scene": {
                "validation": {key: zero_val[key] for key in fields},
                "test": {key: zero["test"][key] for key in fields},
                "checkpoint": f"checkpoints/trajectory_transformer_zero_scene_15x15_seed{seed}.pt",
                "scene_mode": zero["scene_mode"],
                "parameter_count": zero_parameter_count,
                "same_parameter_shapes_as_real": same_parameter_shapes,
            },
            "test_paired_delta_zero_minus_real": {
                key: zero["test"][key] - real["test"][key] for key in fields
            },
        }
    aggregate = {}
    for key in fields:
        real_values = [per_seed[str(seed)]["real_scene"]["test"][key] for seed in SEEDS]
        zero_values = [per_seed[str(seed)]["zero_scene"]["test"][key] for seed in SEEDS]
        deltas = [per_seed[str(seed)]["test_paired_delta_zero_minus_real"][key] for seed in SEEDS]
        aggregate[key] = {
            "real_scene": stat(real_values),
            "zero_scene": stat(zero_values),
            "paired_zero_minus_real": stat(deltas),
        }
    ade_wins = sum(
        per_seed[str(seed)]["test_paired_delta_zero_minus_real"]["trajectory_ade_pixel"] > 0
        for seed in SEEDS
    )
    fde_wins = sum(
        per_seed[str(seed)]["test_paired_delta_zero_minus_real"]["trajectory_fde_pixel"] > 0
        for seed in SEEDS
    )
    zero_ade_wins = len(SEEDS) - ade_wins
    zero_fde_wins = len(SEEDS) - fde_wins
    result = {
        "protocol": {
            "dataset": "JAAD clean 15x15 processed sequences; existing official split",
            "real_scene_results_reused": True,
            "zero_scene_retrained": True,
            "checkpoint_selection": "lowest validation pixel ADE",
            "seeds": list(SEEDS),
            "paired_delta_definition": "zero-scene minus real-scene; positive favors real-scene",
        },
        "per_seed": per_seed,
        "aggregate_mean_sample_std": aggregate,
        "real_scene_better_seed_count": {
            "ade_pixel": f"{ade_wins}/{len(SEEDS)}",
            "fde_pixel": f"{fde_wins}/{len(SEEDS)}",
        },
        "conclusion": (
            "Static video-level scene input improves both ADE and FDE in all three paired seeds under this protocol."
            if ade_wins == len(SEEDS) and fde_wins == len(SEEDS)
            else f"No stable trajectory benefit from real scene input is supported: real-scene ADE is lower in {ade_wins}/3 seeds and FDE is lower in {fde_wins}/3; zero-scene has lower ADE in {zero_ade_wins}/3."
        ),
    }
    rows = []
    for seed in SEEDS:
        r = per_seed[str(seed)]
        rows.append(
            f"| {seed} | {r['real_scene']['test']['trajectory_ade_pixel']:.3f} | "
            f"{r['zero_scene']['test']['trajectory_ade_pixel']:.3f} | "
            f"{r['test_paired_delta_zero_minus_real']['trajectory_ade_pixel']:+.3f} | "
            f"{r['real_scene']['test']['trajectory_fde_pixel']:.3f} | "
            f"{r['zero_scene']['test']['trajectory_fde_pixel']:.3f} | "
            f"{r['test_paired_delta_zero_minus_real']['trajectory_fde_pixel']:+.3f} |"
        )
    md = [
        "# 15×15 trajectory scene ablation",
        "",
        "Pixel ADE/FDE are primary; normalized ADE/FDE and validation metrics are retained in the JSON. "
        "Paired Δ = zero-scene − real-scene, so a positive value favors real-scene input.",
        "",
        "| Seed | Real ADE px | Zero ADE px | ΔADE | Real FDE px | Zero FDE px | ΔFDE |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"- Real ADE: {fmt(aggregate['trajectory_ade_pixel']['real_scene'])} px; "
        f"zero ADE: {fmt(aggregate['trajectory_ade_pixel']['zero_scene'])} px; "
        f"paired ΔADE: {fmt(aggregate['trajectory_ade_pixel']['paired_zero_minus_real'])} px.",
        f"- Real FDE: {fmt(aggregate['trajectory_fde_pixel']['real_scene'])} px; "
        f"zero FDE: {fmt(aggregate['trajectory_fde_pixel']['zero_scene'])} px; "
        f"paired ΔFDE: {fmt(aggregate['trajectory_fde_pixel']['paired_zero_minus_real'])} px.",
        f"- Paired seed outcome: real-scene lower ADE in {ade_wins}/3 and lower FDE in {fde_wins}/3; zero-scene lower ADE in {zero_ade_wins}/3 and lower FDE in {zero_fde_wins}/3.",
        f"- Conclusion: {result['conclusion']}",
        "",
        "The separate same-checkpoint real-vs-zero inference diagnostic is not the retrained no-scene baseline; see `trajectory_zero_scene_inference_diagnostic.json`.",
    ]
    return result, "\n".join(md) + "\n"


def intent_summaries() -> tuple[dict, dict, dict, str]:
    per_seed = {}
    run_metrics = {}
    for seed in SEEDS:
        modes = {}
        for mode in ("target_only", "target_scene"):
            path = PROJECT_ROOT / f"results/intent_{mode}_15x15_seed{seed}/metrics.json"
            data = read_json(path)
            if data["scene_mode"] != mode or int(data["seed"]) != seed:
                raise ValueError(f"Unexpected intent result identity: {path}")
            modes[mode] = data
        run_metrics[str(seed)] = modes
        only_test = modes["target_only"]["test"]
        scene_test = modes["target_scene"]["test"]
        per_seed[str(seed)] = {
            "target_only": {key: only_test[key] for key in ("auc", "brier", "ece_10", "balanced_accuracy", "f1", "selected_threshold")},
            "target_scene": {key: scene_test[key] for key in ("auc", "brier", "ece_10", "balanced_accuracy", "f1", "selected_threshold")},
            "threshold_0_5": {
                mode: {
                    "balanced_accuracy": modes[mode]["test"]["balanced_accuracy_threshold_0_5"],
                    "f1": modes[mode]["test"]["f1_threshold_0_5"],
                }
                for mode in ("target_only", "target_scene")
            },
            "paired_delta_target_scene_minus_target_only": {
                key: scene_test[key] - only_test[key]
                for key in ("auc", "brier", "ece_10", "balanced_accuracy", "f1")
            },
            "calibration": {
                mode: {
                    "selected": modes[mode]["selected_calibration"],
                    "candidates": modes[mode]["validation_calibration_candidates"],
                }
                for mode in ("target_only", "target_scene")
            },
        }
    aggregate = {}
    fields = ("auc", "brier", "ece_10", "balanced_accuracy", "f1")
    for mode in ("target_only", "target_scene"):
        aggregate[mode] = {
            metric: stat([per_seed[str(seed)][mode][metric] for seed in SEEDS]) for metric in fields
        }
    aggregate["paired_delta_target_scene_minus_target_only"] = {
        metric: stat([
            per_seed[str(seed)]["paired_delta_target_scene_minus_target_only"][metric]
            for seed in SEEDS
        ])
        for metric in fields
    }
    auc_better = sum(per_seed[str(seed)]["paired_delta_target_scene_minus_target_only"]["auc"] > 0 for seed in SEEDS)
    brier_better = sum(per_seed[str(seed)]["paired_delta_target_scene_minus_target_only"]["brier"] < 0 for seed in SEEDS)
    result = {
        "protocol": {
            "dataset": "JAAD clean 15x15; existing official split",
            "backbone": "same frozen real-scene SceneTrajectoryTransformer checkpoint per paired seed",
            "classifier_difference": "only scene_context is real for target_scene and zero for target_only",
            "ambiguous_labels_used": False,
            "weighted_sampler": "inverse-frequency WeightedRandomSampler, replacement=True",
            "seeds": list(SEEDS),
        },
        "per_seed": per_seed,
        "aggregate_mean_sample_std": aggregate,
        "scene_auc_better_seed_count": f"{auc_better}/{len(SEEDS)}",
        "scene_brier_better_seed_count": f"{brier_better}/{len(SEEDS)}",
        "mean_paired_delta_auc": aggregate["paired_delta_target_scene_minus_target_only"]["auc"],
        "mean_paired_delta_brier": aggregate["paired_delta_target_scene_minus_target_only"]["brier"],
        "conclusion": (
            "Video-level static scene context raises AUC in all three seeds under this protocol."
            if auc_better == len(SEEDS)
            else f"No stable intent benefit is supported: target+scene AUC is higher in {auc_better}/3 seeds and Brier improves in {brier_better}/3; the mean AUC difference is small relative to seed variability."
        ),
    }
    permutation = {
        str(seed): run_metrics[str(seed)]["target_scene"].get("scene_permutation_diagnostic")
        for seed in SEEDS
    }
    rows = []
    for seed in SEEDS:
        r = per_seed[str(seed)]
        d = r["paired_delta_target_scene_minus_target_only"]
        rows.append(
            f"| {seed} | {r['target_only']['auc']:.4f} | {r['target_scene']['auc']:.4f} | {d['auc']:+.4f} | "
            f"{r['target_only']['brier']:.4f} | {r['target_scene']['brier']:.4f} | {d['brier']:+.4f} | "
            f"{r['target_only']['ece_10']:.4f} | {r['target_scene']['ece_10']:.4f} | "
            f"{r['target_only']['balanced_accuracy']:.4f} | {r['target_scene']['balanced_accuracy']:.4f} | "
            f"{r['target_only']['f1']:.4f} | {r['target_scene']['f1']:.4f} |"
        )
    aggregate_delta = aggregate["paired_delta_target_scene_minus_target_only"]
    calibration_lines = []
    calibration_comparison_lines = []
    for seed in SEEDS:
        for mode in ("target_only", "target_scene"):
            item = run_metrics[str(seed)][mode]
            calibration_lines.append(
                f"| {seed} | {mode} | {item['selected_calibration']['method']} | "
                f"{item['selected_threshold']:.4f} | {item['threshold_selection']['validation_balanced_accuracy']:.4f} |"
            )
            candidates = item["validation_calibration_candidates"]
            calibration_comparison_lines.append(
                f"| {seed} | {mode} | {candidates['temperature']['validation']['brier']:.4f} | "
                f"{candidates['temperature']['validation']['ece_10']:.4f} | "
                f"{candidates['temperature_bias']['validation']['brier']:.4f} | "
                f"{candidates['temperature_bias']['validation']['ece_10']:.4f} | "
                f"{item['selected_calibration']['method']} |"
            )
    md = [
        "# 15×15 intent scene ablation",
        "",
        "Primary comparison uses validation-selected calibration and a validation-selected balanced-accuracy threshold. "
        "Test threshold 0.5 BAcc/F1 are retained per run in JSON for historical comparability.",
        "",
        "| Seed | Target-only AUC | Target+scene AUC | ΔAUC | Target-only Brier | Target+scene Brier | ΔBrier | Target-only ECE | Target+scene ECE | Target-only BAcc | Target+scene BAcc | Target-only F1 | Target+scene F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        f"- Target-only AUC: {fmt(aggregate['target_only']['auc'], 4)}; target+scene AUC: {fmt(aggregate['target_scene']['auc'], 4)}.",
        f"- Mean paired ΔAUC: {fmt(aggregate_delta['auc'], 4)}; scene AUC higher in {auc_better}/3 seeds.",
        f"- Mean paired ΔBrier: {fmt(aggregate_delta['brier'], 4)}; lower Brier is better; Brier improved in {brier_better}/3 seeds.",
        f"- Conclusion: {result['conclusion']}",
        "",
        "## Mean ± sample standard deviation",
        "",
        "| Metric | Target-only | Target+scene | Paired Δ (scene−only) |",
        "|---|---:|---:|---:|",
        *[
            f"| {metric} | {fmt(aggregate['target_only'][metric], 4)} | {fmt(aggregate['target_scene'][metric], 4)} | {fmt(aggregate_delta[metric], 4)} |"
            for metric in fields
        ],
        "",
        "## Validation-only calibration and thresholds",
        "",
        "| Seed | Readout | Selected calibration | Threshold | Validation BAcc |",
        "|---:|---|---|---:|---:|",
        *calibration_lines,
        "",
        "Temperature-only versus temperature+bias calibration; selection uses validation Brier (validation ECE breaks exact ties).",
        "",
        "| Seed | Readout | Temp Brier | Temp ECE | Temp+bias Brier | Temp+bias ECE | Selected |",
        "|---:|---|---:|---:|---:|---:|---|",
        *calibration_comparison_lines,
        "",
        "## Test threshold 0.5 compatibility metrics",
        "",
        "| Seed | Readout | BAcc @ 0.5 | F1 @ 0.5 |",
        "|---:|---|---:|---:|",
        *[
            f"| {seed} | {mode} | {per_seed[str(seed)]['threshold_0_5'][mode]['balanced_accuracy']:.4f} | {per_seed[str(seed)]['threshold_0_5'][mode]['f1']:.4f} |"
            for seed in SEEDS for mode in ("target_only", "target_scene")
        ],
        "",
        "## Scene permutation diagnostic",
        "",
        "See `scene_permutation_diagnostic.json` for real-, shuffled-, and zero-readout-scene AUC per seed. The permutation is test-only diagnostic (seed 9124), not used for training or selection.",
    ]
    return result, permutation, run_metrics, "\n".join(md) + "\n"


def actual_future_prediction_audit(run_metrics: dict, device: torch.device) -> dict:
    test_set = SequenceWithImageSize(PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15/test.npz")
    loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)
    audit = {}
    for seed in SEEDS:
        outputs = {}
        for mode in ("target_only", "target_scene"):
            data = run_metrics[str(seed)][mode]
            trajectory_checkpoint = PROJECT_ROOT / data["trajectory_checkpoint"]
            backbone = load_backbone(trajectory_checkpoint, test_set[0]).to(device)
            model = SceneAblationIntentModel(backbone, scene_mode=mode).to(device)
            payload = torch.load(data["intent_checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(payload["model"], strict=True)
            model.eval()
            outputs[mode] = collect(model, loader, device)["future_pred"]
        difference = np.abs(outputs["target_only"] - outputs["target_scene"])
        audit[str(seed)] = {
            "same_frozen_backbone_state_sha256": run_metrics[str(seed)]["target_only"]["trajectory_backbone_state_sha256"]
            == run_metrics[str(seed)]["target_scene"]["trajectory_backbone_state_sha256"],
            "future_pred_bitwise_equal": bool(np.array_equal(outputs["target_only"], outputs["target_scene"])),
            "future_pred_max_abs_difference": float(difference.max(initial=0.0)),
        }
    return audit


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    trajectory, trajectory_md = trajectory_summaries()
    intent, permutation, run_metrics, intent_md = intent_summaries()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    future_audit = actual_future_prediction_audit(run_metrics, device)

    protocol = {}
    for seed in SEEDS:
        only = run_metrics[str(seed)]["target_only"]
        scene = run_metrics[str(seed)]["target_scene"]
        optimizer_fields = ("name", "learning_rate", "weight_decay", "batch_size", "epochs_max", "patience", "checkpoint_selection")
        protocol[str(seed)] = {
            "same_trajectory_checkpoint_sha256": only["trajectory_checkpoint_sha256"] == scene["trajectory_checkpoint_sha256"],
            "same_trajectory_backbone_state_sha256": only["trajectory_backbone_state_sha256"] == scene["trajectory_backbone_state_sha256"],
            "same_dataset_sha256": only["dataset_sha256"] == scene["dataset_sha256"],
            "same_class_counts": only["class_counts"] == scene["class_counts"],
            "same_sampling": {k: only["sampling"][k] for k in only["sampling"] if k != "seed"}
            == {k: scene["sampling"][k] for k in scene["sampling"] if k != "seed"},
            "same_optimizer_hyperparameters": all(only["optimizer"][k] == scene["optimizer"][k] for k in optimizer_fields),
            "same_classifier_architecture_and_parameter_count": only["architecture"]["classifier_input_dim"] == scene["architecture"]["classifier_input_dim"]
            and only["architecture"]["intent_head_parameter_count"] == scene["architecture"]["intent_head_parameter_count"],
            "no_ambiguous_or_extra_loss": not any((only["ambiguous_supervision_used"], scene["ambiguous_supervision_used"], only["trajectory_loss_used"], scene["trajectory_loss_used"], only["social_loss_used"], scene["social_loss_used"])),
            "same_future_prediction": future_audit[str(seed)],
        }
    legacy_checkpoint = torch.load(
        PROJECT_ROOT / "checkpoints/fixed_base_intent_seed123.pt",
        map_location="cpu",
        weights_only=False,
    )
    legacy_args = legacy_checkpoint.get("args", {})
    legacy_metrics = read_json(PROJECT_ROOT / "results/fixed_base_intent_seed123/metrics.json")
    expected_optimizer = run_metrics["123"]["target_only"]["optimizer"]
    legacy_metadata_checks = {
        "seed": int(legacy_args.get("seed", -1)) == 123,
        "epochs": int(legacy_args.get("epochs", -1)) == expected_optimizer["epochs_max"],
        "batch_size": int(legacy_args.get("batch_size", -1)) == expected_optimizer["batch_size"],
        "learning_rate": float(legacy_args.get("learning_rate", -1)) == expected_optimizer["learning_rate"],
        "weight_decay": float(legacy_args.get("weight_decay", -1)) == expected_optimizer["weight_decay"],
        "patience": int(legacy_args.get("patience", -1)) == expected_optimizer["patience"],
        "class_counts": legacy_metrics["class_counts"] == run_metrics["123"]["target_only"]["class_counts"],
        "data_root": Path(str(legacy_args.get("data_root", ""))).resolve()
        == (PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15").resolve(),
        "trajectory_checkpoint": Path(str(legacy_checkpoint["trajectory_checkpoint"])).name
        == Path(run_metrics["123"]["target_only"]["trajectory_checkpoint"]).name,
    }
    trainer_source = (PROJECT_ROOT / "scripts/train_fixed_base_intent.py").read_text(encoding="utf-8")
    legacy_sampler_code_check = (
        "WeightedRandomSampler" in trainer_source
        and "replacement=True" in trainer_source
        and "1.0 / class_counts[0]" in trainer_source
    )
    protocol_audit = {
        "paired_seed_checks": protocol,
        "seed123_historical_checkpoint_compatibility": {
            "checkpoint_optimizer_metadata_checks": legacy_metadata_checks,
            "all_checkpoint_metadata_checks_pass": all(legacy_metadata_checks.values()),
            "trainer_source_uses_inverse_frequency_weighted_sampler": legacy_sampler_code_check,
            "historical_result_best_validation_auc": legacy_metrics["best_validation_auc_before_calibration"],
            "historical_result_ambiguous_supervision_used": legacy_metrics["ambiguous_supervision_used"],
        },
        "all_protocol_checks_pass": all(
            value
            for seed_item in protocol.values()
            for key, value in seed_item.items()
            if key != "same_future_prediction"
        ) and all(item["same_frozen_backbone_state_sha256"] and item["future_pred_bitwise_equal"] for item in future_audit.values())
        and all(legacy_metadata_checks.values()) and legacy_sampler_code_check,
        "note": "Seed 123 target+scene reuses the historical pilot after checking its saved seed/optimizer metadata, matching clean class counts and trajectory checkpoint, and confirming that its recorded trainer implements inverse-frequency WeightedRandomSampler.",
    }

    (OUT_ROOT / "trajectory_summary.json").write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "trajectory_summary.md").write_text(trajectory_md, encoding="utf-8")
    (OUT_ROOT / "intent_summary.json").write_text(json.dumps(intent, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "intent_summary.md").write_text(intent_md, encoding="utf-8")
    (OUT_ROOT / "scene_permutation_diagnostic.json").write_text(json.dumps(permutation, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "protocol_audit.json").write_text(json.dumps(protocol_audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "trajectory_conclusion": trajectory["conclusion"],
        "intent_conclusion": intent["conclusion"],
        "protocol_audit_pass": protocol_audit["all_protocol_checks_pass"],
        "trajectory_summary": str(OUT_ROOT / "trajectory_summary.md"),
        "intent_summary": str(OUT_ROOT / "intent_summary.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
