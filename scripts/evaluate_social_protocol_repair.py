#!/usr/bin/env python3
"""Evaluate natural-sampling and visibility-aware social checkpoints on JAAD test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_fixed_base_intent import SequenceWithImageSize, ece_10
from train_fixed_base_social_residual import load_base_model
from src.models.fixed_base_social_residual import FixedBaseSocialResidual
from src.models.fixed_base_social_residual_visible import FixedBaseSocialResidualVisible


def bce_per_sample(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))


def intent_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
    predicted = (probabilities >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10": ece_10(labels, probabilities),
    }


def paired_summary(base_logits: np.ndarray, final_logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    improvement = bce_per_sample(base_logits, labels) - bce_per_sample(final_logits, labels)
    return {
        "mean": float(improvement.mean()),
        "median": float(np.median(improvement)),
        "helped_ratio": float((improvement > 0).mean()),
        "hurt_ratio": float((improvement < 0).mean()),
        "unchanged_ratio": float((improvement == 0).mean()),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "abs_mean": float(np.abs(values).mean()),
    }


def load_model(
    run_metrics_path: Path,
    base_checkpoint: Path,
    gate_mode: str,
    dataset: SequenceWithImageSize,
    device: torch.device,
) -> tuple[torch.nn.Module, bool, dict[str, Any]]:
    run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
    checkpoint = Path(run_metrics["checkpoint"])
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    visible = bool(run_metrics.get("visibility_aware", args.get("visibility_aware", False)))
    base = load_base_model(base_checkpoint, dataset, device)
    model_class = FixedBaseSocialResidualVisible if visible else FixedBaseSocialResidual
    model = model_class(
        base,
        gate_mode=gate_mode,
        social_hidden_dim=int(args.get("social_hidden_dim", base.d_model)),
        dropout=float(args.get("dropout", 0.1)),
        social_scale=float(args.get("social_scale", 1.0)),
    )
    model.load_state_dict(payload["model"], strict=True)
    if not model.all_base_parameters_frozen:
        raise RuntimeError(f"Base parameters are trainable in {run_metrics_path}")
    model.to(device).eval()
    return model, visible, run_metrics


@torch.no_grad()
def collect(
    model: torch.nn.Module,
    visible_aware: bool,
    dataset: SequenceWithImageSize,
    device: torch.device,
    batch_size: int,
    neighbor_permutation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    keys = (
        "labels", "base_logits", "base_probabilities", "final_logits", "final_probabilities",
        "delta_logits", "gates", "neighbor_counts", "future_pred", "future_gt", "image_size",
        "visibility_ratio",
    )
    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    offset = 0
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        if neighbor_permutation is None:
            neighbors = batch["neighbor_obs"]
            neighbor_mask = batch["neighbor_mask"]
            visible_mask = batch["neighbor_visible_mask"]
        else:
            indices = torch.as_tensor(
                neighbor_permutation[offset : offset + len(target)], dtype=torch.long
            )
            # Permute the complete neighbor record while preserving each target/scene/label.
            neighbors = dataset.dataset.neighbor_obs[indices]
            neighbor_mask = dataset.dataset.neighbor_mask[indices]
            visible_mask = dataset.dataset.neighbor_visible_mask[indices]
        inputs = (
            target,
            batch["scene_feat"].to(device),
            neighbors.to(device),
            neighbor_mask.to(device),
        )
        if visible_aware:
            output = model(*inputs, visible_mask.to(device))
        else:
            output = model(*inputs)

        valid = (batch["neighbor_mask"].numpy() > 0).astype(np.float32)
        actual_visibility = batch["neighbor_visible_mask"].numpy()
        neighbor_count = valid.sum(axis=1)
        visible_denominator = neighbor_count * actual_visibility.shape[-1]
        visibility_ratio = (actual_visibility * valid[:, :, None]).sum(axis=(1, 2))
        visibility_ratio = np.divide(
            visibility_ratio,
            visible_denominator,
            out=np.zeros_like(visibility_ratio, dtype=np.float32),
            where=visible_denominator > 0,
        )
        batches = {
            "labels": batch["intent_label"].numpy(),
            "base_logits": output["base_logit"].cpu().numpy(),
            "base_probabilities": output["base_probability"].cpu().numpy(),
            "final_logits": output["final_logit"].cpu().numpy(),
            "final_probabilities": output["final_probability"].cpu().numpy(),
            "delta_logits": output["delta_logit"].cpu().numpy(),
            "gates": output["gate"].cpu().numpy(),
            "neighbor_counts": output["neighbor_count"].cpu().numpy(),
            "future_pred": output["future_pred"].cpu().numpy(),
            "future_gt": batch["future_gt"].numpy(),
            "image_size": batch["image_size"].numpy(),
            "visibility_ratio": visibility_ratio,
        }
        for key, value in batches.items():
            collected[key].append(value)
        offset += len(target)
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def model_summary(values: dict[str, np.ndarray]) -> dict[str, Any]:
    metrics = intent_metrics(values["labels"], values["final_probabilities"])
    error = values["future_pred"] - values["future_gt"]
    pixel_error = np.linalg.norm(error * values["image_size"][:, None, :], axis=-1)
    return {
        **metrics,
        "paired_bce_improvement": paired_summary(
            values["base_logits"], values["final_logits"], values["labels"]
        ),
        "delta_logit": distribution(values["delta_logits"]),
        "gate": distribution(values["gates"]),
        "no_neighbor_count": int((values["neighbor_counts"] == 0).sum()),
        "no_neighbor_max_abs_logit_change": float(
            np.abs(values["final_logits"][values["neighbor_counts"] == 0]
                   - values["base_logits"][values["neighbor_counts"] == 0]).max()
        ) if np.any(values["neighbor_counts"] == 0) else None,
        "ade_pixel": float(pixel_error.mean()),
        "fde_pixel": float(pixel_error[:, -1].mean()),
    }


def subset_summary(values: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"sample_count": 0, "metrics": None}
    subset = {key: value[mask] for key, value in values.items()}
    return {"sample_count": int(mask.sum()), **model_summary(subset)}


def stratified_by_neighbor_count(outputs: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    base = next(iter(outputs.values()))
    counts = base["neighbor_counts"]
    groups = {
        "0": counts == 0,
        "1": counts == 1,
        "2-3": (counts >= 2) & (counts <= 3),
        ">=4": counts >= 4,
    }
    result = {}
    for name, mask in groups.items():
        result[name] = {
            "sample_count": int(mask.sum()),
            "models": {mode: subset_summary(values, mask) for mode, values in outputs.items()},
        }
    return result


def stratified_by_visibility(
    outputs: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    base = next(iter(outputs.values()))
    eligible = base["neighbor_counts"] > 0
    ratio = base["visibility_ratio"]
    values = ratio[eligible]
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    collapsed = bool(np.isclose(q1, q2))
    if collapsed:
        # Do not manufacture three groups by arbitrarily splitting equal visibility values.
        masks = {
            "low_visibility": eligible & (ratio < q1),
            "medium_visibility": eligible & (ratio > q1) & (ratio < q2),
            "high_visibility": eligible & (ratio >= q2),
        }
    else:
        masks = {
            "low_visibility": eligible & (ratio <= q1),
            "medium_visibility": eligible & (ratio > q1) & (ratio <= q2),
            "high_visibility": eligible & (ratio > q2),
        }
    return {
        "definition": "visible frames among valid neighbors divided by (valid neighbor count × observation length); zero-neighbor samples excluded",
        "eligible_sample_count": int(eligible.sum()),
        "excluded_zero_neighbor_count": int((~eligible).sum()),
        "mean_ratio": float(values.mean()),
        "tercile_cutpoints": {"q1": float(q1), "q2": float(q2)},
        "cutpoints_tied_and_bins_collapsed": collapsed,
        "strata": {
            name: {
                "sample_count": int(mask.sum()),
                "visibility_min": float(ratio[mask].min()) if mask.any() else None,
                "visibility_max": float(ratio[mask].max()) if mask.any() else None,
                "models": {mode: subset_summary(value, mask) for mode, value in outputs.items()},
            }
            for name, mask in masks.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/social_protocol_repair_analysis/evaluation_seed123.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = SequenceWithImageSize(args.data_root / "test.npz")
    labels = dataset.dataset.intent_label.numpy().astype(np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Test split must contain clean binary labels only")

    outputs: dict[str, dict[str, np.ndarray]] = {}
    run_records = {}
    for protocol, directory in (
        ("natural", "fixed_base_social_{mode}_natural_seed{seed}"),
        ("visible", "fixed_base_social_visible_{mode}_seed{seed}"),
    ):
        for mode in ("always", "uncertainty"):
            metrics_path = args.results_root / directory.format(mode=mode, seed=args.seed) / "metrics.json"
            model, visible_aware, run_metrics = load_model(
                metrics_path, args.base_checkpoint, mode, dataset, device
            )
            key = f"{protocol}_{mode}"
            outputs[key] = collect(
                model, visible_aware, dataset, device, args.batch_size
            )
            run_records[key] = {
                "metrics_path": str(metrics_path),
                "checkpoint": run_metrics["checkpoint"],
                "visibility_aware": visible_aware,
                "selected_epoch": run_metrics["selected_epoch"],
                "epoch0_validation_auc": run_metrics["epoch0_validation_auc"],
                "best_validation_auc": run_metrics["best_validation_auc"],
                "social_improvement_detected": run_metrics["social_improvement_detected"],
            }

    base_logits_ref = outputs["natural_always"]["base_logits"]
    base_consistency = {
        name: {
            "identical": bool(np.array_equal(base_logits_ref, value["base_logits"])),
            "max_abs_difference": float(np.max(np.abs(base_logits_ref - value["base_logits"]))),
        }
        for name, value in outputs.items()
    }
    if not all(entry["identical"] for entry in base_consistency.values()):
        raise RuntimeError(f"Fixed base logits differ across repair experiments: {base_consistency}")

    model_metrics = {mode: model_summary(value) for mode, value in outputs.items()}
    rng = np.random.default_rng(args.seed + 19001)
    permutation = rng.permutation(len(dataset))
    shuffle_results = {}
    for protocol, mode in (
        ("natural", "always"),
        ("natural", "uncertainty"),
        ("visible", "always"),
        ("visible", "uncertainty"),
    ):
        key = f"{protocol}_{mode}"
        model, visible_aware, _ = load_model(
            Path(run_records[key]["metrics_path"]), args.base_checkpoint, mode, dataset, device
        )
        shuffled = collect(
            model, visible_aware, dataset, device, args.batch_size,
            neighbor_permutation=permutation,
        )
        real = outputs[key]
        real_metrics = intent_metrics(labels, real["final_probabilities"])
        shuffled_metrics = intent_metrics(labels, shuffled["final_probabilities"])
        shuffle_results[key] = {
            "real_neighbor": real_metrics,
            "shuffled_neighbor": shuffled_metrics,
            "auc_delta_shuffled_minus_real": shuffled_metrics["auc"] - real_metrics["auc"],
            "brier_delta_shuffled_minus_real": shuffled_metrics["brier"] - real_metrics["brier"],
            "real_paired_bce_improvement": paired_summary(
                real["base_logits"], real["final_logits"], labels
            ),
            "shuffled_paired_bce_improvement": paired_summary(
                shuffled["base_logits"], shuffled["final_logits"], labels
            ),
            "base_logits_identical_after_shuffle": bool(
                np.array_equal(real["base_logits"], shuffled["base_logits"])
            ),
        }

    result = {
        "seed": args.seed,
        "test_sample_count": len(dataset),
        "base_checkpoint": str(args.base_checkpoint),
        "fixed_base_logits_identical_across_all_runs": base_consistency,
        "runs": run_records,
        "models": model_metrics,
        "neighbor_count_stratification": stratified_by_neighbor_count(
            {"base": outputs["natural_always"], **outputs}
        ),
        "visibility_stratification": stratified_by_visibility(
            {"base": outputs["natural_always"], **{
                "visible_always": outputs["visible_always"],
                "visible_uncertainty": outputs["visible_uncertainty"],
            }}
        ),
        "neighbor_shuffle": {
            "permutation_seed": args.seed + 19001,
            "method": "permute complete neighbor_obs, neighbor_mask, neighbor_visible_mask records across samples; target, scene and labels stay fixed",
            "models": shuffle_results,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": model_metrics, "shuffle": shuffle_results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
