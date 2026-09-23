#!/usr/bin/env python3
"""Evaluate fixed-base social residuals with paired, stratified, and shuffle analyses."""

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

from train_fixed_base_intent import SequenceWithImageSize, ece_10, load_backbone
from src.models.fixed_base_social_residual import FixedBaseIntentModel, FixedBaseSocialResidual


MODES = ("none", "always", "uncertainty")


def load_base_model(
    checkpoint_path: Path,
    dataset: SequenceWithImageSize,
    device: torch.device,
) -> FixedBaseIntentModel:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trajectory_path = Path(payload["trajectory_checkpoint"])
    if not trajectory_path.is_absolute():
        trajectory_path = PROJECT_ROOT / trajectory_path
    sample = dataset[0]
    target = torch.cat([sample["target_obs"], sample["target_abs_obs"]], dim=-1)
    backbone, _ = load_backbone(
        trajectory_path,
        input_dim=target.shape[-1],
        scene_dim=sample["scene_feat"].numel(),
        pred_len=sample["future_gt"].shape[0],
        max_obs_len=target.shape[0],
        map_location="cpu",
    )
    model = FixedBaseIntentModel(backbone)
    model.load_state_dict(payload["model"], strict=True)
    model.freeze_base_classifier()
    return model.to(device).eval()


def load_social_model(
    gate_mode: str,
    base_checkpoint: Path,
    social_checkpoint: Path | None,
    dataset: SequenceWithImageSize,
    device: torch.device,
) -> FixedBaseSocialResidual:
    base_model = load_base_model(base_checkpoint, dataset, device)
    if gate_mode == "none":
        return FixedBaseSocialResidual(base_model, gate_mode="none").to(device).eval()
    if social_checkpoint is None or not social_checkpoint.is_file():
        raise FileNotFoundError(f"Missing {gate_mode} social checkpoint: {social_checkpoint}")
    payload = torch.load(social_checkpoint, map_location="cpu", weights_only=False)
    args = payload.get("args", {})
    model = FixedBaseSocialResidual(
        base_model,
        gate_mode=gate_mode,
        social_hidden_dim=int(args.get("social_hidden_dim", base_model.d_model)),
        dropout=float(args.get("dropout", 0.1)),
        social_scale=float(args.get("social_scale", 1.0)),
    )
    model.load_state_dict(payload["model"], strict=True)
    if not model.all_base_parameters_frozen:
        raise RuntimeError(f"Base parameters are not frozen in {gate_mode} model")
    model.to(device).eval()
    return model


def distribution(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64).reshape(-1)
    q = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
    }


@torch.no_grad()
def collect_outputs(
    model: FixedBaseSocialResidual,
    dataset: SequenceWithImageSize,
    device: torch.device,
    batch_size: int,
    neighbor_permutation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    keys = (
        "labels", "base_logit", "base_probability", "entropy", "final_logit",
        "final_probability", "delta_logit", "logit_change", "gate", "effective_gate",
        "has_neighbor", "neighbor_count", "future_pred", "future_gt", "image_size",
    )
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    offset = 0
    model.eval()
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        neighbor_obs = batch["neighbor_obs"]
        neighbor_mask = batch["neighbor_mask"]
        if neighbor_permutation is not None:
            indices = neighbor_permutation[offset : offset + len(target)]
            indices = torch.as_tensor(indices, dtype=torch.long)
            neighbor_obs = dataset.dataset.neighbor_obs[indices]
            neighbor_mask = dataset.dataset.neighbor_mask[indices]
        output = model(
            target,
            batch["scene_feat"].to(device),
            neighbor_obs.to(device),
            neighbor_mask.to(device),
        )
        values["labels"].append(batch["intent_label"].numpy())
        values["base_logit"].append(output["base_logit"].cpu().numpy())
        values["base_probability"].append(output["base_probability"].cpu().numpy())
        values["entropy"].append(output["base_entropy"].cpu().numpy())
        values["final_logit"].append(output["final_logit"].cpu().numpy())
        values["final_probability"].append(output["final_probability"].cpu().numpy())
        values["delta_logit"].append(output["delta_logit"].cpu().numpy())
        values["logit_change"].append(output["logit_change"].cpu().numpy())
        values["gate"].append(output["gate"].cpu().numpy())
        values["effective_gate"].append(output["effective_gate"].cpu().numpy())
        values["has_neighbor"].append(output["has_neighbor"].cpu().numpy())
        values["neighbor_count"].append(output["neighbor_count"].cpu().numpy())
        values["future_pred"].append(output["future_pred"].cpu().numpy())
        values["future_gt"].append(batch["future_gt"].numpy())
        values["image_size"].append(batch["image_size"].numpy())
        offset += len(target)
    return {key: np.concatenate(chunks, axis=0) for key, chunks in values.items()}


def bce_per_sample(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))


def summarize_intent(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10": ece_10(labels, probabilities),
    }


def summarize_trajectory(values: dict[str, np.ndarray]) -> dict[str, float]:
    error = values["future_pred"] - values["future_gt"]
    pixel = np.linalg.norm(error * values["image_size"][:, None, :], axis=-1)
    return {"ade_pixel": float(pixel.mean()), "fde_pixel": float(pixel[:, -1].mean())}


def paired_summary(improvement: np.ndarray) -> dict[str, float]:
    return {
        "mean_improvement": float(improvement.mean()),
        "median_improvement": float(np.median(improvement)),
        "helped_sample_ratio": float((improvement > 0).mean()),
        "hurt_sample_ratio": float((improvement < 0).mean()),
        "unchanged_sample_ratio": float((improvement == 0).mean()),
    }


def summarize_mode(values: dict[str, np.ndarray], base_values: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = values["labels"].astype(np.float64)
    base_losses = bce_per_sample(base_values["base_logit"], labels)
    social_losses = bce_per_sample(values["final_logit"], labels)
    improvement = base_losses - social_losses
    return {
        **summarize_intent(labels, values["final_probability"]),
        **summarize_trajectory(values),
        "gate_distribution": distribution(values["gate"]),
        "effective_gate_distribution": distribution(values["effective_gate"]),
        "delta_logit_distribution": distribution(values["delta_logit"]),
        "delta_logit_abs_mean": float(np.abs(values["delta_logit"]).mean()),
        "logit_change_distribution": distribution(values["logit_change"]),
        "paired_bce_improvement": paired_summary(improvement),
        "no_neighbor_samples": int((values["neighbor_count"] == 0).sum()),
        "no_neighbor_max_abs_logit_change": float(
            np.abs(values["logit_change"][values["neighbor_count"] == 0]).max()
        ) if np.any(values["neighbor_count"] == 0) else None,
    }


def stratified_analysis(
    labels: np.ndarray,
    entropy: np.ndarray,
    outputs: dict[str, dict[str, np.ndarray]],
    base_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    cut1, cut2 = np.quantile(entropy, [1.0 / 3.0, 2.0 / 3.0])
    assignments = np.digitize(entropy, [cut1, cut2], right=True)
    strata = {}
    for index, name in enumerate(("low", "medium", "high")):
        mask = assignments == index
        current = {}
        base_p = base_values["base_probability"][mask]
        current["base"] = {
            **summarize_intent(labels[mask], base_p),
            **paired_summary(np.zeros(mask.sum(), dtype=np.float64)),
        }
        base_loss = bce_per_sample(base_values["base_logit"][mask], labels[mask])
        for mode in ("always", "uncertainty"):
            values = outputs[mode]
            improvement = base_loss - bce_per_sample(values["final_logit"][mask], labels[mask])
            current[mode] = {
                **summarize_intent(labels[mask], values["final_probability"][mask]),
                **paired_summary(improvement),
            }
        strata[name] = {
            "sample_count": int(mask.sum()),
            "entropy_min": float(entropy[mask].min()),
            "entropy_max": float(entropy[mask].max()),
            "metrics": current,
        }
    return {
        "entropy_source": "shared validation-calibrated fixed-base entropy",
        "cutpoints": {"low_medium": float(cut1), "medium_high": float(cut2)},
        "strata": strata,
    }


def neighbor_count_analysis(
    labels: np.ndarray,
    neighbor_count: np.ndarray,
    outputs: dict[str, dict[str, np.ndarray]],
    base_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    groups = {
        "0": neighbor_count == 0,
        "1": neighbor_count == 1,
        "2-3": (neighbor_count >= 2) & (neighbor_count <= 3),
        ">=4": neighbor_count >= 4,
    }
    result = {}
    for name, mask in groups.items():
        group_values = {"base": base_values, **outputs}
        result[name] = {
            "sample_count": int(mask.sum()),
            "models": {
                mode: {
                    **summarize_intent(
                        labels[mask], values["base_probability"][mask]
                        if mode == "base"
                        else values["final_probability"][mask]
                    ),
                    "paired_bce_improvement": paired_summary(
                        np.zeros(mask.sum(), dtype=np.float64)
                        if mode == "base"
                        else bce_per_sample(base_values["base_logit"][mask], labels[mask])
                        - bce_per_sample(values["final_logit"][mask], labels[mask])
                    ),
                }
                for mode, values in group_values.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--social-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/fixed_base_social_analysis")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    dataset = SequenceWithImageSize(args.data_root / "test.npz")
    labels = dataset.dataset.intent_label.numpy().astype(np.int64)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Test set must contain only clean binary intent labels")

    social_checkpoints = {}
    for mode in ("always", "uncertainty"):
        run_metrics_path = args.social_root / f"fixed_base_social_{mode}_seed{args.seed}" / "metrics.json"
        run_metrics = json.loads(run_metrics_path.read_text(encoding="utf-8"))
        path = Path(run_metrics["checkpoint"])
        social_checkpoints[mode] = path if path.is_absolute() else PROJECT_ROOT / path

    outputs = {}
    for mode in MODES:
        model = load_social_model(
            mode,
            args.base_checkpoint,
            social_checkpoints.get(mode),
            dataset,
            device,
        )
        outputs[mode] = collect_outputs(model, dataset, device, args.batch_size)

    base_values = outputs["none"]
    base_logit_reference = base_values["base_logit"]
    base_consistency = {
        mode: {
            "identical": bool(np.array_equal(base_logit_reference, outputs[mode]["base_logit"])),
            "max_abs_difference": float(
                np.max(np.abs(base_logit_reference - outputs[mode]["base_logit"]))
            ),
        }
        for mode in MODES
    }
    if not all(value["identical"] for value in base_consistency.values()):
        raise RuntimeError(f"Shared base logits are not exactly identical: {base_consistency}")

    model_metrics = {
        "base": {
            **summarize_intent(labels, base_values["base_probability"]),
            **summarize_trajectory(base_values),
        },
        "always": summarize_mode(outputs["always"], base_values),
        "uncertainty": summarize_mode(outputs["uncertainty"], base_values),
    }
    entropy_analysis = stratified_analysis(
        labels, base_values["entropy"], outputs, base_values
    )
    count_analysis = neighbor_count_analysis(
        labels, base_values["neighbor_count"], outputs, base_values
    )

    rng = np.random.default_rng(args.seed + 9001)
    permutation = rng.permutation(len(dataset))
    shuffle_metrics = {}
    for mode in MODES:
        model = load_social_model(
            mode,
            args.base_checkpoint,
            social_checkpoints.get(mode),
            dataset,
            device,
        )
        shuffled = collect_outputs(
            model, dataset, device, args.batch_size, neighbor_permutation=permutation
        )
        real_metrics = summarize_intent(labels, outputs[mode]["final_probability"])
        shuffled_metrics = summarize_intent(labels, shuffled["final_probability"])
        shuffle_metrics[mode] = {
            "real_neighbor": real_metrics,
            "shuffled_neighbor": shuffled_metrics,
            "auc_delta_shuffled_minus_real": shuffled_metrics["auc"] - real_metrics["auc"],
            "brier_delta_shuffled_minus_real": shuffled_metrics["brier"] - real_metrics["brier"],
            "base_logit_identical_to_unshuffled": bool(
                np.array_equal(outputs[mode]["base_logit"], shuffled["base_logit"])
            ),
        }
    shuffle_payload = {
        "seed": args.seed,
        "permutation_seed": args.seed + 9001,
        "method": "permute complete neighbor tensor and mask across clean test samples; keep each target, scene, and label fixed",
        "models": shuffle_metrics,
    }
    (args.output_root / "neighbor_shuffle_diagnostic.json").write_text(
        json.dumps(shuffle_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = {
        "seed": args.seed,
        "base_checkpoint": str(args.base_checkpoint),
        "test_sample_count": len(dataset),
        "base_logit_identical_across_modes": base_consistency,
        "base_classifier_and_transformer_frozen": all(
            load_social_model(
                mode, args.base_checkpoint, social_checkpoints.get(mode), dataset, device
            ).all_base_parameters_frozen
            for mode in MODES
        ),
        "models": model_metrics,
        "paired_improvement": {
            "always": model_metrics["always"]["paired_bce_improvement"],
            "uncertainty": model_metrics["uncertainty"]["paired_bce_improvement"],
        },
        "uncertainty_stratification": entropy_analysis,
        "neighbor_count_stratification": count_analysis,
        "neighbor_shuffle_diagnostic": shuffle_payload,
    }
    output_path = args.output_root / f"evaluation_seed{args.seed}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
