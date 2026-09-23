#!/usr/bin/env python3
"""Run post-training ambiguity and uncertainty-stratified diagnostics only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from train_residual_social_joint import SequenceWithImageSize, make_model


MODES = ("none", "always", "uncertainty")
SEEDS = (42, 123, 2024)


def load_trained_model(
    mode: str,
    seed: int,
    stage: str,
    data_root: Path,
    model_root: Path,
    device: torch.device,
):
    run_root = model_root / f"residual_social_stage{stage}_{mode}_seed{seed}"
    metrics_path = run_root / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing completed run metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint_path = metrics.get("checkpoint")
    if not checkpoint_path:
        raise RuntimeError(f"Run has no selected checkpoint: {metrics_path}")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    run_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone_path = Path(run_checkpoint["args"]["trajectory_checkpoint"])
    if not backbone_path.is_absolute():
        backbone_path = PROJECT_ROOT / backbone_path
    dataset = SequenceWithImageSize(data_root / "test.npz")
    model = make_model(
        dataset,
        backbone_path,
        device,
        gate_mode=mode,
        trajectory_residual_scale=float(run_checkpoint["args"].get("trajectory_residual_scale", 0.1)),
        enable_trajectory_residual=(stage == "B"),
    )
    model.load_state_dict(run_checkpoint["model"], strict=True)
    model.eval()
    return model, metrics, dataset


@torch.no_grad()
def collect_outputs(model, dataset: SequenceWithImageSize, device: torch.device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    collected: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "labels",
            "intent_probability",
            "prior_probability",
            "entropy",
            "gate",
            "future_pred",
            "future_gt",
            "image_size",
        )
    }
    model.eval()
    for batch in loader:
        target = torch.cat([batch["target_obs"], batch["target_abs_obs"]], dim=-1).to(device)
        output = model(
            target,
            batch["neighbor_obs"].to(device),
            batch["neighbor_mask"].to(device),
            batch["neighbor_visible_mask"].to(device),
            batch["scene_feat"].to(device),
        )
        collected["labels"].append(batch["intent_label"].numpy())
        collected["intent_probability"].append(output["intent_probability"].cpu().numpy())
        collected["prior_probability"].append(output["prior_probability"].cpu().numpy())
        collected["entropy"].append(output["normalized_entropy"].cpu().numpy())
        collected["gate"].append(output["gate"].cpu().numpy())
        collected["future_pred"].append(output["future_pred"].cpu().numpy())
        collected["future_gt"].append(batch["future_gt"].numpy())
        collected["image_size"].append(batch["image_size"].numpy())
    return {key: np.concatenate(chunks, axis=0) for key, chunks in collected.items()}


def intent_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | None]:
    labels = labels.astype(np.int64)
    valid = (labels == 0) | (labels == 1)
    labels = labels[valid]
    probabilities = np.clip(probabilities[valid], 1e-7, 1.0 - 1e-7)
    result: dict[str, float | None] = {
        "sample_count": int(valid.sum()),
        "positive_count": int((labels == 1).sum()),
        "negative_count": int((labels == 0).sum()),
        "brier": float(brier_score_loss(labels, probabilities)),
        "auc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else None,
    }
    return result


def trajectory_metrics(outputs: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    error = outputs["future_pred"][mask] - outputs["future_gt"][mask]
    scale = outputs["image_size"][mask]
    pixel_error = np.linalg.norm(error * scale[:, None, :], axis=-1)
    return {
        "ade_pixel": float(pixel_error.mean()),
        "fde_pixel": float(pixel_error[:, -1].mean()),
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    values = values.astype(np.float64).reshape(-1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def clean_ambiguous_diagnostic(clean: dict[str, np.ndarray], ambiguous: dict[str, np.ndarray]) -> dict[str, Any]:
    clean_count = len(clean["entropy"])
    labels = np.concatenate([np.zeros(clean_count, dtype=np.int64), np.ones(len(ambiguous["entropy"]), dtype=np.int64)])
    entropy_scores = np.concatenate([clean["entropy"], ambiguous["entropy"]])
    gate_scores = np.concatenate([clean["gate"], ambiguous["gate"]])
    return {
        "interpretation": "diagnostic proxy / ambiguous annotation subset only; crossing=-1 is not uncertainty ground truth",
        "clean_sample_count": clean_count,
        "ambiguous_sample_count": int(len(ambiguous["entropy"])),
        "entropy_auroc_ambiguous_vs_clean": float(roc_auc_score(labels, entropy_scores)),
        "gate_auroc_ambiguous_vs_clean": float(roc_auc_score(labels, gate_scores)),
        "clean_entropy": distribution(clean["entropy"]),
        "ambiguous_entropy": distribution(ambiguous["entropy"]),
        "clean_gate": distribution(clean["gate"]),
        "ambiguous_gate": distribution(ambiguous["gate"]),
    }


def stratified_comparison(
    outputs_by_mode: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    uncertainty = outputs_by_mode["uncertainty"]
    entropy = uncertainty["entropy"]
    cut1, cut2 = np.quantile(entropy, [1.0 / 3.0, 2.0 / 3.0])
    # Keep ties deterministic and ensure all samples map to one of three bins.
    strata_index = np.digitize(entropy, [cut1, cut2], right=True)
    names = ("low", "medium", "high")
    strata: dict[str, Any] = {}
    for index, name in enumerate(names):
        mask = strata_index == index
        if not mask.any():
            strata[name] = {"sample_count": 0, "models": {}}
            continue
        per_mode = {}
        for mode in MODES:
            outputs = outputs_by_mode[mode]
            y = outputs["labels"][mask]
            p = outputs["intent_probability"][mask]
            per_mode[mode] = {
                **intent_metrics(y, p),
                **trajectory_metrics(outputs, mask),
            }
        strata[name] = {
            "sample_count": int(mask.sum()),
            "uncertainty_entropy_range": [float(entropy[mask].min()), float(entropy[mask].max())],
            "models": per_mode,
        }
    return {
        "stratification_source": "normalized prior entropy from the seed-matched uncertainty-gate model on the clean test split",
        "tercile_cutpoints": {"low_to_medium": float(cut1), "medium_to_high": float(cut2)},
        "strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences_scene_15x15")
    parser.add_argument("--ambiguous-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_ambiguous_scene_15x15")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/residual_social_analysis")
    parser.add_argument("--stage", choices=("A", "B"), default="A")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    clean_dataset = SequenceWithImageSize(args.data_root / "test.npz")
    outputs_by_mode: dict[str, dict[str, np.ndarray]] = {}
    for mode in MODES:
        model, _, _ = load_trained_model(
            mode, args.seed, args.stage, args.data_root, args.model_root, device
        )
        outputs_by_mode[mode] = collect_outputs(model, clean_dataset, device, args.batch_size)

    uncertainty_model, _, _ = load_trained_model(
        "uncertainty", args.seed, args.stage, args.data_root, args.model_root, device
    )
    ambiguous_dataset = SequenceWithImageSize(args.ambiguous_root / "test.npz")
    ambiguous_outputs = collect_outputs(uncertainty_model, ambiguous_dataset, device, args.batch_size)
    diagnostic = clean_ambiguous_diagnostic(outputs_by_mode["uncertainty"], ambiguous_outputs)
    diagnostic.update({"seed": args.seed, "stage": args.stage, "coordinate_unit": "pixel"})
    stratified = stratified_comparison(outputs_by_mode)
    stratified.update({"seed": args.seed, "stage": args.stage, "coordinate_unit": "pixel"})

    diagnostic_path = args.output_root / f"clean_ambiguous_diagnostic_seed{args.seed}.json"
    diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    stratified_path = args.output_root / "uncertainty_stratified_metrics.json"
    stratified_path.write_text(json.dumps(stratified, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        f"# Uncertainty-stratified Stage {args.stage} metrics (seed {args.seed})",
        "",
        "Samples are stratified by terciles of the uncertainty model's normalized prior entropy on the clean test split. `crossing=-1` is not used as uncertainty ground truth.",
        "",
        f"Cut points: `{stratified['tercile_cutpoints']}`.",
        "",
        "| Entropy stratum | N | Model | AUC | Brier | ADE (px) | FDE (px) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for name, group in stratified["strata"].items():
        for mode in MODES:
            metrics = group.get("models", {}).get(mode)
            if not metrics:
                continue
            auc = "NA" if metrics["auc"] is None else f"{metrics['auc']:.4f}"
            markdown.append(
                f"| {name} | {group['sample_count']} | {mode} | {auc} | "
                f"{metrics['brier']:.4f} | {metrics['ade_pixel']:.3f} | {metrics['fde_pixel']:.3f} |"
            )
    markdown.extend(
        [
            "",
            "## Clean vs ambiguous diagnostic proxy",
            "",
            f"Entropy AUROC: {diagnostic['entropy_auroc_ambiguous_vs_clean']:.4f}; gate AUROC: {diagnostic['gate_auroc_ambiguous_vs_clean']:.4f}.",
            "",
            f"Clean entropy mean±std: {diagnostic['clean_entropy']['mean']:.4f}±{diagnostic['clean_entropy']['std']:.4f}; ambiguous: {diagnostic['ambiguous_entropy']['mean']:.4f}±{diagnostic['ambiguous_entropy']['std']:.4f}.",
            f"Clean gate mean±std: {diagnostic['clean_gate']['mean']:.4f}±{diagnostic['clean_gate']['std']:.4f}; ambiguous: {diagnostic['ambiguous_gate']['mean']:.4f}±{diagnostic['ambiguous_gate']['std']:.4f}.",
            "",
            "This is a diagnostic proxy on the ambiguous annotation subset, not a claim that the annotation is ground-truth uncertainty.",
            "",
        ]
    )
    (args.output_root / "uncertainty_stratified_metrics.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(json.dumps({"diagnostic": diagnostic, "stratified": stratified}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
