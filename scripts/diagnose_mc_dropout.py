"""Diagnose MC-Dropout uncertainty for clean and ambiguous JAAD samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.uncertainty_social_gate import UncertaintySocialGate


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return -probability * np.log(probability) - (1.0 - probability) * np.log(1.0 - probability)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def collect_mc(
    model,
    loader,
    device,
    target_features: str,
    passes: int,
) -> dict[str, np.ndarray]:
    # There are dropout layers in the proposal/fusion heads. train() activates
    # those layers while no_grad() keeps this inference-only.
    model.train()
    result = {
        "mean_prob": [],
        "predictive_entropy": [],
        "expected_entropy": [],
        "mutual_information": [],
        "prob_std": [],
        "gate_mean": [],
        "gate_std": [],
        "ade": [],
        "fde": [],
    }
    with torch.no_grad():
        for batch in loader:
            target = batch["target_obs"]
            if target_features == "relative_abs":
                target = torch.cat([target, batch["target_abs_obs"]], dim=-1)
            target = target.to(device)
            neighbor_obs = batch["neighbor_obs"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)
            visible_mask = batch["neighbor_visible_mask"].to(device)
            probabilities, gates, trajectories = [], [], []
            for _ in range(passes):
                output = model(target, neighbor_obs, neighbor_mask, visible_mask)
                probabilities.append(torch.sigmoid(output["intent_logit"]).cpu().numpy())
                gates.append(output["gate"].cpu().numpy())
                trajectories.append(output["future_pred"].cpu().numpy())
            probabilities = np.stack(probabilities, axis=0)
            gates = np.stack(gates, axis=0)
            trajectories = np.stack(trajectories, axis=0)
            mean_probability = probabilities.mean(axis=0)
            predictive_entropy = binary_entropy(mean_probability)
            expected_entropy = binary_entropy(probabilities).mean(axis=0)
            mean_trajectory = trajectories.mean(axis=0)
            future_gt = batch["future_gt"].numpy()
            point_error = np.linalg.norm(mean_trajectory - future_gt, axis=-1)
            result["mean_prob"].append(mean_probability)
            result["predictive_entropy"].append(predictive_entropy)
            result["expected_entropy"].append(expected_entropy)
            result["mutual_information"].append(predictive_entropy - expected_entropy)
            result["prob_std"].append(probabilities.std(axis=0))
            result["gate_mean"].append(gates.mean(axis=0))
            result["gate_std"].append(gates.std(axis=0))
            result["ade"].append(point_error.mean(axis=1))
            result["fde"].append(point_error[:, -1])
    return {key: np.concatenate(value) for key, value in result.items()}


def summarize_group(group: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {key: summarize(value) for key, value in group.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences")
    parser.add_argument("--ambiguous-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_ambiguous")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.passes < 2:
        raise ValueError("--passes must be at least 2")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    target_features = saved_args.get("target_features", "relative")
    gate_mode = saved_args.get("gate_mode", "uncertainty")
    hidden_dim = int(saved_args.get("hidden_dim", 128))
    input_dim = 8 if target_features == "relative_abs" else 4
    model = UncertaintySocialGate(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        pred_len=12,
        gate_mode=gate_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    clean_set = JAADSequenceDataset(args.data_root / "test.npz")
    ambiguous_set = JAADSequenceDataset(args.ambiguous_root / "test.npz")
    clean_loader = DataLoader(clean_set, batch_size=args.batch_size, shuffle=False)
    ambiguous_loader = DataLoader(ambiguous_set, batch_size=args.batch_size, shuffle=False)
    clean = collect_mc(model, clean_loader, device, target_features, args.passes)
    ambiguous = collect_mc(model, ambiguous_loader, device, target_features, args.passes)

    labels = np.concatenate([
        np.zeros(clean["predictive_entropy"].shape[0], dtype=np.int64),
        np.ones(ambiguous["predictive_entropy"].shape[0], dtype=np.int64),
    ])
    separation = {}
    for key in ("predictive_entropy", "mutual_information", "prob_std", "gate_std"):
        scores = np.concatenate([clean[key], ambiguous[key]])
        separation[key] = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "passes": args.passes,
        "target_features": target_features,
        "gate_mode": gate_mode,
        "clean_test_count": int(clean["predictive_entropy"].shape[0]),
        "ambiguous_test_count": int(ambiguous["predictive_entropy"].shape[0]),
        "clean_test": summarize_group(clean),
        "ambiguous_test": summarize_group(ambiguous),
        "ambiguous_vs_clean_separation": separation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
