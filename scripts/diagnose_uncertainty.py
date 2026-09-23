"""Diagnose whether the current entropy branch distinguishes ambiguous JAAD samples."""

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
from src.models.scene_social_gate import SceneSocialGate


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p90": float(np.quantile(values, 0.90)),
    }


def collect(model, loader, device, target_features: str, scene_model: bool = False) -> dict[str, np.ndarray]:
    model.eval()
    result = {
        "entropy": [],
        "prior_prob": [],
        "gate": [],
        "intent_prob": [],
        "neighbor_count": [],
        "ade": [],
        "fde": [],
    }
    with torch.no_grad():
        for batch in loader:
            target = batch["target_obs"]
            if target_features == "relative_abs":
                target = torch.cat([target, batch["target_abs_obs"]], dim=-1)
            model_inputs = [
                target.to(device),
                batch["neighbor_obs"].to(device),
                batch["neighbor_mask"].to(device),
                batch["neighbor_visible_mask"].to(device),
            ]
            if scene_model:
                model_inputs.append(batch["scene_feat"].to(device))
            output = model(*model_inputs)
            future_gt = batch["future_gt"].to(device)
            point_error = torch.linalg.vector_norm(output["future_pred"] - future_gt, dim=-1)
            result["entropy"].append(output["entropy"].cpu().numpy())
            result["prior_prob"].append(torch.sigmoid(output["prior_logit"]).cpu().numpy())
            result["gate"].append(output["gate"].cpu().numpy())
            result["intent_prob"].append(torch.sigmoid(output["intent_logit"]).cpu().numpy())
            result["neighbor_count"].append(batch["neighbor_mask"].sum(dim=1).numpy())
            result["ade"].append(point_error.mean(dim=1).cpu().numpy())
            result["fde"].append(point_error[:, -1].cpu().numpy())
    return {key: np.concatenate(value) for key, value in result.items()}


def group_report(values: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {key: summarize(value) for key, value in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences")
    parser.add_argument("--ambiguous-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_ambiguous")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--scene-model", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    target_features = "relative_abs" if args.scene_model else saved_args.get("target_features", "relative")
    gate_mode = saved_args.get("gate_mode", "uncertainty")
    hidden_dim = int(saved_args.get("hidden_dim", 128))
    input_dim = 8 if target_features == "relative_abs" else 4

    if args.scene_model:
        model = SceneSocialGate(
            input_dim=8,
            scene_dim=512,
            hidden_dim=hidden_dim,
            pred_len=12,
            gate_mode=gate_mode,
        ).to(device)
    else:
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
    clean = collect(model, clean_loader, device, target_features, args.scene_model)
    ambiguous = collect(model, ambiguous_loader, device, target_features, args.scene_model)

    labels = np.concatenate([
        np.zeros(clean["entropy"].shape[0], dtype=np.int64),
        np.ones(ambiguous["entropy"].shape[0], dtype=np.int64),
    ])
    separation = {}
    for key in ("entropy", "prior_prob", "gate", "intent_prob"):
        scores = np.concatenate([clean[key], ambiguous[key]])
        separation[key] = {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "target_features": target_features,
        "gate_mode": gate_mode,
        "clean_test_count": int(clean["entropy"].shape[0]),
        "ambiguous_test_count": int(ambiguous["entropy"].shape[0]),
        "clean_test": group_report(clean),
        "ambiguous_test": group_report(ambiguous),
        "ambiguous_vs_clean_separation": separation,
        "interpretation": {
            "entropy_roc_auc_above_0_5": bool(separation["entropy"]["roc_auc"] > 0.5),
            "useful_separation_threshold": 0.65,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
