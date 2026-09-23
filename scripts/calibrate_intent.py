"""Fit validation-set logit calibration and evaluate test calibration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.jaad_sequence_dataset import JAADSequenceDataset
from src.models.uncertainty_social_gate import UncertaintySocialGate
from src.models.scene_social_gate import SceneSocialGate


def ece_score(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    value = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probabilities >= lower) & (
            probabilities < upper if index < bins - 1 else probabilities <= upper
        )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return value


def metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "intent_accuracy": float(accuracy_score(labels, predictions)),
        "intent_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "intent_f1": float(f1_score(labels, predictions, zero_division=0)),
        "intent_auc": float(roc_auc_score(labels, probabilities)),
        "intent_brier": float(brier_score_loss(labels, probabilities)),
        "intent_ece_10": ece_score(labels, probabilities),
        "probability_mean": float(probabilities.mean()),
    }


def collect_logits(model, loader, device, target_features: str, scene_model: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_logits = [], []
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
            all_labels.append(batch["intent_label"].numpy())
            all_logits.append(output["intent_logit"].cpu().numpy())
    return np.concatenate(all_labels).astype(np.int64), np.concatenate(all_logits)


def fit_affine_calibrator(
    labels: np.ndarray, logits: np.ndarray, device: torch.device
) -> tuple[float, float]:
    """Fit logit / temperature + bias.

    The training loader uses inverse-frequency sampling, so a temperature alone
    cannot correct the resulting prior shift. The extra bias handles that shift.
    """
    labels_tensor = torch.from_numpy(labels.astype(np.float32)).to(device)
    logits_tensor = torch.from_numpy(logits.astype(np.float32)).to(device)
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias], lr=0.1, max_iter=100, line_search_fn="strong_wolfe"
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = loss_fn(logits_tensor / temperature + bias, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0).cpu())
    return temperature, float(bias.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/processed/jaad_sequences")
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

    val_set = JAADSequenceDataset(args.data_root / "val.npz")
    test_set = JAADSequenceDataset(args.data_root / "test.npz")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    input_dim = 8 if target_features == "relative_abs" else 4
    if args.scene_model:
        model = SceneSocialGate(
            input_dim=8,
            scene_dim=val_set.scene_feat.shape[1],
            hidden_dim=hidden_dim,
            pred_len=val_set.future_gt.shape[1],
            gate_mode=gate_mode,
        ).to(device)
    else:
        model = UncertaintySocialGate(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            pred_len=val_set.future_gt.shape[1],
            gate_mode=gate_mode,
        ).to(device)
    model.load_state_dict(checkpoint["model"])

    val_labels, val_logits = collect_logits(model, val_loader, device, target_features, args.scene_model)
    test_labels, test_logits = collect_logits(model, test_loader, device, target_features, args.scene_model)
    temperature, bias = fit_affine_calibrator(val_labels, val_logits, device)
    calibrated_test_logits = test_logits / temperature + bias
    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "target_features": target_features,
        "gate_mode": gate_mode,
        "calibration_fit_on": "validation",
        "temperature": temperature,
        "bias": bias,
        "validation_raw": metrics(val_labels, val_logits),
        "test_raw": metrics(test_labels, test_logits),
        "test_affine_calibrated": metrics(test_labels, calibrated_test_logits),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
