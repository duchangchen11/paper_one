#!/usr/bin/env python3
"""Validation-only feature statistics and linear separability for P1 vs M0."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trajectory_preserving_utils import (
    SEEDS,
    TrajectoryIntentDataset,
    load_seed_backbone,
    set_seed,
    sha256_file,
)
from scripts.train_intention_scratch_matched import model_sha256
from src.models.intention_scratch_transformer import IntentionScratchTransformer

RESULTS = ROOT / "results/intention_scratch_matched"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_contexts(model, dataset: TrajectoryIntentDataset, *, pretrained: bool, device: torch.device) -> np.ndarray:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["target"].to(device)
            if pretrained:
                backbone = model.backbone
                temporal = backbone.input_projection(target)
                temporal = temporal + backbone.position_embedding[:, : target.shape[1]]
                context = backbone.temporal_encoder(temporal)[:, -1]
            else:
                context = model.encode_target(target)
            rows.append(context.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def feature_statistics(features: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(features, dtype=np.float64)
    per_dimension_variance = matrix.var(axis=0, ddof=1).tolist() if len(matrix) > 1 else matrix.var(axis=0).tolist()
    return {
        "sample_count": int(matrix.shape[0]),
        "feature_dimension": int(matrix.shape[1]),
        "mean": float(matrix.mean()),
        "std": float(matrix.std(ddof=1)) if matrix.size > 1 else 0.0,
        "mean_l2_norm": float(np.linalg.norm(matrix, axis=1).mean()),
        "mean_per_dimension_variance": float(np.mean(per_dimension_variance)),
        "per_dimension_variance": per_dimension_variance,
    }


def linear_diagnostic(train_features: np.ndarray, train_labels: np.ndarray, val_features: np.ndarray, val_labels: np.ndarray, seed: int) -> dict[str, float]:
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="lbfgs",
        ),
    )
    classifier.fit(train_features, train_labels)
    probability = classifier.predict_proba(val_features)[:, 1]
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

    predicted = (probability >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(val_labels, probability)),
        "brier": float(brier_score_loss(val_labels, probability)),
        "f1": float(f1_score(val_labels, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(val_labels, predicted)),
        "accuracy": float(np.mean(predicted == val_labels)),
        "classifier": "StandardScaler + LogisticRegression(C=1,class_weight=balanced,solver=lbfgs,max_iter=2000)",
        "fit_split": "train only",
        "evaluation_split": "validation only",
        "used_for_checkpoint_selection": False,
    }


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    train_set = TrajectoryIntentDataset(data_root / "train.npz")
    val_set = TrajectoryIntentDataset(data_root / "val.npz")
    validation_count = min(1000, len(val_set))
    results: dict[str, Any] = {}

    for seed in SEEDS:
        set_seed(seed)
        config = json.loads((ROOT / "configs/intention_scratch_matched.json").read_text(encoding="utf-8"))
        model_cfg = config["model"]["transformer"]
        scratch = IntentionScratchTransformer(
            input_dim=8,
            d_model=int(model_cfg["hidden_dimension"]),
            nhead=int(model_cfg["heads"]),
            num_layers=int(model_cfg["layers"]),
            dropout=float(model_cfg["dropout"]),
            max_obs_len=15,
        )
        scratch_checkpoint = ROOT / "checkpoints/intention_scratch_matched" / f"M0_scratch_seed{seed}.pt"
        scratch_payload = torch.load(scratch_checkpoint, map_location="cpu", weights_only=False)
        if scratch_payload.get("pretrained_checkpoint_loaded") is not False:
            raise RuntimeError(f"Seed {seed} M0 checkpoint did not record random initialization")
        scratch.load_state_dict(scratch_payload["model"], strict=True)
        scratch.to(device).eval()

        p1, p1_load_report, p1_checkpoint, p1_checkpoint_sha = load_seed_backbone(
            seed,
            "target",
            device=device,
            input_dim=8,
            scene_dim=512,
            observed_length=15,
            prediction_length=15,
        )
        if not p1_load_report["complete"]:
            raise RuntimeError(f"P1 seed {seed} target encoder checkpoint failed strict mapping")
        p1.eval()

        p1_train = collect_contexts(p1, train_set, pretrained=True, device=device)
        p1_val = collect_contexts(p1, val_set, pretrained=True, device=device)
        m0_train = collect_contexts(scratch, train_set, pretrained=False, device=device)
        m0_val = collect_contexts(scratch, val_set, pretrained=False, device=device)
        train_labels = train_set.intent_label.numpy().astype(np.int64)
        val_labels = val_set.intent_label.numpy().astype(np.int64)
        results[str(seed)] = {
            "m0_initialization_sha256": scratch_payload["initialization_sha256"],
            "m0_selected_checkpoint_sha256": sha256_file(scratch_checkpoint),
            "p1_trajectory_checkpoint_sha256": p1_checkpoint_sha,
            "p1_trajectory_checkpoint": str(p1_checkpoint.relative_to(ROOT)),
            "train_samples": len(train_labels),
            "validation_samples_total": len(val_labels),
            "validation_samples_for_feature_statistics": validation_count,
            "pretrained_frozen_validation_representation": feature_statistics(p1_val[:validation_count]),
            "scratch_trained_validation_representation": feature_statistics(m0_val[:validation_count]),
            "pretrained_frozen_linear_separability": linear_diagnostic(p1_train, train_labels, p1_val, val_labels, seed),
            "scratch_trained_linear_separability": linear_diagnostic(m0_train, train_labels, m0_val, val_labels, seed),
        }

    payload = {
        "analysis": "Validation-only target_context distribution and train-fitted linear intention diagnostic",
        "requested_validation_statistics_samples": 1000,
        "actual_validation_statistics_samples": validation_count,
        "all_validation_used_for_linear_classifier_evaluation": True,
        "no_test_split_loaded": True,
        "no_scene_input": True,
        "seeds": results,
    }
    output = RESULTS / "representation_analysis.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "validation_count_for_stats": validation_count, "per_seed_linear_auc": {seed: {"P1": row["pretrained_frozen_linear_separability"]["roc_auc"], "M0": row["scratch_trained_linear_separability"]["roc_auc"]} for seed, row in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
