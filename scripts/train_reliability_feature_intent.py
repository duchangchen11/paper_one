#!/usr/bin/env python3
"""Train frozen-predictor reliability-as-feature intention ablations.

This script reads only train-OOF, validation, and unlabeled test feature files.
It calibrates models on validation and freezes the complete protocol before the
separate evaluator is allowed to open the released labeled test cache.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reliability_feature_intent_utils import (
    SEEDS,
    VARIANT_FEATURES,
    VARIANT_NAMES,
    add_normalized_reliability,
    fit_train_oof_normalization,
    get_model_inputs,
    load_json,
    prepare_splits,
    sha256_file,
    validation_calibration,
    write_json,
)
from src.models.reliability_feature_intent import ReliabilityFeatureIntent, parameter_count


SOURCE_ROOT = PROJECT_ROOT / "results/reliability_gated_intent_15x15"
OUTPUT_ROOT = PROJECT_ROOT / "results/reliability_feature_intent_15x15"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/reliability_feature_intent_15x15"
VARIANTS = ("A", "B", "C", "D", "E", "D_no_future")
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 5.0
SELECTION_TOLERANCE = 1e-12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_hash(model: nn.Module, prefixes: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(prefixes):
            digest.update(key.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_loader(
    data: dict[str, np.ndarray], variant: str, *, shuffle: bool, seed: int
) -> DataLoader:
    obs, future, reliability = get_model_inputs(data, variant)
    labels = np.asarray(data["intent_label"], dtype=np.float32)
    tensors: list[torch.Tensor] = [torch.from_numpy(obs)]
    if future is not None:
        tensors.append(torch.from_numpy(future))
    if reliability is not None:
        tensors.append(torch.from_numpy(reliability))
    tensors.append(torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def batch_logits(
    model: ReliabilityFeatureIntent,
    batch: tuple[torch.Tensor, ...],
    variant: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cursor = 0
    obs = batch[cursor].to(device, non_blocking=True)
    cursor += 1
    future = None
    reliability = None
    if variant in {"B", "C", "D", "E"}:
        future = batch[cursor].to(device, non_blocking=True)
        cursor += 1
    if VARIANT_FEATURES[variant]:
        reliability = batch[cursor].to(device, non_blocking=True)
        cursor += 1
    labels = batch[cursor].to(device, non_blocking=True)
    logits = model(obs, future, reliability)["final_logit"]
    return logits, labels


@torch.no_grad()
def collect_logits(
    model: ReliabilityFeatureIntent,
    loader: DataLoader,
    variant: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        logit, label = batch_logits(model, batch, variant, device)
        logits.append(logit.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
    return np.concatenate(logits).astype(np.float64), np.concatenate(labels).astype(np.int64)


def model_inputs_record(variant: str) -> dict[str, Any]:
    has_future = variant in {"B", "C", "D", "E"}
    return {
        "description": VARIANT_NAMES[variant],
        "inputs": ["target_obs"]
        + (["future_pred_mean"] if has_future else [])
        + list(VARIANT_FEATURES[variant]),
        "future_branch": has_future,
        "reliability_features": list(VARIANT_FEATURES[variant]),
        "forbidden_inputs": ["future_gt", "ADE", "FDE", "intent_label", "crossing_label"],
    }


def train_one(
    variant: str,
    seed: int,
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    class_weights: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed_everything(seed)
    model = ReliabilityFeatureIntent(variant).to(device)
    initial_observed_sha = state_hash(model, ("observed_encoder.",))
    initial_future_sha = (
        state_hash(model, ("future_encoder.",)) if variant in {"B", "C", "D", "E"} else None
    )
    # Synchronize dropout RNG across matched variants after architecture creation.
    seed_everything(seed)
    train_loader = make_loader(train, variant, shuffle=True, seed=seed)
    val_loader = make_loader(val, variant, shuffle=False, seed=seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    positive_weight = float(class_weights["positive"])
    negative_weight = float(class_weights["negative"])
    checkpoint_path = CHECKPOINT_ROOT / f"model_{variant}" / f"seed{seed}.pt"
    best_auc = -np.inf
    best_brier = np.inf
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            logits, labels = batch_logits(model, batch, variant, device)
            weights = torch.where(
                labels > 0.5,
                torch.as_tensor(positive_weight, device=device),
                torch.as_tensor(negative_weight, device=device),
            )
            loss = (
                nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                * weights
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            sample_count += len(labels)

        val_logits, val_labels = collect_logits(model, val_loader, variant, device)
        val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -60, 60)))
        val_auc = float(roc_auc_score(val_labels, val_prob))
        val_brier = float(brier_score_loss(val_labels, val_prob))
        better = val_auc > best_auc + SELECTION_TOLERANCE
        if abs(val_auc - best_auc) <= SELECTION_TOLERANCE and val_brier < best_brier:
            better = True
        print(
            json.dumps(
                {
                    "variant": variant,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": loss_sum / sample_count,
                    "val_auc": val_auc,
                    "val_brier": val_brier,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if better:
            best_auc, best_brier, best_epoch = val_auc, val_brier, epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "variant": variant,
                    "seed": seed,
                    "model_state": model.state_dict(),
                    "selected_epoch": best_epoch,
                    "validation_auc": best_auc,
                    "validation_brier": best_brier,
                    "initial_observed_sha256": initial_observed_sha,
                    "initial_future_sha256": initial_future_sha,
                    "feature_normalization_sha256": sha256_file(
                        OUTPUT_ROOT / "feature_normalization.json"
                    ),
                    "train_cache_sha256": sha256_file(
                        SOURCE_ROOT / "cache/train_oof_features.npz"
                    ),
                    "validation_cache_sha256": sha256_file(
                        SOURCE_ROOT / "cache/val_features.npz"
                    ),
                    "class_weights": class_weights,
                    "optimizer": {
                        "name": "AdamW",
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                    },
                    "epochs_max": EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "gradient_clip_norm": GRADIENT_CLIP,
                },
                checkpoint_path,
            )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected = ReliabilityFeatureIntent(variant)
    selected.load_state_dict(payload["model_state"], strict=True)
    selected.to(device)
    selected_val_logits, selected_val_labels = collect_logits(
        selected, val_loader, variant, device
    )
    calibration = validation_calibration(selected_val_logits, selected_val_labels)
    scaled = selected_val_logits / calibration["temperature"]
    val_prob_calibrated = 1.0 / (1.0 + np.exp(-np.clip(scaled, -60, 60)))
    val_metrics = {
        "roc_auc": float(roc_auc_score(selected_val_labels, val_prob_calibrated)),
        "brier": float(brier_score_loss(selected_val_labels, val_prob_calibrated)),
    }
    output_record: dict[str, Any] = {
        "variant": variant,
        "variant_name": VARIANT_NAMES[variant],
        "seed": seed,
        "selected_epoch": int(payload["selected_epoch"]),
        "validation_auc_at_selection": float(payload["validation_auc"]),
        "validation_brier_at_selection": float(payload["validation_brier"]),
        **calibration,
        "validation_metrics_calibrated": val_metrics,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_observed_sha256": initial_observed_sha,
        "initial_future_sha256": initial_future_sha,
        "parameter_count": parameter_count(selected),
        "architecture": {
            "observed_gru": {"input": [15, 8], "hidden": 128, "layers": 1},
            "future_gru": {"input": [15, 2], "hidden": 64, "layers": 1}
            if variant in {"B", "C", "D", "E"}
            else None,
            "reliability_mlp_input_dim": len(VARIANT_FEATURES[variant]),
            "fusion_input_dim": selected.fusion_input_dim,
            "classifier": "Linear(fusion,128)-ReLU-Dropout(0.1)-Linear(128,64)-ReLU-Linear(64,1)",
        },
    }
    write_json(
        OUTPUT_ROOT / f"model_{variant}" / f"seed{seed}" / "validation_metrics.json",
        output_record,
    )
    del selected, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_record


def feature_distribution(splits: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    from scripts.reliability_feature_intent_utils import distribution_summary

    return {
        split: {
            "sample_count": int(len(data["sample_index"])),
            "features": {
                "raw_u": distribution_summary(data["u_mean_pixel"]),
                "adjusted_u": distribution_summary(data["adjusted_u"]),
                "motion": distribution_summary(data["observed_motion_pixel"]),
            },
            "labels_used": split != "test_unlabeled",
        }
        for split, data in splits.items()
    }


def collect_importance(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    importance: dict[str, Any] = {
        "diagnostic_only": True,
        "test_labels_used": False,
        "measure": "L2 norm per input column of the reliability MLP first linear layer; not a causal attribution",
        "models": {},
    }
    for variant in VARIANTS:
        if not VARIANT_FEATURES[variant]:
            continue
        importance["models"][variant] = {}
        for seed in SEEDS:
            checkpoint = torch.load(
                PROJECT_ROOT / records[f"{variant}_seed{seed}"]["checkpoint"],
                map_location="cpu",
                weights_only=False,
            )
            weight = checkpoint["model_state"]["reliability_encoder.0.weight"].numpy()
            names = VARIANT_FEATURES[variant]
            importance["models"][variant][str(seed)] = {
                "input_column_l2_norm": {
                    name: float(np.linalg.norm(weight[:, index]))
                    for index, name in enumerate(names)
                },
                "branch_first_layer_frobenius_norm": float(np.linalg.norm(weight)),
                "branch_output_layer_frobenius_norm": float(
                    np.linalg.norm(checkpoint["model_state"]["reliability_encoder.2.weight"].numpy())
                ),
            }
    return importance


def main() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing experiment results: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    splits, manifest_info = prepare_splits(SOURCE_ROOT)
    normalization = fit_train_oof_normalization(
        splits["train_oof"], fit_split="official_train_oof"
    )
    write_json(OUTPUT_ROOT / "feature_normalization.json", normalization)
    for split in splits:
        splits[split] = add_normalized_reliability(splits[split], normalization)
    write_json(OUTPUT_ROOT / "train_distribution.json", feature_distribution(splits))

    previous_weights = load_json(SOURCE_ROOT / "protocol_frozen.json")["class_weighting"]
    train_labels = np.asarray(splits["train_oof"]["intent_label"], dtype=np.int64)
    pos_count, neg_count = int(np.sum(train_labels == 1)), int(np.sum(train_labels == 0))
    if (pos_count, neg_count) != (
        int(previous_weights["positive_count"]),
        int(previous_weights["negative_count"]),
    ):
        raise ValueError("train OOF labels/counts differ from the previous frozen balanced-BCE protocol")
    class_weights = {
        "positive": float(previous_weights["positive"]),
        "negative": float(previous_weights["negative"]),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "sampling": previous_weights["sampling"],
        "source": "previous_frozen_reliability_gated_intent_protocol",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_records: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            record = train_one(
                variant,
                seed,
                splits["train_oof"],
                splits["val"],
                class_weights,
                device,
            )
            model_records[f"{variant}_seed{seed}"] = record

    # Identical random seeds initialize the common observed/future branches for
    # every matched B/C/D/E model. Verify the actual stored initialization IDs.
    for seed in SEEDS:
        for branch in ("initial_observed_sha256", "initial_future_sha256"):
            variants = ("B", "C", "D", "E")
            values = {model_records[f"{variant}_seed{seed}"][branch] for variant in variants}
            if len(values) != 1:
                raise RuntimeError(f"matched initialization invariant failed: seed={seed}, {branch}")

    # Confirm that the released labeled cache is exactly the previously
    # protocol-released artifact; hashing bytes does not open or inspect labels.
    release = load_json(SOURCE_ROOT / "test_features_released_after_protocol.json")
    labeled_test_path = SOURCE_ROOT / "cache/test_features.npz"
    labeled_test_sha = sha256_file(labeled_test_path)
    if labeled_test_sha != release["released_labeled_feature_cache_sha256"]:
        raise ValueError("released labeled test cache SHA does not match its release record")
    if release.get("test_intent_labels_read_after_protocol_freeze") is not True:
        raise ValueError("labeled test cache lacks a prior post-freeze release record")

    normalization_sha = sha256_file(OUTPUT_ROOT / "feature_normalization.json")
    test_cutpoints = np.quantile(
        splits["train_oof"]["adjusted_u"], [1 / 3, 2 / 3], method="linear"
    )
    protocol = {
        "protocol": "reliability-as-feature intention recognition; frozen predictor and OOF features",
        "test_protocol_frozen_before_labeled_test_labels_read": True,
        "test_labels_read_during_training_or_selection": False,
        "data": {
            "train_oof_sample_count": int(len(splits["train_oof"]["sample_index"])),
            "validation_sample_count": int(len(splits["val"]["sample_index"])),
            "test_sample_count_from_unlabeled_cache": int(len(splits["test_unlabeled"]["sample_index"])),
            "train_feature_sha256": manifest_info["cache_file_sha256"]["train_oof_features.npz"],
            "validation_feature_sha256": manifest_info["cache_file_sha256"]["val_features.npz"],
            "unlabeled_test_feature_sha256": manifest_info["cache_file_sha256"]["test_features_unlabeled.npz"],
            "labeled_test_cache_sha256_released_after_prior_protocol": labeled_test_sha,
            "feature_cache_manifest_sha256": manifest_info["feature_cache_manifest_sha256"],
            "feature_cache_manifest_embedded_sha256": manifest_info["feature_cache_manifest_embedded_sha256"],
            "crossfit_manifest_sha256": manifest_info["crossfit_manifest_sha256"],
            "crossfit_manifest_embedded_sha256": manifest_info["crossfit_manifest_embedded_sha256"],
            "trajectory_predictor_retrained": False,
            "jaad_split_changed": False,
        },
        "motion_adjustment": {
            "method": "log1p(u_mean_pixel) residualized on quadratic log1p(observed_motion_pixel)",
            "fit_split": "previously_frozen_official_train_oof_only",
            "transform_sha256": manifest_info["reliability_transform_sha256"],
            "coefficients": manifest_info["reliability_adjustment_coefficients"],
        },
        "feature_normalization": {
            "path": "feature_normalization.json",
            "sha256": normalization_sha,
            "fit_split": "official_train_oof",
            "features": ["u_mean_pixel", "observed_motion_pixel", "adjusted_u"],
            "std_ddof": 0,
            "validation_or_test_fit": False,
        },
        "model_definitions": {variant: model_inputs_record(variant) for variant in VARIANTS},
        "architecture": {
            "observed_gru": {"input_shape": [15, 8], "hidden_dim": 128, "layers": 1},
            "future_gru": {"input_shape": [15, 2], "hidden_dim": 64, "layers": 1},
            "reliability_branch": "Linear(d,32)-ReLU-Linear(32,32)",
            "fusion": "Linear(d,128)-ReLU-Dropout(0.1)-Linear(128,64)-ReLU-Linear(64,1)",
            "all_models_train_end_to_end": True,
        },
        "training": {
            "seeds": list(SEEDS),
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs_max": EPOCHS,
            "gradient_clip_norm": GRADIENT_CLIP,
            "balanced_bce_weights": class_weights,
            "sampling": "natural shuffle=True",
            "checkpoint_selection": "maximum official validation ROC-AUC; exact tie broken by lower validation Brier",
            "test_selection": False,
        },
        "matched_initialization": {
            "required_variants": ["B", "C", "D", "E"],
            "common_observed_and_future_branch_initialization": {
                str(seed): {
                    "observed_sha256": model_records[f"B_seed{seed}"]["initial_observed_sha256"],
                    "future_sha256": model_records[f"B_seed{seed}"]["initial_future_sha256"],
                }
                for seed in SEEDS
            },
        },
        "selected_models": model_records,
        "calibration_and_threshold": {
            "temperature_fit": "official_val only, separately per seed/model",
            "threshold": "maximize balanced accuracy on calibrated official_val probabilities",
            "test_temperature_and_threshold_frozen": True,
        },
        "reliability_tertile_cutpoints_train_oof_adjusted_u": [float(value) for value in test_cutpoints],
        "test_metrics": ["ROC-AUC", "Brier", "ECE-15 equal-width", "Balanced Accuracy", "F1"],
        "paired_comparisons": ["D-B", "D-A", "D-C", "E-D"],
        "bootstrap": {
            "unit": "scene_id video cluster",
            "repetitions": 2000,
            "seed": 9124,
            "paired_sampling": True,
            "ci": "percentile 95%",
            "per_seed": True,
        },
        "test_stratification": "fixed q33/q67 of official train OOF adjusted_u; no test percentile normalization",
        "extra_ablation": {
            "variant": "D_no_future",
            "description": "Observed + adjusted reliability, with no predicted future input",
            "raw_and_adjusted_aliases": {"D_raw": "C", "D_adjusted": "D"},
        },
        "prohibited_information": {
            "future_gt_as_input": False,
            "ADE_or_FDE_as_input": False,
            "test_labels_for_training_selection_or_normalization": False,
            "scene_or_social_features": False,
            "test_percentile_normalization": False,
            "gate_design": False,
        },
        "results_scope": "one unified A/B/C/D/E and D_no_future test pass after this file is frozen",
    }
    protocol_path = OUTPUT_ROOT / "protocol_frozen.json"
    if protocol_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {protocol_path}")
    write_json(protocol_path, protocol)
    (OUTPUT_ROOT / "protocol_frozen.sha256").write_text(
        f"{sha256_file(protocol_path)}  protocol_frozen.json\n", encoding="utf-8"
    )

    write_json(OUTPUT_ROOT / "feature_importance.json", collect_importance(model_records))
    print(
        json.dumps(
            {
                "protocol_frozen": str(protocol_path),
                "protocol_sha256": sha256_file(protocol_path),
                "normalization_sha256": normalization_sha,
                "test_labels_read": False,
                "trained_models": len(model_records),
                "device": str(device),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
