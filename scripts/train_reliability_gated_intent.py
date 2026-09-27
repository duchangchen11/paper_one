#!/usr/bin/env python3
"""Train A/B/C/D controlled intention models and freeze the test protocol."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.reliability_gated_intent import ObservedOnlyIntent, ReliabilityGatedIntent, parameter_count
from scripts.reliability_gated_intent_utils import (
    apply_motion_adjustment,
    array_sha256,
    canonical_json,
    choose_balanced_accuracy_threshold,
    empirical_confidence,
    fit_empirical_reference,
    fit_motion_adjustment,
    fit_temperature,
)

OUTPUT_ROOT = PROJECT_ROOT / "results/reliability_gated_intent_15x15"
CACHE_ROOT = OUTPUT_ROOT / "cache"
CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints/reliability_gated_intent"
SEEDS = (42, 123, 2024)
VARIANTS = ("A_observed_only", "B_always_future", "C_motion_gate", "D_reliability_gate")
SELECTION_TOLERANCE = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def feature_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std()), "median": float(np.median(values)), "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95))}


def prepare_features() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    train = load_cache(CACHE_ROOT / "train_oof_features.npz")
    val = load_cache(CACHE_ROOT / "val_features.npz")
    test = load_cache(CACHE_ROOT / "test_features_unlabeled.npz")
    required = {"target_obs", "future_pred_mean", "u_mean_pixel", "observed_motion_pixel", "image_size", "sample_index", "scene_id", "target_id", "obs_end_frame"}
    if not required.issubset(train) or "intent_label" not in train:
        raise ValueError("train OOF cache is incomplete")
    if not required.issubset(val) or "intent_label" not in val:
        raise ValueError("validation feature cache is incomplete")
    if not required.issubset(test) or "intent_label" in test or "future_gt" in test:
        raise ValueError("test feature cache must remain unlabeled and must not contain future_gt")
    for name, cache in (("train", train), ("val", val)):
        if not set(np.unique(cache["intent_label"]).tolist()).issubset({0, 1}):
            raise ValueError(f"{name} has non-clean intent labels")
    transform = fit_motion_adjustment(train["observed_motion_pixel"], train["u_mean_pixel"])
    train_adjusted = apply_motion_adjustment(train["observed_motion_pixel"], train["u_mean_pixel"], transform)
    val_adjusted = apply_motion_adjustment(val["observed_motion_pixel"], val["u_mean_pixel"], transform)
    test_adjusted = apply_motion_adjustment(test["observed_motion_pixel"], test["u_mean_pixel"], transform)
    motion_cdf = fit_empirical_reference(np.log1p(train["observed_motion_pixel"]), source="official_train_oof_log1p_motion")
    reliability_cdf = fit_empirical_reference(train_adjusted, source="official_train_oof_adjusted_u")
    raw_u_cdf = fit_empirical_reference(train["u_mean_pixel"], source="official_train_oof_raw_u_secondary")
    transform_payload = {
        "polynomial": transform,
        "motion_cdf": {k: v for k, v in motion_cdf.items() if k != "reference"},
        "adjusted_u_cdf": {k: v for k, v in reliability_cdf.items() if k != "reference"},
        "raw_u_cdf_secondary": {k: v for k, v in raw_u_cdf.items() if k != "reference"},
        "cdf_formula": "1 - (count(train_reference <= score) + 0.5) / (N + 1)",
        "fit_source": "official_train_oof_only",
        "validation_or_test_refit": False,
    }
    transform_payload["transform_sha256"] = hashlib.sha256(canonical_json(transform_payload)).hexdigest()
    write_json(OUTPUT_ROOT / "reliability_transform.json", transform_payload)
    # `reference` arrays remain in memory only and are never built from val/test.
    prepared = {
        "train": {**train, "adjusted_u": train_adjusted, "motion_confidence": empirical_confidence(np.log1p(train["observed_motion_pixel"]), motion_cdf), "reliability_confidence": empirical_confidence(train_adjusted, reliability_cdf), "raw_u_confidence": empirical_confidence(train["u_mean_pixel"], raw_u_cdf)},
        "val": {**val, "adjusted_u": val_adjusted, "motion_confidence": empirical_confidence(np.log1p(val["observed_motion_pixel"]), motion_cdf), "reliability_confidence": empirical_confidence(val_adjusted, reliability_cdf), "raw_u_confidence": empirical_confidence(val["u_mean_pixel"], raw_u_cdf)},
        "test": {**test, "adjusted_u": test_adjusted, "motion_confidence": empirical_confidence(np.log1p(test["observed_motion_pixel"]), motion_cdf), "reliability_confidence": empirical_confidence(test_adjusted, reliability_cdf), "raw_u_confidence": empirical_confidence(test["u_mean_pixel"], raw_u_cdf)},
    }
    for split, arrays in prepared.items():
        arrays["reliability_tertile"] = np.searchsorted(np.quantile(train_adjusted, [1 / 3, 2 / 3]), arrays["adjusted_u"], side="right").astype(np.int8)
    audit = {
        split: {
            "sample_count": len(data["u_mean_pixel"]),
            "u_mean_pixel": feature_stats(data["u_mean_pixel"]),
            "observed_motion_pixel": feature_stats(data["observed_motion_pixel"]),
            "adjusted_u": feature_stats(prepared[{"train_oof": "train", "val": "val", "test": "test"}[split]]["adjusted_u"]),
            "adjusted_u_tertile_counts": {str(i): int(np.sum(prepared[{"train_oof": "train", "val": "val", "test": "test"}[split]]["reliability_tertile"] == i)) for i in range(3)},
            "test_intent_label_used": False if split == "test" else None,
        }
        for split, data in (("train_oof", train), ("val", val), ("test", test))
    }
    write_json(OUTPUT_ROOT / "feature_distribution_audit.json", {
        "trajectory_audit": json.loads((OUTPUT_ROOT / "feature_distribution_audit.json").read_text(encoding="utf-8")) if (OUTPUT_ROOT / "feature_distribution_audit.json").exists() else {},
        "reliability_feature_distribution": audit,
        "no_val_or_test_rescaling": True,
        "scale_ratios_vs_train_oof": {
            split: {
                metric: float(audit[split][metric]["median"] / audit["train_oof"][metric]["median"]) if audit["train_oof"][metric]["median"] else None
                for metric in ("u_mean_pixel", "observed_motion_pixel")
            }
            for split in ("val", "test")
        },
    })
    return prepared["train"], prepared["val"], prepared["test"], transform_payload, audit


def create_loader(data: dict[str, np.ndarray], variant: str, *, shuffle: bool, seed: int, batch_size: int = 256) -> DataLoader:
    target = torch.from_numpy(data["target_obs"].astype(np.float32))
    labels = torch.from_numpy(data["intent_label"].astype(np.float32))
    tensors: list[torch.Tensor]
    if variant == "A_observed_only":
        tensors = [target, labels]
    else:
        future = torch.from_numpy(data["future_pred_mean"].astype(np.float32))
        if variant == "B_always_future":
            gate = torch.ones(len(labels), dtype=torch.float32)
        elif variant == "C_motion_gate":
            gate = torch.from_numpy(data["motion_confidence"].astype(np.float32))
        elif variant == "D_reliability_gate":
            gate = torch.from_numpy(data["reliability_confidence"].astype(np.float32))
        else:
            raise ValueError(variant)
        tensors = [target, future, gate, labels]
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def logits_for_batch(model: nn.Module, batch: tuple[torch.Tensor, ...], variant: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if variant == "A_observed_only":
        target, labels = batch
        output = model(target.to(device))
    else:
        target, future, gate, labels = batch
        output = model(target.to(device), future.to(device), gate.to(device))
    return output["final_logit"], labels.to(device)


def evaluate(model: nn.Module, loader: DataLoader, variant: str, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            logit, y = logits_for_batch(model, batch, variant, device)
            logits.append(logit.cpu().numpy())
            labels.append(y.cpu().numpy())
    logit = np.concatenate(logits).astype(np.float64)
    label = np.concatenate(labels).astype(np.int64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
    return {"logits": logit, "labels": label, "roc_auc": float(roc_auc_score(label, probability)), "brier": float(brier_score_loss(label, probability))}


def run_one(variant: str, seed: int, train: dict[str, np.ndarray], val: dict[str, np.ndarray], class_weights: dict[str, float], device: torch.device) -> dict[str, Any]:
    seed_everything(seed)
    train_loader = create_loader(train, variant, shuffle=True, seed=seed)
    val_loader = create_loader(val, variant, shuffle=False, seed=seed)
    if variant == "A_observed_only":
        model: nn.Module = ObservedOnlyIntent().to(device)
    else:
        model = ReliabilityGatedIntent().to(device)
        base_path = CHECKPOINT_ROOT / f"A_observed_only_seed{seed}.pt"
        base_payload = torch.load(base_path, map_location="cpu", weights_only=False)
        base_state = base_payload["base_state"]
        incompatible = model.load_state_dict(base_state, strict=False)
        expected_missing = {key for key in model.state_dict() if key.startswith(("future_encoder.", "residual_head."))}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError("Could not initialize the frozen base branch from model A")
        model.freeze_base()

    initial_branch_hash = None
    if variant != "A_observed_only":
        branch_payload = {
            key: tensor.detach().cpu().numpy()
            for key, tensor in model.state_dict().items()
            if key.startswith(("future_encoder.", "residual_head."))
        }
        initial_branch_hash = hashlib.sha256(b"".join(key.encode() + value.tobytes() for key, value in sorted(branch_payload.items()))).hexdigest()

    learning_rate = 1e-3
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
    pos_weight, neg_weight = class_weights["positive"], class_weights["negative"]
    epochs = 20 if variant == "A_observed_only" else 15
    checkpoint_path = CHECKPOINT_ROOT / f"{variant}_seed{seed}.pt"
    best_auc, best_brier, best_epoch = -np.inf, np.inf, 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for batch in train_loader:
            logits, labels = logits_for_batch(model, batch, variant, device)
            weights = torch.where(labels > 0.5, torch.as_tensor(pos_weight, device=device), torch.as_tensor(neg_weight, device=device))
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(labels)
            train_count += len(labels)
        val_result = evaluate(model, val_loader, variant, device)
        replace_best = val_result["roc_auc"] > best_auc + SELECTION_TOLERANCE
        if abs(val_result["roc_auc"] - best_auc) < SELECTION_TOLERANCE and val_result["brier"] < best_brier:
            replace_best = True
        print(json.dumps({"variant": variant, "seed": seed, "epoch": epoch, "train_loss": train_loss_sum / train_count, "val_auc": val_result["roc_auc"], "val_brier": val_result["brier"]}, ensure_ascii=False), flush=True)
        if replace_best:
            best_auc, best_brier, best_epoch = val_result["roc_auc"], val_result["brier"], epoch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "task_tag": "reliability_gated_intent_15x15_v1",
                "variant": variant,
                "seed": seed,
                "model_state": model.state_dict(),
                "base_state": ({key: value.detach().cpu() for key, value in model.state_dict().items()} if variant == "A_observed_only" else None),
                "selected_epoch": best_epoch,
                "validation_auc": best_auc,
                "validation_brier": best_brier,
                "initial_future_branch_sha256": initial_branch_hash,
                "train_cache_sha256": sha256_file(CACHE_ROOT / "train_oof_features.npz"),
                "validation_cache_sha256": sha256_file(CACHE_ROOT / "val_features.npz"),
                "class_weights": class_weights,
                "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4},
                "epochs_max": epochs,
                "batch_size": 256,
                "selection": "maximum validation ROC-AUC; if within 1e-4, lower validation Brier",
            }
            torch.save(payload, checkpoint_path)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if variant == "A_observed_only":
        selected_model: nn.Module = ObservedOnlyIntent().to(device)
        selected_model.load_state_dict(payload["model_state"], strict=True)
    else:
        selected_model = ReliabilityGatedIntent().to(device)
        selected_model.load_state_dict(payload["model_state"], strict=True)
        selected_model.freeze_base()
    val_eval = evaluate(selected_model, val_loader, variant, device)
    temperature = fit_temperature(val_eval["logits"], val_eval["labels"])
    calibrated = 1.0 / (1.0 + np.exp(-np.clip(val_eval["logits"] / temperature, -60, 60)))
    threshold = choose_balanced_accuracy_threshold(calibrated, val_eval["labels"])
    result = {
        "variant": variant,
        "seed": seed,
        "selected_epoch": int(payload["selected_epoch"]),
        "validation_auc_at_selection": float(payload["validation_auc"]),
        "validation_brier_at_selection": float(payload["validation_brier"]),
        "temperature": temperature,
        "temperature_fit_split": "official_val",
        "threshold": threshold,
        "threshold_fit_split": "official_val",
        "validation_metrics_calibrated": {
            "roc_auc": float(roc_auc_score(val_eval["labels"], calibrated)),
            "brier": float(brier_score_loss(val_eval["labels"], calibrated)),
        },
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "initial_future_branch_sha256": payload.get("initial_future_branch_sha256"),
        "model_parameter_count": parameter_count(selected_model),
        "base_frozen": bool(variant != "A_observed_only" and selected_model.base_is_frozen),
        "model_architecture": "ObservedOnlyIntent" if variant == "A_observed_only" else "ReliabilityGatedIntent",
    }
    result_dir_name = {
        "A_observed_only": "observed_base",
        "B_always_future": "always_future",
        "C_motion_gate": "motion_gate",
        "D_reliability_gate": "reliability_gate",
    }[variant]
    write_json(OUTPUT_ROOT / f"{result_dir_name}_seed{seed}" / "validation_metrics.json", result)
    return result


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    train, val, test, transform_payload, distribution_audit = prepare_features()
    train_labels = train["intent_label"].astype(np.int64)
    count_pos = int((train_labels == 1).sum())
    count_neg = int((train_labels == 0).sum())
    class_weights = {"positive": len(train_labels) / (2.0 * count_pos), "negative": len(train_labels) / (2.0 * count_neg), "positive_count": count_pos, "negative_count": count_neg, "sampling": "natural shuffle=True"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_results: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        for variant in VARIANTS:
            # A must exist before its matched-seed B/C/D models are initialized.
            if variant != "A_observed_only" and f"A_observed_only_seed{seed}" not in run_results:
                a_path = CHECKPOINT_ROOT / f"A_observed_only_seed{seed}.pt"
                if not a_path.exists():
                    raise FileNotFoundError(f"Train model A before {variant}: {a_path}")
            result = run_one(variant, seed, train, val, class_weights, device)
            run_results[f"{variant}_seed{seed}"] = result

    for seed in SEEDS:
        matched = [run_results[f"{variant}_seed{seed}"] for variant in VARIANTS[1:]]
        initial_hashes = {row["initial_future_branch_sha256"] for row in matched}
        parameter_counts = {row["model_parameter_count"] for row in matched}
        if len(initial_hashes) != 1 or len(parameter_counts) != 1 or not all(row["base_frozen"] for row in matched):
            raise RuntimeError(f"B/C/D fairness invariant failed for seed {seed}")

    # All selected checkpoints, temperatures and thresholds are frozen before test labels are read.
    crossfit_path = OUTPUT_ROOT / "crossfit_manifest.json"
    feature_manifest_path = OUTPUT_ROOT / "feature_cache_manifest.json"
    trajectory_summary_path = OUTPUT_ROOT / "trajectory_crossfit_summary.json"
    if not all(path.is_file() for path in (crossfit_path, feature_manifest_path, trajectory_summary_path)):
        raise FileNotFoundError("Required cross-fit/cache manifests are missing")
    adjusted = train["adjusted_u"]
    cutpoints = np.quantile(adjusted, [1 / 3, 2 / 3], method="linear")
    definitions = {
        "A_observed_only": {"input": "15x8 observed target history only", "gate": None, "trainable": "observed GRU and base head"},
        "B_always_future": {"input": "observed history + ensemble-mean predicted future [15,2]", "gate": 1.0, "trainable": "future GRU and residual head; matched A base frozen"},
        "C_motion_gate": {"input": "observed history + ensemble-mean predicted future [15,2]", "gate": "train-defined motion empirical confidence", "trainable": "future GRU and residual head; matched A base frozen"},
        "D_reliability_gate": {"input": "observed history + ensemble-mean predicted future [15,2]", "gate": "train-defined motion-adjusted disagreement empirical confidence", "trainable": "future GRU and residual head; matched A base frozen"},
    }
    protocol = {
        "protocol": "reliability-gated future-trajectory intention experiment; A/B/C/D primary",
        "data": {
            "train_npz_sha256": json.loads(crossfit_path.read_text(encoding="utf-8"))["train_npz_sha256"],
            "validation_npz_sha256": json.loads(crossfit_path.read_text(encoding="utf-8"))["validation_npz_sha256"],
            "test_npz_sha256": json.loads(feature_manifest_path.read_text(encoding="utf-8"))["test_npz_sha256"],
            "test_intent_labels_read_before_freeze": False,
            "test_feature_cache_sha256": sha256_file(CACHE_ROOT / "test_features_unlabeled.npz"),
        },
        "crossfit_manifest_sha256": sha256_file(crossfit_path),
        "feature_cache_manifest_sha256": sha256_file(feature_manifest_path),
        "trajectory_crossfit_checkpoints_sha256": sha256_file(trajectory_summary_path),
        "full_train_trajectory_checkpoint_sha256": json.loads(feature_manifest_path.read_text(encoding="utf-8"))["full_train_zero_scene_checkpoint_sha256"],
        "reliability_transform_sha256": sha256_file(OUTPUT_ROOT / "reliability_transform.json"),
        "reliability_transform": transform_payload,
        "variant_definitions": definitions,
        "class_weighting": class_weights,
        "intent_training": {"batch_size": 256, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "gradient_clip_norm": 5.0, "sampling": "natural shuffle=True", "selection": "max val ROC-AUC; if within 1e-4 select lower val Brier"},
        "selected_models": run_results,
        "reliability_tertile_cutpoints_train_oof_adjusted_u": [float(value) for value in cutpoints],
        "primary_test_metrics_planned": ["ROC-AUC", "Brier", "ECE(15 equal-width bins)", "Balanced Accuracy", "F1", "negative recall/specificity"],
        "paired_comparisons": ["D-A", "D-B", "D-C", "B-A", "C-B"],
        "bootstrap": {"unit": "scene_id video cluster", "repetitions": 2000, "seed": 9124, "ci": "percentile 95%", "paired": True},
        "gate_shuffle": {"within_bins": 10, "bin_cutpoints_source": "official_train_oof motion quantiles", "permutations": 200, "seed": 9124},
        "future_gt_or_ade_or_fde_as_intention_input_loss_gate_or_selection": False,
        "trajectory_predictors_fine_tuned_for_intention": False,
        "test_selection": False,
    }
    protocol_path = OUTPUT_ROOT / "protocol_frozen.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise FileExistsError("A different frozen protocol already exists; refusing to overwrite it")
    else:
        write_json(protocol_path, protocol)
        (OUTPUT_ROOT / "protocol_frozen.sha256").write_text(sha256_file(protocol_path) + "  protocol_frozen.json\n", encoding="utf-8")
    print(json.dumps({"protocol_frozen": str(protocol_path), "protocol_sha256": sha256_file(protocol_path), "models_frozen": len(run_results), "test_labels_read": False}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
