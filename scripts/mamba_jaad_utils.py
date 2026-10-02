"""Matched train/validation tools for the first JAAD Mamba experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reliability_gated_intent_utils import choose_balanced_accuracy_threshold, fit_temperature
from scripts.trajectory_preserving_utils import (
    SEEDS, TrajectoryIntentDataset, intention_metrics, set_seed, sha256_file,
    tensor_state_sha256, trajectory_metrics,
)
from src.models.mamba_intention import MambaIntentionPredictor
from src.models.mamba_trajectory import MambaTrajectoryPredictor
from src.models.trajectory_transformer_target_only import TargetOnlyTrajectoryTransformer

CONFIG_PATH = ROOT / "configs/mamba_baselines.json"
RESULTS_ROOT = ROOT / "results/mamba_baselines"
CHECKPOINT_ROOT = ROOT / "checkpoints/mamba_baselines"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def environment_report() -> dict[str, Any]:
    versions = {}
    for package in ("torch", "mamba-ssm", "triton", "causal-conv1d", "einops", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python": platform.python_version(), "packages": versions,
        "host_libc": platform.libc_ver(), "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
        "float_dtype": "float32", "amp": False,
        "torch_cpu_threads": torch.get_num_threads(),
        "optional_causal_conv1d_installed": versions["causal-conv1d"] is not None,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def initialize_matched_decoder(model: nn.Module, seed: int) -> str:
    """Reset only the shared decoder design with an independent CPU RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 104729)
        for module in model.decoder.modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm)):
                module.reset_parameters()
    return tensor_state_sha256(model.decoder.state_dict())


def make_model(method: str, config: dict[str, Any], num_layers: int) -> nn.Module:
    cfg = config["model"]
    common = dict(input_dim=cfg["input_dim"], d_model=cfg["d_model"], num_layers=num_layers, dropout=cfg["dropout"])
    if method == "trajectory_transformer_target":
        return TargetOnlyTrajectoryTransformer(
            **common, nhead=cfg["transformer"]["nhead"],
            pred_len=cfg["prediction_length"], max_obs_len=cfg["observed_length"],
        )
    if method == "trajectory_mamba":
        return MambaTrajectoryPredictor(**common, **cfg["mamba"], pred_len=cfg["prediction_length"])
    if method == "intention_mamba":
        return MambaIntentionPredictor(**common, **cfg["mamba"])
    raise ValueError(f"Unknown baseline: {method}")


def load_datasets() -> tuple[TrajectoryIntentDataset, TrajectoryIntentDataset, dict[str, Any]]:
    data_root = ROOT / "data/processed/jaad_sequences_scene_15x15"
    train_path, val_path = data_root / "train.npz", data_root / "val.npz"
    train, val = TrajectoryIntentDataset(train_path), TrajectoryIntentDataset(val_path)
    for split, dataset in (("train", train), ("val", val)):
        if tuple(dataset.target.shape[1:]) != (15, 8) or tuple(dataset.future_gt.shape[1:]) != (15, 2):
            raise RuntimeError(f"Unexpected {split} target/future shape")
        for array in (dataset.target, dataset.future_gt, dataset.intent_label, dataset.image_size):
            if not torch.isfinite(array).all():
                raise RuntimeError(f"Nonfinite {split} data")
        if not torch.all((dataset.intent_label == 0) | (dataset.intent_label == 1)):
            raise RuntimeError(f"Invalid {split} intention labels")
    provenance = {
        "train_path": str(train_path.relative_to(ROOT)), "validation_path": str(val_path.relative_to(ROOT)),
        "train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(val_path),
        "train_samples": len(train), "validation_samples": len(val),
        "target_shape": [15, 8], "future_shape": [15, 2],
        "data_splits_loaded": ["train", "val"], "test_accessed": False,
        "split": "existing JAAD default video split, arrays unchanged",
        "scene_feat_consumed_by_model": False,
    }
    return train, val, provenance


def inverse_frequency_weights(labels: torch.Tensor) -> dict[str, float | int]:
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if min(positive, negative) == 0:
        raise ValueError("Training split must contain both intention classes")
    return {
        "positive_count": positive, "negative_count": negative,
        "positive": len(labels) / (2.0 * positive), "negative": len(labels) / (2.0 * negative),
    }


def weighted_intention_loss(logits: torch.Tensor, labels: torch.Tensor, weights: dict[str, Any]) -> torch.Tensor:
    per_sample = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    sample_weights = torch.where(labels > 0.5, weights["positive"], weights["negative"])
    return (per_sample * sample_weights).mean()


def intention_checkpoint_is_better(auc: float, brier: float, best_auc: float, best_brier: float, tolerance: float) -> bool:
    return auc > best_auc + tolerance or (abs(auc - best_auc) <= tolerance and brier < best_brier)


@torch.no_grad()
def evaluate_trajectory(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    prediction, target, image_size = [], [], []
    loss_sum, count = 0.0, 0
    for batch in loader:
        pred = model(batch["target"].to(device))["future_pred"]
        truth = batch["future_gt"].to(device)
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite validation trajectory prediction")
        loss_sum += float(nn.functional.smooth_l1_loss(pred, truth)) * len(pred)
        count += len(pred)
        prediction.append(pred.cpu().numpy())
        target.append(batch["future_gt"].numpy())
        image_size.append(batch["image_size"].numpy())
    return {
        **trajectory_metrics(np.concatenate(prediction), np.concatenate(target), np.concatenate(image_size)),
        "normalized_loss": loss_sum / count,
    }


@torch.no_grad()
def evaluate_intention(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        output = model(batch["target"].to(device))["intent_logit"]
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite validation intention prediction")
        logits.append(output.cpu().numpy())
        labels.append(batch["intent_label"].numpy())
    logit_array = np.concatenate(logits).astype(np.float64)
    label_array = np.concatenate(labels).astype(np.int64)
    return intention_metrics(label_array, logit_array), logit_array, label_array


@torch.no_grad()
def benchmark_inference(model: nn.Module, target: torch.Tensor, device: torch.device) -> dict[str, Any]:
    model.eval()
    results = {"dtype": "float32", "device": str(device), "warmup_iterations": 10, "measured_iterations": 30,
               "excludes_host_to_device_transfer": True, "sequence_length": int(target.shape[1]), "measurements": {}}
    for name, data in (("single_sample", target[:1]), ("batch", target)):
        data = data.to(device)
        for _ in range(10):
            model(data)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timings = []
        for _ in range(30):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            model(data)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(begin.elapsed_time(end))
        results["measurements"][name] = {
            "batch_size": len(data), "mean_ms": float(np.mean(timings)), "median_ms": float(np.median(timings)),
            "p90_ms": float(np.percentile(timings, 90)), "amortized_ms_per_sample": float(np.mean(timings) / len(data)),
            "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
    return results


def train_baseline(method: str) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--num-layers", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    if method != "trajectory_mamba" and args.num_layers != 3:
        parser.error("The single permitted layer sweep applies only to trajectory Mamba")
    if not torch.cuda.is_available():
        raise RuntimeError("Official baseline experiments require the CUDA smoke-tested environment")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    seed = args.seed
    is_intention = method == "intention_mamba"
    run_method = method if args.num_layers == 3 else f"{method}_layers{args.num_layers}"
    run_dir = RESULTS_ROOT / run_method / f"seed{seed}"
    checkpoint = CHECKPOINT_ROOT / run_method / f"seed{seed}.pt"
    if (run_dir / "validation_history.json").exists() or checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {run_method}/seed{seed}")
    set_seed(seed)
    device = torch.device("cuda")
    train, val, provenance = load_datasets()
    train_cfg = config["intention_training" if is_intention else "trajectory_training"]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_loader = DataLoader(train, batch_size=train_cfg["batch_size"], shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=0)
    model = make_model(method, config, args.num_layers)
    decoder_sha = initialize_matched_decoder(model, seed) if not is_intention else None
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Every baseline parameter must be trainable")
    initial_sha = tensor_state_sha256(model.state_dict())
    counts = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    initialization = {
        "method": run_method, "seed": seed, "num_layers": args.num_layers,
        "parameter_count": counts, "model_sha256_before_training": initial_sha,
        "decoder_initialization_sha256": decoder_sha,
        "pretrained_checkpoint_loaded": False, "input_shape": [15, 8],
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad],
    }
    write_json(run_dir / "initialization_report.json", initialization)
    write_json(run_dir / "parameter_count.json", counts)
    write_json(run_dir / "data_provenance.json", provenance)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = None if is_intention else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **train_cfg["scheduler"])
    weights = inverse_frequency_weights(train.intent_label) if is_intention else None
    best_auc, best_brier, best_ade, best_epoch = -float("inf"), float("inf"), float("inf"), 0
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        loss_sum, count, max_gradient_norm = 0.0, 0, 0.0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch in train_loader:
            output = model(batch["target"].to(device))
            if is_intention:
                loss = weighted_intention_loss(output["intent_logit"], batch["intent_label"].to(device), weights)
            else:
                loss = nn.functional.smooth_l1_loss(output["future_pred"], batch["future_gt"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {run_method} seed{seed} epoch{epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), train_cfg["gradient_clip_norm"], error_if_nonfinite=True)
            max_gradient_norm = max(max_gradient_norm, float(gradient_norm))
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch["target"])
            count += len(batch["target"])
        if is_intention:
            metrics, _, _ = evaluate_intention(model, val_loader, device)
            select = intention_checkpoint_is_better(metrics["roc_auc"], metrics["brier"], best_auc, best_brier, train_cfg["selection_tolerance"])
            if select:
                best_auc, best_brier = metrics["roc_auc"], metrics["brier"]
        else:
            metrics = evaluate_trajectory(model, val_loader, device)
            select = metrics["ade_pixel"] < best_ade
            if select:
                best_ade = metrics["ade_pixel"]
            scheduler.step(metrics["ade_pixel"])
        row = {"epoch": epoch, "train_loss": loss_sum / count, "validation": metrics,
               "learning_rate": learning_rate, "max_gradient_norm_before_clip": max_gradient_norm,
               "all_losses_and_gradients_finite": True}
        history.append(row)
        write_json(run_dir / "validation_history.json", {"method": run_method, "seed": seed, "epochs": history})
        print(json.dumps({"method": run_method, "seed": seed, **row}), flush=True)
        if select:
            best_epoch = epoch
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "method": method, "seed": seed, "num_layers": args.num_layers,
                        "selected_epoch": epoch, "config": config, "validation": metrics, "initialization": initialization}, checkpoint)
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - start
    training_peak = torch.cuda.max_memory_allocated(device) / 1024**2
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    calibration = None
    if is_intention:
        selected, logits, labels = evaluate_intention(model, val_loader, device)
        temperature = fit_temperature(logits, labels)
        from scripts.trajectory_preserving_utils import probabilities_from_logits
        threshold = choose_balanced_accuracy_threshold(probabilities_from_logits(logits, temperature), labels)
        calibration = {"temperature": temperature, "threshold": threshold, "fit_split": "validation",
                       "metrics": intention_metrics(labels, logits, temperature=temperature, threshold=threshold)}
    else:
        selected = evaluate_trajectory(model, val_loader, device)
    # Measure inference memory after releasing training state and the loaded
    # checkpoint copy, rather than counting Adam moments as inference memory.
    model.zero_grad(set_to_none=True)
    del optimizer, scheduler, payload, output, loss, gradient_norm
    benchmark = benchmark_inference(model, val.target[:train_cfg["batch_size"]], device)
    result = {
        "method": run_method, "seed": seed, "num_layers": args.num_layers, "best_epoch": best_epoch,
        "selected_validation_metrics": selected, "selected_validation_calibration": calibration,
        "parameter_count": counts, "initialization": initialization,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": sha256_file(checkpoint),
        "data_provenance": provenance, "config_sha256": sha256_file(CONFIG_PATH), "training": train_cfg,
        "class_weights": weights, "environment": environment_report(),
        "training_seconds": training_seconds, "training_gpu_peak_allocated_mb": training_peak,
        "all_losses_and_gradients_finite": True, "test_accessed": False,
        "test_evaluation_status": "withheld_until_protocol_freeze",
    }
    write_json(run_dir / "inference_benchmark.json", benchmark)
    write_json(run_dir / "metrics_validation.json", result)
    print(json.dumps({"completed": run_method, "seed": seed, "best_epoch": best_epoch, "validation": selected}), flush=True)
