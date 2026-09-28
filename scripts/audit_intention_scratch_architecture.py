#!/usr/bin/env python3
"""Compare M0 modules and tensor shapes against the P1 target path and head."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trajectory_preserving_utils import set_seed
from src.models.intention_scratch_transformer import IntentionScratchTransformer
from src.models.trajectory_preserving_joint import TrajectoryPreservingJoint


def transformer_descriptor(encoder: nn.TransformerEncoder) -> dict[str, Any]:
    layers = encoder.layers
    first = layers[0]
    return {
        "class": type(encoder).__name__,
        "layer_count": len(layers),
        "layers": [
            {
                "class": type(layer).__name__,
                "d_model": layer.self_attn.embed_dim,
                "heads": layer.self_attn.num_heads,
                "feedforward_dimension": layer.linear1.out_features,
                "activation": "gelu",
                "norm_first": bool(layer.norm_first),
                "batch_first": bool(layer.self_attn.batch_first),
                "dropout": float(layer.dropout.p),
            }
            for layer in layers
        ],
        "identical_layer_specs": all(
            layer.self_attn.embed_dim == first.self_attn.embed_dim
            and layer.self_attn.num_heads == first.self_attn.num_heads
            and layer.linear1.out_features == first.linear1.out_features
            and bool(layer.norm_first) == bool(first.norm_first)
            and bool(layer.self_attn.batch_first) == bool(first.self_attn.batch_first)
            for layer in layers
        ),
    }


def main() -> None:
    config = json.loads((ROOT / "configs/intention_scratch_matched.json").read_text(encoding="utf-8"))
    model_cfg = config["model"]["transformer"]
    set_seed(42)
    p1 = TrajectoryPreservingJoint(
        input_dim=8,
        scene_dim=512,
        d_model=int(model_cfg["hidden_dimension"]),
        nhead=int(model_cfg["heads"]),
        num_layers=int(model_cfg["layers"]),
        pred_len=15,
        dropout=float(model_cfg["dropout"]),
        max_obs_len=15,
        intent_input="target",
    ).eval()
    m0 = IntentionScratchTransformer(
        input_dim=8,
        d_model=int(model_cfg["hidden_dimension"]),
        nhead=int(model_cfg["heads"]),
        num_layers=int(model_cfg["layers"]),
        dropout=float(model_cfg["dropout"]),
        max_obs_len=15,
    ).eval()

    p1_state = p1.state_dict()
    m0_state = m0.state_dict()
    mapping: list[dict[str, Any]] = []
    prefixes = (
        ("backbone.input_projection.", "input_projection."),
        ("backbone.temporal_encoder.", "temporal_encoder."),
        ("intent_adapter.", "intent_adapter."),
        ("intent_head.", "intent_head."),
    )
    for p1_prefix, m0_prefix in prefixes:
        for p1_name, tensor in p1_state.items():
            if p1_name.startswith(p1_prefix):
                m0_name = m0_prefix + p1_name[len(p1_prefix) :]
                m0_tensor = m0_state.get(m0_name)
                mapping.append(
                    {
                        "p1_tensor": p1_name,
                        "m0_tensor": m0_name,
                        "p1_shape": list(tensor.shape),
                        "m0_shape": None if m0_tensor is None else list(m0_tensor.shape),
                        "shape_match": m0_tensor is not None and tuple(tensor.shape) == tuple(m0_tensor.shape),
                    }
                )
    p1_position = p1_state["backbone.position_embedding"]
    m0_position = m0_state["position_embedding"]
    mapping.append(
        {
            "p1_tensor": "backbone.position_embedding",
            "m0_tensor": "position_embedding",
            "p1_shape": list(p1_position.shape),
            "m0_shape": list(m0_position.shape),
            "shape_match": tuple(p1_position.shape) == tuple(m0_position.shape),
        }
    )

    target_names = (
        "backbone.input_projection.",
        "backbone.temporal_encoder.",
        "intent_adapter.",
        "intent_head.",
    )
    p1_matched_parameter_count = sum(
        parameter.numel()
        for name, parameter in p1.named_parameters()
        if name.startswith(target_names) or name == "backbone.position_embedding"
    )
    m0_parameter_count = sum(parameter.numel() for parameter in m0.parameters())

    random_input = torch.randn(3, 15, 8)
    with torch.no_grad():
        p1_temporal = p1.backbone.input_projection(random_input)
        p1_temporal = p1_temporal + p1.backbone.position_embedding[:, :15]
        p1_context = p1.backbone.temporal_encoder(p1_temporal)[:, -1]
        p1_logit = p1.intent_head(p1.intent_adapter(p1_context)).squeeze(-1)
        m0_output = m0(random_input)

    checks = {
        "input_shape_match": list(random_input.shape) == [3, 15, 8],
        "projection_weight_shape_match": list(p1.backbone.input_projection.weight.shape)
        == list(m0.input_projection.weight.shape),
        "position_embedding_shape_match": list(p1.backbone.position_embedding.shape)
        == list(m0.position_embedding.shape),
        "transformer_architecture_match": transformer_descriptor(p1.backbone.temporal_encoder)
        == transformer_descriptor(m0.temporal_encoder),
        "target_context_shape_match": list(p1_context.shape) == list(m0_output["target_context"].shape)
        == [3, 128],
        "intention_logit_shape_match": list(p1_logit.shape) == list(m0_output["intent_logit"].shape) == [3],
        "intention_head_parameter_shapes_match": all(row["shape_match"] for row in mapping),
        "matched_parameter_count_equal": p1_matched_parameter_count == m0_parameter_count,
        "m0_has_no_scene_or_trajectory_modules": not hasattr(m0, "scene_encoder") and not hasattr(m0, "decoder"),
    }
    payload = {
        "matched": all(checks.values()),
        "checks": checks,
        "input_shape": ["B", 15, 8],
        "p1_target_encoder": {
            "projection": {"weight": list(p1.backbone.input_projection.weight.shape), "bias": list(p1.backbone.input_projection.bias.shape)},
            "position_embedding": list(p1.backbone.position_embedding.shape),
            "transformer": transformer_descriptor(p1.backbone.temporal_encoder),
            "context_shape": ["B", 128],
        },
        "intention_head": [
            {"name": "LayerNorm", "normalized_shape": list(p1.intent_adapter[0].normalized_shape)},
            {"name": "Linear", "weight": list(p1.intent_adapter[1].weight.shape), "bias": list(p1.intent_adapter[1].bias.shape)},
            {"name": "GELU"},
            {"name": "Dropout", "p": float(p1.intent_adapter[3].p)},
            {"name": "Linear", "weight": list(p1.intent_head.weight.shape), "bias": list(p1.intent_head.bias.shape)},
        ],
        "p1_matched_target_encoder_plus_intention_head_parameter_count": p1_matched_parameter_count,
        "m0_total_parameter_count": m0_parameter_count,
        "p1_total_parameter_count_including_unused_scene_and_trajectory_modules": sum(
            parameter.numel() for parameter in p1.parameters()
        ),
        "matched_tensor_map": mapping,
        "pretrained_weights_loaded_for_audit": False,
        "random_input_shape": list(random_input.shape),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload["architecture_sha256"] = hashlib.sha256(canonical).hexdigest()
    output = ROOT / "results/intention_scratch_matched/architecture_equivalence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"matched": payload["matched"], "checks": checks, "architecture_sha256": payload["architecture_sha256"], "output": str(output)}, ensure_ascii=False, indent=2))
    if not payload["matched"]:
        raise SystemExit("M0 and P1 target-path architecture mismatch; stop before training")


if __name__ == "__main__":
    main()
