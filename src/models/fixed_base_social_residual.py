"""Fixed base intent classifier with a controlled neighbor-only logit residual."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.trajectory_transformer import SceneTrajectoryTransformer


class FixedBaseIntentModel(nn.Module):
    """Intent classifier over a permanently frozen scene trajectory Transformer."""

    def __init__(
        self,
        trajectory_backbone: SceneTrajectoryTransformer,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trajectory_backbone = trajectory_backbone
        self.d_model = trajectory_backbone.input_projection.out_features
        self.input_dim = trajectory_backbone.input_projection.in_features
        self.scene_dim = trajectory_backbone.scene_encoder[0].in_features
        self.pred_len = trajectory_backbone.pred_len
        for parameter in self.trajectory_backbone.parameters():
            parameter.requires_grad_(False)
        self.trajectory_backbone.eval()

        self.base_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(self.d_model, 1)
        self.register_buffer("temperature", torch.ones(()))

    @property
    def trajectory_backbone_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.trajectory_backbone.parameters())

    @property
    def base_classifier_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for module in (self.base_fusion, self.base_head)
            for parameter in module.parameters()
        )

    def freeze_base_classifier(self) -> None:
        for module in (self.base_fusion, self.base_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FixedBaseIntentModel:
        super().train(mode)
        self.trajectory_backbone.eval()
        return self

    def encode_contexts(
        self, target_obs: torch.Tensor, scene_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            backbone = self.trajectory_backbone
            obs_len = target_obs.shape[1]
            temporal = backbone.input_projection(target_obs)
            temporal = temporal + backbone.position_embedding[:, :obs_len]
            encoded = backbone.temporal_encoder(temporal)
            target_context = encoded[:, -1]
            scene_context = backbone.scene_encoder(scene_feat)
            future_pred = backbone(target_obs, scene_feat)
        return target_context, scene_context, future_pred

    def forward(self, target_obs: torch.Tensor, scene_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        if target_obs.ndim != 3 or target_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"target_obs must be [B,T,{self.input_dim}], got {tuple(target_obs.shape)}"
            )
        if scene_feat.shape != (target_obs.shape[0], self.scene_dim):
            raise ValueError(
                f"scene_feat must be [B,{self.scene_dim}], got {tuple(scene_feat.shape)}"
            )
        target_context, scene_context, future_pred = self.encode_contexts(target_obs, scene_feat)
        fused_context = self.base_fusion(torch.cat([target_context, scene_context], dim=-1))
        base_raw_logit = self.base_head(fused_context).squeeze(-1)
        calibrated_base_logit = base_raw_logit / self.temperature.clamp(min=1e-4)
        calibrated_probability = torch.sigmoid(calibrated_base_logit)
        eps = torch.finfo(calibrated_probability.dtype).eps
        p = calibrated_probability.clamp(eps, 1.0 - eps)
        normalized_entropy = -(
            p * p.log() + (1.0 - p) * (1.0 - p).log()
        ) / math.log(2.0)
        return {
            "target_context": target_context,
            "scene_context": scene_context,
            "base_context": fused_context,
            "base_raw_logit": base_raw_logit,
            "base_logit": calibrated_base_logit,
            "base_raw_probability": torch.sigmoid(base_raw_logit),
            "base_probability": calibrated_probability,
            "base_entropy": normalized_entropy,
            "future_pred": future_pred,
        }


class FixedBaseSocialResidual(nn.Module):
    """Freeze one base model and learn only a neighbor-conditioned logit residual."""

    def __init__(
        self,
        base_model: FixedBaseIntentModel,
        *,
        gate_mode: str = "uncertainty",
        social_hidden_dim: int | None = None,
        dropout: float = 0.1,
        social_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"none", "always", "uncertainty"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.base_model = base_model
        self.base_model.freeze_base_classifier()
        self.gate_mode = gate_mode
        self.social_scale = float(social_scale)
        hidden_dim = int(social_hidden_dim or base_model.d_model)
        self.neighbor_encoder = nn.GRU(input_size=4, hidden_size=hidden_dim, batch_first=True)
        self.social_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.social_head[-1].weight)
        nn.init.zeros_(self.social_head[-1].bias)

        # Uncertainty gate is constrained to be monotone increasing in calibrated entropy.
        self.raw_gate_slope = nn.Parameter(torch.tensor(math.log(math.expm1(5.0))))
        self.raw_gate_threshold = nn.Parameter(torch.tensor(0.0))
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    @property
    def all_base_parameters_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.base_model.parameters())

    @property
    def gate_slope(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_gate_slope) + 1e-6

    @property
    def gate_threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate_threshold)

    def train(self, mode: bool = True) -> FixedBaseSocialResidual:
        super().train(mode)
        self.base_model.eval()
        return self

    def uncertainty_to_gate(self, entropy: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate_slope * (entropy - self.gate_threshold))

    def forward(
        self,
        target_obs: torch.Tensor,
        scene_feat: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if neighbor_obs.ndim != 4 or neighbor_obs.shape[-1] != 4:
            raise ValueError(f"neighbor_obs must be [B,N,T,4], got {tuple(neighbor_obs.shape)}")
        batch_size, max_neighbors, obs_len, _ = neighbor_obs.shape
        if neighbor_mask.shape != (batch_size, max_neighbors):
            raise ValueError(f"neighbor_mask must be [B,N], got {tuple(neighbor_mask.shape)}")
        if target_obs.shape[:2] != (batch_size, obs_len):
            raise ValueError("neighbor_obs batch/time dimensions must match target_obs")

        with torch.no_grad():
            base = self.base_model(target_obs, scene_feat)
        base_logit = base["base_logit"]
        base_probability = base["base_probability"]
        entropy = base["base_entropy"]
        valid_neighbor = (neighbor_mask > 0)
        has_neighbor = valid_neighbor.any(dim=1)

        if self.gate_mode == "none":
            gate = torch.zeros_like(entropy)
            delta_logit = torch.zeros_like(base_logit)
        else:
            neighbor_input = neighbor_obs.reshape(batch_size * max_neighbors, obs_len, 4)
            _, hidden = self.neighbor_encoder(neighbor_input)
            per_neighbor = hidden[-1].reshape(batch_size, max_neighbors, -1)
            mask = valid_neighbor.to(per_neighbor.dtype).unsqueeze(-1)
            social_context = (per_neighbor * mask).sum(dim=1)
            social_context = social_context / mask.sum(dim=1).clamp_min(1.0)
            delta_logit = self.social_head(social_context).squeeze(-1)
            if self.gate_mode == "always":
                gate = torch.ones_like(entropy)
            else:
                gate = self.uncertainty_to_gate(entropy)

        effective_gate = gate * has_neighbor.to(gate.dtype)
        logit_change = self.social_scale * effective_gate * delta_logit
        final_logit = base_logit + logit_change
        return {
            **base,
            "base_logit": base_logit,
            "final_logit": final_logit,
            "final_probability": torch.sigmoid(final_logit),
            "delta_logit": delta_logit,
            "logit_change": logit_change,
            "gate": gate,
            "effective_gate": effective_gate,
            "has_neighbor": has_neighbor,
            "neighbor_count": valid_neighbor.sum(dim=1),
        }
