"""Frozen Transformer with uncertainty-guided social residual branches."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.trajectory_transformer import SceneTrajectoryTransformer


class ResidualSocialJointModel(nn.Module):
    """Wrap a pretrained scene trajectory model without changing its decoder."""

    def __init__(
        self,
        *,
        input_dim: int = 8,
        scene_dim: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        pred_len: int = 15,
        max_obs_len: int = 15,
        dropout: float = 0.1,
        gate_mode: str = "uncertainty",
        trajectory_residual_scale: float = 0.1,
        enable_trajectory_residual: bool = False,
        trajectory_backbone: SceneTrajectoryTransformer | None = None,
        freeze_trajectory_backbone: bool = True,
    ) -> None:
        super().__init__()
        if gate_mode not in {"none", "always", "uncertainty"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        if not freeze_trajectory_backbone:
            raise ValueError("This experiment requires a fully frozen trajectory backbone")

        self.trajectory_backbone = trajectory_backbone or SceneTrajectoryTransformer(
            input_dim=input_dim,
            scene_dim=scene_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pred_len=pred_len,
            dropout=dropout,
            max_obs_len=max_obs_len,
        )
        self.input_dim = input_dim
        self.scene_dim = scene_dim
        self.d_model = self.trajectory_backbone.input_projection.out_features
        self.pred_len = self.trajectory_backbone.pred_len
        self.gate_mode = gate_mode
        self.trajectory_residual_scale = float(trajectory_residual_scale)
        self.enable_trajectory_residual = bool(enable_trajectory_residual)

        for parameter in self.trajectory_backbone.parameters():
            parameter.requires_grad_(False)
        self.trajectory_backbone.eval()

        self.neighbor_encoder = nn.GRU(
            input_size=4,
            hidden_size=self.d_model,
            batch_first=True,
        )
        self.prior_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prior_head = nn.Linear(self.d_model, 1)

        self.intent_base_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.social_projection = nn.Linear(self.d_model, self.d_model)
        self.intent_head = nn.Linear(self.d_model, 1)

        residual_hidden = self.d_model * 2
        self.social_trajectory_residual_head = nn.Sequential(
            nn.Linear(self.d_model * 3, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, self.pred_len * 2),
        )
        nn.init.zeros_(self.social_trajectory_residual_head[-1].weight)
        nn.init.zeros_(self.social_trajectory_residual_head[-1].bias)

        # softplus(raw_slope) is strictly positive; sigmoid(raw_threshold) is in (0, 1).
        self.raw_gate_slope = nn.Parameter(torch.tensor(math.log(math.expm1(5.0))))
        self.raw_gate_threshold = nn.Parameter(torch.tensor(0.0))

    @property
    def trajectory_backbone_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.trajectory_backbone.parameters())

    @property
    def gate_slope(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_gate_slope) + 1e-6

    @property
    def gate_threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate_threshold)

    def train(self, mode: bool = True) -> ResidualSocialJointModel:
        super().train(mode)
        # Frozen BatchNorm/dropout behavior must remain identical to the pretrained model.
        self.trajectory_backbone.eval()
        return self

    def uncertainty_to_gate(self, normalized_entropy: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            self.gate_slope * (normalized_entropy - self.gate_threshold)
        )

    def _backbone_contexts(
        self, target_obs: torch.Tensor, scene_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone = self.trajectory_backbone
        with torch.no_grad():
            obs_len = target_obs.shape[1]
            temporal = backbone.input_projection(target_obs)
            temporal = temporal + backbone.position_embedding[:, :obs_len]
            encoded = backbone.temporal_encoder(temporal)
            target_context = encoded[:, -1]
            scene_context = backbone.scene_encoder(scene_feat)
            # Keep the pretrained forward path as the sole source of the base prediction.
            base_future_pred = backbone(target_obs, scene_feat)
        return target_context, scene_context, base_future_pred

    def forward(
        self,
        target_obs: torch.Tensor,
        neighbor_obs: torch.Tensor,
        neighbor_mask: torch.Tensor,
        neighbor_visible_mask: torch.Tensor,
        scene_feat: torch.Tensor,
        *,
        enable_trajectory_residual: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if target_obs.ndim != 3 or target_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"target_obs must be [B,T,{self.input_dim}], got {tuple(target_obs.shape)}"
            )
        if scene_feat.ndim != 2 or scene_feat.shape != (target_obs.shape[0], self.scene_dim):
            raise ValueError(
                f"scene_feat must be [B,{self.scene_dim}], got {tuple(scene_feat.shape)}"
            )
        if neighbor_obs.ndim != 4 or neighbor_obs.shape[-1] != 4:
            raise ValueError(
                "neighbor_obs must follow [B,N,T,4]; "
                f"got shape {tuple(neighbor_obs.shape)}"
            )

        batch_size, max_neighbors, obs_len, _ = neighbor_obs.shape
        if (batch_size, obs_len) != tuple(target_obs.shape[:2]):
            raise ValueError(
                "neighbor_obs batch/time axes must match target_obs: "
                f"expected [B, N, {target_obs.shape[1]}, 4], "
                f"got {tuple(neighbor_obs.shape)}"
            )
        if neighbor_mask.shape != (batch_size, max_neighbors):
            raise ValueError(
                f"neighbor_mask must be [B,N], got {tuple(neighbor_mask.shape)}"
            )
        if neighbor_visible_mask.shape != (batch_size, max_neighbors, obs_len):
            raise ValueError(
                "neighbor_visible_mask must be [B,N,T], "
                f"got {tuple(neighbor_visible_mask.shape)}"
            )

        target_context, scene_context, base_future_pred = self._backbone_contexts(
            target_obs, scene_feat
        )

        neighbor_input = neighbor_obs.reshape(
            batch_size * max_neighbors, obs_len, 4
        )
        _, neighbor_hidden = self.neighbor_encoder(neighbor_input)
        per_neighbor_context = neighbor_hidden[-1].reshape(
            batch_size, max_neighbors, self.d_model
        )
        valid_neighbor = (neighbor_mask > 0).to(per_neighbor_context.dtype).unsqueeze(-1)
        social_context = (per_neighbor_context * valid_neighbor).sum(dim=1)
        social_context = social_context / valid_neighbor.sum(dim=1).clamp_min(1.0)

        prior_context = self.prior_fusion(torch.cat([target_context, scene_context], dim=-1))
        prior_logit = self.prior_head(prior_context).squeeze(-1)
        prior_probability = torch.sigmoid(prior_logit)
        epsilon = torch.finfo(prior_probability.dtype).eps
        safe_probability = prior_probability.clamp(epsilon, 1.0 - epsilon)
        entropy = -(
            safe_probability * safe_probability.log()
            + (1.0 - safe_probability) * (1.0 - safe_probability).log()
        )
        normalized_entropy = entropy / math.log(2.0)

        if self.gate_mode == "none":
            gate = torch.zeros_like(normalized_entropy)
        elif self.gate_mode == "always":
            gate = torch.ones_like(normalized_entropy)
        else:
            gate = self.uncertainty_to_gate(normalized_entropy)

        base_intent_context = self.intent_base_fusion(
            torch.cat([target_context, scene_context], dim=-1)
        )
        social_residual = self.social_projection(social_context)
        intent_context = base_intent_context + gate.unsqueeze(-1) * social_residual
        intent_logit = self.intent_head(intent_context).squeeze(-1)

        residual_enabled = (
            self.enable_trajectory_residual
            if enable_trajectory_residual is None
            else enable_trajectory_residual
        )
        if residual_enabled:
            residual_input = torch.cat(
                [target_context, scene_context, social_context], dim=-1
            )
            delta_future = self.social_trajectory_residual_head(residual_input).view(
                batch_size, self.pred_len, 2
            )
            future_pred = (
                base_future_pred
                + self.trajectory_residual_scale
                * gate.view(batch_size, 1, 1)
                * delta_future
            )
        else:
            delta_future = torch.zeros_like(base_future_pred)
            future_pred = base_future_pred

        return {
            "base_future_pred": base_future_pred,
            "future_pred": future_pred,
            "delta_future": delta_future,
            "prior_logit": prior_logit,
            "prior_probability": prior_probability,
            "entropy": normalized_entropy,
            "normalized_entropy": normalized_entropy,
            "gate": gate,
            "intent_logit": intent_logit,
            "intent_probability": torch.sigmoid(intent_logit),
        }
