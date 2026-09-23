"""Visibility-aware neighbor residual that keeps the fixed intent base untouched."""

from __future__ import annotations

import math

import torch
from torch import nn

from src.models.fixed_base_social_residual import FixedBaseIntentModel


class FixedBaseSocialResidualVisible(nn.Module):
    """Encode neighbor trajectories together with an explicit per-frame visibility bit."""

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
        self.visibility_aware = True
        self.social_hidden_dim = int(social_hidden_dim or base_model.d_model)
        self.neighbor_encoder = nn.GRU(
            input_size=5, hidden_size=self.social_hidden_dim, batch_first=True
        )
        self.social_head = nn.Sequential(
            nn.Linear(self.social_hidden_dim, self.social_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.social_hidden_dim, 1),
        )
        nn.init.zeros_(self.social_head[-1].weight)
        nn.init.zeros_(self.social_head[-1].bias)
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

    def train(self, mode: bool = True) -> FixedBaseSocialResidualVisible:
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
        neighbor_visible_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if neighbor_obs.ndim != 4 or neighbor_obs.shape[-1] != 4:
            raise ValueError(
                f"neighbor_obs must be [B,N,T,4], got {tuple(neighbor_obs.shape)}"
            )
        batch_size, max_neighbors, obs_len, _ = neighbor_obs.shape
        if neighbor_visible_mask.shape != (batch_size, max_neighbors, obs_len):
            raise ValueError(
                "neighbor_visible_mask must be [B,N,T], "
                f"got {tuple(neighbor_visible_mask.shape)}"
            )
        if neighbor_mask.shape != (batch_size, max_neighbors):
            raise ValueError(
                f"neighbor_mask must be [B,N], got {tuple(neighbor_mask.shape)}"
            )
        if target_obs.shape[:2] != (batch_size, obs_len):
            raise ValueError("neighbor_obs batch/time dimensions must match target_obs")

        with torch.no_grad():
            base = self.base_model(target_obs, scene_feat)
        base_logit = base["base_logit"]
        entropy = base["base_entropy"]
        valid_neighbor = neighbor_mask > 0
        has_neighbor = valid_neighbor.any(dim=1)
        visible = neighbor_visible_mask.to(dtype=neighbor_obs.dtype)

        if self.gate_mode == "none":
            social_context = neighbor_obs.new_zeros(
                batch_size, self.social_hidden_dim
            )
            delta_logit = torch.zeros_like(base_logit)
            gate = torch.zeros_like(entropy)
        else:
            masked_neighbor_obs = neighbor_obs * visible.unsqueeze(-1)
            neighbor_features = torch.cat(
                [masked_neighbor_obs, visible.unsqueeze(-1)], dim=-1
            )
            neighbor_input = neighbor_features.reshape(
                batch_size * max_neighbors, obs_len, 5
            )
            _, hidden = self.neighbor_encoder(neighbor_input)
            per_neighbor = hidden[-1].reshape(
                batch_size, max_neighbors, self.social_hidden_dim
            )
            valid = valid_neighbor.to(dtype=per_neighbor.dtype).unsqueeze(-1)
            social_context = (per_neighbor * valid).sum(dim=1)
            social_context = social_context / valid.sum(dim=1).clamp_min(1.0)
            delta_logit = self.social_head(social_context).squeeze(-1)
            if self.gate_mode == "always":
                gate = torch.ones_like(entropy)
            else:
                gate = self.uncertainty_to_gate(entropy)

        effective_gate = gate * has_neighbor.to(dtype=gate.dtype)
        logit_change = self.social_scale * effective_gate * delta_logit
        final_logit = base_logit + logit_change
        return {
            **base,
            "social_context": social_context,
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
