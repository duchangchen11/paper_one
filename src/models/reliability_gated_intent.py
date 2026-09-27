"""Controlled future-trajectory residual models for crossing-intention prediction.

The only difference between B/C/D is the externally supplied deterministic gate.
The observed-only branch is shared in structure and can be frozen from model A.
"""

from __future__ import annotations

import torch
from torch import nn


class ObservedOnlyIntent(nn.Module):
    """A-only baseline containing no future-trajectory modules or inputs."""

    def __init__(
        self,
        *,
        input_dim: int = 8,
        observed_hidden_dim: int = 128,
        observed_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.observed_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=observed_hidden_dim,
            num_layers=observed_layers,
            dropout=dropout if observed_layers > 1 else 0.0,
            batch_first=True,
        )
        self.base_head = nn.Sequential(
            nn.Linear(observed_hidden_dim, observed_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(observed_hidden_dim // 2, 1),
        )

    def forward(self, target_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        if target_obs.ndim != 3 or target_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"target_obs must have shape [B,T,{self.input_dim}], got {tuple(target_obs.shape)}"
            )
        _, state = self.observed_encoder(target_obs)
        base_logit = self.base_head(state[-1]).squeeze(-1)
        return {"base_logit": base_logit, "final_logit": base_logit}


class ReliabilityGatedIntent(nn.Module):
    """Observed-history baseline plus an optionally gated future-evidence residual."""

    def __init__(
        self,
        *,
        input_dim: int = 8,
        observed_hidden_dim: int = 128,
        observed_layers: int = 2,
        future_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.observed_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=observed_hidden_dim,
            num_layers=observed_layers,
            dropout=dropout if observed_layers > 1 else 0.0,
            batch_first=True,
        )
        self.base_head = nn.Sequential(
            nn.Linear(observed_hidden_dim, observed_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(observed_hidden_dim // 2, 1),
        )
        self.future_encoder = nn.GRU(
            input_size=2,
            hidden_size=future_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.residual_head = nn.Sequential(
            nn.Linear(observed_hidden_dim + future_hidden_dim, observed_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(observed_hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self._base_frozen = False

    def freeze_base(self) -> None:
        """Freeze the observed encoder and head, including their dropout behavior."""
        for module in (self.observed_encoder, self.base_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.observed_encoder.eval()
        self.base_head.eval()
        self._base_frozen = True

    @property
    def base_is_frozen(self) -> bool:
        return all(
            not parameter.requires_grad
            for module in (self.observed_encoder, self.base_head)
            for parameter in module.parameters()
        )

    def train(self, mode: bool = True) -> ReliabilityGatedIntent:
        super().train(mode)
        if self._base_frozen:
            self.observed_encoder.eval()
            self.base_head.eval()
        return self

    def forward(
        self,
        target_obs: torch.Tensor,
        future_pred_mean: torch.Tensor | None = None,
        gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if target_obs.ndim != 3 or target_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"target_obs must have shape [B,T,{self.input_dim}], got {tuple(target_obs.shape)}"
            )
        _, observed_state = self.observed_encoder(target_obs)
        z_obs = observed_state[-1]
        base_logit = self.base_head(z_obs).squeeze(-1)

        if future_pred_mean is None:
            if gate is not None:
                raise ValueError("gate cannot be supplied without future_pred_mean")
            zero = torch.zeros_like(base_logit)
            return {
                "base_logit": base_logit,
                "delta_logit": zero,
                "gate": zero,
                "final_logit": base_logit,
                "z_obs": z_obs,
            }

        if future_pred_mean.ndim != 3 or future_pred_mean.shape[0] != target_obs.shape[0]:
            raise ValueError("future_pred_mean must have shape [B,T,2]")
        if future_pred_mean.shape[-1] != 2:
            raise ValueError("future_pred_mean last dimension must be 2")
        if gate is None:
            raise ValueError("gate is required when future_pred_mean is supplied")
        gate = gate.reshape(-1).to(dtype=base_logit.dtype, device=base_logit.device)
        if gate.shape != base_logit.shape:
            raise ValueError("gate must contain one scalar per sample")

        _, future_state = self.future_encoder(future_pred_mean)
        z_future = future_state[-1]
        delta_logit = self.residual_head(torch.cat([z_obs, z_future], dim=-1)).squeeze(-1)
        final_logit = base_logit + gate * delta_logit
        return {
            "base_logit": base_logit,
            "delta_logit": delta_logit,
            "gate": gate,
            "final_logit": final_logit,
            "z_obs": z_obs,
            "z_future": z_future,
        }


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return a stable parameter count for model-fairness audits."""
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )
