"""Intention models that optionally consume frozen trajectory reliability features."""

from __future__ import annotations

import torch
from torch import nn


VARIANT_CONFIG = {
    "A": {"use_future": False, "reliability_dim": 0},
    "B": {"use_future": True, "reliability_dim": 0},
    "C": {"use_future": True, "reliability_dim": 1},
    "D": {"use_future": True, "reliability_dim": 1},
    "E": {"use_future": True, "reliability_dim": 2},
    "D_no_future": {"use_future": False, "reliability_dim": 1},
}


class ReliabilityFeatureIntent(nn.Module):
    """Observed/future GRUs with an optional scalar or vector reliability branch.

    The forward signature intentionally exposes only observed history, predicted
    future coordinates, and the protocol-defined reliability vector. Ground-truth
    future trajectories and trajectory errors are not accepted model inputs.
    """

    def __init__(
        self,
        variant: str,
        *,
        observed_input_dim: int = 8,
        future_input_dim: int = 2,
        observed_hidden_dim: int = 128,
        future_hidden_dim: int = 64,
        reliability_hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_CONFIG:
            raise ValueError(f"unknown reliability feature variant: {variant}")
        self.variant = variant
        self.use_future = VARIANT_CONFIG[variant]["use_future"]
        self.reliability_dim = VARIANT_CONFIG[variant]["reliability_dim"]
        self.observed_encoder = nn.GRU(
            input_size=observed_input_dim,
            hidden_size=observed_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        if self.use_future:
            self.future_encoder = nn.GRU(
                input_size=future_input_dim,
                hidden_size=future_hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        if self.reliability_dim:
            self.reliability_encoder = nn.Sequential(
                nn.Linear(self.reliability_dim, reliability_hidden_dim),
                nn.ReLU(),
                nn.Linear(reliability_hidden_dim, reliability_hidden_dim),
            )

        fusion_dim = observed_hidden_dim
        if self.use_future:
            fusion_dim += future_hidden_dim
        if self.reliability_dim:
            fusion_dim += reliability_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    @property
    def fusion_input_dim(self) -> int:
        return int(self.classifier[0].in_features)

    def forward(
        self,
        target_obs: torch.Tensor,
        future_pred_mean: torch.Tensor | None = None,
        reliability_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if target_obs.ndim != 3 or target_obs.shape[1:] != (15, 8):
            raise ValueError(f"target_obs must have shape [B,15,8], got {tuple(target_obs.shape)}")
        _, observed_state = self.observed_encoder(target_obs)
        pieces = [observed_state[-1]]
        output: dict[str, torch.Tensor] = {"z_obs": pieces[0]}

        if self.use_future:
            if future_pred_mean is None or future_pred_mean.ndim != 3:
                raise ValueError(f"variant {self.variant} requires future_pred_mean [B,15,2]")
            if future_pred_mean.shape != (target_obs.shape[0], 15, 2):
                raise ValueError("future_pred_mean must have shape [B,15,2]")
            _, future_state = self.future_encoder(future_pred_mean)
            z_future = future_state[-1]
            pieces.append(z_future)
            output["z_future"] = z_future
        elif future_pred_mean is not None:
            raise ValueError(f"variant {self.variant} must not receive future_pred_mean")

        if self.reliability_dim:
            if reliability_features is None:
                raise ValueError(f"variant {self.variant} requires reliability_features")
            if reliability_features.ndim != 2 or reliability_features.shape != (
                target_obs.shape[0], self.reliability_dim
            ):
                raise ValueError(
                    f"reliability_features must have shape [B,{self.reliability_dim}]"
                )
            z_reliability = self.reliability_encoder(reliability_features)
            pieces.append(z_reliability)
            output["z_reliability"] = z_reliability
        elif reliability_features is not None:
            raise ValueError(f"variant {self.variant} must not receive reliability_features")

        output["final_logit"] = self.classifier(torch.cat(pieces, dim=-1)).squeeze(-1)
        return output


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
