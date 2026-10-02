"""Trajectory prediction from a residual Mamba history encoder."""

from __future__ import annotations

import torch
from torch import nn

from src.models.mamba_sequence_encoder import ResidualMambaEncoder
from src.models.trajectory_transformer_target_only import make_trajectory_decoder


class MambaTrajectoryPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 8,
        d_model: int = 128,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        pred_len: int = 15,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.encoder = ResidualMambaEncoder(
            input_dim=input_dim, d_model=d_model, num_layers=num_layers,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
        )
        self.decoder = make_trajectory_decoder(d_model, pred_len, dropout)

    def forward(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        _, context = self.encoder(target)
        future_pred = self.decoder(context).reshape(target.shape[0], self.pred_len, 2)
        return {"future_pred": future_pred, "target_context": context}
