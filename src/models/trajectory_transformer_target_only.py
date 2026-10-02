"""Input-matched Transformer temporal control for JAAD trajectory prediction."""

from __future__ import annotations

import torch
from torch import nn


def make_trajectory_decoder(d_model: int, pred_len: int, dropout: float) -> nn.Sequential:
    """The identical decoder factory used by TT and MT."""
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model, pred_len * 2),
    )


class TargetOnlyTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        pred_len: int = 15,
        dropout: float = 0.1,
        max_obs_len: int = 15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.max_obs_len = max_obs_len
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_obs_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.decoder = make_trajectory_decoder(d_model, pred_len, dropout)

    def forward(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if target.ndim != 3 or target.shape[-1] != self.input_dim or target.shape[1] < 1:
            raise ValueError(f"Expected nonempty target [B,T,{self.input_dim}], got {tuple(target.shape)}")
        if target.shape[1] > self.max_obs_len:
            raise ValueError(f"Observed length exceeds {self.max_obs_len}")
        sequence = self.input_projection(target) + self.position_embedding[:, :target.shape[1]]
        context = self.temporal_encoder(sequence)[:, -1]
        future_pred = self.decoder(context).reshape(target.shape[0], self.pred_len, 2)
        return {"future_pred": future_pred, "target_context": context}
