"""Transformer baseline dedicated to deterministic trajectory forecasting."""

import torch
from torch import nn


class SceneTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        scene_dim: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        pred_len: int = 15,
        dropout: float = 0.1,
        max_obs_len: int = 32,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_obs_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * 2),
        )

    def forward(self, target_obs: torch.Tensor, scene_feat: torch.Tensor) -> torch.Tensor:
        obs_len = target_obs.shape[1]
        temporal = self.input_projection(target_obs) + self.position_embedding[:, :obs_len]
        encoded = self.temporal_encoder(temporal)
        target_context = encoded[:, -1]
        scene_context = self.scene_encoder(scene_feat)
        return self.decoder(torch.cat([target_context, scene_context], dim=-1)).view(
            -1, self.pred_len, 2
        )
