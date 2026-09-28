"""Scratch-trained intention Transformer matched to the P1 target encoder."""

from __future__ import annotations

import torch
from torch import nn


class IntentionScratchTransformer(nn.Module):
    """A target-only intention model with the P1 8D temporal encoder shape.

    The target projection, learned position embedding, Transformer encoder,
    intention adapter, and intention head match the corresponding P1 modules.
    No scene branch, trajectory decoder, or checkpoint-loading path exists.
    """

    def __init__(
        self,
        *,
        input_dim: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        max_obs_len: int = 15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.max_obs_len = max_obs_len
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
        self.intent_adapter = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.intent_head = nn.Linear(d_model, 1)

    def encode_target(self, target_history: torch.Tensor) -> torch.Tensor:
        if target_history.ndim != 3 or target_history.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected target history [B,T,{self.input_dim}], got {tuple(target_history.shape)}"
            )
        obs_len = target_history.shape[1]
        if obs_len > self.max_obs_len:
            raise ValueError(f"Observed length {obs_len} exceeds {self.max_obs_len}")
        temporal = self.input_projection(target_history) + self.position_embedding[:, :obs_len]
        encoded = self.temporal_encoder(temporal)
        return encoded[:, -1]

    def forward(self, target_history: torch.Tensor) -> dict[str, torch.Tensor]:
        target_context = self.encode_target(target_history)
        intent_logit = self.intent_head(self.intent_adapter(target_context)).squeeze(-1)
        return {"intent_logit": intent_logit, "target_context": target_context}
