"""Independent crossing-intention prediction from Mamba history features."""

from __future__ import annotations

import torch
from torch import nn

from src.models.mamba_sequence_encoder import ResidualMambaEncoder


class MambaIntentionPredictor(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 8,
        d_model: int = 128,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = ResidualMambaEncoder(
            input_dim=input_dim, d_model=d_model, num_layers=num_layers,
            d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout,
        )
        self.intent_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1),
        )

    def forward(self, target: torch.Tensor) -> dict[str, torch.Tensor]:
        _, context = self.encoder(target)
        return {"intent_logit": self.intent_head(context).squeeze(-1), "target_context": context}
