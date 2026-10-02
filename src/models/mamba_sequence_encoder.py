"""Residual temporal encoder using the official Mamba selective SSM."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, dropout: float) -> None:
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError("ResidualMambaEncoder requires a compatible official mamba-ssm installation") from error
        self.norm = nn.LayerNorm(d_model)
        self.mixer = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.dropout(self.mixer(self.norm(sequence)))


class ResidualMambaEncoder(nn.Module):
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
        if min(input_dim, d_model, num_layers, d_state, d_conv, expand) < 1:
            raise ValueError("Encoder dimensions and layer count must be positive")
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            ResidualMambaBlock(d_model, d_state, d_conv, expand, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if target.ndim != 3 or target.shape[-1] != self.input_dim or target.shape[1] < 1:
            raise ValueError(f"Expected nonempty target [B,T,{self.input_dim}], got {tuple(target.shape)}")
        sequence = self.input_projection(target)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.final_norm(sequence)
        return sequence, sequence[:, -1]
