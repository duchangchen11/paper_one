"""Target-only multi-task baseline for JAAD."""

import torch
from torch import nn


class TargetOnlyBaseline(nn.Module):
    """Encode the target history and jointly predict intent and trajectory."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 128,
        pred_len: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )

    def forward(self, target_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.encoder(target_obs)
        context = self.dropout(hidden[-1])
        intent_logit = self.intent_head(context).squeeze(-1)
        future_pred = self.traj_head(context).view(-1, self.pred_len, 2)
        return {"intent_logit": intent_logit, "future_pred": future_pred}
