"""Three-state social-gating model: non-crossing, crossing, ambiguous."""

import torch
from torch import nn


class TriStateSocialGate(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        pred_len: int = 12,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.num_classes = num_classes
        self.target_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.neighbor_encoder = nn.GRU(4, hidden_dim, batch_first=True)
        self.proposal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.proposal_head = nn.Linear(hidden_dim, num_classes)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.intent_head = nn.Linear(hidden_dim, num_classes)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )

    def forward(self, target_obs, neighbor_obs, neighbor_mask, neighbor_visible_mask):
        _, target_hidden = self.target_encoder(target_obs)
        target_context = target_hidden[-1]
        batch_size, obs_len, max_neighbors, feature_dim = neighbor_obs.shape
        neighbor_input = neighbor_obs.permute(0, 2, 1, 3).reshape(
            batch_size * max_neighbors, obs_len, feature_dim
        )
        _, neighbor_hidden = self.neighbor_encoder(neighbor_input)
        neighbor_context = neighbor_hidden[-1].view(batch_size, max_neighbors, -1)
        slot_mask = neighbor_mask.bool()
        masked_context = neighbor_context * slot_mask.unsqueeze(-1)
        denominator = slot_mask.sum(dim=1, keepdim=True).clamp_min(1).to(masked_context.dtype)
        social_context = masked_context.sum(dim=1) / denominator
        social_context = social_context * slot_mask.any(dim=1, keepdim=True).to(social_context.dtype)

        proposal_context = self.proposal_fusion(torch.cat([target_context, social_context], dim=-1))
        prior_logits = self.proposal_head(proposal_context)
        prior_prob = torch.softmax(prior_logits, dim=-1).clamp(1e-6, 1.0)
        entropy = -(prior_prob * torch.log(prior_prob)).sum(dim=-1) / torch.log(
            torch.tensor(float(self.num_classes), device=prior_prob.device)
        )
        gate = self.gate(torch.cat([target_context, entropy.unsqueeze(-1)], dim=-1)).squeeze(-1)
        fused = self.fusion(torch.cat([target_context, gate.unsqueeze(-1) * social_context], dim=-1))
        intent_logits = self.intent_head(fused)
        future_pred = self.traj_head(fused).view(-1, self.pred_len, 2)
        return {
            "intent_logits": intent_logits,
            "future_pred": future_pred,
            "prior_logits": prior_logits,
            "entropy": entropy,
            "gate": gate,
            "neighbor_visible_mask": neighbor_visible_mask,
        }
