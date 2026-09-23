"""Scene-aware social interaction gate for JAAD."""

import torch
from torch import nn


class SceneSocialGate(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        scene_dim: int = 512,
        hidden_dim: int = 128,
        pred_len: int = 12,
        dropout: float = 0.1,
        gate_mode: str = "always",
    ):
        super().__init__()
        if gate_mode not in {"uncertainty", "always", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        self.gate_mode = gate_mode
        self.pred_len = pred_len
        self.target_encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.neighbor_encoder = nn.GRU(4, hidden_dim, batch_first=True)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.proposal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.proposal_head = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.intent_head = nn.Linear(hidden_dim, 1)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )

    def forward(self, target_obs, neighbor_obs, neighbor_mask, neighbor_visible_mask, scene_feat):
        _, target_hidden = self.target_encoder(target_obs)
        target_context = target_hidden[-1]
        scene_context = self.scene_encoder(scene_feat)
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

        proposal_context = self.proposal_fusion(
            torch.cat([target_context, scene_context, social_context], dim=-1)
        )
        prior_logit = self.proposal_head(proposal_context).squeeze(-1)
        prior_prob = torch.sigmoid(prior_logit).clamp(1e-6, 1.0 - 1e-6)
        entropy = -prior_prob * torch.log(prior_prob) - (1.0 - prior_prob) * torch.log(1.0 - prior_prob)
        if self.gate_mode == "uncertainty":
            gate = self.gate(torch.cat([target_context, scene_context, entropy.unsqueeze(-1)], dim=-1)).squeeze(-1)
        elif self.gate_mode == "always":
            gate = torch.ones_like(entropy)
        else:
            gate = torch.zeros_like(entropy)
        fused = self.fusion(
            torch.cat([target_context, scene_context, gate.unsqueeze(-1) * social_context], dim=-1)
        )
        intent_logit = self.intent_head(fused).squeeze(-1)
        future_pred = self.traj_head(fused).view(-1, self.pred_len, 2)
        return {
            "intent_logit": intent_logit,
            "future_pred": future_pred,
            "prior_logit": prior_logit,
            "entropy": entropy,
            "gate": gate,
            "neighbor_visible_mask": neighbor_visible_mask,
        }
