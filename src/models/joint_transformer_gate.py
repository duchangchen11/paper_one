"""Joint intention/trajectory model with a Transformer target encoder."""

import torch
from torch import nn


class JointTransformerSceneGate(nn.Module):
    def __init__(
        self,
        input_dim: int = 8,
        scene_dim: int = 512,
        hidden_dim: int = 128,
        pred_len: int = 15,
        nhead: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        gate_mode: str = "uncertainty",
        max_obs_len: int = 32,
        component_flags_all_enabled: bool = True,
        component_ablation: str | None = None,
    ):
        super().__init__()
        if gate_mode not in {"uncertainty", "always", "none"}:
            raise ValueError(f"Unsupported gate_mode: {gate_mode}")
        valid_ablations = {
            "no_scene",
            "no_social",
            "no_proposal_loss",
            "no_adaptive_gate",
            "no_ambiguity",
        }
        if component_flags_all_enabled and component_ablation not in (None, "full"):
            raise ValueError("all-enabled component flags cannot be combined with an ablation")
        if not component_flags_all_enabled and component_ablation not in valid_ablations:
            raise ValueError("a supported component_ablation is required when flags are disabled")
        self.component_flags_all_enabled = bool(component_flags_all_enabled)
        self.component_ablation = None if self.component_flags_all_enabled else component_ablation
        self.gate_mode = gate_mode
        self.pred_len = pred_len
        self.target_projection = nn.Linear(input_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_obs_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.neighbor_encoder = nn.GRU(4, hidden_dim, batch_first=True)
        self.scene_encoder = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proposal_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.proposal_head = nn.Linear(hidden_dim, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.intent_head = nn.Linear(hidden_dim, 1)
        self.traj_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * 2),
        )

    def forward(self, target_obs, neighbor_obs, neighbor_mask, neighbor_visible_mask, scene_feat):
        obs_len = target_obs.shape[1]
        target_encoded = self.target_projection(target_obs) + self.position_embedding[:, :obs_len]
        target_context = self.target_encoder(target_encoded)[:, -1]
        scene_context = self.scene_encoder(scene_feat)
        if self.component_ablation == "no_scene":
            scene_context = torch.zeros_like(scene_context)

        batch_size, max_neighbors, obs_len, feature_dim = neighbor_obs.shape
        neighbor_input = neighbor_obs.reshape(batch_size * max_neighbors, obs_len, feature_dim)
        _, neighbor_hidden = self.neighbor_encoder(neighbor_input)
        neighbor_context = neighbor_hidden[-1].view(batch_size, max_neighbors, -1)
        slot_mask = neighbor_mask.bool()
        masked_context = neighbor_context * slot_mask.unsqueeze(-1)
        denominator = slot_mask.sum(dim=1, keepdim=True).clamp_min(1).to(masked_context.dtype)
        social_context = masked_context.sum(dim=1) / denominator
        social_context = social_context * slot_mask.any(dim=1, keepdim=True).to(social_context.dtype)
        if self.component_ablation == "no_social":
            social_context = torch.zeros_like(social_context)

        proposal_context = self.proposal_fusion(
            torch.cat([target_context, scene_context, social_context], dim=-1)
        )
        prior_logit = self.proposal_head(proposal_context).squeeze(-1)
        prior_prob = torch.sigmoid(prior_logit).clamp(1e-6, 1.0 - 1e-6)
        entropy = -prior_prob * torch.log(prior_prob) - (1.0 - prior_prob) * torch.log(1.0 - prior_prob)
        if self.component_ablation == "no_adaptive_gate":
            # Neutral interpolation weight: retain the social and non-social
            # branches while removing sample-dependent uncertainty modulation.
            gate = torch.full_like(entropy, 0.5)
        elif self.gate_mode == "uncertainty":
            gate = self.gate(torch.cat([target_context, scene_context, entropy.unsqueeze(-1)], dim=-1)).squeeze(-1)
        elif self.gate_mode == "always":
            gate = torch.ones_like(entropy)
        else:
            gate = torch.zeros_like(entropy)
        fused = self.fusion(
            torch.cat([target_context, scene_context, gate.unsqueeze(-1) * social_context], dim=-1)
        )
        intent_logit = self.intent_head(fused).squeeze(-1)
        future_pred = self.traj_head(torch.cat([fused, target_context], dim=-1)).view(-1, self.pred_len, 2)
        return {
            "intent_logit": intent_logit,
            "future_pred": future_pred,
            "prior_logit": prior_logit,
            "entropy": entropy,
            "gate": gate,
            "effective_scene_context": scene_context,
            "effective_social_context": social_context,
            "neighbor_visible_mask": neighbor_visible_mask,
        }
