"""Joint intent/trajectory wrapper with an exactly frozen trajectory backbone."""

from __future__ import annotations

import torch
from torch import nn

from src.models.trajectory_transformer import SceneTrajectoryTransformer


class TrajectoryPreservingJoint(nn.Module):
    """Reuse the standalone trajectory forward path and train only an intent head.

    ``intent_input='target'`` is P1. ``intent_input='target_scene'`` is P2.
    The original trajectory decoder always receives the original concatenation
    ``[target_context, scene_context]`` and is never connected to the intent logit.
    """

    def __init__(
        self,
        *,
        input_dim: int = 8,
        scene_dim: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        pred_len: int = 15,
        dropout: float = 0.1,
        max_obs_len: int = 32,
        intent_input: str = "target",
    ) -> None:
        super().__init__()
        if intent_input not in {"target", "target_scene"}:
            raise ValueError(f"Unsupported intent_input: {intent_input}")
        self.intent_input = intent_input
        self.d_model = d_model
        self.pred_len = pred_len
        self.backbone = SceneTrajectoryTransformer(
            input_dim=input_dim,
            scene_dim=scene_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            pred_len=pred_len,
            dropout=dropout,
            max_obs_len=max_obs_len,
        )
        intent_dim = d_model if intent_input == "target" else 2 * d_model
        self.intent_adapter = nn.Sequential(
            nn.LayerNorm(intent_dim),
            nn.Linear(intent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.intent_head = nn.Linear(d_model, 1)
        self.freeze_trajectory_backbone()
        self.backbone.eval()

    def freeze_trajectory_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        """Train the intent branch while keeping backbone dropout disabled."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, target_obs: torch.Tensor, scene_feat: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_len = target_obs.shape[1]
        with torch.no_grad():
            temporal = self.backbone.input_projection(target_obs)
            temporal = temporal + self.backbone.position_embedding[:, :obs_len]
            encoded = self.backbone.temporal_encoder(temporal)
            target_context = encoded[:, -1]
            scene_context = self.backbone.scene_encoder(scene_feat)

            # This is intentionally byte-for-byte the feature ordering in the
            # standalone SceneTrajectoryTransformer.forward implementation.
            trajectory_input = torch.cat([target_context, scene_context], dim=-1)
            future_pred = self.backbone.decoder(trajectory_input).view(
                -1, self.pred_len, 2
            )

        if self.intent_input == "target":
            intent_input = target_context
        else:
            intent_input = torch.cat([target_context, scene_context], dim=-1)
        intent_logit = self.intent_head(self.intent_adapter(intent_input)).squeeze(-1)
        return {
            "future_pred": future_pred,
            "intent_logit": intent_logit,
            "target_context": target_context,
            "scene_context": scene_context,
        }
