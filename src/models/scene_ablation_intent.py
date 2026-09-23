"""Matched intent readouts for controlled static-scene ablation."""

from __future__ import annotations

import torch
from torch import nn

from src.models.trajectory_transformer import SceneTrajectoryTransformer


class SceneAblationIntentModel(nn.Module):
    """Use a frozen scene-aware trajectory model with a controlled intent readout.

    ``target_only`` replaces only the classifier's scene context with zeros. The
    frozen trajectory prediction remains scene-aware in both modes, so paired
    intent models use exactly the same trajectory representation/checkpoint.
    Module names and dimensions match ``FixedBaseIntentModel`` for seed-123
    checkpoint compatibility.
    """

    def __init__(
        self,
        trajectory_backbone: SceneTrajectoryTransformer,
        *,
        scene_mode: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if scene_mode not in {"target_only", "target_scene"}:
            raise ValueError(f"Unsupported scene_mode: {scene_mode}")
        self.scene_mode = scene_mode
        self.trajectory_backbone = trajectory_backbone
        self.d_model = trajectory_backbone.input_projection.out_features
        self.input_dim = trajectory_backbone.input_projection.in_features
        self.scene_dim = trajectory_backbone.scene_encoder[0].in_features
        self.pred_len = trajectory_backbone.pred_len
        for parameter in self.trajectory_backbone.parameters():
            parameter.requires_grad_(False)
        self.trajectory_backbone.eval()

        self.base_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(self.d_model, 1)
        self.register_buffer("temperature", torch.ones(()))

    @property
    def trajectory_backbone_frozen(self) -> bool:
        return all(not parameter.requires_grad for parameter in self.trajectory_backbone.parameters())

    def train(self, mode: bool = True) -> SceneAblationIntentModel:
        super().train(mode)
        self.trajectory_backbone.eval()
        return self

    def forward(
        self,
        target_obs: torch.Tensor,
        scene_feat: torch.Tensor,
        *,
        zero_scene_context: bool = False,
    ) -> dict[str, torch.Tensor]:
        if target_obs.ndim != 3 or target_obs.shape[-1] != self.input_dim:
            raise ValueError(
                f"target_obs must be [B,T,{self.input_dim}], got {tuple(target_obs.shape)}"
            )
        if scene_feat.shape != (target_obs.shape[0], self.scene_dim):
            raise ValueError(
                f"scene_feat must be [B,{self.scene_dim}], got {tuple(scene_feat.shape)}"
            )

        with torch.no_grad():
            backbone = self.trajectory_backbone
            obs_len = target_obs.shape[1]
            temporal = backbone.input_projection(target_obs)
            temporal = temporal + backbone.position_embedding[:, :obs_len]
            target_context = backbone.temporal_encoder(temporal)[:, -1]
            if self.scene_mode == "target_scene" and not zero_scene_context:
                scene_context = backbone.scene_encoder(scene_feat)
            else:
                scene_context = target_context.new_zeros(target_context.shape)
            # Intent readout ablation does not alter the pretrained trajectory path.
            future_pred = backbone(target_obs, scene_feat)

        fused_context = self.base_fusion(torch.cat([target_context, scene_context], dim=-1))
        raw_logit = self.base_head(fused_context).squeeze(-1)
        return {
            "target_context": target_context,
            "scene_context": scene_context,
            "base_context": fused_context,
            "base_raw_logit": raw_logit,
            "base_logit": raw_logit / self.temperature.clamp(min=1e-4),
            "future_pred": future_pred,
        }
