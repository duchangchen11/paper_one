"""Decoupled frozen trajectory and trainable pretrained-clone intention branches."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.intention_scratch_transformer import IntentionScratchTransformer
from src.models.trajectory_transformer import SceneTrajectoryTransformer


TARGET_ENCODER_PREFIXES = ("input_projection.", "temporal_encoder.")


def clone_target_encoder(
    trajectory_model: SceneTrajectoryTransformer,
    intention_model: IntentionScratchTransformer,
) -> dict[str, Any]:
    """Copy only the target temporal encoder, reporting exact tensor parity."""
    source_state = trajectory_model.state_dict()
    target_state = intention_model.state_dict()
    names = [
        name
        for name in source_state
        if name == "position_embedding" or name.startswith(TARGET_ENCODER_PREFIXES)
    ]
    target_names = [
        name
        for name in target_state
        if name == "position_embedding" or name.startswith(TARGET_ENCODER_PREFIXES)
    ]
    missing = sorted(set(names) - set(target_names))
    unexpected = sorted(set(target_names) - set(names))
    shape_mismatch = [
        {
            "name": name,
            "source_shape": list(source_state[name].shape),
            "target_shape": list(target_state[name].shape),
        }
        for name in sorted(set(names) & set(target_names))
        if tuple(source_state[name].shape) != tuple(target_state[name].shape)
    ]
    if missing or unexpected or shape_mismatch:
        return {
            "expected_tensor_count": len(names),
            "copied_tensor_count": 0,
            "missing": missing,
            "unexpected": unexpected,
            "shape_mismatch": shape_mismatch,
            "max_abs_tensor_diff": None,
            "clone_pass": False,
        }

    with torch.no_grad():
        for name in names:
            target_state[name].copy_(source_state[name])
    copied_state = intention_model.state_dict()
    differences = [
        float((source_state[name].detach().cpu() - copied_state[name].detach().cpu()).abs().max())
        for name in names
    ]
    max_abs = max(differences, default=0.0)
    return {
        "expected_tensor_count": len(names),
        "copied_tensor_count": len(names),
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
        "max_abs_tensor_diff": max_abs,
        "tensor_names": names,
        "clone_pass": max_abs == 0.0,
    }


def compare_encoder_storage(
    trajectory_model: SceneTrajectoryTransformer,
    intention_model: IntentionScratchTransformer,
) -> dict[str, Any]:
    """Confirm cloned values match while Parameters/storage remain independent."""
    trajectory_parameters = dict(trajectory_model.named_parameters())
    intention_parameters = dict(intention_model.named_parameters())
    names = [
        name
        for name in intention_parameters
        if name == "position_embedding" or name.startswith(TARGET_ENCODER_PREFIXES)
    ]
    missing = [name for name in names if name not in trajectory_parameters]
    shared_object = [
        name
        for name in names
        if name in trajectory_parameters and trajectory_parameters[name] is intention_parameters[name]
    ]
    shared_storage = [
        name
        for name in names
        if name in trajectory_parameters
        and trajectory_parameters[name].untyped_storage().data_ptr()
        == intention_parameters[name].untyped_storage().data_ptr()
    ]
    unequal = [
        name
        for name in names
        if name in trajectory_parameters
        and not torch.equal(trajectory_parameters[name].detach(), intention_parameters[name].detach())
    ]
    return {
        "compared_parameter_count": len(names),
        "missing": missing,
        "initial_value_equal": not unequal and not missing,
        "same_parameter_object": bool(shared_object),
        "same_parameter_object_names": shared_object,
        "same_storage": bool(shared_storage),
        "same_storage_names": shared_storage,
        "unequal_value_names": unequal,
    }


class PretrainedCloneIntent(nn.Module):
    """Independent task branches; only intention parameters are trainable."""

    def __init__(
        self,
        trajectory_branch: SceneTrajectoryTransformer,
        intention_branch: IntentionScratchTransformer,
    ) -> None:
        super().__init__()
        self.trajectory_branch = trajectory_branch
        self.intention_branch = intention_branch
        for parameter in self.trajectory_branch.parameters():
            parameter.requires_grad_(False)
        self.trajectory_branch.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.trajectory_branch.eval()
        return self

    def forward_intention(self, target_history: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.intention_branch(target_history)

    def forward_trajectory(
        self, target_history: torch.Tensor, scene_feat: torch.Tensor
    ) -> torch.Tensor:
        self.trajectory_branch.eval()
        with torch.no_grad():
            return self.trajectory_branch(target_history, scene_feat)

    def forward(
        self, target_history: torch.Tensor, scene_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        intent = self.forward_intention(target_history)
        future = self.forward_trajectory(target_history, scene_feat)
        return {
            "intent_logit": intent["intent_logit"],
            "target_context": intent["target_context"],
            "future_pred": future,
        }
