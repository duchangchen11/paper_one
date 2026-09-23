from __future__ import annotations

import copy

import torch

from src.models.fixed_base_social_residual import (
    FixedBaseIntentModel,
    FixedBaseSocialResidual,
)
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def make_base_model() -> FixedBaseIntentModel:
    backbone = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=6,
        d_model=16,
        nhead=4,
        num_layers=1,
        pred_len=7,
        dropout=0.0,
        max_obs_len=8,
    )
    return FixedBaseIntentModel(backbone, dropout=0.0)


def make_inputs(batch: int = 2, neighbors: int = 3, obs_len: int = 5):
    torch.manual_seed(29)
    target = torch.randn(batch, obs_len, 8)
    scene = torch.randn(batch, 6)
    neighbor = torch.arange(
        batch * neighbors * obs_len * 4, dtype=torch.float32
    ).reshape(batch, neighbors, obs_len, 4)
    mask = torch.ones(batch, neighbors)
    return target, scene, neighbor, mask


def test_base_logits_are_identical_across_modes_from_same_base_state():
    torch.manual_seed(3)
    base_a = make_base_model()
    base_b = make_base_model()
    base_b.load_state_dict(copy.deepcopy(base_a.state_dict()))
    always = FixedBaseSocialResidual(base_a, gate_mode="always", dropout=0.0).eval()
    uncertain = FixedBaseSocialResidual(base_b, gate_mode="uncertainty", dropout=0.0).eval()
    target, scene, neighbor, mask = make_inputs()

    with torch.no_grad():
        out_a = always(target, scene, neighbor, mask)
        out_b = uncertain(target, scene, neighbor, mask)

    torch.testing.assert_close(out_a["base_logit"], out_b["base_logit"], rtol=0, atol=0)
    torch.testing.assert_close(
        out_a["base_probability"], out_b["base_probability"], rtol=0, atol=0
    )
    torch.testing.assert_close(out_a["base_entropy"], out_b["base_entropy"], rtol=0, atol=0)


def test_none_mode_returns_base_logit_exactly():
    model = FixedBaseSocialResidual(make_base_model(), gate_mode="none", dropout=0.0).eval()
    target, scene, neighbor, mask = make_inputs()
    with torch.no_grad():
        output = model(target, scene, neighbor, mask)
    torch.testing.assert_close(output["final_logit"], output["base_logit"], rtol=0, atol=0)
    torch.testing.assert_close(
        output["final_probability"], output["base_probability"], rtol=0, atol=0
    )


def test_zero_initialized_social_head_preserves_base_for_both_gates():
    target, scene, neighbor, mask = make_inputs()
    for mode in ("always", "uncertainty"):
        model = FixedBaseSocialResidual(
            make_base_model(), gate_mode=mode, dropout=0.0
        ).eval()
        final = model.social_head[-1]
        assert torch.count_nonzero(final.weight) == 0
        assert torch.count_nonzero(final.bias) == 0
        with torch.no_grad():
            output = model(target, scene, neighbor, mask)
        torch.testing.assert_close(
            output["final_logit"], output["base_logit"], rtol=0, atol=0
        )


def test_changing_neighbors_never_changes_base_outputs():
    model = FixedBaseSocialResidual(
        make_base_model(), gate_mode="uncertainty", dropout=0.0
    ).eval()
    target, scene, neighbor, mask = make_inputs()
    with torch.no_grad():
        first = model(target, scene, neighbor, mask)
        second = model(target, scene, neighbor.flip((1, 2, 3)) * -17.0, mask)
    for key in ("base_logit", "base_probability", "base_entropy", "future_pred"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_no_neighbor_samples_cannot_receive_a_logit_residual():
    model = FixedBaseSocialResidual(
        make_base_model(), gate_mode="always", dropout=0.0
    ).eval()
    model.social_head[-1].bias.data.fill_(4.0)
    target, scene, neighbor, _ = make_inputs()
    mask = torch.zeros(2, 3)
    with torch.no_grad():
        output = model(target, scene, neighbor, mask)
    assert torch.all(output["delta_logit"] != 0)
    assert torch.count_nonzero(output["effective_gate"]) == 0
    torch.testing.assert_close(output["final_logit"], output["base_logit"], rtol=0, atol=0)


def test_uncertainty_gate_is_monotonic_in_entropy():
    model = FixedBaseSocialResidual(make_base_model(), gate_mode="uncertainty")
    entropy = torch.tensor([0.0, 0.1, 0.3, 0.5, 0.8, 1.0])
    gates = model.uncertainty_to_gate(entropy)
    assert model.gate_slope.item() > 0
    assert torch.all(gates[1:] >= gates[:-1])


def test_neighbor_gru_receives_one_pedestrian_sequence_per_batch_neighbor_pair():
    batch, neighbors, obs_len = 2, 3, 5
    model = FixedBaseSocialResidual(
        make_base_model(), gate_mode="always", dropout=0.0
    ).eval()
    target, scene, neighbor, mask = make_inputs(batch, neighbors, obs_len)
    captured = {}

    def inspect_gru_input(_module, inputs):
        captured["sequence"] = inputs[0].detach().clone()

    hook = model.neighbor_encoder.register_forward_pre_hook(inspect_gru_input)
    with torch.no_grad():
        model(target, scene, neighbor, mask)
    hook.remove()

    expected = neighbor.reshape(batch * neighbors, obs_len, 4)
    assert captured["sequence"].shape == (batch * neighbors, obs_len, 4)
    torch.testing.assert_close(captured["sequence"], expected, rtol=0, atol=0)


def test_base_parameters_remain_frozen_and_in_eval_mode_during_social_training():
    model = FixedBaseSocialResidual(make_base_model(), gate_mode="uncertainty")
    model.train()
    assert model.all_base_parameters_frozen
    assert not model.base_model.training
    assert not model.base_model.trajectory_backbone.training
    assert all(parameter.requires_grad for parameter in model.neighbor_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.social_head.parameters())
