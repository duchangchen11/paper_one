from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.mamba_jaad_utils import (
    TrajectoryIntentDataset, evaluate_trajectory, initialize_matched_decoder,
    intention_checkpoint_is_better, inverse_frequency_weights, weighted_intention_loss,
)
from src.models.mamba_intention import MambaIntentionPredictor
from src.models.mamba_trajectory import MambaTrajectoryPredictor
from src.models.trajectory_transformer_target_only import TargetOnlyTrajectoryTransformer


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("Official Mamba kernels require CUDA")
    import mamba_ssm
    return torch.device("cuda")


def test_target_only_transformer_shape_and_encoder_configuration():
    model = TargetOnlyTrajectoryTransformer()
    output = model(torch.randn(4, 15, 8))
    assert output["future_pred"].shape == (4, 15, 2)
    assert output["target_context"].shape == (4, 128)
    assert len(model.temporal_encoder.layers) == 3
    assert model.temporal_encoder.layers[0].self_attn.num_heads == 4
    assert model.temporal_encoder.layers[0].linear1.out_features == 512
    assert torch.isfinite(output["future_pred"]).all()
    output["future_pred"].square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_trajectory_mamba_matches_transformer_decoder_and_output(cuda_device):
    mt = MambaTrajectoryPredictor()
    tt = TargetOnlyTrajectoryTransformer()
    assert [type(module) for module in mt.decoder] == [type(module) for module in tt.decoder]
    assert {k: tuple(v.shape) for k, v in mt.decoder.state_dict().items()} == {k: tuple(v.shape) for k, v in tt.decoder.state_dict().items()}
    assert initialize_matched_decoder(mt, 42) == initialize_matched_decoder(tt, 42)
    mt.to(cuda_device)
    tt.to(cuda_device)
    target = torch.randn(4, 15, 8, device=cuda_device)
    mout, tout = mt(target), tt(target)
    assert mout["future_pred"].shape == tout["future_pred"].shape == (4, 15, 2)
    assert mout["target_context"].shape == (4, 128)
    mout["future_pred"].square().mean().backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in mt.parameters())


def test_intention_mamba_shapes_finite_gradients_and_target_only_interface(cuda_device):
    model = MambaIntentionPredictor().to(cuda_device)
    out = model(torch.randn(4, 15, 8, device=cuda_device))
    assert out["intent_logit"].shape == (4,)
    assert out["target_context"].shape == (4, 128)
    assert torch.isfinite(out["intent_logit"]).all()
    nn.functional.binary_cross_entropy_with_logits(out["intent_logit"], torch.tensor([0., 1., 0., 1.], device=cuda_device)).backward()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    for cls in (MambaIntentionPredictor, MambaTrajectoryPredictor, TargetOnlyTrajectoryTransformer):
        assert list(inspect.signature(cls.forward).parameters) == ["self", "target"]
    assert not any(any(word in name for word in ("scene", "neighbor", "reliability")) for name, _ in model.named_modules())


def test_decoder_reinitialization_preserves_cpu_random_stream():
    model = TargetOnlyTrajectoryTransformer()
    state = torch.random.get_rng_state().clone()
    initialize_matched_decoder(model, 123)
    assert torch.equal(state, torch.random.get_rng_state())


def test_class_weights_balance_natural_sample_frequency():
    labels = torch.tensor([0., 1., 1., 1.])
    weights = inverse_frequency_weights(labels)
    assert weights["negative"] == 2.0 and weights["positive"] == pytest.approx(2/3)
    logits = torch.zeros(4, requires_grad=True)
    loss = weighted_intention_loss(logits, labels, weights)
    assert loss.item() == pytest.approx(np.log(2))
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="both intention classes"):
        inverse_frequency_weights(torch.ones(3))


def test_intention_selection_matches_m0_tie_break():
    assert intention_checkpoint_is_better(0.701, 0.4, 0.70, 0.2, 1e-4)
    assert intention_checkpoint_is_better(0.70005, 0.1, 0.70, 0.2, 1e-4)
    assert not intention_checkpoint_is_better(0.70, 0.3, 0.70, 0.2, 1e-4)
    assert not intention_checkpoint_is_better(0.699, 0.1, 0.70, 0.2, 1e-4)


def test_existing_dataset_test_guard_is_reused(monkeypatch, tmp_path):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("test data must never be opened")
    monkeypatch.setattr("scripts.trajectory_preserving_utils.np.load", forbidden_load)
    with pytest.raises(RuntimeError, match="Test split"):
        TrajectoryIntentDataset(tmp_path / "test.npz")


def test_trajectory_evaluation_reuses_pixel_scaling_and_passes_only_target():
    class Model(nn.Module):
        def forward(self, target):
            return {"future_pred": torch.ones(len(target), 15, 2)}
    samples = [dict(target=torch.zeros(15, 8), future_gt=torch.zeros(15, 2), image_size=torch.tensor([100., 200.]), scene_feat=torch.tensor([float("nan")])) for _ in range(2)]
    result = evaluate_trajectory(Model(), DataLoader(samples, batch_size=2), torch.device("cpu"))
    assert result["ade_pixel"] == pytest.approx(np.hypot(100, 200))
    assert result["fde_pixel"] == pytest.approx(np.hypot(100, 200))
    assert result["normalized_loss"] == pytest.approx(0.5)
