from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.mamba_sequence_encoder import ResidualMambaEncoder


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip("Official Mamba kernels require CUDA")
    import mamba_ssm  # A broken installation must fail on a CUDA machine.
    return torch.device("cuda")


def test_encoder_rejects_zero_layers():
    with pytest.raises(ValueError, match="positive"):
        ResidualMambaEncoder(num_layers=0)


def test_encoder_gpu_shapes_finite_backward_and_residual_structure(cuda_device):
    model = ResidualMambaEncoder().to(cuda_device)
    target = torch.randn(4, 15, 8, device=cuda_device, requires_grad=True)
    sequence, context = model(target)
    assert sequence.shape == (4, 15, 128)
    assert context.shape == (4, 128)
    assert torch.equal(context, sequence[:, -1])
    assert torch.isfinite(sequence).all() and torch.isfinite(context).all()
    context.square().mean().backward()
    assert torch.isfinite(target.grad).all()
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert isinstance(model.final_norm, nn.LayerNorm)
    assert len(model.blocks) == 3
    assert all(isinstance(block.norm, nn.LayerNorm) and isinstance(block.dropout, nn.Dropout) for block in model.blocks)
    assert not any(isinstance(module, nn.MultiheadAttention) for module in model.modules())
    assert not hasattr(model, "position_embedding")


@pytest.mark.parametrize("shape", [(4, 15, 7), (4, 8), (4, 0, 8)])
def test_encoder_rejects_invalid_inputs_before_kernel(shape, cuda_device):
    model = ResidualMambaEncoder().to(cuda_device)
    with pytest.raises(ValueError, match="Expected nonempty target"):
        model(torch.zeros(shape, device=cuda_device))


def test_mamba_two_layer_sweep_retains_output_contract(cuda_device):
    model = ResidualMambaEncoder(num_layers=2).to(cuda_device)
    sequence, context = model(torch.randn(4, 15, 8, device=cuda_device))
    assert len(model.blocks) == 2
    assert sequence.shape == (4, 15, 128) and context.shape == (4, 128)
