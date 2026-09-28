import math

import torch
from torch import nn

from scripts.train_joint_transformer_gate import (
    DynamicGradientBalance,
    SHARED_GRADIENT_PREFIXES,
    compose_training_objective,
    measure_aligned_task_gradients,
    shared_named_parameters,
)
from src.models.joint_transformer_gate import JointTransformerSceneGate


def test_controller_clamps_lambda_to_configured_bounds():
    upper = DynamicGradientBalance()
    upper_record = upper.observe(2, 10, 1e100, 1e-10)
    assert upper.lambda_value == 300.0
    assert upper_record["update_status"] == "clipped_max"

    lower = DynamicGradientBalance()
    lower.lambda_value = 10.0
    lower_record = lower.observe(2, 10, 1e-8, 1e2)
    assert lower.lambda_value == 10.0
    assert lower_record["update_status"] == "clipped_min"


def test_controller_warmup_holds_lambda_and_interval_gates_updates():
    controller = DynamicGradientBalance(initial_lambda=100.0)
    assert controller.should_measure(9) is False
    assert controller.should_measure(10) is True
    warmup_record = controller.observe(1, 10, 100.0, 0.01)
    assert warmup_record["update_status"] == "warmup"
    assert controller.lambda_value == 100.0

    updated_record = controller.observe(2, 10, 100.0, 0.01)
    assert updated_record["controller_update"] is True
    assert 10.0 <= controller.lambda_value <= 300.0
    assert not math.isclose(controller.lambda_value, 100.0)


def test_invalid_or_tiny_gradients_retain_lambda_without_nan():
    controller = DynamicGradientBalance(initial_lambda=100.0)
    for intent_norm, trajectory_norm in ((0.0, 1.0), (1.0, 0.0), (float("nan"), 1.0)):
        record = controller.observe(2, 10, intent_norm, trajectory_norm)
        assert record["update_status"] == "invalid_gradient_norm"
        assert controller.lambda_value == 100.0
        assert math.isfinite(controller.lambda_value)


def test_fixed_objective_matches_legacy_expression_exactly():
    main = torch.tensor(0.1234567, requires_grad=True)
    proposal = torch.tensor(0.2345678, requires_grad=True)
    trajectory = torch.tensor(0.0034567, requires_grad=True)
    ambiguity = torch.tensor(0.0456789, requires_grad=True)
    legacy = main + 0.5 * proposal
    legacy = legacy + 100.0 * trajectory
    legacy = legacy + 0.2 * ambiguity
    current = compose_training_objective(
        main, proposal, trajectory, ambiguity, 0.5, 100.0, 0.2
    )
    assert torch.equal(current, legacy)


def test_controller_lambda_is_detached_python_state():
    controller = DynamicGradientBalance()
    controller.observe(2, 10, 20.0, 0.01)
    assert isinstance(controller.lambda_value, float)
    assert not isinstance(controller.lambda_value, torch.Tensor)


def test_shared_parameter_scope_matches_audit_prefixes_and_excludes_task_heads():
    model = JointTransformerSceneGate(input_dim=8, scene_dim=4, hidden_dim=16, pred_len=3)
    selected = shared_named_parameters(model)
    names = [name for name, _ in selected]
    assert names == [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(SHARED_GRADIENT_PREFIXES)
    ]
    assert names
    assert all(name.startswith(SHARED_GRADIENT_PREFIXES) for name in names)
    assert all(not name.startswith(("intent_head.", "traj_head.")) for name in names)


def test_aligned_gradient_measurement_does_not_mutate_parameter_grad_buffers():
    shared = nn.Parameter(torch.tensor([2.0, -1.0]))
    head_intent = nn.Parameter(torch.tensor([0.5]))
    head_traj = nn.Parameter(torch.tensor([-0.5]))
    shared.grad = torch.tensor([7.0, 8.0])
    prior_grad = shared.grad.clone()
    intent_loss = (shared.sum() * head_intent).square()
    trajectory_loss = ((shared * head_traj).sum() - 1.0).square()
    result = measure_aligned_task_gradients(
        intent_loss, trajectory_loss, [("fusion.weight", shared)]
    )
    assert shared.grad is not None and torch.equal(shared.grad, prior_grad)
    assert math.isfinite(result["intent_gradient_norm"])
    assert math.isfinite(result["trajectory_gradient_norm_unweighted"])
