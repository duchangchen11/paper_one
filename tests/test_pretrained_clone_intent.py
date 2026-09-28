from __future__ import annotations

import torch
import pytest
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.pretrained_clone_intent_utils import model_sha256, state_sha256
from scripts.analyze_pretrained_clone_intent import paired_indices
from scripts.trajectory_preserving_utils import TrajectoryIntentDataset, set_seed
from src.models.intention_scratch_transformer import IntentionScratchTransformer
from src.models.pretrained_clone_intent import (
    PretrainedCloneIntent,
    clone_target_encoder,
    compare_encoder_storage,
)
from src.models.trajectory_transformer import SceneTrajectoryTransformer


def make_pair(*, d_model: int = 16, layers: int = 1, dropout: float = 0.0, max_obs_len: int = 5):
    trajectory = SceneTrajectoryTransformer(
        input_dim=8,
        scene_dim=6,
        d_model=d_model,
        nhead=4,
        num_layers=layers,
        pred_len=3,
        dropout=dropout,
        max_obs_len=max_obs_len,
    )
    intention = IntentionScratchTransformer(
        input_dim=8,
        d_model=d_model,
        nhead=4,
        num_layers=layers,
        dropout=dropout,
        max_obs_len=max_obs_len,
    )
    return trajectory, intention


def test_pretrained_target_encoder_clones_every_tensor_exactly():
    set_seed(123)
    trajectory, intention = make_pair()
    report = clone_target_encoder(trajectory, intention)
    assert report["clone_pass"] is True
    assert report["expected_tensor_count"] == report["copied_tensor_count"]
    assert report["max_abs_tensor_diff"] == 0.0
    assert report["missing"] == []
    assert report["unexpected"] == []
    assert report["shape_mismatch"] == []


def test_clone_copies_only_target_encoder_not_scene_or_trajectory_decoder():
    set_seed(321)
    trajectory, intention = make_pair()
    report = clone_target_encoder(trajectory, intention)
    assert all(name.startswith(("input_projection.", "temporal_encoder.")) or name == "position_embedding" for name in report["tensor_names"])
    assert not any(name.startswith("scene_encoder.") for name in report["tensor_names"])
    assert not any(name.startswith("decoder.") for name in report["tensor_names"])


def test_cloned_encoders_have_equal_values_but_distinct_parameters_and_storage():
    set_seed(42)
    trajectory, intention = make_pair()
    clone_target_encoder(trajectory, intention)
    result = compare_encoder_storage(trajectory, intention)
    assert result["initial_value_equal"] is True
    assert result["same_parameter_object"] is False
    assert result["same_storage"] is False
    assert result["missing"] == []


def test_m1_random_head_initialization_matches_m0_for_the_same_seed():
    set_seed(123)
    m0 = IntentionScratchTransformer(input_dim=8, d_model=16, nhead=4, num_layers=1, dropout=0.0, max_obs_len=5)
    m0_head = {
        name: tensor.detach().clone()
        for name, tensor in m0.state_dict().items()
        if name.startswith(("intent_adapter.", "intent_head."))
    }
    set_seed(123)
    m1_intention = IntentionScratchTransformer(input_dim=8, d_model=16, nhead=4, num_layers=1, dropout=0.0, max_obs_len=5)
    with torch.random.fork_rng(devices=[]):
        trajectory = SceneTrajectoryTransformer(
            input_dim=8,
            scene_dim=6,
            d_model=16,
            nhead=4,
            num_layers=1,
            pred_len=3,
            dropout=0.0,
            max_obs_len=5,
        )
    m1_head_before_clone = {
        name: tensor.detach().clone()
        for name, tensor in m1_intention.state_dict().items()
        if name.startswith(("intent_adapter.", "intent_head."))
    }
    clone_target_encoder(trajectory, m1_intention)
    m1_head_after_clone = {
        name: tensor.detach()
        for name, tensor in m1_intention.state_dict().items()
        if name.startswith(("intent_adapter.", "intent_head."))
    }
    assert all(torch.equal(m0_head[name], m1_head_before_clone[name]) for name in m0_head)
    assert all(torch.equal(m0_head[name], m1_head_after_clone[name]) for name in m0_head)


def test_optimizer_contains_only_intention_parameters_and_keeps_trajectory_fixed():
    set_seed(7)
    trajectory, intention = make_pair()
    clone_target_encoder(trajectory, intention)
    model = PretrainedCloneIntent(trajectory, intention)
    model.train()
    assert model.trajectory_branch.training is False

    trajectory_before = state_sha256(model.trajectory_branch.state_dict())
    probe_target, probe_scene = torch.randn(4, 5, 8), torch.randn(4, 6)
    trajectory_outputs_before = model.forward_trajectory(probe_target, probe_scene).detach()
    encoder_before = {
        name: tensor.detach().clone()
        for name, tensor in model.intention_branch.state_dict().items()
        if name == "position_embedding" or name.startswith(("input_projection.", "temporal_encoder."))
    }
    optimizer = torch.optim.AdamW(model.intention_branch.parameters(), lr=1e-2)
    trajectory_ids = {id(parameter) for parameter in model.trajectory_branch.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert not (optimizer_ids & trajectory_ids)

    target = torch.randn(8, 5, 8)
    labels = torch.randint(0, 2, (8,), dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(model.forward_intention(target)["intent_logit"], labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    encoder_after = model.intention_branch.state_dict()
    assert any(not torch.equal(encoder_after[name], value) for name, value in encoder_before.items())
    assert state_sha256(model.trajectory_branch.state_dict()) == trajectory_before
    torch.testing.assert_close(
        model.forward_trajectory(probe_target, probe_scene),
        trajectory_outputs_before,
        rtol=0,
        atol=0,
    )


def test_wrapper_train_mode_keeps_trajectory_eval_and_all_parameters_frozen():
    trajectory, intention = make_pair()
    clone_target_encoder(trajectory, intention)
    model = PretrainedCloneIntent(trajectory, intention)
    model.train()
    assert model.trajectory_branch.training is False
    assert all(not parameter.requires_grad for parameter in model.trajectory_branch.parameters())
    assert all(parameter.requires_grad for parameter in model.intention_branch.parameters())


def test_trajectory_forward_matches_reference_for_same_inputs_before_and_after_intent_step():
    set_seed(21)
    trajectory, intention = make_pair()
    clone_target_encoder(trajectory, intention)
    model = PretrainedCloneIntent(trajectory, intention).eval()
    target, scene = torch.randn(4, 5, 8), torch.randn(4, 6)
    with torch.no_grad():
        reference = model.trajectory_branch(target, scene)
        before = model.forward_trajectory(target, scene)
    torch.testing.assert_close(before, reference, rtol=0, atol=0)

    optimizer = torch.optim.AdamW(model.intention_branch.parameters(), lr=1e-2)
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = nn.functional.binary_cross_entropy_with_logits(model.forward_intention(target)["intent_logit"], labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = model.forward_trajectory(target, scene)
    torch.testing.assert_close(after, reference, rtol=0, atol=0)


def test_m1_intention_input_and_architecture_match_m0():
    trajectory, m0 = make_pair()
    clone_target_encoder(trajectory, m0)
    m1 = IntentionScratchTransformer(input_dim=8, d_model=16, nhead=4, num_layers=1, dropout=0.0, max_obs_len=5)
    clone_target_encoder(trajectory, m1)
    m0_shapes = {name: tuple(value.shape) for name, value in m0.state_dict().items()}
    m1_shapes = {name: tuple(value.shape) for name, value in m1.state_dict().items()}
    assert m0_shapes == m1_shapes
    output = m1(torch.randn(3, 5, 8))
    assert output["intent_logit"].shape == (3,)
    assert output["target_context"].shape == (3, 16)
    with pytest.raises(ValueError, match="Expected target history"):
        m1(torch.randn(3, 5, 4))


def test_training_dataset_guard_rejects_test_split_before_loading(monkeypatch, tmp_path):
    load_calls = []

    def forbidden_load(*args, **kwargs):
        load_calls.append(args)
        raise AssertionError("test split must not load before freeze")

    monkeypatch.setattr("scripts.trajectory_preserving_utils.np.load", forbidden_load)
    with pytest.raises(RuntimeError, match="Test split may only be loaded"):
        TrajectoryIntentDataset(tmp_path / "test.npz")
    assert load_calls == []


def test_seed_reproducibility_includes_initial_head_and_dataloader_order():
    def build(seed: int):
        set_seed(seed)
        trajectory, intention = make_pair()
        clone_target_encoder(trajectory, intention)
        head = {
            name: tensor.detach().clone()
            for name, tensor in intention.state_dict().items()
            if name.startswith(("intent_adapter.", "intent_head."))
        }
        loader = DataLoader(
            TensorDataset(torch.arange(24)),
            batch_size=5,
            shuffle=True,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            num_workers=0,
        )
        order = torch.cat([batch[0] for batch in loader])
        return state_sha256(head), model_sha256(intention), order

    first = build(123)
    second = build(123)
    third = build(2024)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert first[0] != third[0]
    assert not torch.equal(first[2], third[2])


def test_paired_prediction_ids_align_reordered_rows():
    reference = {
        "scene_id": torch.tensor([10, 20, 30]).numpy(),
        "video_id": torch.tensor([1, 2, 3]).numpy(),
        "target_id": torch.tensor([101, 202, 303]).numpy(),
        "obs_end_frame": torch.tensor([5, 6, 7]).numpy(),
    }
    target = {name: values[[2, 0, 1]] for name, values in reference.items()}
    assert paired_indices(reference, target).tolist() == [2, 0, 1]


def test_paired_prediction_ids_reject_missing_or_duplicate_rows():
    reference = {
        "scene_id": ["a", "b"], "video_id": ["1", "2"],
        "target_id": ["x", "y"], "obs_end_frame": [3, 4],
    }
    missing = {**reference, "target_id": ["x", "z"]}
    duplicate = {
        "scene_id": ["a", "a"], "video_id": ["1", "1"],
        "target_id": ["x", "x"], "obs_end_frame": [3, 3],
    }
    with pytest.raises(ValueError, match="identities differ"):
        paired_indices(reference, missing)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_indices(reference, duplicate)
