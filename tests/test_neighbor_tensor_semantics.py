import pytest
import torch

from src.models.scene_social_gate import SceneSocialGate
from src.models.uncertainty_social_gate import UncertaintySocialGate


def model_factories():
    return [
        pytest.param(
            lambda: UncertaintySocialGate(
                input_dim=8, hidden_dim=16, pred_len=3, dropout=0.0, gate_mode="always"
            ),
            False,
            id="uncertainty-social-gate",
        ),
        pytest.param(
            lambda: SceneSocialGate(
                input_dim=8,
                scene_dim=6,
                hidden_dim=16,
                pred_len=3,
                dropout=0.0,
                gate_mode="always",
            ),
            True,
            id="scene-social-gate",
        ),
    ]


@pytest.mark.parametrize("model_factory,uses_scene", model_factories())
def test_each_neighbor_gru_sequence_is_one_pedestrian_over_time(model_factory, uses_scene):
    batch_size, max_neighbors, obs_len, feature_dim = 2, 3, 5, 4
    torch.manual_seed(7)
    # Distinct values on all axes make an accidental neighbor/time swap observable.
    neighbor_obs = torch.arange(
        batch_size * max_neighbors * obs_len * feature_dim, dtype=torch.float32
    ).reshape(batch_size, max_neighbors, obs_len, feature_dim)
    target_obs = torch.zeros(batch_size, obs_len, 8)
    neighbor_mask = torch.ones(batch_size, max_neighbors)
    visible_mask = torch.ones(batch_size, max_neighbors, obs_len)
    captured = {}
    expected = neighbor_obs.reshape(batch_size * max_neighbors, obs_len, feature_dim)

    def inspect_gru_input(_module, inputs):
        sequence = inputs[0].detach().clone()
        assert sequence.shape == (batch_size * max_neighbors, obs_len, feature_dim)
        torch.testing.assert_close(sequence, expected)
        captured["sequence"] = sequence

    model = model_factory().eval()
    hook = model.neighbor_encoder.register_forward_pre_hook(inspect_gru_input)
    with torch.no_grad():
        if uses_scene:
            model(
                target_obs,
                neighbor_obs,
                neighbor_mask,
                visible_mask,
                torch.zeros(batch_size, 6),
            )
        else:
            model(target_obs, neighbor_obs, neighbor_mask, visible_mask)
    hook.remove()

    assert captured["sequence"].shape == (batch_size * max_neighbors, obs_len, feature_dim)
    torch.testing.assert_close(captured["sequence"], expected)


@pytest.mark.parametrize("model_factory,uses_scene", model_factories())
def test_neighbor_mask_shapes_must_follow_batch_neighbor_time_axes(model_factory, uses_scene):
    batch_size, max_neighbors, obs_len, feature_dim = 2, 3, 5, 4
    model = model_factory().eval()
    target_obs = torch.zeros(batch_size, obs_len, 8)
    neighbor_obs = torch.zeros(batch_size, max_neighbors, obs_len, feature_dim)
    neighbor_mask = torch.ones(batch_size, max_neighbors)
    wrong_visible_mask = torch.ones(batch_size, obs_len, max_neighbors)

    with pytest.raises(ValueError, match="neighbor_visible_mask"):
        with torch.no_grad():
            if uses_scene:
                model(
                    target_obs,
                    neighbor_obs,
                    neighbor_mask,
                    wrong_visible_mask,
                    torch.zeros(batch_size, 6),
                )
            else:
                model(target_obs, neighbor_obs, neighbor_mask, wrong_visible_mask)
