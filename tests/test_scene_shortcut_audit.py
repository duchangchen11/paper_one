import numpy as np

from scripts.run_scene_shortcut_audit import (
    build_scene_swap_indices,
    cosine_rows,
    pair_summary,
    target_groups,
)


def test_cosine_similarity_handles_zero_vectors_and_matches():
    left = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    right = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(cosine_rows(left, right), [1.0, 0.0])
    summary = pair_summary(left, right, population=2)
    assert summary["exact_feature_match_fraction"] == 1.0
    assert summary["zero_vector_pair_fraction"] == 0.5


def test_same_video_mapping_is_deterministic_and_never_self_donates():
    scenes = np.asarray(["v1"] * 4 + ["v2"] * 3)
    donor_a, manifest_a = build_scene_swap_indices(scenes, "same_video", seed=3)
    donor_b, manifest_b = build_scene_swap_indices(scenes, "same_video", seed=3)
    np.testing.assert_array_equal(donor_a, donor_b)
    assert manifest_a["mapping_sha256"] == manifest_b["mapping_sha256"]
    assert np.all(donor_a != np.arange(len(scenes)))
    np.testing.assert_array_equal(scenes[donor_a], scenes)


def test_cross_video_mapping_is_deterministic_and_always_changes_video():
    scenes = np.asarray(["v1"] * 3 + ["v2"] * 2 + ["v3"] * 4)
    donor, _ = build_scene_swap_indices(scenes, "cross_video", seed=10)
    np.testing.assert_array_equal(scenes[donor], np.asarray(["v2"] * 3 + ["v3"] * 2 + ["v1"] * 4))


def test_target_groups_keep_person_windows_together():
    data = {
        "scene_id": np.asarray(["v1", "v1", "v1", "v1"]),
        "target_id": np.asarray(["p1", "p1", "p2", "p2"]),
        "intent_label": np.asarray([1, 1, 0, 0]),
    }
    groups, labels = target_groups(data)
    assert set(groups) == {("v1", "p1"), ("v1", "p2")}
    np.testing.assert_array_equal(groups[("v1", "p1")], [0, 1])
    assert labels == {("v1", "p1"): 1, ("v1", "p2"): 0}
