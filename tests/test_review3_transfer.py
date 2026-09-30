import copy

import numpy as np
import pytest
from pyriemann.utils.mean import mean_covariance

from expose.review3_transfer import (
    C_GRID, fit_mdwm, fit_mdwm_from_source_means, fit_recenter_references,
    fit_ts_lr_grid, mdwm_source_class_means,
    recenter_source, recenter_target, training_membership_report,
)


def spd(seed, count, channels=3):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(count, channels, channels))
    return factors @ factors.transpose(0, 2, 1) + np.eye(channels)


def balanced(count):
    return np.tile([0, 1], count // 2)


def test_official_mdwm_orientation_has_independent_geodesic_endpoints_and_integer_labels():
    source, target = spd(1, 20), spd(2, 4)
    y_source, y_target = balanced(20), np.array([1, 0, 1, 0])
    source_endpoint = fit_mdwm(source, y_source, target, y_target, lambda_target=0)
    target_endpoint = fit_mdwm(source, y_source, target, y_target, lambda_target=1)
    expected_source = np.stack([
        mean_covariance(source[y_source == label], metric="riemann") for label in (0, 1)
    ])
    expected_target = np.stack([
        mean_covariance(target[y_target == label], metric="riemann") for label in (0, 1)
    ])
    np.testing.assert_allclose(source_endpoint.covmeans_, expected_source, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(target_endpoint.covmeans_, expected_target, rtol=1e-8, atol=1e-10)
    assert source_endpoint.estimator.classes_.tolist() == ["0", "1"]
    assert source_endpoint.predict(source[:3]).dtype.kind in "iu"


def test_mdwm_requires_balanced_binary_target_and_h0_does_not_read_target_labels():
    source, y_source = spd(3, 20), balanced(20)
    target = spd(4, 4)
    with pytest.raises(ValueError, match="equal nonzero counts"):
        fit_mdwm(source, y_source, target, np.array([0, 0, 0, 1]))

    class ExplodingLabels:
        def __array__(self, *args, **kwargs):
            raise AssertionError("h0 target labels were inspected")

    with pytest.raises(ValueError, match="forbidden"):
        fit_mdwm(source, y_source, None, ExplodingLabels())
    h0 = fit_mdwm(source, y_source)
    assert h0.mode == "pooled_mdm_h0" and h0.lambda_target == 0
    assert h0.target_means_ is None


def test_mdwm_prediction_does_not_mutate_fitted_centroids():
    source, target, evaluation = spd(5, 20), spd(6, 4), spd(7, 6)
    model = fit_mdwm(source, balanced(20), target, balanced(4), lambda_target=.5)
    before = (model.covmeans_.copy(), model.source_means_.copy(), model.target_means_.copy())
    probabilities = model.predict_proba(evaluation)
    assert probabilities.shape == (6, 2)
    for observed, expected in zip(
            (model.covmeans_, model.source_means_, model.target_means_), before):
        np.testing.assert_array_equal(observed, expected)


def test_cached_mdwm_source_means_match_official_implementation():
    source, target, evaluation = spd(51, 20), spd(52, 4), spd(53, 7)
    official = fit_mdwm(source, balanced(20), target, balanced(4), lambda_target=.5)
    cached = fit_mdwm_from_source_means(
        mdwm_source_class_means(source, balanced(20)), target, balanced(4), .5)
    np.testing.assert_allclose(cached.covmeans_, official.covmeans_, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(cached.predict_proba(evaluation),
                               official.predict_proba(evaluation), rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(cached.predict(evaluation), official.predict(evaluation))


def test_recentring_is_per_source_person_and_target_history_only():
    source = np.concatenate([spd(8, 10), 3 * spd(9, 10)])
    subjects = np.repeat([11, 22], 10)
    history, evaluation = spd(10, 100), spd(11, 6)
    references = fit_recenter_references(source, subjects, history)
    transformed_source = recenter_source(source, subjects, references)
    transformed_history = recenter_target(history, references)
    identity = np.eye(source.shape[1])
    for subject in (11, 22):
        observed = mean_covariance(transformed_source[subjects == subject], metric="riemann")
        np.testing.assert_allclose(observed, identity, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        mean_covariance(transformed_history, metric="riemann"), identity, rtol=1e-7, atol=1e-8)

    frozen = references.target_whitener.copy()
    first = recenter_target(evaluation, references)
    changed = evaluation.copy()
    changed[0] *= 5
    second = recenter_target(changed, references)
    np.testing.assert_allclose(first[1:], second[1:], rtol=0, atol=0)
    np.testing.assert_array_equal(references.target_whitener, frozen)
    with pytest.raises(ValueError):
        references.target_whitener[0, 0] = 0


def test_recentring_retains_symmetry_at_inverse_volt_scale():
    source = 1e-16 * spd(218, 20)
    history = 1e-16 * spd(219, 100)
    references = fit_recenter_references(source, np.repeat([11, 22], 10), history)
    for whitener in [*references.source_whiteners.values(), references.target_whitener]:
        np.testing.assert_array_equal(whitener, whitener.T)
    transformed = recenter_target(history, references)
    np.testing.assert_allclose(mean_covariance(transformed, metric="riemann"),
                               np.eye(source.shape[1]), rtol=1e-7, atol=1e-8)


def test_source_frozen_ts_lr_grid_and_fixed_membership_roles():
    source, personal_a, personal_b = spd(12, 100), spd(13, 4), spd(14, 4)
    y_source, y_personal = balanced(100), balanced(4)
    first = fit_ts_lr_grid(source, y_source, personal_a, y_personal, "source_frozen")
    second = fit_ts_lr_grid(source, y_source, personal_b, y_personal, "source_frozen")
    assert first.C_values == C_GRID and set(first.classifiers) == set(C_GRID)
    np.testing.assert_allclose(first.transform(source), second.transform(source), rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="C grid"):
        fit_ts_lr_grid(source, y_source, personal_a, y_personal, C_values=(.1, 1, 10, 100))

    source_ids = [f"source-{index}" for index in range(100)]
    reference_ids = [f"history-{index}" for index in range(100)]
    personal_ids = reference_ids[:4]
    evaluation_ids = [f"evaluation-{index}" for index in range(100)]
    report = training_membership_report(
        source_ids, personal_ids, reference_ids, evaluation_ids, "source_frozen")
    assert report["roles"]["representation_fit_trials"]["trial_ids"] == source_ids
    assert report["roles"]["classifier_fit_trials"]["trial_ids"] == source_ids + personal_ids
    assert report["target_reference_is_label_free"] is True
    assert report["evaluation_used_for_fit_or_selection"] is False
    changed = copy.deepcopy(evaluation_ids)
    changed[0] = personal_ids[0]
    with pytest.raises(ValueError, match="evaluation"):
        training_membership_report(source_ids, personal_ids, reference_ids, changed, "source_frozen")
