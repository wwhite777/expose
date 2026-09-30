import numpy as np
import pytest
from expose.baselines import (assert_disjoint_trial_ids, make_baseline,
                              nested_history_indices, validate_epochs)


def fixture_epochs():
    rng = np.random.default_rng(43)
    y = np.tile([0, 1], 20)
    X = rng.normal(0, 1e-6, size=(40, 8, 200))
    X[y == 1, 0] *= 4
    X[y == 0, 1] *= 4
    return X, y


@pytest.mark.parametrize("name", ["csp_lda", "ts_lr"])
def test_real_pipeline_separates_known_covariance_case_and_predict_does_not_refit(name):
    X, y = fixture_epochs()
    validate_epochs(X, y)
    pipeline = make_baseline(name).fit(X[:30], y[:30])
    if name == "ts_lr":
        frozen = [pipeline.named_steps["tangentspace"].reference_.copy(),
                  pipeline.named_steps["standardscaler"].mean_.copy()]
    else:
        frozen = [pipeline.named_steps["csp"].filters_.copy()]
    probs = pipeline.predict_proba(X[30:])
    assert probs.shape == (10, 2) and np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert np.mean(pipeline.predict(X[30:]) == y[30:]) >= .9
    pipeline.predict_proba(X[30:] * 20)
    current = ([pipeline.named_steps["tangentspace"].reference_,
                pipeline.named_steps["standardscaler"].mean_] if name == "ts_lr"
               else [pipeline.named_steps["csp"].filters_])
    for before, after in zip(frozen, current):
        np.testing.assert_array_equal(before, after)


def test_history_is_nested_and_balanced_without_evaluation():
    y = np.tile([0, 1], 50)
    small = nested_history_indices(y, 5, 20260914)
    large = nested_history_indices(y, 30, 20260914)
    assert set(small) <= set(large)
    np.testing.assert_array_equal(np.bincount(y[large]), [30, 30])
    assert len(nested_history_indices(y, 0, 20260914)) == 0
    with pytest.raises(ValueError, match="insufficient"):
        nested_history_indices(y, 51, 20260914)


@pytest.mark.parametrize("train,evaluation", [([], ["q"]), (["s", "s"], ["q"]),
                                                   (["s"], ["q", "q"]), (["s"], ["s"])])
def test_split_detector_rejects_relevant_defects(train, evaluation):
    with pytest.raises(ValueError):
        assert_disjoint_trial_ids(train, evaluation)


def test_split_detector_accepts_valid_and_epochs_reject_invalid():
    assert_disjoint_trial_ids(["s"], ["q"])
    X, y = fixture_epochs()
    with pytest.raises(ValueError, match="both classes"):
        validate_epochs(X, y * 0)
    with pytest.raises(ValueError, match="binary"):
        validate_epochs(X, y + 1)
    X[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_epochs(X, y)
