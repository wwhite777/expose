import numpy as np
import pytest

from expose.review3_data import covariance_record, fit_review3_whitener, legacy_anchor_check


def balanced_epochs(seed=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(100, 20, 750))
    y = np.repeat([0, 1], 50)
    return X, y


def test_compact_record_has_covariances_labels_and_no_epochs():
    X, y = balanced_epochs()
    record = covariance_record(X, y, fit_review3_whitener(X, y))
    assert set(record) == {"cov", "ea_cov", "y"}
    assert record["cov"].shape == record["ea_cov"].shape == (100, 20, 20)
    assert np.array_equal(record["y"], y)


def test_session1_whitener_is_reused_without_session2_refit():
    X, y = balanced_epochs()
    W = fit_review3_whitener(X, y)
    future = X + 0.25
    assert np.array_equal(covariance_record(future, y, W)["y"], y)
    assert not np.array_equal(covariance_record(future, y, W)["ea_cov"], covariance_record(future, y, fit_review3_whitener(future, y))["ea_cov"])


def test_original_eight_anchor_accepts_exact_and_rejects_real_change():
    X, _ = balanced_epochs()
    channels = ("FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")
    positions = [channels.index(name) for name in ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CPz", "CP4")]
    assert legacy_anchor_check(X, X[:, positions, :], channels)["exact"]
    changed = X[:, positions, :].copy(); changed[0, 0, 0] += 1e-6
    with pytest.raises(ValueError, match="frozen anchor"):
        legacy_anchor_check(X, changed, channels)
