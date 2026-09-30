"""Compact covariance-only records for the guarded Review3 sensitivity."""
import hashlib

import numpy as np

from .review2_controls import apply_ea, fit_ea_whitener, oas_covariances, validate_epochs


def covariance_record(epochs, labels, whitener, expected_channels=20):
    """Return fixed OAS and historical-EA/OAS covariances, never retained epochs."""
    X = np.asarray(epochs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    validate_epochs(X, y)
    if expected_channels not in (8, 20) or X.shape != (100, expected_channels, 750) or y.shape != (100,):
        raise ValueError("Review3 records require exactly 100x8|20x750 epochs and 100 labels")
    if np.bincount(y, minlength=2).tolist() != [50, 50]:
        raise ValueError("Review3 records require exactly 50 trials of each class")
    W = np.asarray(whitener, dtype=np.float64)
    if W.shape != (expected_channels, expected_channels):
        raise ValueError("Review3 EA whitener has the wrong channel count")
    result = {"cov": oas_covariances(X), "ea_cov": oas_covariances(apply_ea(X, W)), "y": y}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Review3 covariance record contains nonfinite values")
    return result


def fit_review3_whitener(session1_epochs, session1_labels, expected_channels=20):
    """Fit the exact all-100 historical EA reference, without labels in its estimator."""
    X = np.asarray(session1_epochs, dtype=np.float64)
    y = np.asarray(session1_labels, dtype=np.int64)
    validate_epochs(X, y)
    if (expected_channels not in (8, 20) or X.shape != (100, expected_channels, 750)
            or np.bincount(y, minlength=2).tolist() != [50, 50]):
        raise ValueError("EA fitting requires all 100 balanced session-1 epochs with 8 or 20 channels")
    return fit_ea_whitener(X)


def legacy_anchor_check(derived_20, old_8, channels):
    """Confirm the original eight processed channels remain the frozen cache anchor."""
    X20, X8 = np.asarray(derived_20), np.asarray(old_8)
    positions = [tuple(channels).index(name) for name in ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CPz", "CP4")]
    selected = X20[:, positions, :]
    if selected.shape != X8.shape:
        raise ValueError("original-eight cache shape differs from derived 20-channel selection")
    difference = float(np.max(np.abs(selected - X8)))
    exact = bool(np.array_equal(selected, X8))
    if not exact and difference > 1e-12:
        raise ValueError(f"derived original-eight values differ from frozen anchor: max_abs={difference}")
    return {"exact": exact, "max_abs_difference": difference, "tolerance": 1e-12,
            "selected_channel_positions": positions}


def whitener_sha256(whitener):
    value = np.ascontiguousarray(np.asarray(whitener, dtype=np.float64))
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()
