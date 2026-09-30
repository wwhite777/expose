import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.covariance import oas


PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_review4_alignment.py"
SPEC = importlib.util.spec_from_file_location("review4_alignment", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_affine_distance_is_congruence_invariant():
    left = np.array([[3.0, 0.4], [0.4, 1.5]])
    right = np.array([[1.8, -0.2], [-0.2, 2.2]])
    transform = np.array([[1.3, 0.5], [-0.1, 0.9]])
    expected = MOD.affine_distance(left, right)
    observed = MOD.affine_distance(transform @ left @ transform.T,
                                   transform @ right @ transform.T)
    assert observed == pytest.approx(expected, abs=1e-12)


def test_reference_inverse_sqrt_whitens_to_identity():
    empirical = np.array([
        [[2.0, 0.2], [0.2, 1.0]],
        [[1.0, -0.1], [-0.1, 3.0]],
    ])
    reference = MOD.empirical_reference(empirical)
    whitener = MOD.inverse_sqrt(reference)
    assert whitener @ reference @ whitener == pytest.approx(np.eye(2), abs=1e-12)


def test_ea_then_oas_matches_direct_centered_epoch_computation():
    rng = np.random.default_rng(20260924)
    epochs = rng.normal(size=(3, 2, MOD.SAMPLES))
    empirical = MOD.empirical_ml(epochs)
    whitener = MOD.inverse_sqrt(MOD.empirical_reference(empirical))
    observed = MOD.ea_before_oas(empirical, whitener)
    transformed = np.einsum("ij,njt->nit", whitener, epochs)
    expected = np.stack([oas(trial.T, assume_centered=False)[0]
                         for trial in transformed])
    assert observed == pytest.approx(expected, rel=1e-12, abs=1e-12)
