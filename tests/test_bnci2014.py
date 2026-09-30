"""Synthetic BNCI2014-001 MAT-contract tests; no public EEG is accessed."""

from dataclasses import replace
import json

import numpy as np
import pytest
from sklearn.covariance import oas

import expose.bnci2014 as bnci
from expose.bnci2014 import BNCI2014Config, EEG_CHANNELS, load_bnci2014_epochs
from expose.review2_controls import oas_covariances
from scripts.prepare_review3_bnci import _empirical_covariances


def _run(*, empty=False):
    classes = np.array(["left hand", "right hand", "feet", "tongue"], dtype=object)
    if empty:
        return {
            "X": np.zeros((1, 25), dtype=np.float32),
            "trial": np.array([], dtype=np.int64),
            "y": np.array([], dtype=np.int64),
            "fs": 250,
            "classes": classes,
            "artifacts": np.array([], dtype=np.uint8),
        }
    trial = 1 + 1000 * np.arange(48)
    sample = np.arange(48500, dtype=np.float32)[:, None]
    channel = 100000 * np.arange(25, dtype=np.float32)[None, :]
    artifacts = np.zeros(48, dtype=np.uint8)
    artifacts[[0, 1, 7]] = 1
    return {
        "X": sample + channel,
        "trial": trial,
        "y": np.tile([1, 2, 3, 4], 12),
        "fs": 250,
        "classes": classes,
        "artifacts": artifacts,
    }


@pytest.fixture
def mat_data():
    return {"data": np.array([_run(empty=True) for _ in range(3)] + [_run() for _ in range(6)], dtype=object)}


def _patch_mat(monkeypatch, mat_data):
    calls = []

    def fake_loadmat(path, **kwargs):
        calls.append(kwargs)
        return mat_data

    monkeypatch.setattr(bnci, "loadmat", fake_loadmat)
    monkeypatch.setattr(bnci, "sosfiltfilt", lambda sos, x, **kwargs: x)
    monkeypatch.setattr(bnci, "resample_poly", lambda x, up, down, **kwargs: x)
    return calls


def test_contract_timing_labels_channels_artifacts_and_identity(tmp_path, mat_data, monkeypatch):
    calls = _patch_mat(monkeypatch, mat_data)
    path = tmp_path / "A01T.mat"
    result = load_bnci2014_epochs(path, subject=1, session="T")

    assert calls == [{"variable_names": ["data"], "simplify_cells": True}]
    assert result.X.shape == (144, 22, 750)
    assert result.X.dtype == np.float64
    np.testing.assert_array_equal(result.y[:4], [1, 0, 1, 0])
    assert np.bincount(result.y).tolist() == [72, 72]
    # Trial onset 1 -> zero-based 0; cue +2 s, output starts +0.5 s.
    assert result.X[0, 0, 0] == pytest.approx(625e-6)
    assert result.X[0, 21, 0] == pytest.approx((2100000 + 625) * 1e-6)
    assert result.metadata["eog_channels_excluded"] == ["EOG1", "EOG2", "EOG3"]
    assert result.metadata["channel_names"] == list(EEG_CHANNELS)
    assert result.metadata["skipped_empty_run_indices_zero_based"] == [0, 1, 2]
    assert result.metadata["trial_records"][0]["cue_onset_sample_zero_based"] == 500
    assert result.metadata["trial_records"][0]["output_start_sample_zero_based"] == 625
    assert result.metadata["trial_records"][0]["output_stop_sample_zero_based_exclusive"] == 1375
    assert result.metadata["trial_records"][0]["artifact_flag"] == 1
    assert result.artifact_flags.sum() == 12  # two selected flagged trials per MI run
    assert len(result.artifact_flags) == len(result.y)
    assert result.metadata["checks"]["artifact_dropped_trials"] == 0
    assert result.trial_ids[0] == "bnci2014-001:s01:sessionT:run01:trial001"
    assert len(set(result.trial_ids)) == 144
    json.dumps(result.metadata, allow_nan=False)


def test_evaluation_session_identity_and_official_url(tmp_path, mat_data, monkeypatch):
    _patch_mat(monkeypatch, mat_data)
    result = load_bnci2014_epochs(tmp_path / "A09E.mat", subject=9, session="E")
    assert result.metadata["session_role"] == "evaluation"
    assert result.metadata["sessions_recorded_on_different_days"] is True
    assert result.metadata["official_url"].endswith("/A09E.mat")
    assert all(":s09:sessionE:" in trial_id for trial_id in result.trial_ids)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda runs: runs[3].update(fs=500), "documented 250 Hz"),
        (lambda runs: runs[3].update(X=runs[3]["X"][:, :24]), "samples by 25 channels"),
        (lambda runs: runs[3].update(artifacts=np.zeros(47)), "equal lengths"),
        (lambda runs: runs[3].update(artifacts=np.r_[2, np.zeros(47)]), "only 0 or 1"),
        (lambda runs: runs[3].update(classes=np.array(["right hand", "left hand", "feet", "tongue"])),
         "unexpected class order"),
        (lambda runs: runs[3].pop("artifacts"), "missing fields"),
    ],
)
def test_invalid_mat_contract_is_rejected(tmp_path, mat_data, monkeypatch, mutate, message):
    runs = list(mat_data["data"])
    mutate(runs)
    _patch_mat(monkeypatch, {"data": np.array(runs, dtype=object)})
    with pytest.raises(ValueError, match=message):
        load_bnci2014_epochs(tmp_path / "A01T.mat", subject=1, session="T")


@pytest.mark.parametrize(
    ("subject", "session", "name", "message"),
    [
        (0, "T", "A00T.mat", "subject"),
        (1, "X", "A01X.mat", "session"),
        (1, "T", "wrong.mat", "filename"),
    ],
)
def test_identity_guards_precede_io(tmp_path, monkeypatch, subject, session, name, message):
    monkeypatch.setattr(bnci, "loadmat", lambda *args, **kwargs: pytest.fail("unexpected I/O"))
    with pytest.raises((TypeError, ValueError), match=message):
        load_bnci2014_epochs(tmp_path / name, subject=subject, session=session)


@pytest.mark.parametrize(
    "config",
    [
        replace(BNCI2014Config(), target_fs=251),
        replace(BNCI2014Config(), target_fs=100.5),
        replace(BNCI2014Config(), tmin=-0.1),
        replace(BNCI2014Config(), tmax=4.1),
        replace(BNCI2014Config(), l_freq=30),
        replace(BNCI2014Config(), filter_order=0),
    ],
)
def test_invalid_recipe_precedes_io(tmp_path, monkeypatch, config):
    monkeypatch.setattr(bnci, "loadmat", lambda *args, **kwargs: pytest.fail("unexpected I/O"))
    with pytest.raises((TypeError, ValueError)):
        load_bnci2014_epochs(tmp_path / "A01T.mat", subject=1, session="T", config=config)


def test_missing_data_struct_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(bnci, "loadmat", lambda *args, **kwargs: {})
    with pytest.raises(ValueError, match="data struct"):
        load_bnci2014_epochs(tmp_path / "A01T.mat", subject=1, session="T")


def test_empirical_and_oas_covariances_match_separate_analytic_definitions():
    rng = np.random.default_rng(20260923)
    epochs = rng.normal(size=(3, 22, 750))
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    expected_empirical = np.einsum("nct,ndt->ncd", centered, centered) / 750
    np.testing.assert_allclose(
        _empirical_covariances(epochs), expected_empirical, rtol=1e-14, atol=1e-14
    )

    expected_oas = np.stack([oas(epoch.T, assume_centered=False)[0] for epoch in epochs])
    actual_oas = oas_covariances(epochs)
    np.testing.assert_allclose(actual_oas, expected_oas, rtol=1e-14, atol=1e-14)
    assert not np.allclose(actual_oas, expected_empirical, rtol=1e-7, atol=1e-10)
