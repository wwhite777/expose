"""Synthetic BNCI2014-004 contract tests; no public EEG or network is accessed."""

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
from sklearn.covariance import oas

import expose.bnci2b as bnci2b
from expose.bnci2b import BNCI2BConfig, EEG_CHANNELS, load_bnci2b_screening_sessions
from scripts import prepare_review4_bnci2b as prepare


def _screening_session():
    trials = 1 + 1750 * np.arange(120)
    sample_count = int(trials[-1] - 1 + 750 + 1000)
    samples = np.arange(sample_count, dtype=np.float32)[:, None]
    channels = 10000 * np.arange(6, dtype=np.float32)[None, :]
    artifacts = np.zeros(120, dtype=np.uint8)
    artifacts[[0, 7]] = 1
    return {
        "X": samples + channels,
        "trial": trials,
        "y": np.tile([1, 2], 60),
        "fs": 250,
        "classes": np.array(["left hand", "right hand"], dtype=object),
        "artifacts": artifacts,
    }


@pytest.fixture
def mat_data():
    return {
        "data": np.array(
            [_screening_session(), _screening_session(), {"excluded": "03T"}],
            dtype=object,
        )
    }


def _patch_loader(monkeypatch, mat_data):
    calls = []

    def fake_loadmat(stream, **kwargs):
        calls.append(kwargs)
        return mat_data

    monkeypatch.setattr(bnci2b, "loadmat", fake_loadmat)
    monkeypatch.setattr(bnci2b, "sosfiltfilt", lambda sos, values, **kwargs: values)
    return calls


def test_two_screening_sessions_timing_channels_labels_and_artifacts(mat_data, monkeypatch):
    calls = _patch_loader(monkeypatch, mat_data)
    result = load_bnci2b_screening_sessions(b"synthetic", subject=1)

    assert calls == [{"variable_names": ["data"], "simplify_cells": True}]
    assert len(result.sessions) == 2
    first, second = result.sessions
    assert first.X.shape == second.X.shape == (120, 3, 750)
    assert first.X.dtype == np.float64
    np.testing.assert_array_equal(first.y[:4], [1, 0, 1, 0])
    assert np.bincount(first.y).tolist() == [60, 60]
    assert first.X[0, 0, 0] == pytest.approx(875e-6)
    assert first.X[0, 2, 0] == pytest.approx((20000 + 875) * 1e-6)
    assert first.metadata["channel_names"] == list(EEG_CHANNELS)
    assert first.metadata["eog_channels_excluded"] == ["EOG1", "EOG2", "EOG3"]
    assert first.metadata["trial_records"][0]["trial_onset_sample_zero_based"] == 0
    assert first.metadata["trial_records"][0]["cue_onset_sample_zero_based"] == 750
    assert first.metadata["trial_records"][0]["output_start_sample_zero_based"] == 875
    assert first.metadata["trial_records"][0]["output_stop_sample_zero_based_exclusive"] == 1625
    assert first.artifact_flags.sum() == 2
    assert len(first.artifact_flags) == len(first.y)
    assert first.metadata["checks"]["artifact_dropped_trials"] == 0
    assert first.trial_ids[0] == "bnci2014-004:s01:session01:trial001"
    assert second.trial_ids[0] == "bnci2014-004:s01:session02:trial001"
    assert result.metadata["selected_official_sessions"] == ["01T", "02T"]
    assert result.metadata["excluded_official_sessions"] == ["03T"]
    json.dumps(first.metadata, allow_nan=False)


def test_aggregate_must_contain_three_genuine_training_sessions(mat_data, monkeypatch):
    _patch_loader(monkeypatch, {"data": mat_data["data"][:2]})
    with pytest.raises(ValueError, match="exactly three"):
        load_bnci2b_screening_sessions(b"synthetic", subject=1)


def test_documented_nan_gaps_outside_trial_support_are_allowed(mat_data, monkeypatch):
    mat_data["data"][0]["X"][10, :] = np.nan
    _patch_loader(monkeypatch, mat_data)
    result = load_bnci2b_screening_sessions(b"synthetic", subject=1)
    assert result.sessions[0].X.shape == (120, 3, 750)


def test_nonfinite_value_inside_selected_support_is_rejected(mat_data, monkeypatch):
    mat_data["data"][0]["X"][800, 0] = np.nan
    _patch_loader(monkeypatch, mat_data)
    with pytest.raises(ValueError, match="nonfinite EEG"):
        load_bnci2b_screening_sessions(b"synthetic", subject=1)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda run: run.update(fs=500), "documented 250 Hz"),
        (lambda run: run.update(X=run["X"][:, :5]), "samples by 6 channels"),
        (lambda run: run.update(y=run["y"][:-1]), "equal lengths"),
        (lambda run: run.update(artifacts=np.r_[2, np.zeros(119)]), "only 0 or 1"),
        (
            lambda run: run.update(classes=np.array(["right hand", "left hand"])),
            "unexpected class order",
        ),
        (lambda run: run.pop("artifacts"), "missing fields"),
    ],
)
def test_unexpected_screening_schema_is_rejected(mat_data, monkeypatch, mutate, message):
    mutate(mat_data["data"][0])
    _patch_loader(monkeypatch, mat_data)
    with pytest.raises(ValueError, match=message):
        load_bnci2b_screening_sessions(b"synthetic", subject=1)


@pytest.mark.parametrize(
    "config",
    [
        replace(BNCI2BConfig(), tmin=-0.1),
        replace(BNCI2BConfig(), tmax=4.1),
        replace(BNCI2BConfig(), l_freq=30),
        replace(BNCI2BConfig(), filter_order=0),
    ],
)
def test_invalid_recipe_precedes_mat_read(monkeypatch, config):
    monkeypatch.setattr(bnci2b, "loadmat", lambda *args, **kwargs: pytest.fail("unexpected I/O"))
    with pytest.raises((TypeError, ValueError)):
        load_bnci2b_screening_sessions(b"synthetic", subject=1, config=config)


def test_official_identity_and_subject_guard_precede_mat_read(monkeypatch):
    monkeypatch.setattr(bnci2b, "loadmat", lambda *args, **kwargs: pytest.fail("unexpected I/O"))
    assert bnci2b.official_file_url(9).endswith("/004-2014/B09T.mat")
    with pytest.raises(ValueError, match="subject"):
        load_bnci2b_screening_sessions(b"synthetic", subject=0)


def test_covariance_arrays_match_independent_definitions():
    rng = np.random.default_rng(20260924)
    epochs = rng.normal(size=(4, 3, 750))
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    actual = prepare._covariance_arrays(epochs, labels, np.eye(3))

    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    expected_empirical = np.einsum("nct,ndt->ncd", centered, centered) / 750
    expected_oas = np.stack([oas(epoch.T, assume_centered=False)[0] for epoch in epochs])
    np.testing.assert_allclose(actual["empirical_cov"], expected_empirical, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual["cov"], expected_oas, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual["ea_cov"], expected_oas, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual["y"], labels)


def test_dataset_index_requires_exact_external_two_session_cells():
    records = []
    for subject in range(1, 10):
        for session in (1, 2):
            records.append({
                "subject": subject,
                "session": session,
                "role": "external",
                "npz_path": f"data/derived/example/subj{subject:02d}_sess{session}.npz",
                "sha256": "a" * 64,
                "trial_ids": [f"s{subject:02d}:session{session}:trial{i:03d}" for i in range(120)],
            })
    index = {"dataset": "bnci2014b", "montage": "3ch", "records": records}
    prepare._validate_dataset_index(index)
    records[-1]["role"] = "development"
    with pytest.raises(ValueError, match="role"):
        prepare._validate_dataset_index(index)


def test_contract_hash_guard_fails_before_any_network(tmp_path, monkeypatch):
    contract = tmp_path / "contract.json"
    contract.write_text("{}\n")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "_fetch_bytes", lambda *args: pytest.fail("unexpected network"))
    with pytest.raises(ValueError, match="contract hash"):
        prepare._load_approved_contract(contract, "0" * 64, hashlib.sha256(b"x").hexdigest())


def test_checked_contract_and_code_hashes_are_internally_consistent():
    contract = prepare.ROOT / "research/review4_20260924/BNCI2B_DATA_CONTRACT_v1.json"
    contract_sha256 = "d3f57674ae94e6530e5de61349d263b910282b1faf816a6638d289f9fbdc4015"
    script_sha256 = "329676adf39cd86fe5e9c955c6ebc0eb93619b80fdfb03d184c93e427e1a8a49"
    value, checked_path, checked_script_sha256 = prepare._load_approved_contract(
        contract, contract_sha256, script_sha256
    )
    assert value["loader_sha256"] == "26eef40447b5d2f3c4c4a74c61e8ed1b0de798084e3d5a1b5bfe908a171cf7c9"
    assert checked_path == contract
    assert checked_script_sha256 == script_sha256
