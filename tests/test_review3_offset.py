"""Synthetic tests for the additive OpenBMI offset path; no project EEG reads."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_review3_offset.py"
spec = importlib.util.spec_from_file_location("review3_offset", SCRIPT)
offset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offset)


def synthetic_struct():
    channels = list(offset.MOTOR_CHANNELS) + [f"unused{i:02d}" for i in range(54)]
    # Deliberately nonstationary samples make t and t-1 distinguishable.
    signal = np.arange(402000 * 62, dtype=np.float64).reshape(402000, 62)
    matlab_t = 1 + np.arange(100, dtype=np.int64) * 4010
    labels = np.tile([1, 2], 50)
    smt = np.stack([signal[t:t + 4000, :] for t in matlab_t], axis=1)
    data = {"x": signal, "t": matlab_t, "fs": 1000, "y_dec": labels,
            "chan": np.asarray(channels, dtype=object),
            "class": np.asarray([["1", "right"], ["2", "left"]], dtype=object), "smt": smt}
    rows = [{"trial_index_zero_based": str(i), "event_sample_matlab": str(t),
             "event_sample_zero_based": str(t - 1), "label": str(labels[i] - 1),
             "trial_id": f"synthetic:t{i:03d}"} for i, t in enumerate(matlab_t)]
    return data, rows


def test_support_starts_at_stored_matlab_t_and_matches_smt_all_channels():
    data, rows = synthetic_struct()
    supports, y, metadata = offset.validate_and_extract_raw_supports(data, rows)
    first_t = int(data["t"][0])
    selected = metadata["channel_indices_zero_based"]
    assert np.array_equal(supports[0].T, data["x"][first_t:first_t + 4000, selected])
    assert not np.array_equal(supports[0].T, data["x"][first_t - 1:first_t - 1 + 4000, selected])
    assert metadata["smt_all_trials_all_channels_exact"] is True
    assert np.array_equal(y, data["y_dec"] - 1)


def test_support_rejects_one_value_mismatch_in_supplied_smt():
    data, rows = synthetic_struct()
    data["smt"][0, 72, 31] += 1
    with pytest.raises(ValueError, match="trial 72"):
        offset.validate_and_extract_raw_supports(data, rows)


def test_support_rejects_manifest_event_or_label_disagreement():
    data, rows = synthetic_struct()
    rows[9]["event_sample_zero_based"] = str(int(rows[9]["event_sample_zero_based"]) + 1)
    with pytest.raises(ValueError, match="trial 9"):
        offset.validate_and_extract_raw_supports(data, rows)


def test_preprocessing_contract_is_exact_and_returns_750_samples():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(100, 8, 4000))
    X, contract = offset.preprocess_supports(raw)
    assert X.shape == (100, 8, 750)
    assert X.dtype == np.float64
    assert contract["bandpass_hz"] == [8.0, 30.0]
    assert contract["butterworth_order"] == 4
    assert contract["filter_padlen_raw_samples"] == 500
    assert contract["resample"] == {"method": "resample_poly", "up": 1, "down": 4,
                                    "window": ["kaiser", 5.0], "padtype": "line"}
    assert contract["retained_interval_relative_to_support_seconds"] == [0.5, 3.5]


def test_frozen_offset_audit_requires_all_42_exact_records(tmp_path):
    expected = {(i, 1): "source" for i in range(1, 43)}
    rows = [{"subject_id": i, "session": 1, "field_present": True, "comparable": True,
             "shape": [4000, 100, 62], "trials": 100,
             "exact_matches_alternate_t": 100, "exact_matches_legacy_t_minus_1": 0}
            for i in range(1, 43)]
    path = tmp_path / "audit.json"
    path.write_text(__import__("json").dumps(rows), encoding="utf-8")
    assert len(offset.validate_offset_audit(path, expected)) == 42
    rows[7]["exact_matches_alternate_t"] = 99
    path.write_text(__import__("json").dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="exact alternate support"):
        offset.validate_offset_audit(path, expected)
