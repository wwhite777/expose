"""Small synthetic MAT fixtures test parsing, not scientific performance."""

from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.io import savemat

import expose.lee as lee
from expose.lee import EpochConfig, MOTOR_CHANNELS, load_offline_epochs


@pytest.fixture
def training():
    time = np.arange(9000) / 1000
    frequencies = np.array([12, 15, 18, 20, 22, 25, 10, 16])
    return {
        "x": np.sin(2 * np.pi * time[:, None] * frequencies) * np.arange(10, 18),
        "t": np.array([1, 5001]),
        "fs": 1000,
        "chan": np.array(MOTOR_CHANNELS, dtype=object),
        "y_dec": np.array([2, 1]),
        "y_class": np.array(["left", "right"], dtype=object),
        "y_logic": np.array([[0, 1], [1, 0]], dtype=np.uint8),
        "class": np.array([["1", "right"], ["2", "left"]], dtype=object),
    }


def save_training(tmp_path, training, *, name="synthetic.mat", online=None):
    path = tmp_path / name
    savemat(path, {"EEG_MI_train": training, "EEG_MI_test": {"x": np.nan} if online is None else online})
    return path


def test_default_epochs_shape_labels_channels_and_metadata(tmp_path, training):
    epochs = load_offline_epochs(save_training(tmp_path, training))
    assert epochs.X.shape == (2, 8, 750)
    assert epochs.X.dtype == np.float64
    assert np.all(np.isfinite(epochs.X))
    np.testing.assert_array_equal(epochs.y, [1, 0])
    assert epochs.metadata["channel_names"] == list(MOTOR_CHANNELS)
    assert epochs.metadata["class_names"] == ["right_hand", "left_hand"]
    assert epochs.metadata["class_counts"] == {"right_hand": 1, "left_hand": 1}
    assert epochs.metadata["event_samples_zero_based"] == [0, 5000]
    assert epochs.metadata["epoch_start_samples_zero_based"] == [500, 5500]
    assert epochs.metadata["epoch_stop_samples_zero_based_exclusive"] == [3500, 8500]
    assert epochs.metadata["first_output_time_s"] == 0.5
    assert epochs.metadata["last_output_time_s"] == 3.496
    assert epochs.metadata["fs"] == 250
    assert epochs.metadata["raw_fs"] == 1000
    # A 12 Hz, 10 uV signal should retain its physical amplitude in volts.
    expected = 10e-6 * np.sin(2 * np.pi * 12 * (0.5 + np.arange(750) / 250))
    np.testing.assert_allclose(epochs.X[0, 0], expected, atol=0.2e-6)
    json.dumps(epochs.metadata, allow_nan=False)


def test_one_based_event_conversion_exact_samples_and_requested_order(tmp_path, training, monkeypatch):
    # Isolate indexing from filtering: the sample value itself encodes its index.
    training["x"] = np.arange(9000)[:, None] + 10000 * np.arange(8)[None, :]
    monkeypatch.setattr(lee, "sosfiltfilt", lambda sos, x, **kwargs: x)
    config = EpochConfig(channels=("C4", "C3"), target_fs=1000)
    epochs = load_offline_epochs(save_training(tmp_path, training), config)
    assert epochs.X.shape == (2, 2, 3000)
    np.testing.assert_allclose(epochs.X[0, 0], (20000 + np.arange(500, 3500)) * 1e-6)
    np.testing.assert_allclose(epochs.X[1, 1], np.arange(5500, 8500) * 1e-6)
    assert epochs.metadata["channel_indices_zero_based"] == [2, 0]


def test_loadmat_only_deserializes_offline_training(tmp_path, training, monkeypatch):
    path = save_training(tmp_path, training, online={"t": [-999], "y_dec": [999], "x": [np.nan]})
    real_loadmat = lee.loadmat
    calls = []

    def checked_loadmat(path, **kwargs):
        calls.append(kwargs)
        result = real_loadmat(path, **kwargs)
        assert "EEG_MI_test" not in result
        return result

    monkeypatch.setattr(lee, "loadmat", checked_loadmat)
    load_offline_epochs(path)
    assert calls == [{"variable_names": ["EEG_MI_train"], "simplify_cells": True}]


def test_other_trials_and_intertrial_samples_cannot_change_first_epoch(tmp_path, training):
    original = load_offline_epochs(save_training(tmp_path, training, name="original.mat"))
    training["x"][4000:] = 1e12 * np.random.default_rng(7).normal(size=(5000, 8))
    changed = load_offline_epochs(save_training(tmp_path, training, name="changed.mat"))
    np.testing.assert_array_equal(changed.X[0], original.X[0])
    assert not np.array_equal(changed.X[1], original.X[1])
    for key in ("trial_raw_support_sha256", "trial_epoch_sha256"):
        assert changed.metadata[key][0] == original.metadata[key][0]
        assert changed.metadata[key][1] != original.metadata[key][1]
        assert all(len(digest) == 64 for digest in changed.metadata[key])


def test_nondefault_recorded_sampling_rate_is_used(tmp_path, training):
    training["fs"] = 500
    training["t"] = np.array([1, 2501])
    epochs = load_offline_epochs(save_training(tmp_path, training))
    assert epochs.X.shape == (2, 8, 750)
    assert epochs.metadata["raw_fs"] == 500
    assert epochs.metadata["filter"]["padlen_raw_samples"] == 250
    assert epochs.metadata["resample"]["down"] == 2
    assert epochs.metadata["epoch_start_samples_zero_based"] == [250, 2750]


@pytest.mark.parametrize("field", ["x", "t", "fs", "y_dec", "chan", "class"])
def test_missing_required_fields_raise(tmp_path, training, field):
    del training[field]
    with pytest.raises(ValueError, match="missing fields"):
        load_offline_epochs(save_training(tmp_path, training))


@pytest.mark.parametrize(("field", "value", "message"), [
    ("t", [0, 5001], "positive MATLAB"),
    ("t", [1.5, 5001], "finite integers"),
    ("t", [5001, 1], "strictly increasing"),
    ("t", [1, 4000], "overlap"),
    ("t", [1, 5002], "extends beyond"),
    ("t", [1], "equal trial counts"),
    ("y_dec", [1, 3], "declared class codes"),
    ("y_dec", [1, np.nan], "finite integers"),
    ("y_class", np.array(["right", "left"], dtype=object), "y_class disagrees"),
    ("y_logic", [[1, 0], [0, 1]], "y_logic disagrees"),
    ("fs", 0, "positive integer sampling rate"),
    ("fs", 1000.5, "finite integers"),
    ("fs", [500, 1000], "one positive integer"),
    ("class", np.array([[1, "left"], [2, "right"]], dtype=object), "unexpected class mapping"),
    ("class", np.array([[1, "right"], [1, "left"]], dtype=object), "distinct scalars"),
])
def test_invalid_fields_raise(tmp_path, training, field, value, message):
    training[field] = value
    with pytest.raises(ValueError, match=message):
        load_offline_epochs(save_training(tmp_path, training))


def test_no_offline_struct_does_not_fall_back_to_online(tmp_path):
    path = tmp_path / "online_only.mat"
    savemat(path, {"EEG_MI_test": {"x": np.zeros((100, 8))}})
    with pytest.raises(ValueError, match="EEG_MI_train struct"):
        load_offline_epochs(path)


def test_numeric_class_table_codes_are_also_supported(tmp_path, training):
    training["class"] = np.array([[1, "right"], [2, "left"]], dtype=object)
    epochs = load_offline_epochs(save_training(tmp_path, training))
    np.testing.assert_array_equal(epochs.y, [1, 0])


def test_noncanonical_class_code_string_is_rejected(tmp_path, training):
    training["class"][0, 0] = "1.5"
    with pytest.raises(ValueError, match="class code strings"):
        load_offline_epochs(save_training(tmp_path, training))


def test_fc_z_is_not_substituted_or_interpolated(tmp_path, training):
    with pytest.raises(ValueError, match="absent.*FCz"):
        load_offline_epochs(save_training(tmp_path, training), EpochConfig(channels=("FCz",)))


def test_duplicate_channels_are_rejected(tmp_path, training):
    training["chan"][1] = "C3"
    with pytest.raises(ValueError, match="duplicate channel"):
        load_offline_epochs(save_training(tmp_path, training))


def test_transposed_eeg_is_rejected(tmp_path, training):
    training["x"] = training["x"].T
    with pytest.raises(ValueError, match="samples-by-channels"):
        load_offline_epochs(save_training(tmp_path, training))


def test_nonfinite_selected_trial_eeg_is_rejected(tmp_path, training):
    training["x"][6000, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite EEG.*trial 1"):
        load_offline_epochs(save_training(tmp_path, training))


@pytest.mark.parametrize("update", [
    {"target_fs": 2000}, {"target_fs": 0}, {"target_fs": 100.5},
    {"tmin": -0.1}, {"tmax": 4.1}, {"tmin": 3.6}, {"tmin": 0.501},
    {"l_freq": 35}, {"h_freq": 125}, {"filter_order": 0}, {"filter_order": 2.5},
    {"channels": ()}, {"channels": ("C3", "C3")},
])
def test_invalid_recipe_is_rejected(tmp_path, training, update):
    with pytest.raises(ValueError):
        load_offline_epochs(save_training(tmp_path, training), replace(EpochConfig(), **update))
