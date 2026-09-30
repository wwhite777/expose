"""Strict offline-only Lee2019/OpenBMI MI epochs; see research/day1/LOADER_NOTES.md."""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


# FCz is absent from the published Lee2019 montage. Never synthesize it.
MOTOR_CHANNELS = ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CPz", "CP4")
RAW_CLASSES = {1: "right", 2: "left"}
CLASS_NAMES = ("right_hand", "left_hand")
TRIAL_DURATION_S = 4.0


@dataclass(frozen=True)
class EpochConfig:
    channels: tuple[str, ...] = MOTOR_CHANNELS
    l_freq: float = 8.0
    h_freq: float = 30.0
    tmin: float = 0.5
    tmax: float = 3.5
    target_fs: int = 250
    filter_order: int = 4


@dataclass
class OfflineEpochs:
    X: np.ndarray
    y: np.ndarray
    metadata: dict


def _integer_vector(value, name):
    values = np.asarray(value).squeeze()
    if values.ndim > 1 or values.size == 0 or values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty numeric vector")
    values = np.atleast_1d(values)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{name} must contain finite integers")
    if np.any(values < 0) or np.any(values >= np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains out-of-range integers")
    return values.astype(np.int64)


def _string(value, name):
    value = np.asarray(value).squeeze()
    if value.ndim != 0 or not isinstance(value.item(), str) or not value.item().strip():
        raise ValueError(f"{name} must contain nonempty strings")
    return value.item().strip()


def _sample_count(seconds, fs, name):
    value = seconds * fs
    if not np.isfinite(value) or not np.isclose(value, round(value), rtol=0, atol=1e-8):
        raise ValueError(f"{name} must fall exactly on the sampling grid")
    return int(round(value))


def load_offline_epochs(path: str | Path, config: EpochConfig = EpochConfig()) -> OfflineEpochs:
    """Load only EEG_MI_train; return volts with shape (trials, channels, time).

    Event indices are converted from MATLAB's 1-based t to Python's t - 1.
    Each independent four-second imagery segment is filtered and resampled
    before retaining the half-open interval [tmin, tmax). Labels are 0=right
    and 1=left. No trials are silently discarded and no statistics are fitted.
    """
    if not isinstance(config, EpochConfig):
        raise TypeError("config must be an EpochConfig")
    if not config.channels or len(set(config.channels)) != len(config.channels):
        raise ValueError("requested channels must be nonempty and unique")
    if not all(isinstance(ch, str) and ch for ch in config.channels):
        raise ValueError("requested channels must be nonempty strings")
    if not (0 <= config.tmin < config.tmax <= TRIAL_DURATION_S):
        raise ValueError("epoch interval must satisfy 0 <= tmin < tmax <= 4 seconds")
    if isinstance(config.target_fs, bool) or not isinstance(config.target_fs, int) or config.target_fs <= 0:
        raise ValueError("target_fs must be a positive integer")
    if isinstance(config.filter_order, bool) or not isinstance(config.filter_order, int) or config.filter_order <= 0:
        raise ValueError("filter_order must be a positive integer")

    # Selecting the variable at load time avoids deserializing online EEG/labels.
    mat = loadmat(path, variable_names=["EEG_MI_train"], simplify_cells=True)
    data = mat.get("EEG_MI_train")
    if not isinstance(data, dict):
        raise ValueError("MAT file must contain an EEG_MI_train struct")
    missing = sorted({"x", "t", "fs", "y_dec", "chan", "class"} - data.keys())
    if missing:
        raise ValueError(f"EEG_MI_train missing fields: {missing}")

    fs_values = _integer_vector(data["fs"], "fs")
    if fs_values.size != 1 or fs_values[0] <= 0:
        raise ValueError("fs must be one positive integer sampling rate")
    raw_fs = int(fs_values[0])
    if config.target_fs > raw_fs:
        raise ValueError("target_fs cannot exceed the recorded fs")
    if not (0 < config.l_freq < config.h_freq < min(raw_fs, config.target_fs) / 2):
        raise ValueError("bandpass must lie strictly below both sampling Nyquist frequencies")

    channel_values = np.asarray(data["chan"], dtype=object).squeeze()
    if channel_values.ndim > 1:
        raise ValueError("chan must be a vector of channel names")
    channel_names = [_string(ch, "chan") for ch in np.atleast_1d(channel_values)]
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("chan contains duplicate channel names")
    absent = [ch for ch in config.channels if ch not in channel_names]
    if absent:
        raise ValueError(f"requested channels absent from chan: {absent}")
    selected = [channel_names.index(ch) for ch in config.channels]
    signal = np.asarray(data["x"])
    if signal.ndim != 2 or signal.shape[1] != len(channel_names) or signal.dtype.kind not in "iuf":
        raise ValueError("x must be a real numeric samples-by-channels matrix matching chan")

    class_table = np.asarray(data["class"], dtype=object)
    if class_table.shape != (2, 2):
        raise ValueError("class must be a two-row [numeric code, class name] table")
    file_classes = {}
    for code, name in class_table:
        scalar_code = np.asarray(code).squeeze()
        if scalar_code.ndim == 0 and isinstance(scalar_code.item(), str):
            # The original MAT files encode these numeric codes as char cells.
            code_text = _string(code, "class code")
            if code_text not in ("1", "2"):
                raise ValueError("class code strings must be '1' or '2'")
            code = int(code_text)
        codes = _integer_vector(code, "class code")
        if codes.size != 1 or int(codes[0]) in file_classes:
            raise ValueError("class codes must be distinct scalars")
        file_classes[int(codes[0])] = _string(name, "class name").lower()
    if file_classes != RAW_CLASSES:
        raise ValueError(f"unexpected class mapping: {file_classes}; expected {RAW_CLASSES}")

    labels = _integer_vector(data["y_dec"], "y_dec")
    onsets_matlab = _integer_vector(data["t"], "t")
    if labels.size != onsets_matlab.size:
        raise ValueError("t and y_dec must have equal trial counts")
    if not np.all(np.isin(labels, list(RAW_CLASSES))):
        raise ValueError("y_dec must contain only the declared class codes 1 and 2")
    if "y_class" in data:
        names = np.asarray(data["y_class"], dtype=object).squeeze()
        if names.ndim > 1 or names.size != labels.size:
            raise ValueError("y_class must have one class name per trial")
        names = [_string(name, "y_class").lower() for name in np.atleast_1d(names)]
        if names != [RAW_CLASSES[int(label)] for label in labels]:
            raise ValueError("y_class disagrees with y_dec")
    if "y_logic" in data:
        logical = np.asarray(data["y_logic"])
        if labels.size == 1:
            logical = logical.reshape(-1, 1)
        expected_logical = np.vstack([labels == code for code in (1, 2)])
        if not np.array_equal(logical, expected_logical):
            raise ValueError("y_logic disagrees with y_dec or is not a two-row one-hot matrix")
    if np.any(onsets_matlab < 1) or np.any(np.diff(onsets_matlab) <= 0):
        raise ValueError("t must contain strictly increasing, positive MATLAB 1-based indices")
    onsets = onsets_matlab - 1
    raw_trial_samples = _sample_count(TRIAL_DURATION_S, raw_fs, "trial duration")
    if onsets[-1] > signal.shape[0] - raw_trial_samples:
        raise ValueError("four-second imagery segment extends beyond x")
    if np.any(np.diff(onsets) < raw_trial_samples):
        raise ValueError("four-second imagery segments overlap")

    # Reject ambiguous rounding instead of shifting the analysis window.
    raw_start = _sample_count(config.tmin, raw_fs, "tmin at raw fs")
    raw_stop = _sample_count(config.tmax, raw_fs, "tmax at raw fs")
    out_start = _sample_count(config.tmin, config.target_fs, "tmin at target fs")
    out_stop = _sample_count(config.tmax, config.target_fs, "tmax at target fs")
    padlen = _sample_count(0.5, raw_fs, "filter padding")
    sos = butter(config.filter_order, [config.l_freq, config.h_freq], btype="bandpass", fs=raw_fs, output="sos")
    common = gcd(raw_fs, config.target_fs)
    up, down = config.target_fs // common, raw_fs // common
    X = np.empty((labels.size, len(selected), out_stop - out_start), dtype=np.float64)
    raw_support_hashes = []
    epoch_hashes = []
    for i, onset in enumerate(onsets):
        raw_support = np.ascontiguousarray(signal[onset:onset + raw_trial_samples, selected])
        raw_support_hashes.append(sha256(raw_support.tobytes(order="C")).hexdigest())
        trial = raw_support.T.astype(np.float64) * 1e-6
        if not np.all(np.isfinite(trial)):
            raise ValueError(f"nonfinite EEG in selected channels of trial {i}")
        filtered = sosfiltfilt(sos, trial, axis=-1, padtype="odd", padlen=padlen)
        resampled = resample_poly(filtered, up, down, axis=-1, window=("kaiser", 5.0), padtype="line")
        X[i] = resampled[:, out_start:out_stop]
        epoch_hashes.append(sha256(X[i].tobytes(order="C")).hexdigest())
    if not np.all(np.isfinite(X)):
        raise ValueError("preprocessing produced nonfinite epochs")
    y = labels - 1

    metadata = {
        "path": str(Path(path).resolve()),
        "mat_variable": "EEG_MI_train",
        "loaded_variables": ["EEG_MI_train"],
        "raw_fs": raw_fs,
        "fs": config.target_fs,
        "units": "V",
        "raw_units": "uV",
        "channel_names": list(config.channels),
        "raw_channel_names": channel_names,
        "channel_indices_zero_based": selected,
        "class_names": list(CLASS_NAMES),
        "raw_class_mapping": {str(k): v for k, v in file_classes.items()},
        "label_mapping": {"1": 0, "2": 1},
        "raw_labels": labels.tolist(),
        "class_counts": {name: int(np.sum(y == i)) for i, name in enumerate(CLASS_NAMES)},
        "trial_indices_zero_based": list(range(labels.size)),
        "event_samples_matlab": onsets_matlab.tolist(),
        "event_samples_zero_based": onsets.tolist(),
        "epoch_start_samples_zero_based": (onsets + raw_start).tolist(),
        "epoch_stop_samples_zero_based_exclusive": (onsets + raw_stop).tolist(),
        "filter_input_start_samples_zero_based": onsets.tolist(),
        "filter_input_stop_samples_zero_based_exclusive": (onsets + raw_trial_samples).tolist(),
        "epoch_interval": "[tmin, tmax)",
        "first_output_time_s": config.tmin,
        "last_output_time_s": (out_stop - 1) / config.target_fs,
        "shape": list(X.shape),
        "trial_raw_support_sha256": raw_support_hashes,
        "trial_epoch_sha256": epoch_hashes,
        "hash_encoding": {"byte_order": "array dtype byte order", "array_order": "C",
                          "raw_support_dtype": signal.dtype.str,
                          "raw_support_shape": [raw_trial_samples, len(selected)],
                          "epoch_dtype": X.dtype.str, "epoch_shape": list(X.shape[1:])},
        "config": {**asdict(config), "channels": list(config.channels)},
        "filter": {"design": "Butterworth", "prototype_order": config.filter_order,
                   "sos": sos.tolist(), "direction": "forward-backward",
                   "padtype": "odd", "padlen_raw_samples": padlen,
                   "input_interval_s": [0.0, TRIAL_DURATION_S], "scope": "one trial only"},
        "resample": {"method": "resample_poly", "up": up, "down": down,
                     "window": ["kaiser", 5.0], "padtype": "line", "scope": "one trial only"},
        "checks": {"labels_and_events_match": True, "class_mapping_verified": True,
                   "y_class_crosschecked": "y_class" in data, "y_logic_crosschecked": "y_logic" in data,
                   "requested_channels_present": True, "selected_trial_input_finite": True,
                   "epochs_finite": True, "segments_in_bounds_and_nonoverlapping": True,
                   "learned_preprocessing_statistics": False, "dropped_trials": 0},
    }
    return OfflineEpochs(X=X, y=y, metadata=metadata)
