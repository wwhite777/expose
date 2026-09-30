"""Strict local loader for BNCI2014-001 / BCI Competition IV 2a MAT files.

The loader performs no network access.  It uses the public BNCI MAT contract
implemented by MOABB: ``run.trial`` is a MATLAB 1-based trial-onset index and
the motor-imagery interval is two through six seconds after that onset.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt


BASE_URL = "https://lampx.tugraz.at/~bci/database/001-2014"
EEG_CHANNELS = (
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
)
EOG_CHANNELS = ("EOG1", "EOG2", "EOG3")
RAW_CLASS_NAMES = ("left_hand", "right_hand", "feet", "tongue")
LABEL_MAPPING = {2: 0, 1: 1}  # EXPOSE convention: 0=right, 1=left.
CLASS_NAMES = ("right_hand", "left_hand")
RECORDED_FS = 250
TRIAL_ONSET_TO_CUE_S = 2.0
IMAGERY_DURATION_S = 4.0
EXPECTED_MI_RUNS = 6
EXPECTED_TRIALS_PER_RUN = 48


@dataclass(frozen=True)
class BNCI2014Config:
    l_freq: float = 8.0
    h_freq: float = 30.0
    tmin: float = 0.5
    tmax: float = 3.5
    target_fs: int = 250
    filter_order: int = 4


@dataclass
class BNCI2014Epochs:
    X: np.ndarray
    y: np.ndarray
    artifact_flags: np.ndarray
    trial_ids: tuple[str, ...]
    metadata: dict


def official_file_url(subject: int, session: str) -> str:
    """Return the official upstream MAT URL without accessing the network."""
    subject, session = _validate_identity(subject, session)
    return f"{BASE_URL}/A{subject:02d}{session}.mat"


def _validate_identity(subject, session):
    if isinstance(subject, bool) or not isinstance(subject, (int, np.integer)):
        raise TypeError("subject must be an integer from 1 through 9")
    subject = int(subject)
    if subject not in range(1, 10):
        raise ValueError("subject must be from 1 through 9")
    if session not in ("T", "E"):
        raise ValueError("session must be 'T' or 'E'")
    return subject, session


def _integer_vector(value, name, *, allow_empty=False):
    values = np.asarray(value).squeeze()
    if values.size == 0 and allow_empty:
        return np.empty(0, dtype=np.int64)
    if values.ndim > 1 or values.size == 0 or values.dtype.kind not in "iufb":
        raise ValueError(f"{name} must be a numeric vector")
    values = np.atleast_1d(values)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)):
        raise ValueError(f"{name} must contain finite integers")
    if np.any(values < 0) or np.any(values >= np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains out-of-range integers")
    return values.astype(np.int64)


def _sample_count(seconds, fs, name):
    value = seconds * fs
    if not np.isfinite(value) or not np.isclose(value, round(value), rtol=0, atol=1e-8):
        raise ValueError(f"{name} must fall exactly on the sampling grid")
    return int(round(value))


def _runs(value):
    if isinstance(value, dict):
        return [value]
    array = np.asarray(value, dtype=object).squeeze()
    if array.ndim == 0:
        return [array.item()]
    if array.ndim != 1:
        raise ValueError("data must be a scalar or one-dimensional run struct array")
    return list(array)


def _normalized_class(value):
    value = np.asarray(value).squeeze()
    if value.ndim != 0 or not isinstance(value.item(), str):
        raise ValueError("classes must contain strings")
    return "_".join(value.item().strip().lower().replace("both ", "").split())


def _validate_config(config):
    if not isinstance(config, BNCI2014Config):
        raise TypeError("config must be a BNCI2014Config")
    if not (0 <= config.tmin < config.tmax <= IMAGERY_DURATION_S):
        raise ValueError("epoch interval must satisfy 0 <= tmin < tmax <= 4 seconds")
    if isinstance(config.target_fs, bool) or not isinstance(config.target_fs, int) or config.target_fs <= 0:
        raise ValueError("target_fs must be a positive integer")
    if config.target_fs > RECORDED_FS:
        raise ValueError("target_fs cannot exceed the recorded 250 Hz")
    if isinstance(config.filter_order, bool) or not isinstance(config.filter_order, int) or config.filter_order <= 0:
        raise ValueError("filter_order must be a positive integer")
    if not (0 < config.l_freq < config.h_freq < min(RECORDED_FS, config.target_fs) / 2):
        raise ValueError("bandpass must lie strictly below both sampling Nyquist frequencies")


def load_bnci2014_epochs(
    path: str | Path,
    *,
    subject: int,
    session: str,
    config: BNCI2014Config = BNCI2014Config(),
) -> BNCI2014Epochs:
    """Load all left/right trials from one official session MAT file.

    The returned signal is float64 volts with shape ``(trials, 22, time)``.
    Raw labels 2 (right) and 1 (left) map to 0 and 1.  Expert artifact flags
    remain aligned with the returned trials and do not cause exclusion.
    Filtering and resampling are performed independently on each four-second
    cue-to-end imagery segment before retaining ``[tmin, tmax)``.
    """
    subject, session = _validate_identity(subject, session)
    _validate_config(config)
    path = Path(path)
    expected_name = f"A{subject:02d}{session}.mat"
    if path.name != expected_name:
        raise ValueError(f"filename must be {expected_name} for the declared subject/session")

    mat = loadmat(path, variable_names=["data"], simplify_cells=True)
    run_values = mat.get("data")
    if run_values is None:
        raise ValueError("MAT file must contain the BNCI data struct")
    all_runs = _runs(run_values)

    raw_support_samples = _sample_count(IMAGERY_DURATION_S, RECORDED_FS, "imagery duration")
    cue_offset = _sample_count(TRIAL_ONSET_TO_CUE_S, RECORDED_FS, "cue offset")
    raw_output_start = _sample_count(config.tmin, RECORDED_FS, "tmin at raw fs")
    raw_output_stop = _sample_count(config.tmax, RECORDED_FS, "tmax at raw fs")
    output_start = _sample_count(config.tmin, config.target_fs, "tmin at target fs")
    output_stop = _sample_count(config.tmax, config.target_fs, "tmax at target fs")
    padlen = _sample_count(0.5, RECORDED_FS, "filter padding")
    sos = butter(
        config.filter_order,
        [config.l_freq, config.h_freq],
        btype="bandpass",
        fs=RECORDED_FS,
        output="sos",
    )
    common = gcd(RECORDED_FS, config.target_fs)
    up, down = config.target_fs // common, RECORDED_FS // common

    epochs = []
    output_labels = []
    output_artifacts = []
    trial_ids = []
    raw_labels = []
    raw_support_hashes = []
    epoch_hashes = []
    trial_records = []
    skipped_empty_runs = []
    mi_run_number = 0
    raw_support_dtype = None

    for source_run_index, run in enumerate(all_runs):
        if not isinstance(run, dict):
            raise ValueError(f"data run {source_run_index} is not a struct")
        missing = sorted({"X", "trial", "fs"} - run.keys())
        if missing:
            raise ValueError(f"data run {source_run_index} missing fields: {missing}")

        trials = _integer_vector(run["trial"], f"run {source_run_index} trial", allow_empty=True)
        if trials.size == 0:
            skipped_empty_runs.append(source_run_index)
            continue
        missing = sorted({"y", "classes", "artifacts"} - run.keys())
        if missing:
            raise ValueError(f"data run {source_run_index} missing fields: {missing}")
        mi_run_number += 1
        if trials.size != EXPECTED_TRIALS_PER_RUN:
            raise ValueError(f"MI run {mi_run_number} must contain 48 trials")
        if np.any(trials < 1) or np.any(np.diff(trials) <= 0):
            raise ValueError(f"run {source_run_index} trial must be increasing MATLAB 1-based indices")

        fs = _integer_vector(run["fs"], f"run {source_run_index} fs")
        if fs.size != 1 or int(fs[0]) != RECORDED_FS:
            raise ValueError(f"run {source_run_index} fs must equal the documented 250 Hz")
        classes = tuple(_normalized_class(value) for value in np.atleast_1d(run["classes"]))
        if classes != RAW_CLASS_NAMES:
            raise ValueError(f"run {source_run_index} has unexpected class order {classes}")

        labels = _integer_vector(run["y"], f"run {source_run_index} y")
        artifacts = _integer_vector(run["artifacts"], f"run {source_run_index} artifacts")
        if labels.size != trials.size or artifacts.size != trials.size:
            raise ValueError(f"run {source_run_index} trial, y and artifacts must have equal lengths")
        if not np.all(np.isin(labels, (1, 2, 3, 4))):
            raise ValueError(f"run {source_run_index} y must contain only class codes 1 through 4")
        if not np.all(np.isin(artifacts, (0, 1))):
            raise ValueError(f"run {source_run_index} artifacts must contain only 0 or 1")

        signal = np.asarray(run["X"])
        if signal.ndim != 2 or signal.shape[1] != len(EEG_CHANNELS) + len(EOG_CHANNELS):
            raise ValueError(f"run {source_run_index} X must be samples by 25 channels")
        if signal.dtype.kind not in "iuf":
            raise ValueError(f"run {source_run_index} X must be real numeric")

        trial_onsets = trials - 1
        cue_onsets = trial_onsets + cue_offset
        if cue_onsets[-1] > signal.shape[0] - raw_support_samples:
            raise ValueError(f"run {source_run_index} four-second imagery segment extends beyond X")
        if np.any(np.diff(cue_onsets) < raw_support_samples):
            raise ValueError(f"run {source_run_index} four-second imagery segments overlap")

        for trial_index, (trial_onset, cue_onset, label, artifact) in enumerate(
            zip(trial_onsets, cue_onsets, labels, artifacts)
        ):
            if int(label) not in LABEL_MAPPING:
                continue
            raw_support = np.ascontiguousarray(
                signal[cue_onset:cue_onset + raw_support_samples, :len(EEG_CHANNELS)]
            )
            if raw_support_dtype is None:
                raw_support_dtype = raw_support.dtype.str
            trial = raw_support.T.astype(np.float64) * 1e-6
            if not np.all(np.isfinite(trial)):
                raise ValueError(
                    f"nonfinite EEG in MI run {mi_run_number}, trial {trial_index + 1}"
                )
            filtered = sosfiltfilt(sos, trial, axis=-1, padtype="odd", padlen=padlen)
            resampled = resample_poly(
                filtered,
                up,
                down,
                axis=-1,
                window=("kaiser", 5.0),
                padtype="line",
            )
            epoch = np.ascontiguousarray(resampled[:, output_start:output_stop], dtype=np.float64)
            if epoch.shape != (len(EEG_CHANNELS), output_stop - output_start):
                raise ValueError("preprocessing produced an unexpected epoch shape")
            if not np.all(np.isfinite(epoch)):
                raise ValueError("preprocessing produced nonfinite epochs")

            label = int(label)
            artifact = int(artifact)
            trial_id = (
                f"bnci2014-001:s{subject:02d}:session{session}:"
                f"run{mi_run_number:02d}:trial{trial_index + 1:03d}"
            )
            epochs.append(epoch)
            output_labels.append(LABEL_MAPPING[label])
            output_artifacts.append(artifact)
            trial_ids.append(trial_id)
            raw_labels.append(label)
            raw_support_hashes.append(sha256(raw_support.tobytes(order="C")).hexdigest())
            epoch_hashes.append(sha256(epoch.tobytes(order="C")).hexdigest())
            trial_records.append(
                {
                    "trial_id": trial_id,
                    "source_run_index_zero_based": source_run_index,
                    "mi_run_index_one_based": mi_run_number,
                    "run_trial_index_one_based": trial_index + 1,
                    "trial_onset_sample_matlab": int(trial_onset + 1),
                    "trial_onset_sample_zero_based": int(trial_onset),
                    "cue_onset_sample_zero_based": int(cue_onset),
                    "filter_input_start_sample_zero_based": int(cue_onset),
                    "filter_input_stop_sample_zero_based_exclusive": int(
                        cue_onset + raw_support_samples
                    ),
                    "output_start_sample_zero_based": int(
                        cue_onset + raw_output_start
                    ),
                    "output_stop_sample_zero_based_exclusive": int(
                        cue_onset + raw_output_stop
                    ),
                    "raw_label": label,
                    "label": LABEL_MAPPING[label],
                    "artifact_flag": artifact,
                }
            )

    if mi_run_number != EXPECTED_MI_RUNS:
        raise ValueError(f"file must contain exactly 6 nonempty MI runs; found {mi_run_number}")
    if len(epochs) != 144 or np.bincount(output_labels, minlength=2).tolist() != [72, 72]:
        raise ValueError("left/right subset must contain 144 trials, 72 per class")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("constructed trial identities are not unique")

    X = np.stack(epochs).astype(np.float64, copy=False)
    y = np.asarray(output_labels, dtype=np.int64)
    artifact_flags = np.asarray(output_artifacts, dtype=np.uint8)
    metadata = {
        "dataset": "BNCI2014-001 / BCI Competition IV 2a",
        "path": str(path.resolve()),
        "official_url": official_file_url(subject, session),
        "subject": subject,
        "session": session,
        "session_role": "training/history" if session == "T" else "evaluation",
        "sessions_recorded_on_different_days": True,
        "mat_variable": "data",
        "loaded_variables": ["data"],
        "raw_fs": RECORDED_FS,
        "fs": config.target_fs,
        "raw_units": "uV",
        "units": "V",
        "raw_channel_names": list(EEG_CHANNELS + EOG_CHANNELS),
        "channel_names": list(EEG_CHANNELS),
        "eog_channels_excluded": list(EOG_CHANNELS),
        "eeg_channel_indices_zero_based": list(range(len(EEG_CHANNELS))),
        "class_names": list(CLASS_NAMES),
        "raw_class_names": list(RAW_CLASS_NAMES),
        "label_mapping": {"2": 0, "1": 1},
        "raw_labels": raw_labels,
        "class_counts": {
            "right_hand": int(np.sum(y == 0)),
            "left_hand": int(np.sum(y == 1)),
        },
        "artifact_flags": artifact_flags.tolist(),
        "artifact_policy": "retain all labeled left/right trials; expose expert flags; no exclusion",
        "artifact_flagged_count": int(artifact_flags.sum()),
        "trial_ids": trial_ids,
        "trial_records": trial_records,
        "source_run_count": len(all_runs),
        "mi_run_count": mi_run_number,
        "skipped_empty_run_indices_zero_based": skipped_empty_runs,
        "trial_onset_to_cue_s": TRIAL_ONSET_TO_CUE_S,
        "filter_input_interval_relative_to_cue_s": [0.0, IMAGERY_DURATION_S],
        "output_interval_relative_to_cue_s": [config.tmin, config.tmax],
        "epoch_interval": "[tmin, tmax)",
        "first_output_time_relative_to_cue_s": config.tmin,
        "last_output_time_relative_to_cue_s": config.tmin + (X.shape[-1] - 1) / config.target_fs,
        "shape": list(X.shape),
        "trial_raw_support_sha256": raw_support_hashes,
        "trial_epoch_sha256": epoch_hashes,
        "hash_encoding": {
            "array_order": "C",
            "raw_support_dtype": raw_support_dtype,
            "raw_support_shape": [raw_support_samples, len(EEG_CHANNELS)],
            "epoch_dtype": X.dtype.str,
            "epoch_shape": list(X.shape[1:]),
        },
        "config": asdict(config),
        "filter": {
            "design": "Butterworth",
            "prototype_order": config.filter_order,
            "band_hz": [config.l_freq, config.h_freq],
            "sos": sos.tolist(),
            "direction": "forward-backward",
            "padtype": "odd",
            "padlen_raw_samples": padlen,
            "scope": "one four-second imagery trial only",
        },
        "resample": {
            "method": "resample_poly",
            "up": up,
            "down": down,
            "window": ["kaiser", 5.0],
            "padtype": "line",
            "scope": "one four-second imagery trial only",
        },
        "checks": {
            "official_filename_matches_declared_identity": True,
            "six_mi_runs": True,
            "forty_eight_trials_per_run": True,
            "class_order_verified": True,
            "labels_events_artifacts_match": True,
            "recorded_sampling_rate_verified": True,
            "twenty_two_eeg_plus_three_eog_channels": True,
            "eog_excluded": True,
            "selected_trial_input_finite": True,
            "epochs_finite": True,
            "segments_in_bounds_and_nonoverlapping": True,
            "artifact_trials_retained": True,
            "learned_preprocessing_statistics": False,
            "artifact_dropped_trials": 0,
            "nontarget_class_trials_excluded": 144,
        },
    }
    return BNCI2014Epochs(
        X=X,
        y=y,
        artifact_flags=artifact_flags,
        trial_ids=tuple(trial_ids),
        metadata=metadata,
    )
