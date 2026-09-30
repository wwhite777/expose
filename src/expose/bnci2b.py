"""Strict in-memory loader for BNCI2014-004 screening sessions 01T and 02T.

The loader performs no network or filesystem I/O.  It accepts the bytes of one
official ``BxxT.mat`` aggregate and returns only the first two genuine,
no-feedback sessions.  ``run.trial`` is interpreted as a MATLAB 1-based trial
onset; the cue begins three seconds later.
"""

from dataclasses import asdict, dataclass
from io import BytesIO

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from .bnci2014 import _integer_vector, _normalized_class, _runs, _sample_count


BASE_URL = "https://lampx.tugraz.at/~bci/database/004-2014"
EEG_CHANNELS = ("C3", "Cz", "C4")
EOG_CHANNELS = ("EOG1", "EOG2", "EOG3")
RAW_CLASS_NAMES = ("left_hand", "right_hand")
CLASS_NAMES = ("right_hand", "left_hand")
LABEL_MAPPING = {2: 0, 1: 1}  # EXPOSE convention: 0=right, 1=left.
RECORDED_FS = 250
TRIAL_ONSET_TO_CUE_S = 3.0
FILTER_SUPPORT_S = 4.0
SCREENING_SESSION_NAMES = ("01T", "02T")
EXPECTED_AGGREGATE_SESSIONS = 3
EXPECTED_SCREENING_TRIALS = 120


@dataclass(frozen=True)
class BNCI2BConfig:
    l_freq: float = 8.0
    h_freq: float = 30.0
    tmin: float = 0.5
    tmax: float = 3.5
    filter_order: int = 4


@dataclass
class BNCI2BSession:
    X: np.ndarray
    y: np.ndarray
    artifact_flags: np.ndarray
    trial_ids: tuple[str, ...]
    metadata: dict


@dataclass
class BNCI2BScreening:
    sessions: tuple[BNCI2BSession, BNCI2BSession]
    metadata: dict


def official_file_url(subject: int) -> str:
    """Return the current MOABB upstream URL without network access."""
    subject = _validate_subject(subject)
    return f"{BASE_URL}/B{subject:02d}T.mat"


def _validate_subject(subject) -> int:
    if isinstance(subject, bool) or not isinstance(subject, (int, np.integer)):
        raise TypeError("subject must be an integer from 1 through 9")
    subject = int(subject)
    if subject not in range(1, 10):
        raise ValueError("subject must be from 1 through 9")
    return subject


def _validate_config(config: BNCI2BConfig) -> None:
    if not isinstance(config, BNCI2BConfig):
        raise TypeError("config must be a BNCI2BConfig")
    if not (0 <= config.tmin < config.tmax <= FILTER_SUPPORT_S):
        raise ValueError("epoch interval must satisfy 0 <= tmin < tmax <= 4 seconds")
    if (
        isinstance(config.filter_order, bool)
        or not isinstance(config.filter_order, int)
        or config.filter_order <= 0
    ):
        raise ValueError("filter_order must be a positive integer")
    if not (0 < config.l_freq < config.h_freq < RECORDED_FS / 2):
        raise ValueError("bandpass must lie strictly below the recorded Nyquist frequency")


def _session(
    run: dict,
    *,
    subject: int,
    session_number: int,
    config: BNCI2BConfig,
    sos: np.ndarray,
) -> BNCI2BSession:
    if not isinstance(run, dict):
        raise ValueError(f"data session {session_number} is not a struct")
    required = {"X", "trial", "y", "fs", "classes", "artifacts"}
    missing = sorted(required - run.keys())
    if missing:
        raise ValueError(f"data session {session_number} missing fields: {missing}")

    trials = _integer_vector(run["trial"], f"session {session_number} trial")
    labels = _integer_vector(run["y"], f"session {session_number} y")
    artifacts = _integer_vector(run["artifacts"], f"session {session_number} artifacts")
    if trials.size != EXPECTED_SCREENING_TRIALS:
        raise ValueError(f"screening session {session_number} must contain 120 trials")
    if labels.size != trials.size or artifacts.size != trials.size:
        raise ValueError(f"session {session_number} trial, y and artifacts must have equal lengths")
    if np.any(trials < 1) or np.any(np.diff(trials) <= 0):
        raise ValueError(f"session {session_number} trial must be increasing MATLAB 1-based indices")
    if not np.all(np.isin(labels, tuple(LABEL_MAPPING))):
        raise ValueError(f"session {session_number} y must contain only class codes 1 and 2")
    if np.bincount(labels, minlength=3)[1:3].tolist() != [60, 60]:
        raise ValueError(f"screening session {session_number} must contain 60 trials per class")
    if not np.all(np.isin(artifacts, (0, 1))):
        raise ValueError(f"session {session_number} artifacts must contain only 0 or 1")

    fs = _integer_vector(run["fs"], f"session {session_number} fs")
    if fs.size != 1 or int(fs[0]) != RECORDED_FS:
        raise ValueError(f"session {session_number} fs must equal the documented 250 Hz")
    classes = tuple(_normalized_class(value) for value in np.atleast_1d(run["classes"]))
    if classes != RAW_CLASS_NAMES:
        raise ValueError(f"session {session_number} has unexpected class order {classes}")

    signal = np.asarray(run["X"])
    if signal.ndim != 2 or signal.shape[1] != len(EEG_CHANNELS) + len(EOG_CHANNELS):
        raise ValueError(f"session {session_number} X must be samples by 6 channels")
    if signal.dtype.kind not in "iuf":
        raise ValueError(f"session {session_number} X must be real numeric data")

    cue_offset = _sample_count(TRIAL_ONSET_TO_CUE_S, RECORDED_FS, "cue offset")
    support_samples = _sample_count(FILTER_SUPPORT_S, RECORDED_FS, "filter support")
    output_start = _sample_count(config.tmin, RECORDED_FS, "tmin")
    output_stop = _sample_count(config.tmax, RECORDED_FS, "tmax")
    padlen = _sample_count(0.5, RECORDED_FS, "filter padding")
    trial_onsets = trials - 1
    cue_onsets = trial_onsets + cue_offset
    if cue_onsets[-1] > signal.shape[0] - support_samples:
        raise ValueError(f"session {session_number} four-second cue support extends beyond X")
    if np.any(np.diff(cue_onsets) < support_samples):
        raise ValueError(f"session {session_number} four-second cue supports overlap")

    epochs = []
    trial_ids = []
    trial_records = []
    for trial_index, (trial_onset, cue_onset, label, artifact) in enumerate(
        zip(trial_onsets, cue_onsets, labels, artifacts), start=1
    ):
        support = signal[
            cue_onset : cue_onset + support_samples, : len(EEG_CHANNELS)
        ].T.astype(np.float64)
        support *= 1e-6
        if not np.isfinite(support).all():
            raise ValueError(
                f"nonfinite EEG in session {session_number}, trial {trial_index}"
            )
        filtered = sosfiltfilt(
            sos, support, axis=-1, padtype="odd", padlen=padlen
        )
        epoch = np.ascontiguousarray(
            filtered[:, output_start:output_stop], dtype=np.float64
        )
        if epoch.shape != (len(EEG_CHANNELS), output_stop - output_start):
            raise ValueError("preprocessing produced an unexpected epoch shape")
        if not np.isfinite(epoch).all():
            raise ValueError("preprocessing produced nonfinite epochs")
        trial_id = (
            f"bnci2014-004:s{subject:02d}:session{session_number:02d}:"
            f"trial{trial_index:03d}"
        )
        epochs.append(epoch)
        trial_ids.append(trial_id)
        trial_records.append(
            {
                "trial_id": trial_id,
                "session_trial_index_one_based": trial_index,
                "trial_onset_sample_matlab": int(trial_onset + 1),
                "trial_onset_sample_zero_based": int(trial_onset),
                "cue_onset_sample_zero_based": int(cue_onset),
                "filter_input_stop_sample_zero_based_exclusive": int(
                    cue_onset + support_samples
                ),
                "output_start_sample_zero_based": int(cue_onset + output_start),
                "output_stop_sample_zero_based_exclusive": int(cue_onset + output_stop),
                "artifact_flag": int(artifact),
            }
        )

    X = np.stack(epochs).astype(np.float64, copy=False)
    y = np.asarray([LABEL_MAPPING[int(value)] for value in labels], dtype=np.int64)
    artifact_flags = artifacts.astype(np.uint8)
    session_name = SCREENING_SESSION_NAMES[session_number - 1]
    metadata = {
        "dataset": "BNCI2014-004 / BCI Competition IV 2b",
        "subject": subject,
        "source_aggregate": f"B{subject:02d}T.mat",
        "source_mat_struct_index_zero_based": session_number - 1,
        "session": session_number,
        "official_session_name": session_name,
        "session_role": "source_person_or_target_history" if session_number == 1 else "target_evaluation",
        "feedback": False,
        "sessions_recorded_on_different_days": True,
        "documented_runs": 6,
        "run_boundaries_used": False,
        "raw_fs": RECORDED_FS,
        "fs": RECORDED_FS,
        "raw_units": "uV",
        "units": "V",
        "raw_channel_names": list(EEG_CHANNELS + EOG_CHANNELS),
        "channel_names": list(EEG_CHANNELS),
        "eog_channels_excluded": list(EOG_CHANNELS),
        "class_names": list(CLASS_NAMES),
        "raw_class_names": list(RAW_CLASS_NAMES),
        "label_mapping": {"2": 0, "1": 1},
        "class_counts": {
            "right_hand": int(np.sum(y == 0)),
            "left_hand": int(np.sum(y == 1)),
        },
        "artifact_flags": artifact_flags.tolist(),
        "artifact_flagged_count": int(artifact_flags.sum()),
        "artifact_policy": "retain all labeled trials; expose expert flags; no exclusion",
        "trial_ids": trial_ids,
        "trial_records": trial_records,
        "trial_order": "increasing run.trial onset within the session struct",
        "trial_onset_to_cue_s": TRIAL_ONSET_TO_CUE_S,
        "filter_input_interval_relative_to_cue_s": [0.0, FILTER_SUPPORT_S],
        "output_interval_relative_to_cue_s": [config.tmin, config.tmax],
        "epoch_interval": "[tmin, tmax)",
        "shape": list(X.shape),
        "config": asdict(config),
        "filter": {
            "design": "Butterworth",
            "prototype_order": config.filter_order,
            "band_hz": [config.l_freq, config.h_freq],
            "sos": sos.tolist(),
            "direction": "forward-backward",
            "function": "scipy.signal.sosfiltfilt",
            "padtype": "odd",
            "padlen_samples": padlen,
            "scope": "one independent four-second cue-anchored trial support",
        },
        "checks": {
            "one_hundred_twenty_trials": True,
            "sixty_trials_per_class": True,
            "class_order_verified": True,
            "labels_events_artifacts_match": True,
            "recorded_sampling_rate_verified": True,
            "three_eeg_plus_three_eog_channels": True,
            "artifact_trials_retained": True,
            "artifact_dropped_trials": 0,
            "learned_preprocessing_statistics": False,
            "segments_in_bounds_and_nonoverlapping": True,
        },
    }
    return BNCI2BSession(
        X=X,
        y=y,
        artifact_flags=artifact_flags,
        trial_ids=tuple(trial_ids),
        metadata=metadata,
    )


def load_bnci2b_screening_sessions(
    mat_bytes: bytes,
    *,
    subject: int,
    config: BNCI2BConfig = BNCI2BConfig(),
) -> BNCI2BScreening:
    """Load sessions 01T and 02T from one official aggregate held in memory."""
    subject = _validate_subject(subject)
    _validate_config(config)
    if not isinstance(mat_bytes, bytes) or not mat_bytes:
        raise TypeError("mat_bytes must be nonempty immutable bytes")
    mat = loadmat(
        BytesIO(mat_bytes), variable_names=["data"], simplify_cells=True
    )
    run_values = mat.get("data")
    if run_values is None:
        raise ValueError("MAT file must contain the BNCI data struct")
    aggregate_sessions = _runs(run_values)
    if len(aggregate_sessions) != EXPECTED_AGGREGATE_SESSIONS:
        raise ValueError("BxxT aggregate must contain exactly three training-session structs")
    sos = butter(
        config.filter_order,
        [config.l_freq, config.h_freq],
        btype="bandpass",
        fs=RECORDED_FS,
        output="sos",
    )
    sessions = tuple(
        _session(
            aggregate_sessions[index],
            subject=subject,
            session_number=index + 1,
            config=config,
            sos=sos,
        )
        for index in range(2)
    )
    if len({trial for session in sessions for trial in session.trial_ids}) != 240:
        raise ValueError("constructed trial identities are not globally unique within subject")
    metadata = {
        "dataset": "BNCI2014-004 / BCI Competition IV 2b",
        "subject": subject,
        "official_url": official_file_url(subject),
        "source_aggregate_session_count": len(aggregate_sessions),
        "selected_struct_indices_zero_based": [0, 1],
        "selected_official_sessions": list(SCREENING_SESSION_NAMES),
        "excluded_official_sessions": ["03T"],
        "selection_reason": "exactly the two early no-feedback screening sessions",
    }
    return BNCI2BScreening(sessions=sessions, metadata=metadata)
