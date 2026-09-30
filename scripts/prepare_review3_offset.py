"""Prepare the OpenBMI eight-channel one-sample offset sensitivity records.

This additive path leaves the historical t-1 loader and caches untouched.  It
opens only the frozen source/development EEG_MI_train records, requires the
previous all-42-file smt audit, and reconstructs independent four-second
supports beginning at Python index ``MATLAB t``.
"""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

# Set numerical-library thread controls before importing project modules that
# import NumPy/SciPy.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expose.development import load_config
from expose.lee import EpochConfig, MOTOR_CHANNELS, RAW_CLASSES, _integer_vector, _string
from expose.provenance import file_sha256
from expose.review3_data import covariance_record, fit_review3_whitener, whitener_sha256

RAW_SAMPLES = 4000
RAW_FS = 1000
TARGET_FS = 250
WINDOW_START = 125
WINDOW_STOP = 875
OFFSET_AUDIT = Path("result/review3_20260923/openbmi20_r001/SEGMENTED_FIELD_OFFSET_AUDIT.json")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    return file_sha256(Path(path))


def expected_roles(cfg):
    return ({(int(s), 1): "source" for s in cfg["source_subjects"]} |
            {(int(s), session): "development" for s in cfg["development_subjects"] for session in (1, 2)})


def validate_record_roles(records, expected):
    by_key = {(int(row["subject"]), int(row["session"])): row for row in records}
    if len(by_key) != len(records) or set(by_key) != set(expected):
        raise ValueError("records do not equal the frozen source/development role set")
    if any(by_key[key]["role"] != role for key, role in expected.items()):
        raise ValueError("record role disagrees with the frozen role set")
    return by_key


def read_plan(cfg):
    records = json.loads((ROOT / "result/day3/preparation_r001/schema.json").read_text(encoding="utf-8"))["records"]
    expected = expected_roles(cfg)
    by_key = validate_record_roles(records, expected)
    rows = list(csv.DictReader((ROOT / "result/day3/preparation_r001/SPLIT_MANIFEST.csv").open(
        encoding="utf-8", newline="")))
    grouped = {}
    for row in rows:
        key = (int(row["subject_id"]), int(row["session"]))
        if key not in expected or row["role"] != expected[key] or row["mat_variable"] != "EEG_MI_train":
            raise ValueError("split mapping contains an unapproved record, role, or MAT variable")
        grouped.setdefault(key, []).append(row)
    if len(rows) != 4200 or set(grouped) != set(expected) or any(len(value) != 100 for value in grouped.values()):
        raise ValueError("expected the exact 42-file, 4200-trial frozen split mapping")
    for key, value in grouped.items():
        value.sort(key=lambda row: int(row["trial_index_zero_based"]))
        if [int(row["trial_index_zero_based"]) for row in value] != list(range(100)):
            raise ValueError(f"trial order is not exactly 0..99 for {key}")
    return by_key, grouped


def validate_offset_audit(path, expected):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, list) or len(values) != 42:
        raise ValueError("offset audit must contain exactly 42 records")
    by_key = {(int(row["subject_id"]), int(row["session"])): row for row in values}
    if len(by_key) != 42 or set(by_key) != set(expected):
        raise ValueError("offset audit keys differ from the frozen role set")
    for key, row in by_key.items():
        if (row.get("field_present") is not True or row.get("comparable") is not True
                or row.get("shape") != [4000, 100, 62]
                or row.get("trials") != 100
                or row.get("exact_matches_alternate_t") != 100
                or row.get("exact_matches_legacy_t_minus_1") != 0):
            raise ValueError(f"offset audit does not establish exact alternate support for {key}")
    return by_key


def load_train_only(path):
    from scipy.io import loadmat
    value = loadmat(path, variable_names=["EEG_MI_train"], simplify_cells=True).get("EEG_MI_train")
    if not isinstance(value, dict):
        raise ValueError("MAT file must contain EEG_MI_train")
    return value


def validate_class_table(value):
    import numpy as np
    table = np.asarray(value, dtype=object)
    if table.shape != (2, 2):
        raise ValueError("class must be a two-row [numeric code, class name] table")
    found = {}
    for code, name in table:
        scalar = np.asarray(code).squeeze()
        if scalar.ndim == 0 and isinstance(scalar.item(), str):
            text = _string(code, "class code")
            if text not in ("1", "2"):
                raise ValueError("class code strings must be '1' or '2'")
            numeric = int(text)
        else:
            values = _integer_vector(code, "class code")
            if values.size != 1:
                raise ValueError("class code must be scalar")
            numeric = int(values[0])
        if numeric in found:
            raise ValueError("class codes must be distinct")
        found[numeric] = _string(name, "class name").lower()
    if found != RAW_CLASSES:
        raise ValueError(f"unexpected class mapping: {found}; expected {RAW_CLASSES}")
    return found


def validate_and_extract_raw_supports(data, split_rows, channels=MOTOR_CHANNELS):
    """Return exact smt-matched supports beginning at Python index MATLAB t.

    The returned signal remains in raw microvolts.  No label or target outcome
    is used to estimate the support, filter, or offset.
    """
    import numpy as np
    required = {"x", "t", "fs", "y_dec", "chan", "class", "smt"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"EEG_MI_train missing fields: {missing}")
    fs = _integer_vector(data["fs"], "fs")
    if fs.size != 1 or int(fs[0]) != RAW_FS:
        raise ValueError("offset sensitivity requires the verified 1000 Hz OpenBMI recording")
    names_value = np.asarray(data["chan"], dtype=object).squeeze()
    if names_value.ndim > 1:
        raise ValueError("chan must be a vector")
    names = [_string(value, "chan") for value in np.atleast_1d(names_value)]
    if len(names) != 62 or len(set(names)) != 62:
        raise ValueError("expected 62 unique OpenBMI channel names")
    absent = [name for name in channels if name not in names]
    if absent:
        raise ValueError(f"requested channels absent from chan: {absent}")
    selected = [names.index(name) for name in channels]
    signal = np.asarray(data["x"])
    if signal.ndim != 2 or signal.shape[1] != 62 or signal.dtype.kind not in "iuf":
        raise ValueError("x must be a real samples-by-62-channels matrix")
    validate_class_table(data["class"])
    labels = _integer_vector(data["y_dec"], "y_dec")
    matlab_t = _integer_vector(data["t"], "t")
    if labels.size != 100 or matlab_t.size != 100 or not np.all(np.isin(labels, (1, 2))):
        raise ValueError("offset sensitivity requires exactly 100 declared class-1/class-2 trials")
    if np.bincount(labels - 1, minlength=2).tolist() != [50, 50]:
        raise ValueError("offset sensitivity requires 50 trials of each class")
    if np.any(matlab_t < 1) or np.any(np.diff(matlab_t) <= 0):
        raise ValueError("t must be strictly increasing positive MATLAB indices")
    if np.any(np.diff(matlab_t) < RAW_SAMPLES):
        raise ValueError("alternate four-second supports overlap")
    if matlab_t[-1] > signal.shape[0] - RAW_SAMPLES:
        raise ValueError("alternate four-second support extends beyond x")
    for i, row in enumerate(split_rows):
        if (int(row["trial_index_zero_based"]) != i
                or int(row["event_sample_matlab"]) != int(matlab_t[i])
                or int(row["event_sample_zero_based"]) != int(matlab_t[i]) - 1
                or int(row["label"]) != int(labels[i]) - 1):
            raise ValueError(f"frozen split row disagrees with MAT event/label at trial {i}")
    supplied = np.asarray(data["smt"])
    if supplied.shape != (RAW_SAMPLES, 100, 62) or supplied.dtype.kind not in "iuf":
        raise ValueError("smt must have exact shape (4000, 100, 62)")
    supports = np.empty((100, len(selected), RAW_SAMPLES), dtype=signal.dtype)
    support_hashes = []
    for i, onset in enumerate(matlab_t):
        full = signal[int(onset):int(onset) + RAW_SAMPLES, :]
        if not np.array_equal(full, supplied[:, i, :]):
            raise ValueError(f"supplied smt does not exactly equal alternate t support at trial {i}")
        chosen = np.ascontiguousarray(full[:, selected])
        if not np.all(np.isfinite(chosen)):
            raise ValueError(f"nonfinite selected EEG at trial {i}")
        supports[i] = chosen.T
        support_hashes.append(sha256(chosen.tobytes(order="C")).hexdigest())
    return supports, labels - 1, {
        "raw_fs": RAW_FS,
        "raw_units": "uV",
        "channel_names": list(channels),
        "raw_channel_names": names,
        "channel_indices_zero_based": selected,
        "event_samples_matlab": matlab_t.tolist(),
        "filter_input_start_samples_zero_based": matlab_t.tolist(),
        "filter_input_stop_samples_zero_based_exclusive": (matlab_t + RAW_SAMPLES).tolist(),
        "support_definition": "x[t:t+4000] where t is the stored MATLAB index value",
        "smt_all_trials_all_channels_exact": True,
        "trial_selected_support_sha256": support_hashes,
    }


def preprocess_supports(raw_supports, config=EpochConfig()):
    """Apply the historical filter/resample/window recipe to alternate supports."""
    from math import gcd
    import numpy as np
    from scipy.signal import butter, resample_poly, sosfiltfilt
    X = np.asarray(raw_supports)
    if X.shape != (100, 8, RAW_SAMPLES) or X.dtype.kind not in "iuf" or not np.all(np.isfinite(X)):
        raise ValueError("raw supports must be finite with shape (100, 8, 4000)")
    if config != EpochConfig(channels=MOTOR_CHANNELS):
        raise ValueError("offset sensitivity uses the exact frozen original-eight preprocessing config")
    sos = butter(config.filter_order, [config.l_freq, config.h_freq], btype="bandpass", fs=RAW_FS, output="sos")
    common = gcd(RAW_FS, TARGET_FS)
    up, down = TARGET_FS // common, RAW_FS // common
    epochs = np.empty((100, 8, WINDOW_STOP - WINDOW_START), dtype=np.float64)
    for i in range(100):
        trial = X[i].astype(np.float64) * 1e-6
        filtered = sosfiltfilt(sos, trial, axis=-1, padtype="odd", padlen=500)
        resampled = resample_poly(filtered, up, down, axis=-1, window=("kaiser", 5.0), padtype="line")
        if resampled.shape != (8, 1000):
            raise ValueError("four-second support did not resample to exactly 1000 samples")
        epochs[i] = resampled[:, WINDOW_START:WINDOW_STOP]
    if not np.all(np.isfinite(epochs)):
        raise ValueError("preprocessing produced nonfinite epochs")
    return epochs, {
        "bandpass_hz": [8.0, 30.0], "butterworth_order": 4,
        "filter_direction": "forward-backward", "filter_padtype": "odd",
        "filter_padlen_raw_samples": 500, "filter_scope": "independent 4-second trial support",
        "resample": {"method": "resample_poly", "up": 1, "down": 4,
                     "window": ["kaiser", 5.0], "padtype": "line"},
        "retained_interval_relative_to_support_seconds": [0.5, 3.5],
        "retained_interval_convention": "half-open", "output_fs": 250,
        "output_shape": [100, 8, 750], "units": "V",
    }


def output_manifest(out):
    return {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()
            and path.name not in {"COMPLETED.json", "receipt.json"}}


def apply_caps():
    memory = 16 * 1024 ** 3
    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
    signal.alarm(1800)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    apply_caps()
    out = args.run_dir.resolve()
    if out.exists():
        raise FileExistsError("run directory must be new and exclusive")
    out.mkdir(parents=True)
    started, cpu = time.monotonic(), time.process_time()
    receipt = {"run_id": out.name, "status": "running", "pid": os.getpid(), "interpreter": sys.executable,
               "started_utc": datetime.now(timezone.utc).isoformat(), "raw_data_accessed": False,
               "online_phase_accessed": False, "confirmation_data_accessed": False, "model_fits": 0,
               "sensitivity": "one_raw_sample_later_support_only"}
    atomic_json(out / "receipt.json", receipt)
    try:
        import numpy as np
        cfg = load_config(ROOT)
        config = ROOT / "research/protocols/DAY3_CONFIG_v1.json"
        if config.with_suffix(config.suffix + ".sha256").read_text().split()[0] != sha(config):
            raise ValueError("frozen config checksum mismatch")
        roles = ROOT / cfg["role_manifest"]
        if sha(roles) != cfg["role_manifest_sha256"]:
            raise ValueError("role manifest checksum mismatch")
        records, split = read_plan(cfg)
        audit_path = ROOT / OFFSET_AUDIT
        audit = validate_offset_audit(audit_path, expected_roles(cfg))
        inputs = {
            "research/protocols/DAY3_CONFIG_v1.json": sha(config),
            str(roles.relative_to(ROOT)): sha(roles),
            "result/day3/preparation_r001/schema.json": sha(ROOT / "result/day3/preparation_r001/schema.json"),
            "result/day3/preparation_r001/SPLIT_MANIFEST.csv": sha(ROOT / "result/day3/preparation_r001/SPLIT_MANIFEST.csv"),
            str(OFFSET_AUDIT): sha(audit_path),
            "src/expose/lee.py": sha(ROOT / "src/expose/lee.py"),
            "src/expose/review3_data.py": sha(ROOT / "src/expose/review3_data.py"),
            "src/expose/review2_controls.py": sha(ROOT / "src/expose/review2_controls.py"),
            "scripts/prepare_review3_offset.py": sha(Path(__file__)),
        }
        for record in records.values():
            inputs[record["raw"]["path"]] = record["raw"]["sha256"]
            inputs[record["cache"]["array_path"]] = record["cache"]["array_sha256"]
        atomic_json(out / "INPUT_MANIFEST.json", inputs)
        target_whiteners, output_records, support_checks = {}, [], []
        for key, record in sorted(records.items()):
            raw, cache = record["raw"], record["cache"]
            raw_path = ROOT / raw["path"]
            old_path = ROOT / cache["array_path"]
            if raw_path.stat().st_size != raw["bytes"] or sha(raw_path) != raw["sha256"]:
                raise ValueError(f"raw identity changed for {key}")
            if sha(old_path) != cache["array_sha256"]:
                raise ValueError(f"historical original-eight cache identity changed for {key}")
            data = load_train_only(raw_path)
            receipt["raw_data_accessed"] = True
            raw_supports, y, support_metadata = validate_and_extract_raw_supports(data, split[key])
            if audit[key]["exact_matches_alternate_t"] != 100:
                raise ValueError(f"frozen offset audit changed for {key}")
            X, processing = preprocess_supports(raw_supports)
            with np.load(old_path, allow_pickle=False) as old:
                if not np.array_equal(y, old["y"]):
                    raise ValueError(f"alternate labels differ from historical cache for {key}")
            if record["role"] == "source":
                W, origin = fit_review3_whitener(X, y, expected_channels=8), "own_source_session1_all_100"
            elif key[1] == 1:
                W, origin = fit_review3_whitener(X, y, expected_channels=8), "own_development_session1_all_100"
                target_whiteners[key[0]] = W
            else:
                if key[0] not in target_whiteners:
                    raise ValueError("development session-2 must reuse its alternate session-1 whitener")
                W, origin = target_whiteners[key[0]], "reused_own_development_session1_all_100"
            compact = covariance_record(X, y, W, expected_channels=8)
            npz = out / f"subj{key[0]:02d}_sess{key[1]}_offline_8ch_offset_plus1.npz"
            with npz.open("xb") as handle:
                np.savez_compressed(handle, **compact)
            meta = npz.with_suffix(".json")
            # Labels remain only in the NPZ and downstream metric records.
            atomic_json(meta, {
                "subject": key[0], "session": key[1], "role": record["role"],
                "raw_path": raw["path"], "raw_sha256": raw["sha256"],
                "historical_cache_path": cache["array_path"], "historical_cache_sha256": cache["array_sha256"],
                "offset_definition": support_metadata["support_definition"],
                "smt_all_trials_all_channels_exact": support_metadata["smt_all_trials_all_channels_exact"],
                "channel_names": list(MOTOR_CHANNELS), "processing": processing,
                "stored_arrays": {name: list(value.shape) for name, value in compact.items()},
                "ea_whitener": W.tolist(), "ea_whitener_sha256": whitener_sha256(W),
                "ea_whitener_origin": origin,
            })
            trial_ids = [row["trial_id"] for row in split[key]]
            output_records.append({"subject": key[0], "session": key[1], "role": record["role"],
                                   "npz_path": str(npz.relative_to(ROOT)), "sha256": sha(npz),
                                   "trial_ids": trial_ids})
            support_checks.append({"subject": key[0], "session": key[1], "role": record["role"],
                                   "trials": 100, "all_62_channels_exact_to_smt": True,
                                   "support_hashes_sha256": sha256(json.dumps(
                                       support_metadata["trial_selected_support_sha256"], separators=(",", ":")
                                   ).encode()).hexdigest()})
            del data, raw_supports, X, y, compact
        atomic_json(out / "SUPPLIED_SMT_SUPPORT_CHECKS.json", support_checks)
        index = {"dataset": "openbmi_offset_plus1", "montage": "8ch", "records": output_records}
        atomic_json(out / "DATASET_INDEX.json", index)
        receipt.update(status="completed", files=42, trials=4200, channels=8,
                       epoch_shape=[100, 8, 750], offset_raw_samples=1, offset_seconds=0.001,
                       supplied_smt_exact_trials=4200, records=output_records,
                       output_manifest=output_manifest(out))
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        traceback.print_exc()
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    if receipt["status"] == "completed":
        atomic_json(out / "COMPLETED.json", {"run_id": out.name, "receipt_sha256": sha(out / "receipt.json"),
                                               "outputs": output_manifest(out)})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
