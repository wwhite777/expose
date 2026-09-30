"""Successor raw-reference verifier with fail-fast nonfinite support diagnostics.

Informed Builder QA: no ``expose.*`` imports, saved-model loading, or confirmation access.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
RAW_CLASSES = {1: "right", 2: "left"}
CLASS_NAMES = ("right_hand", "left_hand")
CHANNELS = ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CPz", "CP4")
DAY3_CONFIG_SHA256 = "32691c9fa6902bbd219e349bdc5f7fc70c7d1a0e059ba099bc854f18eeeac63f"
REQUIREMENTS_SHA256 = "44904d8d56802ed58456609fff51f9b9e21a8eeb93ba1cd35968ae1037da4687"
PREDICTION_FIELDS = ("operation_id", "arm", "model", "draw", "dose_per_class", "prior_weight",
                     "subject_id", "trial_id", "y_true", "y_pred", "p_class0_right", "p_class1_left")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_hashes(path):
    md5, sha, size = hashlib.md5(), hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            size += len(block); md5.update(block); sha.update(block)
    return {"bytes": size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def array_sha256(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def typed_array_sha256(value, dtype):
    import numpy as np
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(path, fieldnames, values):
    path = Path(path); temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader(); writer.writerows(values)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def rooted_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe relative path: {relative!r}")
    root = Path(root).resolve(); candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"symlink input rejected: {relative}")
    resolved = candidate.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"input escapes root or is not a file: {relative}")
    return resolved


def trial_id(subject, session, index):
    return f"lee2019:s{subject:03d}:session{session}:offline:t{index:03d}"


def operation_ids(cfg):
    result = []
    def add(arm, model, draw, dose, target=None, prior=None):
        name = f"{arm}__{model}__r{draw if draw is not None else 'shared'}_n{dose}"
        if target is not None: name += f"_s{target:03d}"
        if prior is not None: name += f"_w{prior}"
        result.append(name)
    for arm in cfg["arms"]:
        for model in cfg["model_order"]:
            for draw in ([None] if arm == "practical" else range(cfg["draws"])): add(arm, model, draw, 0)
            for draw in range(cfg["draws"]):
                for dose in cfg["history_doses_per_class"][1:]:
                    for subject in cfg["development_subjects"]: add(arm, model, draw, dose, subject)
    for model in cfg["model_order"]:
        for draw in range(cfg["draws"]): add("removal", model, draw, 30)
    for draw in range(cfg["draws"]):
        for dose in cfg["history_doses_per_class"][1:]:
            for subject in cfg["development_subjects"]: add("target_only", cfg["target_only_model"], draw, dose, subject)
    add("calibration", "centroid", None, 0)
    for draw in range(cfg["draws"]):
        for dose in cfg["history_doses_per_class"][1:]:
            for subject in cfg["development_subjects"]:
                for prior in cfg["calibration_prior_grid"]: add("calibration", "centroid", draw, dose, subject, prior)
    if len(result) != 808 or len(set(result)) != 808: raise ValueError("operation identity count differs")
    return result


def expected_grid_outputs(cfg):
    result = {"receipt.json", "INPUT_MANIFEST.json", "predictions.csv", "operation_membership.csv", "timing.csv"}
    for operation in operation_ids(cfg):
        result.update(operation + suffix for suffix in (".json", ".joblib", ".predictions.csv", ".membership.csv"))
    if len(result) != 3237: raise ValueError("grid output identity count differs")
    return result


def validate_stage(root, directory, required, expected_script):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    supervisor = json.loads((directory / "supervisor_receipt.json").read_text())
    if supervisor.get("status") != "completed" or supervisor.get("exit_code") != 0:
        raise ValueError("producer supervisor did not complete")
    receipt = json.loads((directory / "receipt.json").read_text())
    marker = json.loads((directory / "COMPLETED.json").read_text())
    child_pid = supervisor.get("child_pid")
    if (type(child_pid) is not int or receipt.get("status") != "completed" or receipt.get("exit_code") != 0
            or receipt.get("pid") != child_pid or receipt.get("run_id") != directory.name):
        raise ValueError("producer receipt identity differs")
    receipt_hash = sha256(directory / "receipt.json")
    if (marker.get("run_id") != directory.name or marker.get("receipt_sha256") != receipt_hash
            or supervisor.get("run_id") != directory.name or supervisor.get("receipt_sha256") != receipt_hash
            or supervisor.get("completion_sha256") != sha256(directory / "COMPLETED.json")):
        raise ValueError("producer completion/supervisor identity differs")
    if Path(supervisor.get("run_dir", "")).resolve() != directory or Path(supervisor.get("cwd", "")).resolve() != root:
        raise ValueError("producer cwd/run directory differs")
    argv = supervisor.get("argv")
    if not isinstance(argv, list) or len(argv) < 2 or Path(argv[1]).resolve() != (root / expected_script).resolve():
        raise ValueError("producer command identity differs")
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(required): raise ValueError("producer output set differs")
    for name, expected in outputs.items():
        if Path(name).name != name or name in {"COMPLETED.json", "supervisor_receipt.json"}: raise ValueError("unsafe output")
        path = directory / name
        if path.is_symlink() or sha256(path) != expected: raise ValueError(f"producer output hash differs: {name}")
    dependencies = json.loads((directory / "INPUT_MANIFEST.json").read_text())
    if not isinstance(dependencies, dict) or not dependencies: raise ValueError("producer dependency manifest missing")
    for relative, expected in dependencies.items():
        if sha256(rooted_file(root, relative)) != expected: raise ValueError(f"producer dependency differs: {relative}")
    return {"supervisor": supervisor, "receipt": receipt, "marker": marker}


def validate_config(root):
    path = Path(root) / "research/protocols/DAY3_CONFIG_v1.json"
    observed = sha256(path); sidecar = path.with_suffix(".json.sha256").read_text().split()[0]
    if observed != DAY3_CONFIG_SHA256 or sidecar != DAY3_CONFIG_SHA256: raise ValueError("config checksum differs")
    cfg = json.loads(path.read_text())
    if (len(cfg.get("source_subjects", [])) != 18 or len(cfg.get("development_subjects", [])) != 12
            or cfg.get("expected_preparation_files") != 42 or cfg.get("expected_preparation_trials") != 4200
            or cfg.get("expected_operations") != 808 or cfg.get("expected_prediction_rows") != 110400
            or cfg.get("model_order") != ["csp_lda", "ts_lr", "mdm"]
            or cfg.get("history_doses_per_class") != [0, 5, 15, 30]
            or cfg.get("history_draw_seeds") != [20260915, 20260916]
            or cfg.get("calibration_prior_grid") != [0, 5, 20, 100]):
        raise ValueError("unsupported frozen config")
    prep = cfg.get("preprocessing", {})
    required = {"channels": list(CHANNELS), "raw_fs_expected": 1000, "units": "V", "l_freq": 8.0,
                "h_freq": 30.0, "tmin": 0.5, "tmax": 3.5, "target_fs": 250, "filter_order": 4,
                "event_index": "MATLAB t-1", "expected_shape": [100, 8, 750]}
    if any(prep.get(key) != value for key, value in required.items()): raise ValueError("preprocessing contract differs")
    if sha256(rooted_file(root, cfg["role_manifest"])) != cfg["role_manifest_sha256"]: raise ValueError("role hash differs")
    if sha256(rooted_file(root, "research/environment/requirements.lock.txt")) != REQUIREMENTS_SHA256:
        raise ValueError("environment requirements identity differs")
    return cfg


def integer_text(value, name, minimum=None):
    try: number = int(value)
    except (TypeError, ValueError) as error: raise ValueError(f"{name} must be an integer") from error
    if str(number) != str(value).strip() or (minimum is not None and number < minimum): raise ValueError(f"{name} must be exact integer")
    return number


def validate_roles(root, cfg):
    roles = {}
    for row in read_csv(rooted_file(root, cfg["role_manifest"])):
        subject, role = integer_text(row.get("subject_id"), "role subject", 1), row.get("role")
        if subject in roles or role not in {"source", "development", "confirmation"}: raise ValueError("invalid role row")
        roles[subject] = role
    if (sorted(roles) != list(range(1, 55))
            or sorted(s for s, r in roles.items() if r == "source") != sorted(cfg["source_subjects"])
            or sorted(s for s, r in roles.items() if r == "development") != sorted(cfg["development_subjects"])
            or sum(r == "confirmation" for r in roles.values()) != 24): raise ValueError("role allocation differs")
    return roles


def validate_prepared_manifests(root, cfg, roles, data_rows, split_rows, schema):
    expected = {(s, 1) for s in cfg["source_subjects"]} | {(s, t) for s in cfg["development_subjects"] for t in (1, 2)}
    if len(data_rows) != 42 or len(split_rows) != 4200: raise ValueError("manifest row count differs")
    data = {}
    for row in data_rows:
        subject = integer_text(row.get("subject_id"), "data subject", 1); session = integer_text(row.get("session"), "session", 1)
        key, role = (subject, session), roles.get(subject)
        if key in data or key not in expected or role not in {"source", "development"} or row.get("role") != role:
            raise ValueError("confirmation, duplicate, or unexpected data row")
        if (row.get("dataset") != "Lee2019_MI" or row.get("used_variable") != "EEG_MI_train"
                or [(integer_text(row.get(k), k), v) for k, v in (("trials", 100), ("right_trials", 50),
                   ("left_trials", 50), ("raw_fs", 1000), ("output_fs", 250)) if integer_text(row.get(k), k) != v]):
            raise ValueError("data schema/count differs")
        exact_raw = f"data/raw/lee2019/session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
        if row.get("raw_path") != exact_raw: raise ValueError("raw path identity differs")
        for name, length in (("published_md5", 32), ("local_sha256", 64), ("derived_sha256", 64), ("metadata_sha256", 64)):
            digest = row.get(name, "")
            if len(digest) != length or any(ch not in "0123456789abcdef" for ch in digest): raise ValueError(f"malformed {name}")
        for name in ("raw_path", "derived_path", "metadata_path"): rooted_file(root, row[name])
        data[key] = row
    if set(data) != expected: raise ValueError("prepared subject/session set differs")

    split, grouped = {}, defaultdict(list)
    for row in split_rows:
        subject = integer_text(row.get("subject_id"), "split subject", 1); session = integer_text(row.get("session"), "session", 1)
        index = integer_text(row.get("trial_index_zero_based"), "trial index", 0); label = integer_text(row.get("label"), "label", 0)
        key, tid, role = (subject, session), row.get("trial_id"), roles.get(subject)
        usage = "source_training" if role == "source" else "development_history" if session == 1 else "development_evaluation"
        if (key not in expected or role not in {"source", "development"} or row.get("role") != role or not 0 <= index < 100
                or label not in (0, 1) or tid != trial_id(subject, session, index) or tid in split
                or row.get("mat_variable") != "EEG_MI_train" or row.get("class_name") != CLASS_NAMES[label]
                or row.get("usage") != usage): raise ValueError("split identity/role/class differs")
        event = integer_text(row.get("event_sample_matlab"), "MATLAB event", 1)
        zero = integer_text(row.get("event_sample_zero_based"), "zero event", 0)
        f0 = integer_text(row.get("filter_start_sample"), "filter start", 0); f1 = integer_text(row.get("filter_stop_sample_exclusive"), "filter stop", 1)
        e0 = integer_text(row.get("epoch_start_sample"), "epoch start", 0); e1 = integer_text(row.get("epoch_stop_sample_exclusive"), "epoch stop", 1)
        if not (zero == event - 1 == f0 and f1 == f0 + 4000 and e0 == f0 + 500 and e1 == f0 + 3500):
            raise ValueError("split event/index interval differs")
        for name in ("raw_support_sha256", "epoch_sha256"):
            digest = row.get(name, "")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest): raise ValueError(f"malformed {name}")
        split[tid] = row; grouped[key].append(row)
    if set(grouped) != expected: raise ValueError("split cells differ")
    for values in grouped.values():
        values.sort(key=lambda row: int(row["trial_index_zero_based"])); events = [int(row["event_sample_zero_based"]) for row in values]
        if ([int(row["trial_index_zero_based"]) for row in values] != list(range(100))
                or Counter(int(row["label"]) for row in values) != Counter({0: 50, 1: 50})
                or any(b - a < 4000 for a, b in zip(events, events[1:]))): raise ValueError("split balance/order/events differ")

    records = schema.get("records") if isinstance(schema, dict) else None
    if not isinstance(records, list) or len(records) != 42: raise ValueError("schema record count differs")
    record_map = {}
    for record in records:
        key = (record.get("subject"), record.get("session"))
        if key in record_map or key not in expected or record.get("role") != roles[key[0]]: raise ValueError("schema role/cell differs")
        row = data[key]; raw = record.get("raw", {}); cache = record.get("cache", {}); metadata = record.get("metadata", {})
        if (raw.get("path") != row["raw_path"] or raw.get("sha256") != row["local_sha256"]
                or raw.get("published_md5") != row["published_md5"] or int(raw.get("actual_bytes", -1)) != int(row["bytes"])
                or cache.get("array_path") != row["derived_path"] or cache.get("array_sha256") != row["derived_sha256"]
                or cache.get("metadata_path") != row["metadata_path"] or cache.get("metadata_sha256") != row["metadata_sha256"]
                or metadata.get("mat_variable") != "EEG_MI_train"): raise ValueError("schema/data lineage differs")
        record_map[key] = record
    if set(record_map) != expected: raise ValueError("schema cells differ")
    return data, split, record_map


def preflight(root):
    root = Path(root).resolve(); cfg = validate_config(root); roles = validate_roles(root, cfg)
    prep = root / "result/day3/preparation_r001"
    prep_stage = validate_stage(root, prep, {"receipt.json", "INPUT_MANIFEST.json", "DATA_MANIFEST.csv", "SPLIT_MANIFEST.csv", "schema.json"}, "scripts/prepare_day3.py")
    if prep_stage["receipt"].get("files") != 42 or prep_stage["receipt"].get("trials") != 4200: raise ValueError("preparation count differs")
    schema = json.loads((prep / "schema.json").read_text())
    if schema.get("confirmation_data_accessed") is not False or schema.get("model_fits") != 0: raise ValueError("preparation scope differs")
    data, split, records = validate_prepared_manifests(root, cfg, roles, read_csv(prep / "DATA_MANIFEST.csv"),
                                                       read_csv(prep / "SPLIT_MANIFEST.csv"), schema)
    grid = root / "result/day3/grid_r002"; grid_stage = validate_stage(root, grid, expected_grid_outputs(cfg), "scripts/run_day3.py")
    receipt, checks = grid_stage["receipt"], grid_stage["receipt"].get("checks", {})
    if (receipt.get("counts") != {"fit": 520, "target_update": 288} or checks.get("operations") != 808
            or checks.get("prediction_rows") != 110400 or checks.get("prepared_trials") != 4200
            or checks.get("confirmation_accessed") is not False): raise ValueError("grid count/scope differs")
    return {"cfg": cfg, "roles": roles, "data": data, "split": split, "records": records,
            "prep": prep, "grid": grid, "prep_stage": prep_stage, "grid_stage": grid_stage}


def numeric_vector(value, name):
    import numpy as np
    values = np.asarray(value).squeeze()
    if values.ndim > 1 or values.size == 0 or values.dtype.kind not in "iuf": raise ValueError(f"{name} must be numeric vector")
    values = np.atleast_1d(values)
    if not np.all(np.isfinite(values)) or not np.all(values == np.floor(values)) or np.any(values < 0):
        raise ValueError(f"{name} must contain finite nonnegative integers")
    return values.astype(np.int64)


def scalar_string(value, name):
    import numpy as np
    value = np.asarray(value).squeeze()
    if value.ndim != 0 or not isinstance(value.item(), str) or not value.item().strip(): raise ValueError(f"{name} must be string")
    return value.item().strip()


def reconstruct_raw_struct(data, cfg, *, build_epochs=True):
    import numpy as np
    from scipy.signal import butter, resample_poly, sosfiltfilt
    required = {"x", "t", "fs", "y_dec", "chan", "class", "y_class", "y_logic", "smt"}
    if not isinstance(data, dict) or required - set(data): raise ValueError("EEG_MI_train missing required fields")
    fs = numeric_vector(data["fs"], "fs")
    if fs.size != 1 or int(fs[0]) != 1000: raise ValueError("fs differs")
    channel_values = np.asarray(data["chan"], dtype=object).squeeze()
    if channel_values.ndim > 1: raise ValueError("chan must be vector")
    names = [scalar_string(v, "chan") for v in np.atleast_1d(channel_values)]
    if len(names) != len(set(names)) or any(name not in names for name in CHANNELS): raise ValueError("channel identity differs")
    selected = [names.index(name) for name in CHANNELS]; signal_data = np.asarray(data["x"])
    if signal_data.ndim != 2 or signal_data.shape[1] != len(names) or signal_data.dtype.kind not in "iuf": raise ValueError("x schema differs")
    class_table = np.asarray(data["class"], dtype=object)
    if class_table.shape != (2, 2): raise ValueError("class table shape differs")
    classes = {}
    for raw_code, raw_name in class_table:
        code_value = np.asarray(raw_code).squeeze()
        if code_value.ndim == 0 and isinstance(code_value.item(), str):
            text = scalar_string(raw_code, "class code")
            if text not in {"1", "2"}: raise ValueError("class code differs")
            code = int(text)
        else:
            codes = numeric_vector(raw_code, "class code")
            if codes.size != 1: raise ValueError("class code must be scalar")
            code = int(codes[0])
        if code in classes: raise ValueError("class code repeats")
        classes[code] = scalar_string(raw_name, "class name").lower()
    if classes != RAW_CLASSES: raise ValueError("class code/name mapping differs")
    labels, events_matlab = numeric_vector(data["y_dec"], "y_dec"), numeric_vector(data["t"], "t")
    if labels.size != 100 or events_matlab.size != 100 or Counter(labels.tolist()) != Counter({1: 50, 2: 50}):
        raise ValueError("trial count/class balance differs")
    class_values = np.asarray(data["y_class"], dtype=object).squeeze()
    if class_values.ndim > 1 or class_values.size != 100: raise ValueError("y_class shape differs")
    class_values = [scalar_string(v, "y_class").lower() for v in np.atleast_1d(class_values)]
    if class_values != [RAW_CLASSES[int(label)] for label in labels]: raise ValueError("y_class disagrees")
    logical = np.asarray(data["y_logic"]); expected_logical = np.vstack([labels == code for code in (1, 2)])
    if logical.shape != (2, 100) or not np.array_equal(logical, expected_logical): raise ValueError("y_logic disagrees")
    if np.any(events_matlab < 1) or np.any(np.diff(events_matlab) <= 0): raise ValueError("events not increasing positive integers")
    events = events_matlab - 1
    if events[-1] + 4000 > signal_data.shape[0] or np.any(np.diff(events) < 4000): raise ValueError("events out of bounds/overlap")
    smt = np.asarray(data["smt"])
    if smt.shape != (4000, 100, len(names)) or smt.dtype.kind not in "iuf": raise ValueError("smt schema differs")
    supports = []
    for index, event in enumerate(events):
        support = np.ascontiguousarray(signal_data[event:event + 4000, selected])
        if support.shape != (4000, 8): raise ValueError(f"selected raw epoch support shape differs: trial {index}")
        if not np.all(np.isfinite(support)):
            raise ValueError(f"selected raw epoch support contains nonfinite values: trial {index}")
        supports.append(support)
    minus_one = zero = 0
    for index, event in enumerate(events_matlab):
        expected_smt = smt[:, index, :][:, selected]
        minus_one += int(np.array_equal(expected_smt, signal_data[event - 1:event - 1 + 4000, selected]))
        zero += int(np.array_equal(expected_smt, signal_data[event:event + 4000, selected]))
    if minus_one != 0 or zero != 100: raise ValueError("smt one-sample discrepancy differs")
    raw_hash_list, epoch_hash_list, epochs = [], [], []
    sos = butter(4, [8.0, 30.0], btype="bandpass", fs=1000, output="sos")
    for index, support in enumerate(supports):
        raw_hash_list.append(array_sha256(support))
        if build_epochs:
            filtered = sosfiltfilt(sos, support.T.astype(np.float64) * 1e-6, axis=-1, padtype="odd", padlen=500)
            resampled = resample_poly(filtered, 1, 4, axis=-1, window=("kaiser", 5.0), padtype="line")
            epoch = np.ascontiguousarray(resampled[:, 125:875], dtype=np.float64)
            if epoch.shape != (8, 750) or not np.all(np.isfinite(epoch)): raise ValueError("epoch invalid")
            epochs.append(epoch); epoch_hash_list.append(array_sha256(epoch))
    return {"X": np.stack(epochs) if build_epochs else None, "y": labels - 1, "events_matlab": events_matlab,
            "events_zero": events, "raw_support_sha256": raw_hash_list, "epoch_sha256": epoch_hash_list,
            "raw_channel_names": names, "channel_indices_zero_based": selected,
            "smt_matches": {"matlab_t_minus_1": minus_one, "python_t": zero}}


def parse_raw(path, cfg):
    from scipy.io import loadmat
    mat = loadmat(path, variable_names=["EEG_MI_train"], simplify_cells=True)
    if set(key for key in mat if not key.startswith("__")) != {"EEG_MI_train"}: raise ValueError("MAT variable scope differs")
    return reconstruct_raw_struct(mat.get("EEG_MI_train"), cfg, build_epochs=True)


def compare_reconstruction(key, rebuilt, record, row, split, root):
    import numpy as np
    subject, session = key; metadata = record["metadata"]
    fixed = {"mat_variable": "EEG_MI_train", "loaded_variables": ["EEG_MI_train"], "raw_fs": 1000, "fs": 250,
             "units": "V", "raw_units": "uV", "channel_names": list(CHANNELS), "class_names": list(CLASS_NAMES),
             "raw_class_mapping": {"1": "right", "2": "left"}, "label_mapping": {"1": 0, "2": 1}, "shape": [100, 8, 750]}
    stored_metadata = json.loads(rooted_file(root, row["metadata_path"]).read_text())
    if stored_metadata != metadata: raise ValueError(f"metadata file/receipt differs: {key}")
    if Path(metadata.get("path", "")).resolve() != rooted_file(root, row["raw_path"]): raise ValueError(f"metadata raw path differs: {key}")
    expected_config = {"channels": list(CHANNELS), "l_freq": 8.0, "h_freq": 30.0, "tmin": 0.5,
                       "tmax": 3.5, "target_fs": 250, "filter_order": 4}
    if (any(metadata.get(name) != value for name, value in fixed.items())
            or metadata.get("config") != expected_config
            or metadata.get("class_counts") != {"right_hand": 50, "left_hand": 50}):
        raise ValueError(f"metadata contract differs: {key}")
    if (metadata.get("raw_channel_names") != rebuilt["raw_channel_names"]
            or metadata.get("channel_indices_zero_based") != rebuilt["channel_indices_zero_based"]
            or metadata.get("raw_labels") != (rebuilt["y"] + 1).tolist()
            or metadata.get("event_samples_matlab") != rebuilt["events_matlab"].tolist()
            or metadata.get("event_samples_zero_based") != rebuilt["events_zero"].tolist()
            or metadata.get("trial_raw_support_sha256") != rebuilt["raw_support_sha256"]
            or metadata.get("trial_epoch_sha256") != rebuilt["epoch_sha256"]): raise ValueError(f"rebuilt metadata differs: {key}")
    schema = record.get("schema", {})
    if schema.get("provided_smt_exact_matches_selected_channels_by_python_t_offset") != {"-1": 0, "0": 100}:
        raise ValueError(f"saved smt discrepancy differs: {key}")
    for index in range(100):
        saved = split[trial_id(subject, session, index)]
        if (int(saved["label"]) != int(rebuilt["y"][index])
                or int(saved["event_sample_matlab"]) != int(rebuilt["events_matlab"][index])
                or int(saved["event_sample_zero_based"]) != int(rebuilt["events_zero"][index])
                or saved["raw_support_sha256"] != rebuilt["raw_support_sha256"][index]
                or saved["epoch_sha256"] != rebuilt["epoch_sha256"][index]): raise ValueError("rebuilt split identity differs")
    with np.load(rooted_file(root, row["derived_path"]), allow_pickle=False) as cache:
        if set(cache.files) != {"X", "y"}: raise ValueError("cache fields differ")
        cached_X, cached_y = cache["X"], cache["y"]
    if cached_X.shape != (100, 8, 750) or cached_X.dtype != np.dtype("float64") or not np.array_equal(cached_y, rebuilt["y"]):
        raise ValueError("cache X/y schema differs")
    maximum = float(np.max(np.abs(cached_X - rebuilt["X"])))
    if not math.isfinite(maximum) or maximum > 1e-12: raise ValueError(f"epoch tolerance exceeded: {key} {maximum}")
    exact = array_sha256(cached_X) == array_sha256(rebuilt["X"])
    if not exact: raise ValueError(f"epoch hash differs: {key}")
    rebuilt_X_hash = typed_array_sha256(rebuilt["X"], "<f8"); cached_X_hash = typed_array_sha256(cached_X, "<f8")
    rebuilt_y_hash = typed_array_sha256(rebuilt["y"], "<i8"); cached_y_hash = typed_array_sha256(cached_y, "<i8")
    return {"subject_id": subject, "session": session, "trials": 100, "max_abs_epoch_difference": maximum,
            "labels_exact": True, "array_hash_exact": exact, "trial_epoch_hashes_exact": True,
            "trial_raw_support_hashes_exact": True,
            "rebuilt_X_float64_c_sha256": rebuilt_X_hash, "cached_X_float64_c_sha256": cached_X_hash,
            "rebuilt_y_int64_c_sha256": rebuilt_y_hash, "cached_y_int64_c_sha256": cached_y_hash,
            "smt_t_minus_1_matches": rebuilt["smt_matches"]["matlab_t_minus_1"],
            "smt_python_t_matches": rebuilt["smt_matches"]["python_t"]}


def reconstructed_split(cfg, rebuilt):
    expected = {(s, 1) for s in cfg["source_subjects"]} | {(s, t) for s in cfg["development_subjects"] for t in (1, 2)}
    if set(rebuilt) != expected: raise ValueError("rebuilt subject/session cells differ")
    result = {}
    for (subject, session), item in sorted(rebuilt.items()):
        role = "source" if subject in cfg["source_subjects"] else "development"
        usage = "source_training" if role == "source" else "development_history" if session == 1 else "development_evaluation"
        for index in range(100):
            tid = trial_id(subject, session, index); label = int(item["y"][index])
            result[tid] = {"trial_id": tid, "subject_id": subject, "session": session, "trial_index_zero_based": index,
                           "role": role, "label": label, "class_name": CLASS_NAMES[label], "usage": usage,
                           "event_sample_matlab": int(item["events_matlab"][index]),
                           "event_sample_zero_based": int(item["events_zero"][index]),
                           "raw_support_sha256": item["raw_support_sha256"][index],
                           "epoch_sha256": item["epoch_sha256"][index]}
    return result


def compare_reconstructed_split(rebuilt_split, saved_split):
    if set(rebuilt_split) != set(saved_split): raise ValueError("rebuilt/saved split identities differ")
    fields = ("subject_id", "session", "trial_index_zero_based", "label", "event_sample_matlab", "event_sample_zero_based")
    for tid, rebuilt in rebuilt_split.items():
        saved = saved_split[tid]
        if (any(int(saved[field]) != rebuilt[field] for field in fields)
                or any(saved[field] != rebuilt[field] for field in ("role", "class_name", "usage", "raw_support_sha256", "epoch_sha256"))):
            raise ValueError(f"rebuilt/saved split row differs: {tid}")
    return True


def history_ids(split, subject, seed_value, dose=30):
    import numpy as np
    pool = sorted(tid for tid, row in split.items() if int(row["subject_id"]) == subject and int(row["session"]) == 1)
    if len(pool) != 100: raise ValueError("history pool differs")
    labels = np.array([int(split[tid]["label"]) for tid in pool]); rng = np.random.default_rng(np.random.SeedSequence([seed_value, subject]))
    selected = []
    for label in (0, 1): selected.extend(pool[int(index)] for index in rng.permutation(np.flatnonzero(labels == label))[:dose])
    if len(selected) != 2 * dose: raise ValueError("history dose differs")
    return selected


def make_model(name):
    from mne.decoding import CSP
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "csp_lda":
        return make_pipeline(CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False, cov_est="concat",
                                 component_order="mutual_info"), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if name == "ts_lr":
        return make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann", tsupdate=False), StandardScaler(),
                             LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=20260914))
    if name == "mdm": return make_pipeline(Covariances(estimator="oas"), MDM(metric="riemann", n_jobs=1))
    raise ValueError(f"unknown model: {name}")


class SourceCentroid:
    def fit(self, X, y):
        import numpy as np
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.representation = make_pipeline(Covariances(estimator="oas"), TangentSpace(metric="riemann", tsupdate=False), StandardScaler())
        z = self.representation.fit_transform(X); self.source_centroids = np.stack([z[y == c].mean(axis=0) for c in (0, 1)])
        self.classes_ = np.array([0, 1]); return self

    def adapted_centroids(self, X, y, prior):
        import numpy as np
        z = self.representation.transform(X)
        return np.stack([(prior * self.source_centroids[c] + z[y == c].sum(axis=0)) /
                         (prior + np.count_nonzero(y == c)) for c in (0, 1)])

    def predict_proba(self, X, centroids=None):
        import numpy as np
        centroids = self.source_centroids if centroids is None else centroids; z = self.representation.transform(X)
        score = -np.sum((z[:, None, :] - centroids[None, :, :]) ** 2, axis=2); score -= score.max(axis=1, keepdims=True)
        probability = np.exp(score); return probability / probability.sum(axis=1, keepdims=True)


def membership_rows(operation, source, history, evaluation, calibration_update=False):
    return ([{"operation_id": operation, "usage": "source_representation" if calibration_update else "source_training", "trial_id": t} for t in source]
            + [{"operation_id": operation, "usage": "target_update" if calibration_update else "history_training", "trial_id": t} for t in history]
            + [{"operation_id": operation, "usage": "evaluation", "trial_id": t} for t in evaluation])


def prediction_rows(operation, arm, model, draw, dose, prior, evaluation, probability, labels, split):
    import numpy as np
    probability = np.asarray(probability)
    if (probability.shape != (len(evaluation), 2) or not np.all(np.isfinite(probability)) or np.any(probability < 0)
            or np.any(probability > 1) or np.max(np.abs(probability.sum(axis=1) - 1)) > 1e-10): raise ValueError("probability schema differs")
    predicted, result = probability.argmax(axis=1), []
    for i, tid in enumerate(evaluation):
        truth = int(labels[i])
        if truth != int(split[tid]["label"]): raise ValueError("evaluation label differs")
        result.append({"operation_id": operation, "arm": arm, "model": model, "draw": "" if draw is None else draw,
                       "dose_per_class": dose, "prior_weight": "" if prior is None else prior,
                       "subject_id": int(split[tid]["subject_id"]), "trial_id": tid, "y_true": truth,
                       "y_pred": int(predicted[i]), "p_class0_right": float(probability[i, 0]),
                       "p_class1_left": float(probability[i, 1])})
    return result


def build_reference_predictions(guarded, rebuilt, split=None):
    import mne
    import numpy as np
    from threadpoolctl import threadpool_info, threadpool_limits
    mne.set_log_level("WARNING"); cfg = guarded["cfg"]
    split = reconstructed_split(cfg, rebuilt) if split is None else split
    source = sorted(t for t, row in split.items() if row["role"] == "source" and int(row["session"]) == 1)
    target = min(cfg["development_subjects"]); history = history_ids(split, target, cfg["history_draw_seeds"][0])
    evaluation_all = sorted(t for t, row in split.items() if row["role"] == "development" and int(row["session"]) == 2)
    evaluation_target = [t for t in evaluation_all if int(split[t]["subject_id"]) == target]
    if tuple(map(len, (source, history, evaluation_all, evaluation_target))) != (1800, 60, 1200, 100): raise ValueError("subset membership count differs")
    epochs, labels = {}, {}
    for (subject, session), item in rebuilt.items():
        for i in range(100):
            tid = trial_id(subject, session, i); epochs[tid] = item["X"][i]; labels[tid] = int(item["y"][i])
    if set(epochs) != set(split): raise ValueError("rebuilt identity set differs")
    def stack(ids): return np.stack([epochs[t] for t in ids]), np.array([labels[t] for t in ids])
    source_X, source_y = stack(source); history_X, history_y = stack(history)
    all_X, all_y = stack(evaluation_all); target_X, target_y = stack(evaluation_target)
    predictions, memberships, timings = [], [], []
    def classical(operation, model_name, draw, dose, train_X, train_y, evaluation, query_X, query_y):
        fw, fc = time.perf_counter(), time.process_time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always"); model = make_model(model_name).fit(train_X, train_y)
        fit_wall, fit_cpu = time.perf_counter() - fw, time.process_time() - fc
        if any("converg" in str(w.message).lower() for w in caught) or model.classes_.tolist() != [0, 1]: raise ValueError("fit warning/class order differs")
        pw, pc = time.perf_counter(), time.process_time(); probability = model.predict_proba(query_X)
        predict_wall, predict_cpu = time.perf_counter() - pw, time.process_time() - pc
        predictions.extend(prediction_rows(operation, "practical", model_name, draw, dose, None, evaluation, probability, query_y, split))
        timings.append({"operation_id": operation, "operation_kind": "fit", "model": model_name, "source_trials": 1800,
                        "history_trials": len(train_y) - 1800, "evaluation_trials": len(evaluation), "fit_cpu_seconds": fit_cpu,
                        "fit_wall_seconds": fit_wall, "predict_cpu_seconds": predict_cpu, "predict_wall_seconds": predict_wall,
                        "warnings": len(caught)})
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools): raise ValueError("thread limit differs")
        for model_name in cfg["model_order"]:
            operation = f"practical__{model_name}__rshared_n0"; classical(operation, model_name, None, 0, source_X, source_y, evaluation_all, all_X, all_y)
            memberships.extend(membership_rows(operation, source, [], evaluation_all))
            operation = f"practical__{model_name}__r0_n30_s{target:03d}"
            classical(operation, model_name, 0, 30, np.concatenate([source_X, history_X]), np.concatenate([source_y, history_y]),
                      evaluation_target, target_X, target_y); memberships.extend(membership_rows(operation, source, history, evaluation_target))
        operation = "calibration__centroid__rshared_n0"; fw, fc = time.perf_counter(), time.process_time(); centroid = SourceCentroid().fit(source_X, source_y)
        fit_wall, fit_cpu = time.perf_counter() - fw, time.process_time() - fc; pw, pc = time.perf_counter(), time.process_time(); probability = centroid.predict_proba(all_X)
        predict_wall, predict_cpu = time.perf_counter() - pw, time.process_time() - pc
        predictions.extend(prediction_rows(operation, "calibration", "centroid", None, 0, None, evaluation_all, probability, all_y, split))
        memberships.extend(membership_rows(operation, source, [], evaluation_all)); timings.append({"operation_id": operation, "operation_kind": "fit", "model": "centroid",
            "source_trials": 1800, "history_trials": 0, "evaluation_trials": 1200, "fit_cpu_seconds": fit_cpu, "fit_wall_seconds": fit_wall,
            "predict_cpu_seconds": predict_cpu, "predict_wall_seconds": predict_wall, "warnings": 0})
        for prior in (0, 100):
            operation = f"calibration__centroid__r0_n30_s{target:03d}_w{prior}"; fw, fc = time.perf_counter(), time.process_time()
            centroids = centroid.adapted_centroids(history_X, history_y, prior); fit_wall, fit_cpu = time.perf_counter() - fw, time.process_time() - fc
            pw, pc = time.perf_counter(), time.process_time(); probability = centroid.predict_proba(target_X, centroids)
            predict_wall, predict_cpu = time.perf_counter() - pw, time.process_time() - pc
            predictions.extend(prediction_rows(operation, "calibration", "centroid", 0, 30, prior, evaluation_target, probability, target_y, split))
            memberships.extend(membership_rows(operation, source, history, evaluation_target, True)); timings.append({"operation_id": operation,
                "operation_kind": "target_update", "model": "centroid", "source_trials": 1800, "history_trials": 60,
                "evaluation_trials": 100, "fit_cpu_seconds": fit_cpu, "fit_wall_seconds": fit_wall,
                "predict_cpu_seconds": predict_cpu, "predict_wall_seconds": predict_wall, "warnings": 0})
    if len(predictions) != 5300 or len(timings) != 9: raise ValueError("physical operation/row count differs")
    return predictions, memberships, timings, {"source": source, "history": history, "evaluation_all": evaluation_all,
                                                "evaluation_target": evaluation_target, "threadpools": pools, "target_subject": target}


def collapse_original_predictions(values, operation, expected_combinations):
    if not values or set(values[0]) != set(PREDICTION_FIELDS): raise ValueError("original prediction columns differ")
    grouped = defaultdict(list)
    for row in values:
        if row.get("operation_id") != operation or (row.get("draw"), row.get("prior_weight")) not in expected_combinations:
            raise ValueError("original logical identity differs")
        truth, predicted = integer_text(row["y_true"], "y_true", 0), integer_text(row["y_pred"], "y_pred", 0)
        probability = (float(row["p_class0_right"]), float(row["p_class1_left"]))
        if (truth not in (0, 1) or predicted not in (0, 1) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probability)
                or abs(sum(probability) - 1) > 1e-10 or predicted != (0 if probability[0] >= probability[1] else 1)):
            raise ValueError("original probability/class mapping differs")
        grouped[row["trial_id"]].append(row)
    collapsed = {}
    for tid, duplicates in grouped.items():
        combinations = {(row["draw"], row["prior_weight"]) for row in duplicates}
        signatures = {(row["subject_id"], row["trial_id"], row["y_true"], row["y_pred"],
                       row["p_class0_right"], row["p_class1_left"]) for row in duplicates}
        if combinations != expected_combinations or len(duplicates) != len(expected_combinations) or len(signatures) != 1:
            raise ValueError(f"shared n0 duplicates disagree: {operation} {tid}")
        collapsed[tid] = duplicates[0]
    return collapsed


def balanced_accuracy(values):
    scores = []
    for label in (0, 1):
        selected = [row for row in values if int(row["y_true"]) == label]
        if not selected: raise ValueError("balanced accuracy lacks class")
        scores.append(sum(int(row["y_pred"]) == label for row in selected) / len(selected))
    return sum(scores) / 2


def compare_predictions(reference_values, original_values, operation, expected_combinations, tolerance=1e-10):
    selected = [row for row in reference_values if row["operation_id"] == operation]
    reference = {row["trial_id"]: row for row in selected}; original = collapse_original_predictions(original_values, operation, expected_combinations)
    if len(reference) != len(selected) or set(reference) != set(original): raise ValueError("physical prediction identities differ")
    maximum = 0.0
    for tid in sorted(reference):
        left, right = reference[tid], original[tid]
        if int(left["y_true"]) != int(right["y_true"]) or int(left["y_pred"]) != int(right["y_pred"]):
            raise ValueError(f"prediction label differs: {operation} {tid}")
        maximum = max(maximum, abs(float(left["p_class0_right"]) - float(right["p_class0_right"])),
                      abs(float(left["p_class1_left"]) - float(right["p_class1_left"])))
    if maximum > tolerance: raise ValueError(f"probability tolerance exceeded: {operation} {maximum}")
    left_ba, right_ba = balanced_accuracy(reference.values()), balanced_accuracy(original.values())
    if left_ba != right_ba: raise ValueError("balanced accuracy differs")
    return {"operation_id": operation, "physical_rows": len(reference), "logical_rows": len(original_values),
            "logical_multiplicity": len(expected_combinations), "max_abs_probability_difference": maximum,
            "predicted_labels_exact": True, "balanced_accuracy": left_ba, "balanced_accuracy_exact": True}


def compare_membership(reference_values, original_values, operation):
    reference = [(row["operation_id"], row["usage"], row["trial_id"]) for row in reference_values if row["operation_id"] == operation]
    original = [(row.get("operation_id"), row.get("usage"), row.get("trial_id")) for row in original_values]
    if reference != original or len(reference) != len(set(reference)): raise ValueError(f"membership differs: {operation}")
    return len(reference)


def operation_combinations(operation):
    if operation == "calibration__centroid__rshared_n0": return {(str(d), str(w)) for d in (0, 1) for w in (0, 5, 20, 100)}
    if operation.endswith("rshared_n0"): return {(str(d), "") for d in (0, 1)}
    if operation.endswith("_w0"): return {("0", "0")}
    if operation.endswith("_w100"): return {("0", "100")}
    return {("0", "")}


def compare_with_grid(grid, predictions, memberships):
    comparisons = []
    for operation in sorted({row["operation_id"] for row in predictions}):
        result = compare_predictions(predictions, read_csv(grid / f"{operation}.predictions.csv"), operation,
                                     operation_combinations(operation))
        result["membership_rows"] = compare_membership(memberships, read_csv(grid / f"{operation}.membership.csv"), operation)
        comparisons.append(result)
    if len(comparisons) != 9 or sum(row["physical_rows"] for row in comparisons) != 5300:
        raise ValueError("comparison operation/row count differs")
    return comparisons


def selected_operation_ids(cfg):
    return sorted({*(f"practical__{model}__rshared_n0" for model in cfg["model_order"]),
                   *(f"practical__{model}__r0_n30_s002" for model in cfg["model_order"]),
                   "calibration__centroid__rshared_n0", "calibration__centroid__r0_n30_s002_w0",
                   "calibration__centroid__r0_n30_s002_w100"})


def reference_input_paths(root, guarded):
    paths = ["research/day5/COMPLETION_CONTRACT_v1.md", "research/protocols/DAY3_CONFIG_v1.json",
             "research/protocols/DAY3_CONFIG_v1.json.sha256", guarded["cfg"]["role_manifest"],
             "research/environment/requirements.lock.txt", "scripts/verify_day5_raw_reference.py",
             "scripts/verify_day5_raw_reference_v2.py",
             "scripts/launch_day5_raw_reference.py", "src/expose/runtime.py"]
    for directory in (guarded["prep"], guarded["grid"]):
        paths.extend(str((directory / name).relative_to(root)) for name in
                     ("supervisor_receipt.json", "COMPLETED.json", "receipt.json", "INPUT_MANIFEST.json"))
    paths.extend(str((guarded["prep"] / name).relative_to(root)) for name in ("DATA_MANIFEST.csv", "SPLIT_MANIFEST.csv", "schema.json"))
    for row in guarded["data"].values(): paths.extend((row["raw_path"], row["derived_path"], row["metadata_path"]))
    for operation in selected_operation_ids(guarded["cfg"]):
        paths.extend((str((guarded["grid"] / f"{operation}.predictions.csv").relative_to(root)),
                      str((guarded["grid"] / f"{operation}.membership.csv").relative_to(root))))
    if len(paths) != len(set(paths)): raise ValueError("input paths repeat")
    return sorted(paths)


def collect_input_manifest(root, guarded):
    manifest, raw_checks = {}, []; raw_paths = {row["raw_path"]: row for row in guarded["data"].values()}
    for relative in reference_input_paths(root, guarded):
        path = rooted_file(root, relative)
        if relative in raw_paths:
            observed, row = raw_hashes(path), raw_paths[relative]
            if (observed["bytes"] != int(row["bytes"]) or observed["md5"] != row["published_md5"]
                    or observed["sha256"] != row["local_sha256"]): raise ValueError(f"raw hash differs: {relative}")
            raw_checks.append({"subject_id": int(row["subject_id"]), "session": int(row["session"]), "path": relative,
                               **observed, "published_md5_exact": True, "manifest_sha256_exact": True})
            manifest[relative] = observed["sha256"]
        else: manifest[relative] = sha256(path)
    return manifest, sorted(raw_checks, key=lambda row: (row["subject_id"], row["session"]))


def inputs_unchanged(root, manifest):
    changed = [relative for relative, expected in manifest.items() if sha256(rooted_file(root, relative)) != expected]
    if changed: raise ValueError(f"inputs changed: {changed}")
    return True


def prepare_output(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    owned = {"receipt.json", "COMPLETED.json", "INPUT_MANIFEST.json", "raw_checks.csv", "array_checks.csv",
             "predictions.csv", "membership.csv", "timing.csv", "PREDICTION_LOCK.json", "comparison.json", "checks.json"}
    existing = owned & {path.name for path in out.iterdir()}
    if existing: raise FileExistsError(f"existing raw-reference evidence: {sorted(existing)}")
    return out


def write_completion(out):
    out = Path(out)
    required = ("receipt.json", "INPUT_MANIFEST.json", "raw_checks.csv", "array_checks.csv", "predictions.csv",
                "membership.csv", "timing.csv", "PREDICTION_LOCK.json", "comparison.json", "checks.json")
    receipt = json.loads((out / "receipt.json").read_text())
    if receipt.get("status") != "completed" or receipt.get("exit_code") != 0 or receipt.get("run_id") != out.name:
        raise ValueError("cannot complete unsuccessful raw-reference run")
    outputs = {name: sha256(out / name) for name in required}
    marker = {"run_id": out.name, "receipt_sha256": outputs["receipt.json"], "outputs": outputs}
    atomic_json(out / "COMPLETED.json", marker)
    return marker


def execute(root, out):
    root, out = Path(root).resolve(), Path(out).resolve(); guarded = preflight(root)
    manifest, raw_checks = collect_input_manifest(root, guarded); atomic_json(out / "INPUT_MANIFEST.json", manifest)
    atomic_csv(out / "raw_checks.csv", list(raw_checks[0]), raw_checks)
    rebuilt = {key: parse_raw(rooted_file(root, guarded["data"][key]["raw_path"]), guarded["cfg"]) for key in sorted(guarded["data"])}
    independent_split = reconstructed_split(guarded["cfg"], rebuilt)
    compare_reconstructed_split(independent_split, guarded["split"])
    array_checks = [compare_reconstruction(key, rebuilt[key], guarded["records"][key], guarded["data"][key],
                                           guarded["split"], root) for key in sorted(rebuilt)]
    atomic_csv(out / "array_checks.csv", list(array_checks[0]), array_checks)
    predictions, memberships, timings, subset = build_reference_predictions(guarded, rebuilt, independent_split)
    # The independent physical outputs exist before corresponding Day3 CSV content is opened below.
    atomic_csv(out / "predictions.csv", PREDICTION_FIELDS, predictions)
    atomic_csv(out / "membership.csv", ("operation_id", "usage", "trial_id"), memberships)
    atomic_csv(out / "timing.csv", list(timings[0]), timings)
    prediction_lock = {"created_utc": utc_now(), "physical_operations": 9, "physical_prediction_rows": 5300,
                       "files": {name: sha256(out / name) for name in ("predictions.csv", "membership.csv", "timing.csv")}}
    atomic_json(out / "PREDICTION_LOCK.json", prediction_lock)
    comparisons = compare_with_grid(guarded["grid"], predictions, memberships)
    comparison = {"probability_tolerance": 1e-10, "operations": comparisons, "physical_operations": len(comparisons),
                  "physical_prediction_rows": sum(row["physical_rows"] for row in comparisons),
                  "logical_prediction_rows_read": sum(row["logical_rows"] for row in comparisons),
                  "maximum_probability_difference": max(row["max_abs_probability_difference"] for row in comparisons),
                  "all_predicted_labels_exact": all(row["predicted_labels_exact"] for row in comparisons),
                  "all_balanced_accuracies_exact": all(row["balanced_accuracy_exact"] for row in comparisons)}
    atomic_json(out / "comparison.json", comparison); inputs_unchanged(root, manifest)
    checks = {"scope": "informed shared-library Builder QA; raw reconstruction plus nine deterministic physical operations",
              "confirmation_accessed": False, "raw_files": len(raw_checks), "rebuilt_trials": 4200,
              "source_subjects": 18, "development_subjects": 12, "target_subject": subset["target_subject"],
              "source_trials": len(subset["source"]), "history_trials": len(subset["history"]),
              "evaluation_trials_source_models": len(subset["evaluation_all"]),
              "evaluation_trials_target_models": len(subset["evaluation_target"]),
              "physical_operations": len(timings), "physical_prediction_rows": len(predictions),
              "max_abs_epoch_difference": max(row["max_abs_epoch_difference"] for row in array_checks),
              "all_array_hashes_exact": all(row["array_hash_exact"] for row in array_checks), "comparison": comparison,
              "prediction_lock_sha256": sha256(out / "PREDICTION_LOCK.json"),
              "input_manifest_sha256": sha256(out / "INPUT_MANIFEST.json"), "input_files": len(manifest),
              "inputs_unchanged": True, "numerical_thread_limit": 1, "threadpools": subset["threadpools"], "gpu_hours": 0}
    atomic_json(out / "checks.json", checks); return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv); out = prepare_output(args.run_dir); started_wall, started_cpu = time.monotonic(), time.process_time()
    receipt = {"run_id": out.name, "status": "running", "pid": os.getpid(), "started_utc": utc_now(),
               "command": [sys.executable, *sys.argv], "interpreter": sys.executable,
               "scope": "informed shared-library Builder QA; not independent authorship or all-grid replication"}
    atomic_json(out / "receipt.json", receipt)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3,) * 2); resource.setrlimit(resource.RLIMIT_CPU, (1200, 1210))
        def expired(signum, _frame): raise TimeoutError(f"Day5 raw-reference limit exceeded: signal {signum}")
        signal.signal(signal.SIGALRM, expired); signal.signal(signal.SIGXCPU, expired); signal.alarm(1800)
        checks = execute(ROOT, out); receipt.update(status="completed", exit_code=0, checks_sha256=sha256(out / "checks.json"),
            input_manifest_sha256=checks["input_manifest_sha256"], physical_operations=checks["physical_operations"],
            physical_prediction_rows=checks["physical_prediction_rows"], confirmation_accessed=False)
    except BaseException as error:
        receipt.update(status="failed", exit_code=1, error=f"{type(error).__name__}: {error}"); traceback.print_exc()
    finally:
        signal.alarm(0); receipt.update(finished_utc=utc_now(), wall_seconds=time.monotonic() - started_wall,
            cpu_seconds=time.process_time() - started_cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    if receipt["status"] == "completed":
        write_completion(out)
    return receipt["exit_code"]


if __name__ == "__main__": raise SystemExit(main())
