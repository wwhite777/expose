"""Prepare a guarded development-only OpenBMI signal audit and 20-channel cache.

This is a prospective review3 preparation tool.  It deliberately requires an
explicit channel-approval JSON before it opens a MAT file; the 20-channel list
below is a candidate, not a silently adopted montage.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expose.development import load_config
from expose.lee import EpochConfig, load_offline_epochs
from expose.provenance import file_sha256
from expose.review3_data import covariance_record, fit_review3_whitener, legacy_anchor_check, whitener_sha256

PROPOSED_20_MOTOR_CHANNELS = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz",
    "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)
LEGACY_CHANNELS = ("C3", "Cz", "C4", "FC3", "FC4", "CP3", "CPz", "CP4")
RAW_SAMPLES = 4000
RAW_FS = 1000
WINDOW_START, WINDOW_STOP = 500, 3500


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha(path):
    return file_sha256(Path(path))


def expected_roles(cfg):
    return ({(int(s), 1): "source" for s in cfg["source_subjects"]} |
            {(int(s), session): "development" for s in cfg["development_subjects"] for session in (1, 2)})


def read_approval(path):
    """Read an explicit root-authored selection; accept only the proposed primary list."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    channels = tuple(value.get("channels", [])) if isinstance(value, dict) else ()
    if channels != PROPOSED_20_MOTOR_CHANNELS:
        raise ValueError("approval must select the exact proposed 20-channel ordering")
    if not isinstance(value.get("approved_by"), str) or not value["approved_by"].strip():
        raise ValueError("approval requires nonempty approved_by")
    return value


def validate_record_roles(records, expected):
    by_key = {(int(r["subject"]), int(r["session"])): r for r in records}
    if len(by_key) != len(records) or set(by_key) != set(expected):
        raise ValueError("preparation records do not equal frozen source/development role set")
    if any(by_key[key]["role"] != role for key, role in expected.items()):
        raise ValueError("preparation record role disagrees with frozen role set")
    return by_key


def validate_channel_names(channels, actual_names):
    if len(channels) != 20 or len(set(channels)) != 20:
        raise ValueError("wider montage must contain exactly 20 unique channels")
    missing = [name for name in channels if name not in actual_names]
    if missing:
        raise ValueError(f"approved channels absent from file: {missing}")
    return [actual_names.index(name) for name in channels]


def read_plan(cfg):
    """Use frozen Day3 preparation records and role/split manifests, never cache arrays."""
    schema_path = ROOT / "result/day3/preparation_r001/schema.json"
    split_path = ROOT / "result/day3/preparation_r001/SPLIT_MANIFEST.csv"
    records = json.loads(schema_path.read_text(encoding="utf-8"))["records"]
    expected = expected_roles(cfg)
    by_key = validate_record_roles(records, expected)
    rows = list(csv.DictReader(split_path.open(encoding="utf-8", newline="")))
    if len(rows) != 4200:
        raise ValueError("frozen split manifest must contain 4200 trials")
    grouped = {}
    for row in rows:
        key = (int(row["subject_id"]), int(row["session"]))
        if key not in expected or row["role"] != expected[key] or row["mat_variable"] != "EEG_MI_train":
            raise ValueError("split manifest contains an unapproved role, session, or MAT variable")
        grouped.setdefault(key, []).append(row)
    if set(grouped) != set(expected) or any(len(v) != 100 for v in grouped.values()):
        raise ValueError("split manifest must provide 100 trials for each exact allowed file")
    return by_key, grouped


def verify_metadata(records, channels):
    """Hash JSON metadata and prove the approved names occur before raw opening."""
    facts = []
    for key, record in sorted(records.items()):
        cache = record["cache"]
        metadata_path = ROOT / cache["metadata_path"]
        if sha(metadata_path) != cache["metadata_sha256"]:
            raise ValueError(f"cache metadata hash changed for {key}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        names = metadata.get("raw_channel_names")
        if not isinstance(names, list):
            raise ValueError(f"metadata lacks raw channel names for {key}")
        indexes = validate_channel_names(channels, names)
        original_indexes = [names.index(channel) for channel in LEGACY_CHANNELS]
        facts.append({"subject_id": key[0], "session": key[1], "metadata_path": cache["metadata_path"],
                      "metadata_sha256": cache["metadata_sha256"], "channel_indices_zero_based": indexes,
                      "original_8_indices_zero_based": original_indexes})
    return facts


def intervals_for_sample(sample, split_rows):
    """Map a raw zero-based sample to all legacy supports/windows without exclusion."""
    hits = []
    for row in split_rows:
        start = int(row["event_sample_zero_based"])
        support = start <= sample < start + RAW_SAMPLES
        legacy_window = start + WINDOW_START <= sample < start + WINDOW_STOP
        alternate_window = start + 1 + WINDOW_START <= sample < start + 1 + WINDOW_STOP
        if support:
            hits.append({"trial_id": row["trial_id"], "trial_index_zero_based": int(row["trial_index_zero_based"]),
                         "in_raw_4s_support": True, "in_legacy_0p5_to_3p5_window": legacy_window,
                         "in_alternate_offset_plus_1_window": alternate_window})
    return hits


def selected_extreme(signal, channel_names, selected_indexes):
    import numpy as np
    selected = signal[:, selected_indexes]
    if not np.all(np.isfinite(selected)):
        raise ValueError("selected raw channels contain nonfinite samples")
    flat = int(np.nanargmax(np.abs(selected)))
    sample, selected_position = np.unravel_index(flat, selected.shape)
    return {"raw_sample_zero_based": int(sample), "channel": channel_names[selected_indexes[selected_position]],
            "channel_index_zero_based": int(selected_indexes[selected_position]),
            "amplitude_microvolts": float(selected[sample, selected_position]),
            "absolute_amplitude_microvolts": float(abs(selected[sample, selected_position]))}


def audit_segmented_field(data, signal, selected_indexes, split_rows):
    """Compare supplied smt only; legacy t-1 remains the processing anchor."""
    import numpy as np
    smt = data.get("smt")
    result = {"field_present": smt is not None, "legacy_anchor": "MATLAB t-1", "alternate": "MATLAB t (offset +1 sensitivity)"}
    if smt is None:
        return result
    smt = np.asarray(smt)
    if smt.ndim != 3 or smt.shape[0] != RAW_SAMPLES or smt.shape[1] != len(split_rows):
        result.update({"shape": list(smt.shape), "comparable": False})
        return result
    if smt.shape[2] != signal.shape[1]:
        raise ValueError("smt channel axis does not match x")
    legacy_matches = alternate_matches = 0
    for i, row in enumerate(sorted(split_rows, key=lambda r: int(r["trial_index_zero_based"]))):
        matlab = int(row["event_sample_matlab"])
        legacy = matlab - 1
        supplied = smt[:, i, :][:, selected_indexes]
        legacy_matches += bool(np.array_equal(supplied, signal[legacy:legacy + RAW_SAMPLES, selected_indexes]))
        alternate_matches += bool(np.array_equal(supplied, signal[matlab:matlab + RAW_SAMPLES, selected_indexes]))
    result.update({"shape": list(smt.shape), "comparable": True, "trials": len(split_rows),
                   "exact_matches_legacy_t_minus_1": legacy_matches,
                   "exact_matches_alternate_t": alternate_matches})
    return result


def load_train_only(path):
    from scipy.io import loadmat
    value = loadmat(path, variable_names=["EEG_MI_train"], simplify_cells=True).get("EEG_MI_train")
    if not isinstance(value, dict):
        raise ValueError("MAT file must contain EEG_MI_train only")
    return value


def output_manifest(out):
    return {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*")) if path.is_file()
            and path.name not in {"COMPLETED.json", "receipt.json"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--approved-channels-json", required=True, type=Path,
                        help="root-authored approval of the exact candidate 20-channel montage")
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if out.exists():
        raise FileExistsError("run directory must be new and exclusive")
    out.mkdir(parents=True)
    start, cpu = time.monotonic(), time.process_time()
    receipt = {"run_id": out.name, "status": "running", "pid": os.getpid(),
               "started_utc": datetime.now(timezone.utc).isoformat(), "interpreter": sys.executable,
               "confirmation_data_accessed": False, "online_phase_accessed": False, "model_fits": 0}
    atomic_json(out / "receipt.json", receipt)
    try:
        cfg = load_config(ROOT)
        config_path = ROOT / "research/protocols/DAY3_CONFIG_v1.json"
        config_sha_path = config_path.with_suffix(config_path.suffix + ".sha256")
        if config_sha_path.read_text(encoding="utf-8").split()[0] != sha(config_path):
            raise ValueError("frozen config checksum mismatch")
        role_path = ROOT / cfg["role_manifest"]
        if sha(role_path) != cfg["role_manifest_sha256"]:
            raise ValueError("frozen role manifest checksum mismatch")
        approval = read_approval(args.approved_channels_json)
        channels = tuple(approval["channels"])
        records, split = read_plan(cfg)
        metadata_facts = verify_metadata(records, channels)
        inputs = {str(config_path.relative_to(ROOT)): sha(config_path), str(role_path.relative_to(ROOT)): sha(role_path),
                  "result/day3/preparation_r001/schema.json": sha(ROOT / "result/day3/preparation_r001/schema.json"),
                  "result/day3/preparation_r001/SPLIT_MANIFEST.csv": sha(ROOT / "result/day3/preparation_r001/SPLIT_MANIFEST.csv"),
                  str(args.approved_channels_json.resolve()): sha(args.approved_channels_json),
                  "src/expose/lee.py": sha(ROOT / "src/expose/lee.py"),
                  "src/expose/review3_data.py": sha(ROOT / "src/expose/review3_data.py"),
                  "src/expose/review2_controls.py": sha(ROOT / "src/expose/review2_controls.py"),
                  "scripts/prepare_review3_openbmi.py": sha(Path(__file__))}
        atomic_json(out / "INPUT_MANIFEST.json", inputs)
        mapping, sensitivity_extremes, original_extremes, offsets, epoch_records = [], [], [], [], []
        target_session1_whiteners = {}
        for key, record in sorted(records.items()):
            raw = record["raw"]
            raw_path = ROOT / raw["path"]
            if raw_path.stat().st_size != raw["bytes"] or sha(raw_path) != raw["sha256"]:
                raise ValueError(f"raw identity changed for {key}")
            data = load_train_only(raw_path)  # only after all role/metadata/hash guards
            channel_names = [str(x).strip() for x in data["chan"]]
            selected = validate_channel_names(channels, channel_names)
            original_selected = [channel_names.index(channel) for channel in LEGACY_CHANNELS]
            signal = data["x"]
            for montage, indexes, collection in (("original_8_primary", original_selected, original_extremes),
                                                 ("anatomy_informed_20_channel_sensitivity", selected, sensitivity_extremes)):
                extreme = selected_extreme(signal, channel_names, indexes)
                extreme.update({"montage": montage, "subject_id": key[0], "session": key[1], "role": record["role"],
                                "raw_path": raw["path"], "trial_membership": intervals_for_sample(extreme["raw_sample_zero_based"], split[key])})
                collection.append(extreme)
            offsets.append({"subject_id": key[0], "session": key[1], **audit_segmented_field(data, signal, selected, split[key])})
            del data, signal
            epochs = load_offline_epochs(raw_path, EpochConfig(channels=channels))
            cache = record["cache"]
            old_array = ROOT / cache["array_path"]
            if sha(old_array) != cache["array_sha256"]:
                raise ValueError(f"frozen original-eight cache hash changed for {key}")
            import numpy as np
            with np.load(old_array, allow_pickle=False) as old:
                anchor = legacy_anchor_check(epochs.X, old["X"], channels)
                if not np.array_equal(epochs.y, old["y"]):
                    raise ValueError(f"derived labels differ from frozen original-eight cache for {key}")
            if record["role"] == "source":
                whitener = fit_review3_whitener(epochs.X, epochs.y)
                whitener_origin = "own_source_session1_all_100"
            elif key[1] == 1:
                whitener = fit_review3_whitener(epochs.X, epochs.y)
                target_session1_whiteners[key[0]] = whitener
                whitener_origin = "own_development_session1_all_100"
            else:
                if key[0] not in target_session1_whiteners:
                    raise ValueError("development session-2 must follow and reuse its session-1 EA whitener")
                whitener = target_session1_whiteners[key[0]]
                whitener_origin = "reused_own_development_session1_all_100"
            compact = covariance_record(epochs.X, epochs.y, whitener)
            npz = out / f"subj{key[0]:02d}_sess{key[1]}_offline_20ch.npz"
            with npz.open("xb") as handle:
                np.savez_compressed(handle, **compact)
            metadata = {"source_epoch_metadata": epochs.metadata, "review3_channel_selection": "approved_candidate_20",
                        "stored_arrays": {name: list(value.shape) for name, value in compact.items()},
                        "stored_dtype": "float64", "ea_whitener": whitener.tolist(),
                        "ea_whitener_sha256": whitener_sha256(whitener), "ea_whitener_origin": whitener_origin,
                        "original_8_anchor": anchor, "original_8_cache_path": cache["array_path"],
                        "original_8_cache_sha256": cache["array_sha256"]}
            meta_path = npz.with_suffix(".json")
            atomic_json(meta_path, metadata)
            epoch_records.append({"subject_id": key[0], "session": key[1], "role": record["role"],
                                  "array_path": npz.name, "array_sha256": sha(npz), "metadata_path": meta_path.name,
                                  "metadata_sha256": sha(meta_path), "shape": list(epochs.X.shape),
                                  "trial_ids": [row["trial_id"] for row in sorted(split[key], key=lambda row: int(row["trial_index_zero_based"]))]})
            mapping.extend(split[key])
            del epochs
        atomic_json(out / "CHANNEL_APPROVAL.json", approval)
        atomic_json(out / "METADATA_CHANNEL_GUARDS.json", metadata_facts)
        original_global = max(original_extremes, key=lambda value: value["absolute_amplitude_microvolts"])
        sensitivity_global = max(sensitivity_extremes, key=lambda value: value["absolute_amplitude_microvolts"])
        atomic_json(out / "GLOBAL_SELECTED_AMPLITUDE_EXTREMES.json",
                    {"original_8_primary": {"global_selected_channel_extreme": original_global,
                                             "per_file_extremes": original_extremes},
                     "anatomy_informed_20_channel_sensitivity": {"global_selected_channel_extreme": sensitivity_global,
                                                                  "per_file_extremes": sensitivity_extremes}})
        atomic_json(out / "SEGMENTED_FIELD_OFFSET_AUDIT.json", offsets)
        with (out / "SPLIT_MAPPING.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mapping[0])); writer.writeheader(); writer.writerows(mapping)
        index = {"schema": "review3_covariance_dataset_v1", "dataset": "Lee2019_MI_offline_development_only",
                 "montage": {"name": "anatomy_informed_20_channel_sensitivity", "channels": list(channels),
                             "original_8_primary_channels": list(LEGACY_CHANNELS)},
                 "records": [{"subject": row["subject_id"], "session": row["session"], "role": row["role"],
                              "npz_path": str((out / row["array_path"]).relative_to(ROOT)), "sha256": row["array_sha256"],
                              "trial_ids": row["trial_ids"], "labels_in_npz": True} for row in epoch_records]}
        atomic_json(out / "DATASET_INDEX.json", index)
        receipt.update(status="completed", files=42, trials=4200, channels=20, epoch_shape=[100, 20, 750],
                       estimated_uncompressed_float64_bytes=42 * 100 * 20 * 750 * 8,
                       stored_covariance_elements_per_record=100 * 20 * 20 * 2,
                       records=epoch_records, output_manifest=output_manifest(out))
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        traceback.print_exc()
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - start,
                       cpu_seconds=time.process_time() - cpu,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    if receipt["status"] == "completed":
        atomic_json(out / "COMPLETED.json", {"run_id": out.name, "receipt_sha256": sha(out / "receipt.json"),
                                               "outputs": output_manifest(out)})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
