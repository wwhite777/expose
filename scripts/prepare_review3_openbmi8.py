"""Prepare compact original-eight covariance records from frozen allowed caches only."""
import argparse
import csv
from datetime import datetime, timezone
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
from expose.provenance import file_sha256
from expose.review3_data import covariance_record, fit_review3_whitener, whitener_sha256
from prepare_review3_openbmi import LEGACY_CHANNELS, atomic_json, expected_roles, output_manifest, validate_record_roles


def sha(path):
    return file_sha256(Path(path))


def read_plan(cfg):
    records = json.loads((ROOT / "result/day3/preparation_r001/schema.json").read_text(encoding="utf-8"))["records"]
    expected = expected_roles(cfg)
    by_key = validate_record_roles(records, expected)
    rows = list(csv.DictReader((ROOT / "result/day3/preparation_r001/SPLIT_MANIFEST.csv").open(encoding="utf-8", newline="")))
    grouped = {}
    for row in rows:
        key = (int(row["subject_id"]), int(row["session"]))
        if key not in expected or row["role"] != expected[key]:
            raise ValueError("split mapping contains an unapproved record")
        grouped.setdefault(key, []).append(row)
    if len(rows) != 4200 or set(grouped) != set(expected) or any(len(value) != 100 for value in grouped.values()):
        raise ValueError("expected exact 42-file, 4200-trial frozen split mapping")
    return by_key, grouped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if out.exists():
        raise FileExistsError("run directory must be new and exclusive")
    out.mkdir(parents=True)
    started, cpu = time.monotonic(), time.process_time()
    receipt = {"run_id": out.name, "status": "running", "pid": os.getpid(), "interpreter": sys.executable,
               "started_utc": datetime.now(timezone.utc).isoformat(), "raw_data_accessed": False,
               "online_phase_accessed": False, "confirmation_data_accessed": False, "model_fits": 0}
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
        metadata_guards, inputs = [], {
            "research/protocols/DAY3_CONFIG_v1.json": sha(config), str(roles.relative_to(ROOT)): sha(roles),
            "result/day3/preparation_r001/schema.json": sha(ROOT / "result/day3/preparation_r001/schema.json"),
            "result/day3/preparation_r001/SPLIT_MANIFEST.csv": sha(ROOT / "result/day3/preparation_r001/SPLIT_MANIFEST.csv"),
            "src/expose/review3_data.py": sha(ROOT / "src/expose/review3_data.py"),
            "src/expose/review2_controls.py": sha(ROOT / "src/expose/review2_controls.py"),
            "scripts/prepare_review3_openbmi.py": sha(ROOT / "scripts/prepare_review3_openbmi.py"),
            "scripts/prepare_review3_openbmi8.py": sha(Path(__file__))}
        for key, record in sorted(records.items()):
            cache = record["cache"]
            metadata = ROOT / cache["metadata_path"]
            array = ROOT / cache["array_path"]
            if sha(metadata) != cache["metadata_sha256"] or sha(array) != cache["array_sha256"]:
                raise ValueError(f"frozen original-eight cache identity changed for {key}")
            inputs[cache["array_path"]] = cache["array_sha256"]
            inputs[cache["metadata_path"]] = cache["metadata_sha256"]
            data = json.loads(metadata.read_text(encoding="utf-8"))
            if tuple(data.get("channel_names", [])) != LEGACY_CHANNELS or data.get("shape") != [100, 8, 750]:
                raise ValueError(f"original-eight metadata contract differs for {key}")
            metadata_guards.append({"subject_id": key[0], "session": key[1], "role": record["role"],
                                    "array_path": cache["array_path"], "array_sha256": cache["array_sha256"],
                                    "metadata_path": cache["metadata_path"], "metadata_sha256": cache["metadata_sha256"]})
        atomic_json(out / "INPUT_MANIFEST.json", inputs)
        target_whiteners, output_records = {}, []
        for key, record in sorted(records.items()):
            cache = record["cache"]
            with np.load(ROOT / cache["array_path"], allow_pickle=False) as values:
                X, y = values["X"], values["y"]
            if record["role"] == "source":
                W, origin = fit_review3_whitener(X, y, expected_channels=8), "own_source_session1_all_100"
            elif key[1] == 1:
                W, origin = fit_review3_whitener(X, y, expected_channels=8), "own_development_session1_all_100"
                target_whiteners[key[0]] = W
            else:
                if key[0] not in target_whiteners:
                    raise ValueError("development session-2 must reuse a prior session-1 whitener")
                W, origin = target_whiteners[key[0]], "reused_own_development_session1_all_100"
            compact = covariance_record(X, y, W, expected_channels=8)
            npz = out / f"subj{key[0]:02d}_sess{key[1]}_offline_8ch.npz"
            with npz.open("xb") as handle:
                np.savez_compressed(handle, **compact)
            meta = npz.with_suffix(".json")
            atomic_json(meta, {"source_cache": cache, "stored_arrays": {name: list(value.shape) for name, value in compact.items()},
                               "stored_dtype": "float64", "ea_whitener": W.tolist(), "ea_whitener_sha256": whitener_sha256(W),
                               "ea_whitener_origin": origin})
            output_records.append({"subject": key[0], "session": key[1], "role": record["role"],
                                   "npz_path": str(npz.relative_to(ROOT)), "sha256": sha(npz),
                                   "trial_ids": [row["trial_id"] for row in sorted(split[key], key=lambda row: int(row["trial_index_zero_based"]))],
                                   "labels_in_npz": True})
            del X, y, compact
        atomic_json(out / "METADATA_CACHE_GUARDS.json", metadata_guards)
        atomic_json(out / "DATASET_INDEX.json", {"schema": "review3_covariance_dataset_v1",
                         "dataset": "Lee2019_MI_offline_development_only", "montage": {"name": "original_8_primary", "channels": list(LEGACY_CHANNELS)},
                         "records": output_records})
        receipt.update(status="completed", files=42, trials=4200, channels=8,
                       nominal_uncompressed_npz_payload_bytes=42 * (100 * 8 * 8 * 8 * 2 + 100 * 8),
                       records=output_records, output_manifest=output_manifest(out))
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        traceback.print_exc()
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - started,
                       cpu_seconds=time.process_time() - cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    if receipt["status"] == "completed":
        atomic_json(out / "COMPLETED.json", {"run_id": out.name, "receipt_sha256": sha(out / "receipt.json"),
                                               "outputs": output_manifest(out)})
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
