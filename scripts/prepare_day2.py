"""Prepare the frozen Day2 source/development files without model evaluation."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
THREAD_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
INDEX_NOTE = "Cue uses MATLAB t-1; supplied smt begins one raw sample later at Python t. At 1000 Hz the difference is 1 ms. smt is audited, not used for preprocessing."


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    count = 0
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            count += len(chunk)
            md5.update(chunk)
            sha.update(chunk)
    return {"actual_bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def sha256(path):
    return file_hashes(path)["sha256"]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def write_csv(path, rows):
    require(bool(rows), "cannot publish an empty manifest")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_plan(plan, roles, inventory, checksums, source_subjects, development_subjects):
    sources, development = set(source_subjects), set(development_subjects)
    require(len(sources) == len(source_subjects) and len(development) == len(development_subjects), "duplicate configured participant")
    require(not sources & development, "source/development overlap")
    require(all(roles.get(sid) == "source" for sid in sources), "source role violation")
    require(all(roles.get(sid) == "development" for sid in development), "development role violation; confirmation access forbidden")
    expected = {(sid, 1) for sid in sources} | {(sid, session) for sid in development for session in (1, 2)}
    seen, items = set(), []
    for item in plan:
        relative = item["key"].split("/100542/", 1)[-1]
        match = re.fullmatch(r"session([12])/s([0-9]+)/sess([0-9]{2})_subj([0-9]{2})_EEG_MI\.mat", relative)
        require(match is not None, "invalid MI file identity")
        session, subject, session_name, subject_name = map(int, match.groups())
        identity = (subject, session)
        require((session, subject) == (session_name, subject_name), "path subject/session disagreement")
        require(identity in expected and identity not in seen, "unselected, confirmation, or duplicate file")
        seen.add(identity)
        official = inventory.get(relative)
        require(official is not None, "file absent from official inventory")
        require((int(official["subject_id"]), int(official["session_id"])) == identity, "inventory subject/session mismatch")
        require(item["url"] == official["url"], "source URL differs from frozen inventory")
        require(int(item["bytes"]) == int(official["bytes"]) > 0, "byte size differs from frozen inventory")
        require(item["etag"].strip('"') == official["etag"].strip('"'), "ETag differs from inventory")
        published = checksums.get(relative)
        require(published is not None and re.fullmatch(r"[a-f0-9]{32}", published) is not None, "missing published MD5")
        require(published == official["archive_md5"], "published MD5 differs from inventory")
        require(item.get("published_md5", published) == published, "plan published MD5 differs")
        items.append({"subject": subject, "session": session, "role": roles[subject],
                      "path": "data/raw/lee2019/" + relative, "url": item["url"],
                      "bytes": int(item["bytes"]), "etag": item["etag"], "published_md5": published})
    require(seen == expected, "plan does not contain exactly the selected files")
    return sorted(items, key=lambda item: (item["role"] != "source", item["subject"], item["session"]))


def verify_existing_raw(path, item, prior):
    require(prior is not None, "existing raw file lacks an original integrity receipt")
    require(prior["md5"] == prior["published_md5"] == item["published_md5"], "prior published MD5 mismatch")
    require(prior["actual_bytes"] == prior["bytes"] == item["bytes"], "prior byte-count mismatch")
    observed = file_hashes(path)
    require(observed["actual_bytes"] == item["bytes"], "existing raw byte count mismatch")
    require(observed["md5"] == item["published_md5"], "existing raw MD5 mismatch")
    require(observed["sha256"] == prior["sha256"], "existing raw SHA256 mismatch")
    return observed


def stream_download(path, item, *, get, check_storage):
    path = Path(path)
    partial = path.with_suffix(".mat.part")
    if path.exists() or partial.exists():
        raise FileExistsError(f"preserve existing file/partial before a successor attempt: {path}")
    check_storage(item["bytes"])
    path.parent.mkdir(parents=True, exist_ok=True)
    md5, sha, count = hashlib.md5(), hashlib.sha256(), 0
    with get(item["url"], stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        require(response.headers.get("ETag", "").strip('"') == item["etag"].strip('"'), "remote ETag mismatch")
        with partial.open("xb") as stream:
            for chunk in response.iter_content(4 * 1024**2):
                if not chunk:
                    continue
                count += len(chunk)
                require(count <= item["bytes"], "remote file exceeded frozen byte count")
                stream.write(chunk)
                md5.update(chunk)
                sha.update(chunk)
                if count % (64 * 1024**2) < len(chunk):
                    check_storage(item["bytes"] - count)
            stream.flush()
            os.fsync(stream.fileno())
    require(count == item["bytes"], "download byte-count mismatch")
    require(md5.hexdigest() == item["published_md5"], "download published MD5 mismatch")
    partial.rename(path)
    return {"actual_bytes": count, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def assert_unique_trials(records):
    counts = {}
    for key in ("trial_raw_support_sha256", "trial_epoch_sha256"):
        all_hashes = []
        for record in records:
            hashes = record["metadata"][key]
            require(len(hashes) == len(record["metadata"]["raw_labels"]), "trial hash/label length mismatch")
            require(all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) for value in hashes), "malformed trial hash")
            all_hashes.extend(hashes)
        require(len(all_hashes) == len(set(all_hashes)), f"duplicate selected trial content: {key}")
        counts[key] = {"trials": len(all_hashes), "unique_hashes": len(set(all_hashes)), "duplicate_hashes": {}}
    return counts


def storage_bytes(paths):
    total, seen = 0, set()
    for base in paths:
        for current, directories, files in os.walk(base, followlinks=False):
            for name in files:
                stat = (Path(current) / name).stat(follow_symlinks=False)
                identity = (stat.st_dev, stat.st_ino)
                if identity not in seen:
                    total += stat.st_size
                    seen.add(identity)
    return total


def enforce_resources():
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (20 * 3600, 20 * 3600 + 10))
    def timeout(signum, frame):
        raise TimeoutError(f"owned preparation process exceeded its resource cap: signal {signum}")
    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGXCPU, timeout)
    signal.alarm(4 * 3600)


def load_and_audit(path):
    import numpy as np
    import expose.lee as lee
    original = lee.loadmat
    schema = {}
    def capture(file, **kwargs):
        mat = original(file, **kwargs)
        require("EEG_MI_test" not in mat, "online variable was deserialized")
        data = mat["EEG_MI_train"]
        channels = [list(data["chan"]).index(name) for name in lee.MOTOR_CHANNELS]
        smt = data["smt"]
        matches = {"-1": 0, "0": 0}
        for trial, event in enumerate(np.asarray(data["t"]).ravel().astype(int)):
            expected = smt[:, trial, :][:, channels]
            for offset in (-1, 0):
                observed = data["x"][event + offset:event + offset + smt.shape[0], channels]
                matches[str(offset)] += int(np.array_equal(expected, observed))
        schema.update({"raw_x_shape": list(data["x"].shape), "raw_smt_shape": list(smt.shape),
                       "offline_fields": sorted(data), "mat_save_header_not_acquisition_date": mat.get("__header__", b"").decode("ascii", errors="replace"),
                       "calendar_acquisition_date": "unverified", "provided_time_interval": np.asarray(data["time_interval"]).tolist(),
                       "provided_ival_first_last": [int(data["ival"][0]), int(data["ival"][-1])],
                       "provided_smt_exact_matches_selected_channels_by_python_t_offset": matches,
                       "indexing_discrepancy": INDEX_NOTE})
        return mat
    lee.loadmat = capture
    try:
        epochs = lee.load_offline_epochs(path)
    finally:
        lee.loadmat = original
    require(schema["provided_smt_exact_matches_selected_channels_by_python_t_offset"]["0"] == 100, "provided smt offset changed; preserve and review schema")
    return epochs.X, epochs.y, epochs.metadata, schema


def validate_epochs(X, y, metadata, preprocessing):
    import numpy as np
    require(X.shape == (100, 8, 750) and X.dtype == np.dtype("float64"), "unexpected epoch shape/dtype")
    require(np.all(np.isfinite(X)) and np.array_equal(np.bincount(y), [50, 50]), "nonfinite or unbalanced epochs")
    require(metadata["raw_fs"] == 1000 and metadata["fs"] == 250, "sampling-rate drift")
    require(metadata["mat_variable"] == "EEG_MI_train" and metadata["loaded_variables"] == ["EEG_MI_train"], "offline scope drift")
    require(metadata["class_names"] == ["right_hand", "left_hand"] and metadata["units"] == "V", "class/unit drift")
    require(np.array_equal(y, np.asarray(metadata["raw_labels"]) - 1), "raw and processed labels disagree")
    require(metadata["channel_names"] == preprocessing["channels"], "channel configuration drift")
    require(all(metadata["config"][key] == preprocessing[key] for key in ("l_freq", "h_freq", "tmin", "tmax", "target_fs", "filter_order")), "epoch configuration drift")
    observed = [hashlib.sha256(epoch.tobytes(order="C")).hexdigest() for epoch in X]
    require(observed == metadata["trial_epoch_sha256"], "cached epoch content/hash disagreement")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="preparation_r001")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"preparation_r[0-9]{3}", args.run_id), "invalid run identifier")
    out = ROOT / "result/day2" / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    enforce_resources()
    start, cpu_start = time.monotonic(), time.process_time()
    receipt = {"status": "running", "run_id": args.run_id, "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
               "interpreter": sys.executable, "scope": "34 frozen source/development offline files; no model fits or scores", "records": [], "raw_files": [], "active_file": None}
    write_json(out / "receipt.json", receipt)
    try:
        sys.path.insert(0, str(ROOT / "src"))
        import numpy as np
        import requests
        from threadpoolctl import threadpool_info, threadpool_limits
        from expose.provenance import verify_cache
        config_path = ROOT / "research/protocols/DAY2_CONFIG_v1.json"
        plan_path = ROOT / "research/day2/DOWNLOAD_PLAN.json"
        roles_path = ROOT / "research/PARTICIPANT_ROLES_v1.csv"
        require(sha256(config_path) == config_path.with_suffix(".json.sha256").read_text().split()[0], "Day2 config hash mismatch")
        require(sha256(roles_path) == roles_path.with_suffix(".csv.sha256").read_text().split()[0], "role hash mismatch")
        config = json.loads(config_path.read_text())
        require(config["source_sessions"] == [1] and config["history_session"] == 1 and config["evaluation_session"] == 2, "session configuration drift")
        require(config["expected_preparation_files"] == 34 and config["expected_preparation_trials"] == 3400, "preparation count drift")
        expected_resources = {"numerical_threads": 1, "process_memory_gib": 16, "day2_cpu_core_hours_cap": 20,
                              "project_plus_environment_storage_gb_cap": 30, "job_wall_seconds_cap": 14400, "gpu_hours": 0}
        require(all(config["resource"][key] == value for key, value in expected_resources.items()), "frozen resource configuration differs from enforced caps")
        require(config["role_manifest_sha256"] == sha256(roles_path), "configured role identity mismatch")
        require(config["loader_file"] == "src/expose/lee.py" and config["loader_sha256"] == sha256(ROOT / config["loader_file"]), "configured loader identity mismatch")
        roles = {int(row["subject_id"]): row["role"] for row in csv.DictReader(roles_path.open())}
        sources = config["source_subjects"]
        development = config["development_subjects"]
        require(sources == sorted(sid for sid, role in roles.items() if role == "source"), "Day2 must use all 18 frozen sources")
        require(development == sorted(sid for sid, role in roles.items() if role == "development")[:8], "Day2 must use the smallest 8 development IDs")
        inventory_path = ROOT / "research/day1/sources/lee2019_mi_remote_inventory.csv"
        inventory = {row["relative_path"]: row for row in csv.DictReader(inventory_path.open())}
        checksum_path = ROOT / "research/day1/sources/100542.md5"
        checksums = {line.split()[1].removeprefix("./"): line.split()[0] for line in checksum_path.read_text().splitlines() if len(line.split()) == 2}
        items = validate_plan(json.loads(plan_path.read_text()), roles, inventory, checksums, sources, development)
        require(len(items) == 34, "unexpected file count")
        old_download_path = ROOT / "result/day1/download_receipt.json"
        old_schema_path = ROOT / "result/day1/schema_validation.json"
        old_download = json.loads(old_download_path.read_text())
        old_schema = json.loads(old_schema_path.read_text())
        require(old_download["status"] == "completed" and old_download["exit_code"] == 0, "Day1 download receipt is not completed")
        old_manifest = json.loads((ROOT / "result/day1/smoke_r001/INPUT_MANIFEST.json").read_text())
        require(sha256(ROOT / "src/expose/lee.py") == old_manifest["src/expose/lee.py"], "frozen Day1 loader changed")
        require(sha256(old_download_path) == old_manifest["result/day1/download_receipt.json"], "original download receipt changed")
        require(sha256(old_schema_path) == old_manifest["result/day1/schema_validation.json"], "original schema receipt changed")
        inputs = [config_path, config_path.with_suffix(".json.sha256"), plan_path, roles_path, inventory_path, checksum_path,
                  old_download_path, old_schema_path, ROOT / "result/day1/smoke_r001/INPUT_MANIFEST.json", ROOT / "src/expose/lee.py", ROOT / "src/expose/provenance.py", Path(__file__),
                  ROOT / "research/environment/requirements.lock.txt"]
        manifest = {str(path.relative_to(ROOT)): sha256(path) for path in inputs}
        write_json(out / "INPUT_MANIFEST.json", manifest)
        receipt["input_manifest_sha256"] = sha256(out / "INPUT_MANIFEST.json")
        original_raw = {row["path"]: (row, str(old_download_path.relative_to(ROOT))) for row in old_download["files"]}
        previous_caches = {}
        for prior_path in sorted((ROOT / "result/day2").glob("preparation_r*/receipt.json")):
            if prior_path.parent == out:
                continue
            prior_receipt = json.loads(prior_path.read_text())
            for raw in prior_receipt.get("raw_files", []):
                original_raw.setdefault(raw["path"], (raw, str(prior_path.relative_to(ROOT))))
            for row in prior_receipt.get("records", []):
                original_raw.setdefault(row["raw"]["path"], (row["raw"], str(prior_path.relative_to(ROOT))))
                previous_caches.setdefault((row["subject"], row["session"]), (row, str(prior_path.relative_to(ROOT))))
        environment = Path(sys.executable).parent.parent
        def check_storage(reserve_bytes=0):
            used = storage_bytes([ROOT, environment])
            require(used + reserve_bytes <= 30_000_000_000, "30 GB project-plus-environment storage cap would be exceeded")
            require(shutil.disk_usage(ROOT).free >= reserve_bytes + 256 * 1024**2, "insufficient filesystem reserve")
            return used
        receipt["initial_storage_bytes_including_environment"] = check_storage(sum(item["bytes"] for item in items if not (ROOT / item["path"]).exists()) + len(items) * 8 * 1024**2)
        data_rows, split_rows = [], []
        cache_dir = ROOT / "data/derived/day2"
        cache_dir.mkdir(parents=True, exist_ok=True)
        with threadpool_limits(limits=1), requests.Session() as client:
            receipt["threadpools"] = threadpool_info()
            require(all(pool["num_threads"] == 1 for pool in receipt["threadpools"]), "numerical thread cap violated")
            for number, item in enumerate(items, 1):
                receipt["active_file"] = item
                write_json(out / "receipt.json", receipt)
                began, cpu_began = time.monotonic(), time.process_time()
                path = ROOT / item["path"]
                prior, prior_path = original_raw.get(item["path"], (None, None))
                reused_raw = path.exists()
                observed = verify_existing_raw(path, item, prior) if reused_raw else stream_download(path, item, get=client.get, check_storage=check_storage)
                raw = {**item, **observed, "reused": reused_raw, "original_receipt": prior_path if reused_raw else None}
                receipt["raw_files"].append(raw)
                write_json(out / "receipt.json", receipt)
                old_records = [row for row in old_schema["records"] if (row["subject"], row["session"]) == (item["subject"], item["session"])]
                if old_records:
                    old = old_records[0]
                    cache = dict(old["cache"])
                    with np.load(ROOT / cache["array_path"], allow_pickle=False) as loaded:
                        X, y = loaded["X"], loaded["y"]
                    metadata = json.loads((ROOT / cache["metadata_path"]).read_text())
                    verify_cache(ROOT, item["subject"], item["session"], item["role"], X, y, metadata, old_schema, old_download)
                    schema = old["schema"]
                    cache.update(reused=True, original_receipt=str(old_schema_path.relative_to(ROOT)))
                elif (item["subject"], item["session"]) in previous_caches:
                    previous, previous_path = previous_caches[item["subject"], item["session"]]
                    require(previous["role"] == item["role"] and previous["status"] == "pass_with_documented_smt_one_sample_offset", "prior cache role/status differs")
                    cache = dict(previous["cache"])
                    require(cache["array_path"] == f"data/derived/day2/subj{item['subject']:02d}_sess{item['session']:02d}_offline.npz", "prior cache identity differs")
                    require(cache["metadata_path"] == str(Path(cache["array_path"]).with_suffix(".json")), "prior metadata identity differs")
                    require(sha256(ROOT / cache["array_path"]) == cache["array_sha256"] and sha256(ROOT / cache["metadata_path"]) == cache["metadata_sha256"], "prior cache file hash differs")
                    with np.load(ROOT / cache["array_path"], allow_pickle=False) as loaded:
                        X, y = loaded["X"], loaded["y"]
                    metadata = json.loads((ROOT / cache["metadata_path"]).read_text())
                    require(metadata == previous["metadata"] and Path(metadata["path"]).resolve() == path.resolve(), "prior metadata lineage differs")
                    require(previous["raw"]["sha256"] == raw["sha256"], "prior raw/cache binding differs")
                    schema = previous["schema"]
                    cache.update(reused=True, original_receipt=previous_path)
                else:
                    check_storage(8 * 1024**2)
                    X, y, metadata, schema = load_and_audit(path)
                    array_path = cache_dir / f"subj{item['subject']:02d}_sess{item['session']:02d}_offline.npz"
                    metadata_path = array_path.with_suffix(".json")
                    require(not array_path.exists() and not metadata_path.exists(), "preserve existing Day2 cache before successor attempt")
                    temporary = array_path.with_suffix(".npz.part")
                    with temporary.open("xb") as stream:
                        np.savez_compressed(stream, X=X, y=y)
                    temporary.rename(array_path)
                    write_json(metadata_path, metadata)
                    cache = {"array_path": str(array_path.relative_to(ROOT)), "array_sha256": sha256(array_path),
                             "metadata_path": str(metadata_path.relative_to(ROOT)), "metadata_sha256": sha256(metadata_path),
                             "reused": False, "original_receipt": None}
                validate_epochs(X, y, metadata, config["preprocessing"])
                record = {"subject": item["subject"], "session": item["session"], "role": item["role"],
                          "status": "pass_with_documented_smt_one_sample_offset", "raw": raw, "cache": cache,
                          "metadata": metadata, "schema": schema,
                          "wall_seconds": time.monotonic() - began, "cpu_seconds": time.process_time() - cpu_began}
                receipt["records"].append(record)
                assert_unique_trials(receipt["records"])
                write_json(out / "receipt.json", receipt)
                data_rows.append({"dataset": "Lee2019_MI", "subject_id": item["subject"], "role": item["role"], "session": item["session"],
                                  "raw_path": item["path"], "source_url": item["url"], "bytes": observed["actual_bytes"],
                                  "published_md5": item["published_md5"], "local_sha256": observed["sha256"], "license": "CC0-1.0",
                                  "used_variable": "EEG_MI_train", "trials": 100, "right_trials": 50, "left_trials": 50,
                                  "raw_fs": 1000, "output_fs": 250, "epoch_shape": "100x8x750",
                                  "derived_path": cache["array_path"], "derived_sha256": cache["array_sha256"],
                                  "metadata_path": cache["metadata_path"], "metadata_sha256": cache["metadata_sha256"],
                                  "reused_raw": reused_raw, "reused_cache": cache["reused"], "calendar_acquisition_date": "unverified"})
                usage = "source_training" if item["role"] == "source" else ("development_history" if item["session"] == 1 else "development_evaluation")
                for i in range(100):
                    split_rows.append({"trial_id": f"lee2019:s{item['subject']:03d}:session{item['session']}:offline:t{i:03d}",
                                       "subject_id": item["subject"], "role": item["role"], "session": item["session"], "mat_variable": "EEG_MI_train",
                                       "trial_index_zero_based": i, "label": int(y[i]), "class_name": metadata["class_names"][int(y[i])],
                                       "event_sample_matlab": metadata["event_samples_matlab"][i], "event_sample_zero_based": metadata["event_samples_zero_based"][i],
                                       "filter_start_sample": metadata["filter_input_start_samples_zero_based"][i], "filter_stop_sample_exclusive": metadata["filter_input_stop_samples_zero_based_exclusive"][i],
                                       "epoch_start_sample": metadata["epoch_start_samples_zero_based"][i], "epoch_stop_sample_exclusive": metadata["epoch_stop_samples_zero_based_exclusive"][i],
                                       "raw_support_sha256": metadata["trial_raw_support_sha256"][i], "epoch_sha256": metadata["trial_epoch_sha256"][i], "usage": usage})
                progress = {"completed_files": number, "required_files": len(items), "subject": item["subject"], "session": item["session"],
                                  "reused_raw": reused_raw, "reused_cache": cache["reused"], "wall_seconds": round(record["wall_seconds"], 3),
                                  "cpu_seconds": round(record["cpu_seconds"], 3)}
                with (out / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(progress) + "\n")
                print(json.dumps(progress), flush=True)
                del X, y
        require(len(data_rows) == 34 and len(split_rows) == 3400, "required files/trials missing")
        require(len({row["trial_id"] for row in split_rows}) == 3400, "duplicate trial IDs")
        receipt["duplicate_checks"] = assert_unique_trials(receipt["records"])
        require(all(sha256(ROOT / path) == expected for path, expected in manifest.items()), "frozen preparation input changed during execution")
        for row in receipt["records"]:
            require(sha256(ROOT / row["cache"]["array_path"]) == row["cache"]["array_sha256"], "cache changed during preparation")
            require(sha256(ROOT / row["cache"]["metadata_path"]) == row["cache"]["metadata_sha256"], "sidecar changed during preparation")
        schema = {"created_utc": datetime.now(timezone.utc).isoformat(), "records": receipt["records"], "duplicate_checks": receipt["duplicate_checks"],
                  "config": config["preprocessing"], "confirmation_data_accessed": False, "online_data_deserialized": False, "model_fits": 0, "indexing_discrepancy": INDEX_NOTE}
        write_json(out / "schema.json", schema)
        for filename, rows in (("DATA_MANIFEST.csv", data_rows), ("SPLIT_MANIFEST.csv", split_rows)):
            write_csv(out / filename, rows)
            canonical = ROOT / "research/day2" / filename
            require(not canonical.exists(), "preserve existing canonical Day2 manifest")
            shutil.copyfile(out / filename, canonical)
        receipt["final_storage_bytes_including_environment"] = check_storage()
        receipt.update(status="completed", exit_code=0, active_file=None, checks={"files": 34, "trials": 3400, "balanced_trials_per_file": True,
                       "unique_raw_support_and_epochs": True, "frozen_inputs_unchanged": True, "confirmation_accessed": False, "model_fits": 0})
    except BaseException as error:
        receipt.update(status="failed", exit_code=1, error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - start,
                       cpu_seconds=time.process_time() - cpu_start, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(out / "receipt.json", receipt)
        signal.alarm(0)
    if receipt["status"] == "completed":
        outputs = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "COMPLETED.json"}
        for filename in ("DATA_MANIFEST.csv", "SPLIT_MANIFEST.csv"):
            path = ROOT / "research/day2" / filename
            outputs[str(path.relative_to(ROOT))] = sha256(path)
        write_json(out / "COMPLETED.json", {"run_id": args.run_id, "status": "completed", "exit_code": 0,
                   "receipt_sha256": sha256(out / "receipt.json"), "outputs": outputs, "input_manifest_sha256": receipt["input_manifest_sha256"]})
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
