#!/usr/bin/env python3
"""Guarded source-freeze and future one-shot OpenBMI confirmation runner."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from expose.lee_memory import download_verified, load_offline_epochs_from_bytes
from expose.review2_controls import fit_representation, oas_covariances, train_classifier


INVENTORY = ROOT / "research/day1/sources/lee2019_mi_remote_inventory.csv"
ROLES = ROOT / "research/PARTICIPANT_ROLES_v1.csv"
SOURCE_INDEX = ROOT / "research/review3_20260923/OPENBMI8_GRID_INDEX.json"
SCORE_FIELDS = ["operation_id", "participant_id", "draw", "condition", "setting", "source_n",
                "added_n", "h_total", "C", "balanced_accuracy", "evaluation_n"]
PREDICTION_FIELDS = ["operation_id", "participant_id", "draw", "condition", "trial_id",
                     "y_true", "y_pred", "p0", "p1"]
LIMITS = {"cpu_seconds": 3600, "wall_seconds": 14400, "memory_gib": 16,
          "minimum_free_output_bytes": 128 * 1024 * 1024}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seeded_rng(seed, *tokens):
    suffix = int(hashlib.sha256(json.dumps(tokens, sort_keys=True, default=str).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(np.random.SeedSequence([seed, suffix & 0xffffffff, suffix >> 32]))


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def file_identity(path):
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": sha256(path),
            "bytes": Path(path).stat().st_size}


def code_files():
    return {
        "runner": ROOT / "scripts/run_review4_confirmation.py",
        "memory_loader": ROOT / "src/expose/lee_memory.py",
        "numeric_loader": ROOT / "src/expose/lee.py",
        "package_init": ROOT / "src/expose/__init__.py",
        "baseline_helpers": ROOT / "src/expose/baselines.py",
        "model_primitives": ROOT / "src/expose/review2_controls.py",
        "analyzer": ROOT / "scripts/analyze_review4_confirmation.py",
        "runner_tests": ROOT / "tests/test_review4_confirmation_runner.py",
        "analyzer_tests": ROOT / "tests/test_review4_confirmation.py",
    }


def dependency_versions():
    import numpy
    import pyriemann
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "pyriemann": pyriemann.__version__}


def validate_roles(protected_ids):
    with ROLES.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if reader.fieldnames != ["subject_id", "role", "day1_selected", "selection_rule"]:
        raise ValueError("participant-role schema differs")
    roles = {}
    for row in rows:
        subject = int(row["subject_id"])
        if subject in roles or subject not in range(1, 55):
            raise ValueError("participant-role identity differs")
        roles[subject] = row["role"]
    if (set(roles) != set(range(1, 55))
            or Counter(roles.values()) != {"source": 18, "development": 12, "confirmation": 24}
            or {subject for subject, role in roles.items() if role == "confirmation"}
            != set(protected_ids)):
        raise ValueError("participant-role allocation differs")
    return roles


def validate_packet(packet_dir):
    receipt_path = packet_dir / "PACKET_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed_source_only_packet" or receipt.get("protected_accessed") is not False:
        raise ValueError("confirmation packet receipt differs")
    for name, identity in receipt["outputs"].items():
        path = packet_dir / name
        if sha256(path) != identity["sha256"] or path.stat().st_size != identity["bytes"]:
            raise ValueError("confirmation packet file differs: " + name)
    machine = json.loads((packet_dir / "MACHINE_CONFIG.json").read_text(encoding="utf-8"))
    if (machine.get("adoption_required") is not True
            or machine.get("protected_access_authorized") is not False
            or len(machine.get("protected_participant_ids", [])) != 24):
        raise ValueError("confirmation packet authorization boundary differs")
    operations = load_gzip_json(packet_dir / "operations.json.gz")
    memberships = load_gzip_json(packet_dir / "membership_sets.json.gz")
    if (len(operations) != 1440 or len(memberships) != 971
            or any(fingerprint(values) != key for key, values in memberships.items())):
        raise ValueError("confirmation packet inventory differs")
    seen = set()
    coordinates = set()
    allowed_conditions = {"base", "own", "pooled", "single_0", "single_1", "single_2"}
    for op in operations:
        condition = op.get("condition")
        coordinate = (op.get("participant_id"), op.get("draw"), condition)
        if (op.get("operation_id") in seen or coordinate in coordinates
                or op.get("participant_id") not in machine["protected_participant_ids"]
                or op.get("draw") not in range(10) or condition not in allowed_conditions
                or op.get("setting") != "tuned_C_source_frozen" or op.get("source_n") != 100
                or op.get("source_membership") not in memberships
                or (condition != "own" and op.get("added_membership") not in memberships)
                or (condition == "own" and op.get("added_membership") is not None)
                or not isinstance(op.get("C"), (int, float)) or not np.isfinite(op["C"])
                or float(op["C"]) <= 0):
            raise ValueError("confirmation packet operation schema differs")
        expected_added = 0 if condition == "base" else 60
        expected_h = 0 if condition == "base" else 60
        source_ids = memberships[op["source_membership"]]
        added_ids = [] if condition == "own" else memberships[op["added_membership"]]
        expected_added_membership = 60 if condition in {"pooled", "single_0", "single_1", "single_2"} else 0
        if (op.get("added_n") != expected_added or op.get("h_total") != expected_h
                or len(source_ids) != 100 or len(set(source_ids)) != 100
                or len(added_ids) != expected_added_membership
                or len(set(added_ids)) != expected_added_membership
                or set(source_ids) & set(added_ids)):
            raise ValueError("confirmation packet operation dose differs")
        seen.add(op["operation_id"]); coordinates.add(coordinate)
    expected_coordinates = {(participant, draw, condition)
                            for participant in machine["protected_participant_ids"]
                            for draw in range(10) for condition in allowed_conditions}
    if coordinates != expected_coordinates:
        raise ValueError("confirmation packet operation coverage differs")
    return receipt_path, receipt, machine, operations, memberships


def validate_public_inventory(protected_ids):
    with INVENTORY.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); rows = list(reader)
    expected_fields = ["subject_id", "session_id", "relative_path", "url", "bytes",
                       "archive_md5", "etag", "last_modified"]
    if reader.fieldnames != expected_fields or len(rows) != 108:
        raise ValueError("published remote inventory schema/count differs")
    result = {}
    for row in rows:
        subject, session = int(row["subject_id"]), int(row["session_id"])
        key = (subject, session)
        if key in result or not 1 <= subject <= 54 or session not in (1, 2):
            raise ValueError("published inventory coordinate differs")
        if (not row["url"].startswith("https://s3.ap-northeast-1.wasabisys.com/")
                or int(row["bytes"]) <= 0 or len(row["archive_md5"]) != 32):
            raise ValueError("published inventory identity differs")
        result[key] = {**row, "bytes": int(row["bytes"])}
    if set(result) != {(subject, session) for subject in range(1, 55) for session in (1, 2)}:
        raise ValueError("published inventory coverage differs")
    return {key: result[key] for key in result if key[0] in protected_ids}


def freeze(packet_dir, output):
    if output.exists():
        raise FileExistsError("code-freeze output must be new")
    receipt_path, _, machine, _, _ = validate_packet(packet_dir)
    public = validate_public_inventory(machine["protected_participant_ids"])
    validate_roles(machine["protected_participant_ids"])
    payload = {
        "schema": "review4-confirmation-code-freeze-v1",
        "status": "complete_code_freeze_requires_author_adoption",
        "adoption_required": True, "protected_access_authorized": False,
        "packet_receipt": file_identity(receipt_path),
        "source_inventory": file_identity(INVENTORY), "participant_roles": file_identity(ROLES),
        "source_index": file_identity(SOURCE_INDEX),
        "code": {name: file_identity(path) for name, path in code_files().items()},
        "dependencies": dependency_versions(),
        "resource_limits": LIMITS, "public_objects_bound": len(public),
        "execution_contract": {
            "requires_external_authorization_overlay": True,
            "raw_public_MAT_storage": "memory_only_never_persisted",
            "aggregate_analysis": "not_called_by_runner",
            "stdout": "status_and_counts_only",
            "outcome_suppression": "procedural: partial plaintext outputs must remain unread until staged lock",
        },
    }
    atomic_json(output, payload)
    return payload


def validate_authorization(path, freeze_path, packet_receipt):
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "author_adopted", "protected_access_authorized",
                "code_freeze_sha256", "packet_receipt_sha256", "adoption_record",
                "adoption_record_sha256", "decision"}
    record_name = value.get("adoption_record")
    record = (ROOT / record_name).resolve() if isinstance(record_name, str) else None
    if (set(value) != required or value["schema"] != "review4-confirmation-authorization-v1"
            or value["author_adopted"] is not True or value["protected_access_authorized"] is not True
            or value["code_freeze_sha256"] != sha256(freeze_path)
            or value["packet_receipt_sha256"] != sha256(packet_receipt)
            or value.get("decision") != "adopt_and_authorize_protected_confirmation_execution"
            or record is None or ROOT not in record.parents or not record.is_file()
            or value.get("adoption_record_sha256") != sha256(record)):
        raise PermissionError("explicit packet/code-freeze-bound author authorization is absent")
    return value


class SourceStore:
    def __init__(self, memberships, roles):
        index = json.loads(SOURCE_INDEX.read_text(encoding="utf-8"))
        if index.get("dataset") != "openbmi" or index.get("montage") != "8ch":
            raise ValueError("source index dataset/montage differs")
        self.trials = {}
        source_subjects = set()
        record_coordinates = set()
        for record in index["records"]:
            coordinate = (int(record["subject"]), int(record["session"]))
            if (coordinate in record_coordinates or record["role"] not in {"source", "development"}
                    or roles.get(coordinate[0]) != record["role"]):
                raise ValueError("source index record role/identity differs")
            record_coordinates.add(coordinate)
            if record["role"] != "source":
                continue
            if record["session"] != 1 or record["subject"] in source_subjects:
                raise ValueError("source index role/session differs")
            source_subjects.add(record["subject"])
            if roles.get(int(record["subject"])) != "source":
                raise ValueError("source index conflicts with frozen participant roles")
            path = ROOT / record["npz_path"]
            if sha256(path) != record["sha256"]:
                raise ValueError("source covariance archive differs")
            with np.load(path, allow_pickle=False) as archive:
                cov, y = archive["cov"].copy(), archive["y"].astype(int).copy()
            if cov.shape != (100, 8, 8) or Counter(y.tolist()) != {0: 50, 1: 50}:
                raise ValueError("source covariance schema differs")
            for trial, matrix, label in zip(record["trial_ids"], cov, y):
                if trial in self.trials:
                    raise ValueError("duplicate source trial")
                self.trials[trial] = (matrix, int(label), int(record["subject"]))
        if len(source_subjects) != 18 or len(self.trials) != 1800:
            raise ValueError("source index inventory differs")
        if source_subjects != {subject for subject, role in roles.items() if role == "source"}:
            raise ValueError("source index omits or adds a source-role participant")
        expected_coordinates = ({(subject, 1) for subject, role in roles.items() if role == "source"}
                                | {(subject, session) for subject, role in roles.items()
                                   if role == "development" for session in (1, 2)})
        if record_coordinates != expected_coordinates:
            raise ValueError("source index record coverage differs")
        used = {trial for values in memberships.values() for trial in values}
        if not used <= set(self.trials):
            raise ValueError("packet source-side membership includes a non-source trial")

    def cov(self, trials):
        return np.stack([self.trials[trial][0] for trial in trials])

    def labels(self, trials):
        return np.asarray([self.trials[trial][1] for trial in trials], dtype=int)


def balanced_accuracy(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or Counter(truth.tolist()) != {0: 50, 1: 50}:
        raise ValueError("evaluation labels differ from fixed 50/class schema")
    return float(np.mean([np.mean(predicted[truth == label] == label) for label in (0, 1)]))


def prepare_public_session(entry, subject, session, access_handle):
    before = time.monotonic()
    access_handle.write(json.dumps({"event": "public_object_access_attempt",
                                    "subject": subject, "session": session,
                                    "url": entry["url"], "expected_bytes": entry["bytes"],
                                    "expected_md5": entry["archive_md5"]}, sort_keys=True) + "\n")
    access_handle.flush()
    try:
        raw, identity = download_verified(entry["url"], entry["bytes"], entry["archive_md5"])
    except OSError as error:
        access_handle.write(json.dumps({"event": "public_object_access_failed",
                                        "subject": subject, "session": session,
                                        "url": entry["url"], "error_type": type(error).__name__,
                                        "error": str(error),
                                        "wall_seconds": time.monotonic() - before}, sort_keys=True) + "\n")
        access_handle.flush()
        raise
    access_handle.write(json.dumps({"event": "public_object_verified_in_memory",
                                    "subject": subject, "session": session,
                                    "url": entry["url"], **identity,
                                    "wall_seconds": time.monotonic() - before}, sort_keys=True) + "\n")
    try:
        loaded = load_offline_epochs_from_bytes(raw, entry["url"], subject, session)
    finally:
        del raw
    if loaded.X.shape != (100, 8, 750) or Counter(loaded.y.astype(int).tolist()) != {0: 50, 1: 50}:
        raise ValueError("protected offline epoch schema differs")
    cov = oas_covariances(loaded.X)
    y = loaded.y.astype(int).copy()
    metadata = {"subject": subject, "session": session, "public_uri": entry["url"],
                "raw_identity": identity, "epoch_shape": list(loaded.X.shape),
                "class_counts": loaded.metadata["class_counts"],
                "trial_epoch_sha256": loaded.metadata["trial_epoch_sha256"]}
    del loaded
    return cov, y, metadata


def history_indices(labels, draw, participant):
    selected = []
    for label in (0, 1):
        pool = np.flatnonzero(labels == label)
        permutation = seeded_rng(2026092303, "history", "openbmi_main", draw,
                                 participant, label).permutation(pool).tolist()
        selected.extend(permutation[:30])
    if len(selected) != 60 or len(set(selected)) != 60:
        raise ValueError("protected history selection differs")
    return selected


def materialize_protected_memberships(participant, s1_y):
    histories = {}
    for draw in range(10):
        indices = history_indices(s1_y, draw, participant)
        histories[str(draw)] = {
            "indices_zero_based": indices,
            "trial_ids": [f"lee2019:s{participant:03d}:session1:offline:t{index:03d}"
                          for index in indices],
        }
    evaluation = [f"lee2019:s{participant:03d}:session2:offline:t{index:03d}"
                  for index in range(100)]
    value = {"participant_id": participant, "histories": histories,
             "evaluation_trial_ids": evaluation}
    value["membership_sha256"] = fingerprint(value)
    return value


def score_person(participant, operations, memberships, source, s1_cov, s1_y, s2_cov, s2_y,
                 representations, protected_memberships):
    score_rows, prediction_rows = [], []
    for op in sorted(operations, key=lambda item: (item["draw"], item["condition"])):
        base_ids = memberships[op["source_membership"]]
        key = op["source_membership"]
        if key not in representations:
            base_cov = source.cov(base_ids); base_y = source.labels(base_ids)
            representations[key] = (fit_representation(base_cov, 100, "source_frozen"),
                                    base_cov, base_y)
        representation, base_cov, base_y = representations[key]
        condition = op["condition"]
        if condition == "base":
            added_cov = np.empty((0, 8, 8)); added_y = np.empty(0, dtype=int)
        elif condition == "own":
            indices = protected_memberships["histories"][str(op["draw"])]["indices_zero_based"]
            added_cov, added_y = s1_cov[indices], s1_y[indices]
        else:
            added_ids = memberships[op["added_membership"]]
            added_cov, added_y = source.cov(added_ids), source.labels(added_ids)
        if len(added_y) != op["added_n"] or (len(added_y) and Counter(added_y.tolist()) != {0: 30, 1: 30}):
            raise ValueError("added membership count/classes differ")
        train_cov = base_cov if not len(added_y) else np.concatenate([base_cov, added_cov])
        train_y = base_y if not len(added_y) else np.concatenate([base_y, added_y])
        classifier = train_classifier(representation.transform(train_cov), train_y, 100,
                                      float(op["C"]), "pooled")
        z_eval = representation.transform(s2_cov)
        predicted, probabilities = classifier.predict(z_eval), classifier.predict_proba(z_eval)
        if (predicted.shape != (100,) or probabilities.shape != (100, 2)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)):
            raise ValueError("prediction schema differs")
        score_rows.append({"operation_id": op["operation_id"], "participant_id": participant,
                           "draw": op["draw"], "condition": condition, "setting": op["setting"],
                           "source_n": 100, "added_n": op["added_n"], "h_total": op["h_total"],
                           "C": op["C"], "balanced_accuracy": balanced_accuracy(s2_y, predicted),
                           "evaluation_n": 100})
        for index, (truth, pred, probability) in enumerate(zip(s2_y, predicted, probabilities)):
            prediction_rows.append({"operation_id": op["operation_id"], "participant_id": participant,
                                    "draw": op["draw"], "condition": condition,
                                    "trial_id": f"lee2019:s{participant:03d}:session2:offline:t{index:03d}",
                                    "y_true": int(truth), "y_pred": int(pred),
                                    "p0": float(probability[0]), "p1": float(probability[1])})
    if len(score_rows) != 60 or len(prediction_rows) != 6000:
        raise ValueError("per-person score inventory differs")
    return score_rows, prediction_rows


def run(packet_dir, freeze_path, authorization_path, output):
    if output.exists():
        raise FileExistsError("runner output must be new")
    receipt_path, packet_receipt, machine, operations, memberships = validate_packet(packet_dir)
    freeze_value = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (freeze_value.get("schema") != "review4-confirmation-code-freeze-v1"
            or freeze_value.get("status") != "complete_code_freeze_requires_author_adoption"
            or freeze_value.get("adoption_required") is not True
            or freeze_value.get("protected_access_authorized") is not False
            or freeze_value.get("packet_receipt") != file_identity(receipt_path)
            or freeze_value.get("resource_limits") != LIMITS
            or freeze_value.get("public_objects_bound") != 48):
        raise ValueError("code freeze differs")
    frozen_inputs = {"source_inventory": INVENTORY, "participant_roles": ROLES,
                     "source_index": SOURCE_INDEX}
    for name, path in frozen_inputs.items():
        if file_identity(path) != freeze_value.get(name):
            raise ValueError("frozen input changed after code freeze: " + name)
    if freeze_value.get("dependencies") != dependency_versions():
        raise ValueError("dependency versions changed after code freeze")
    for name, item in freeze_value["code"].items():
        if name not in code_files() or file_identity(code_files()[name]) != item:
            raise ValueError("code changed after freeze: " + name)
    if set(freeze_value["code"]) != set(code_files()):
        raise ValueError("code-freeze file inventory differs")
    validate_authorization(authorization_path, freeze_path, receipt_path)
    public = validate_public_inventory(machine["protected_participant_ids"])
    roles = validate_roles(machine["protected_participant_ids"])
    if shutil_disk_free(output.parent) < LIMITS["minimum_free_output_bytes"]:
        raise OSError("insufficient reserved output storage")

    output.mkdir(parents=True); cov_dir = output / "protected_covariances"; cov_dir.mkdir()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "packet_receipt_sha256": sha256(receipt_path), "code_freeze_sha256": sha256(freeze_path),
               "runner_sha256": sha256(__file__),
               "authorization_sha256": sha256(authorization_path), "limits": LIMITS,
               "aggregate_analysis_called": False, "raw_MAT_persisted": False}
    atomic_json(output / "receipt.json", receipt)
    score_partial = output / "scores.csv.partial"
    prediction_partial = output / "trial_predictions.csv.gz.partial"
    access = score_handle = raw_prediction = prediction_handle = None
    old_xcpu = old_alarm = None
    try:
        old_xcpu = signal.signal(
            signal.SIGXCPU, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU limit")))
        old_alarm = signal.signal(
            signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall limit")))
        resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_seconds"], LIMITS["cpu_seconds"] + 10))
        resource.setrlimit(resource.RLIMIT_AS, (LIMITS["memory_gib"] * 1024**3,) * 2)
        signal.alarm(LIMITS["wall_seconds"])
        access = (output / "access_log.jsonl").open("x", encoding="utf-8")
        score_handle = score_partial.open("x", encoding="utf-8", newline="")
        score_writer = csv.DictWriter(score_handle, fieldnames=SCORE_FIELDS); score_writer.writeheader()
        raw_prediction = prediction_partial.open("xb")
        compressed = gzip.GzipFile(fileobj=raw_prediction, mode="wb", mtime=0)
        prediction_handle = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        prediction_writer = csv.DictWriter(
            prediction_handle, fieldnames=PREDICTION_FIELDS); prediction_writer.writeheader()
        statuses, metadata_guards, protected_memberships, representations = [], {}, {}, {}
        scores_written = predictions_written = 0
        source = SourceStore(memberships, roles)
        by_person = {participant: [op for op in operations if op["participant_id"] == participant]
                     for participant in machine["protected_participant_ids"]}
    except BaseException as error:
        for handle in (prediction_handle, raw_prediction, score_handle, access):
            try:
                if handle is not None:
                    handle.close()
            except Exception:
                pass
        signal.alarm(0)
        if old_xcpu is not None:
            signal.signal(signal.SIGXCPU, old_xcpu)
        if old_alarm is not None:
            signal.signal(signal.SIGALRM, old_alarm)
        receipt.update(status="failed_setup", finished_utc=datetime.now(timezone.utc).isoformat(),
                       cpu_seconds=time.process_time() - start_cpu,
                       wall_seconds=time.monotonic() - start_wall,
                       error={"type": type(error).__name__, "message": str(error)},
                       partial_outputs={path.name: sha256(path) for path in
                                        (score_partial, prediction_partial) if path.exists()})
        atomic_json(output / "receipt.json", receipt)
        raise
    try:
        for participant in machine["protected_participant_ids"]:
            before = time.monotonic()
            try:
                sessions, guards = {}, {}
                for session in (1, 2):
                    cov, labels, metadata = prepare_public_session(
                        public[(participant, session)], participant, session, access)
                    sessions[session] = (cov, labels); guards[str(session)] = metadata
            except OSError as error:
                statuses.append({"participant_id": participant, "status": "mechanical_failure",
                                 "operations": 0, "predictions": 0,
                                 "reason": f"public preparation failed: {type(error).__name__}: {error}",
                                 "wall_seconds": time.monotonic() - before})
                score_handle.flush(); prediction_handle.flush(); access.flush()
                if "sessions" in locals():
                    del sessions
                continue

            # Output failures are fatal and must not be recoded as participant missingness.
            path = cov_dir / f"s{participant:03d}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle, session1_cov=sessions[1][0], session1_y=sessions[1][1],
                                    session2_cov=sessions[2][0], session2_y=sessions[2][1])
            metadata_guards[str(participant)] = {"sessions": guards,
                                                 "covariance_path": str(path.relative_to(output)),
                                                 "covariance_sha256": sha256(path)}
            protected_memberships[str(participant)] = materialize_protected_memberships(
                participant, sessions[1][1])
            scores, predictions = score_person(
                participant, by_person[participant], memberships, source,
                sessions[1][0], sessions[1][1], sessions[2][0], sessions[2][1], representations,
                protected_memberships[str(participant)])
            score_writer.writerows(scores); prediction_writer.writerows(predictions)
            scores_written += len(scores); predictions_written += len(predictions)
            statuses.append({"participant_id": participant, "status": "complete",
                             "operations": 60, "predictions": 6000, "reason": "",
                             "wall_seconds": time.monotonic() - before})
            del sessions, scores, predictions
            score_handle.flush(); prediction_handle.flush(); access.flush()
        score_handle.close(); prediction_handle.close(); raw_prediction.close(); access.close()
        with (output / "person_status.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(statuses[0])); writer.writeheader(); writer.writerows(statuses)
        atomic_json(output / "protected_covariance_guards.json", metadata_guards)
        atomic_json(output / "protected_memberships.json", protected_memberships)
        covariance_bytes = sum(path.stat().st_size for path in cov_dir.iterdir())
        if covariance_bytes > 5 * 1024 * 1024:
            raise RuntimeError("persisted protected covariance artifacts exceed 5 MiB")
        completed = sum(row["status"] == "complete" for row in statuses)
        if (len(statuses) != 24 or scores_written != completed * 60
                or predictions_written != completed * 6000):
            raise RuntimeError("written score/prediction inventory differs")
        staged = {"status": "staged_inventory_locked_before_analysis",
                  "complete_people": completed, "failed_people": 24 - completed,
                  "scores_written": scores_written, "predictions_written": predictions_written,
                  "scores_partial_sha256": sha256(score_partial),
                  "predictions_partial_sha256": sha256(prediction_partial),
                  "status_sha256": sha256(output / "person_status.csv"),
                  "covariance_guards_sha256": sha256(output / "protected_covariance_guards.json"),
                  "protected_memberships_sha256": sha256(output / "protected_memberships.json"),
                  "access_log_sha256": sha256(output / "access_log.jsonl")}
        atomic_json(output / "STAGED_INTEGRITY.json", staged)
        os.replace(score_partial, output / "scores.csv")
        os.replace(prediction_partial, output / "trial_predictions.csv.gz")
        receipt.update(status="completed_score_inventory_no_analysis",
                       finished_utc=datetime.now(timezone.utc).isoformat(),
                       cpu_seconds=time.process_time() - start_cpu,
                       wall_seconds=time.monotonic() - start_wall,
                       complete_people=completed, failed_people=24 - completed,
                       operations=completed * 60, predictions=completed * 6000,
                       protected_covariance_bytes=covariance_bytes,
                       outputs={name: sha256(output / name) for name in
                                ("scores.csv", "trial_predictions.csv.gz", "person_status.csv",
                                 "protected_covariance_guards.json", "protected_memberships.json",
                                 "access_log.jsonl", "STAGED_INTEGRITY.json")})
        atomic_json(output / "receipt.json", receipt)
        atomic_json(output / "COMPLETED.json", {"status": receipt["status"],
                                                 "receipt_sha256": sha256(output / "receipt.json")})
        return {key: receipt[key] for key in ("status", "complete_people", "failed_people",
                                              "operations", "predictions")}
    except BaseException as error:
        for handle in (score_handle, prediction_handle, raw_prediction, access):
            try: handle.close()
            except Exception: pass
        receipt.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat(),
                       cpu_seconds=time.process_time() - start_cpu,
                       wall_seconds=time.monotonic() - start_wall,
                       error={"type": type(error).__name__, "message": str(error)},
                       partial_outputs={path.name: sha256(path) for path in
                                        (score_partial, prediction_partial) if path.exists()})
        atomic_json(output / "receipt.json", receipt)
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGXCPU, old_xcpu)
        signal.signal(signal.SIGALRM, old_alarm)


def shutil_disk_free(path):
    import shutil
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--packet-dir", required=True, type=Path)
    freeze_parser.add_argument("--out", required=True, type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--packet-dir", required=True, type=Path)
    run_parser.add_argument("--code-freeze", required=True, type=Path)
    run_parser.add_argument("--authorization", required=True, type=Path)
    run_parser.add_argument("--out-dir", required=True, type=Path)
    run_parser.add_argument("--execute-protected", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze(args.packet_dir.resolve(), args.out.resolve())
        print(json.dumps({"status": result["status"], "protected_access_authorized": False}, sort_keys=True))
    else:
        if not args.execute_protected:
            raise PermissionError("protected execution requires explicit --execute-protected plus bound authorization")
        result = run(args.packet_dir.resolve(), args.code_freeze.resolve(),
                     args.authorization.resolve(), args.out_dir.resolve())
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
