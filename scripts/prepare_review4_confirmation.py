#!/usr/bin/env python3
"""Create a source-only confirmation packet without protected signal access."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GRID_SEED = 2026092303
STUDY_TOKEN = "openbmi_main"
SETTING = "tuned_C_source_frozen"
ROLE_FILE = ROOT / "research/PARTICIPANT_ROLES_v1.csv"
INDEX_FILE = ROOT / "research/review3_20260923/OPENBMI8_GRID_INDEX.json"
CONFIG_FILE = ROOT / "research/review3_20260923/OPENBMI8_CONFIG.json"
GRID_PLAN = ROOT / "research/review3_20260923/openbmi8_plan_r003"
SELECTED_C = ROOT / "result/review3_20260923/openbmi8_grid_r003/selected_c.csv"


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


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_gzip_json(path, value):
    with path.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write((json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False) + "\n").encode())


def add_membership(sets, values):
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError("membership contains duplicate trials")
    key = fingerprint(values)
    if key in sets and sets[key] != values:
        raise ValueError("membership hash collision")
    sets[key] = values
    return key


def balanced_block(pool, labels, rng):
    block = []
    for label in (0, 1):
        values = [trial for trial in pool if labels[trial] == label]
        if len(values) < 30:
            raise ValueError("source donor lacks 30/class outside base")
        block.extend(rng.permutation(values).tolist()[:30])
    return block


def load_roles():
    with ROLE_FILE.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 54 or set(rows[0]) != {"subject_id", "role", "day1_selected", "selection_rule"}:
        raise ValueError("participant-role schema differs")
    by_role = {}
    for row in rows:
        subject = int(row["subject_id"])
        if subject in by_role or row["role"] not in ("source", "development", "confirmation"):
            raise ValueError("duplicate or invalid participant role")
        by_role[subject] = row["role"]
    counts = {role: sum(value == role for value in by_role.values())
              for role in ("source", "development", "confirmation")}
    if counts != {"source": 18, "development": 12, "confirmation": 24}:
        raise ValueError("participant role counts differ")
    return by_role


def load_source_index(role_map, inputs):
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    if index.get("dataset") != "openbmi" or index.get("montage") != "8ch":
        raise ValueError("OpenBMI grid index identity differs")
    records, labels = {}, {}
    for row in index["records"]:
        subject = int(row["subject"])
        if row["role"] != "source":
            continue
        if role_map.get(subject) != "source" or int(row["session"]) != 1 or subject in records:
            raise ValueError("source record role/session differs")
        path = ROOT / row["npz_path"]
        if sha256(path) != row["sha256"]:
            raise ValueError("source archive hash differs")
        with np.load(path, allow_pickle=False) as archive:
            y = np.asarray(archive["y"])
        if y.shape != (100,) or set(y.astype(int).tolist()) != {0, 1} or len(row["trial_ids"]) != 100:
            raise ValueError("source labels or trial inventory differ")
        records[subject] = list(row["trial_ids"])
        for trial, value in zip(row["trial_ids"], y.astype(int)):
            if trial in labels:
                raise ValueError("duplicate source trial ID")
            labels[trial] = int(value)
        inputs[f"source_s{subject}_npz"] = path
    if len(records) != 18 or sorted(records) != sorted(subject for subject, role in role_map.items() if role == "source"):
        raise ValueError("source-person inventory differs")
    return records, labels


def load_frozen_bases(labels):
    receipt = json.loads((GRID_PLAN / "PLAN_RECEIPT.json").read_text(encoding="utf-8"))
    if (receipt.get("status") != "planned_no_fits" or receipt.get("schema") != "review3-grid-v1"
            or receipt.get("config_sha256") != sha256(CONFIG_FILE)
            or receipt.get("index_sha256") != sha256(INDEX_FILE)):
        raise ValueError("frozen grid-plan receipt binding differs")
    run_receipt = json.loads((SELECTED_C.parent / "receipt.json").read_text(encoding="utf-8"))
    if (run_receipt.get("status") != "completed"
            or run_receipt.get("plan_receipt_sha256") != sha256(GRID_PLAN / "PLAN_RECEIPT.json")):
        raise ValueError("selected-C run is not bound to the frozen grid plan")
    for name, digest in receipt["files"].items():
        if sha256(GRID_PLAN / name) != digest:
            raise ValueError("frozen grid-plan file changed")
    operations = read_gzip_json(GRID_PLAN / "operations.json.gz")
    memberships = read_gzip_json(GRID_PLAN / "membership_sets.json.gz")
    if any(fingerprint(values) != key for key, values in memberships.items()):
        raise ValueError("frozen membership hash differs")
    selected = {}
    with SELECTED_C.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["tuning_id", "selected_C"]:
            raise ValueError("selected-C schema differs")
        for row in reader:
            value = float(row["selected_C"])
            if row["tuning_id"] in selected or value not in (0.1, 1.0, 10.0):
                raise ValueError("selected-C value differs")
            selected[row["tuning_id"]] = value
    bases = {}
    for draw in range(10):
        candidates = [op for op in operations if op["study_id"] == STUDY_TOKEN
                      and op["method"] == "plain_ts" and op["source_size"] == 100
                      and op["h_total"] == 60 and op["draw"] == draw and op["donor_count"] == "all"]
        source_refs = {op["memberships"]["source"] for op in candidates}
        tuning_ids = {op["tuning_id"] for op in candidates}
        donors = {tuple(op["donor_subjects"]) for op in candidates}
        if len(candidates) != 12 or len(source_refs) != 1 or len(tuning_ids) != 1 or len(donors) != 1:
            raise ValueError("shared source anchor differs")
        reference, tuning_id = source_refs.pop(), tuning_ids.pop()
        base = memberships[reference]
        if len(base) != 100 or sum(labels[trial] == 0 for trial in base) != 50:
            raise ValueError("source base is not 50/class")
        bases[draw] = {"trial_ids": base, "source_membership": reference,
                       "tuning_id": tuning_id, "selected_C": selected[tuning_id],
                       "donor_subjects": list(donors.pop())}
    return bases


def build_packet(role_map, source_records, labels, bases):
    protected = sorted(subject for subject, role in role_map.items() if role == "confirmation")
    memberships, rows, operations = {}, [], []
    empty_ref = add_membership(memberships, [])
    for draw in range(10):
        base = bases[draw]["trial_ids"]
        base_ref = add_membership(memberships, base)
        if base_ref != bases[draw]["source_membership"]:
            raise ValueError("re-added base membership differs")
        outside = [trial for subject in bases[draw]["donor_subjects"]
                   for trial in source_records[subject] if trial not in set(base)]
        for target in protected:
            pooled = balanced_block(outside, labels, seeded_rng(
                GRID_SEED, "review3_donor_pooled", STUDY_TOKEN, draw, target))
            eligible = []
            for subject in bases[draw]["donor_subjects"]:
                available = [trial for trial in source_records[subject] if trial not in set(base)]
                if min(sum(labels[trial] == label for trial in available) for label in (0, 1)) >= 30:
                    eligible.append(subject)
            if len(eligible) < 3:
                raise ValueError("fewer than three eligible source donors")
            chosen = sorted(seeded_rng(GRID_SEED, "review3_donor_choice", STUDY_TOKEN,
                                       draw, target).permutation(eligible)[:3].tolist())
            blocks = {"base": [], "pooled": pooled}
            for index, subject in enumerate(chosen):
                available = [trial for trial in source_records[subject] if trial not in set(base)]
                blocks[f"single_{index}"] = balanced_block(
                    available, labels, seeded_rng(GRID_SEED, "review3_single_donor_trials",
                                                  STUDY_TOKEN, draw, target, subject))
            refs = {name: add_membership(memberships, value) for name, value in blocks.items()}
            if any(set(base) & set(value) for value in blocks.values()):
                raise ValueError("added source-side membership overlaps common base")
            history_rule = {
                "algorithm": "per_class_seeded_permutation_prefix",
                "seed_function": "seeded_rng(2026092303,'history','openbmi_main',draw,protected_id,label)",
                "take_per_class": 30, "session": 1, "phase": "offline",
                "generated_only_after_authorized_schema_read": True,
            }
            evaluation_rule = {"session": 2, "phase": "offline", "all_trials": 100,
                               "generated_only_after_authorized_schema_read": True}
            rows.append({"participant_id": target, "draw": draw,
                         "source_membership": base_ref, "pooled_membership": refs["pooled"],
                         "single_0_membership": refs["single_0"],
                         "single_1_membership": refs["single_1"],
                         "single_2_membership": refs["single_2"],
                         "single_0_donor": chosen[0], "single_1_donor": chosen[1],
                         "single_2_donor": chosen[2], "eligible_donors": json.dumps(eligible),
                         "tuning_id": bases[draw]["tuning_id"], "selected_C": bases[draw]["selected_C"],
                         "history_rule_id": fingerprint(history_rule),
                         "evaluation_rule_id": fingerprint(evaluation_rule)})
            for condition in ("base", "own", "pooled", "single_0", "single_1", "single_2"):
                added_ref = (None if condition == "own" else refs.get(condition, empty_ref))
                operations.append({
                    "operation_id": f"openbmi_confirmation__d{draw}__t{target}__{condition}",
                    "participant_id": target, "draw": draw, "condition": condition,
                    "setting": SETTING, "source_n": 100,
                    "added_n": 0 if condition == "base" else 60,
                    "h_total": 0 if condition == "base" else 60,
                    "C": bases[draw]["selected_C"], "source_membership": base_ref,
                    "added_membership": added_ref,
                    "future_history_rule": history_rule if condition == "own" else None,
                    "future_evaluation_rule": evaluation_rule,
                    "donor_subject": (chosen[int(condition[-1])]
                                      if condition.startswith("single_") else None),
                    "primary": condition in ("own", "single_0", "single_1", "single_2"),
                })
    if len(rows) != 240 or len(operations) != 1440:
        raise ValueError("confirmation operation inventory differs")
    return protected, memberships, rows, operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(); args.out_dir = args.out_dir.resolve()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    if args.out_dir.exists():
        raise FileExistsError("output directory must be new")
    inputs = {"roles": ROLE_FILE, "index": INDEX_FILE, "config": CONFIG_FILE,
              "grid_plan_receipt": GRID_PLAN / "PLAN_RECEIPT.json",
              "grid_operations": GRID_PLAN / "operations.json.gz",
              "grid_memberships": GRID_PLAN / "membership_sets.json.gz",
              "selected_C": SELECTED_C, "selected_C_run_receipt": SELECTED_C.parent / "receipt.json"}
    before = {name: sha256(path) for name, path in inputs.items()}
    roles = load_roles()
    source_records, labels = load_source_index(roles, inputs)
    for name, path in inputs.items():
        before.setdefault(name, sha256(path))
    bases = load_frozen_bases(labels)
    protected, memberships, packet_rows, operations = build_packet(roles, source_records, labels, bases)

    args.out_dir.mkdir(parents=True)
    write_gzip_json(args.out_dir / "membership_sets.json.gz", memberships)
    write_gzip_json(args.out_dir / "operations.json.gz", operations)
    fields = list(packet_rows[0])
    with (args.out_dir / "source_packet.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(packet_rows)
    machine = {
        "schema": "review4-confirmation-source-packet-v1",
        "status": "source_packet_only_not_adopted_not_executable",
        "adoption_required": True, "protected_access_authorized": False,
        "protected_participant_ids": protected, "draws": 10,
        "primary_conditions": ["own", "single_0", "single_1", "single_2"],
        "optional_conditions": ["base", "pooled"],
        "expected_primary_scores": 960, "expected_optional_scores": 480,
        "setting": SETTING, "source_n": 100, "added_n": 60,
        "history_selection": {
            "algorithm": "review3 seeded per-class permutation prefix",
            "seed": GRID_SEED, "tokens": ["history", STUDY_TOKEN, "draw", "protected_id", "label"],
            "take_per_class": 30,
            "exact_memberships_deferred_until_separately_authorized_protected_schema_read": True},
        "evaluation": {"session": 2, "phase": "offline", "expected_trials": 100,
                       "exact_memberships_deferred": True},
        "unimplemented_required_integration": [
            "authorized protected offline-session preparation with schema validation",
            "exact protected history/evaluation membership materialization and hash binding",
            "source-frozen TS-LR fit/score runner using this packet",
            "outcome suppression until inventory and membership integrity lock"],
    }
    (args.out_dir / "MACHINE_CONFIG.json").write_text(
        json.dumps(machine, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("input changed during packet creation")
    output_names = ("membership_sets.json.gz", "operations.json.gz", "source_packet.csv", "MACHINE_CONFIG.json")
    receipt = {
        "status": "completed_source_only_packet", "protected_accessed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "threads": 1,
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256_before": before[name],
                          "sha256_after": after[name], "bytes": path.stat().st_size}
                   for name, path in inputs.items()},
        "code": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__)},
        "counts": {"protected_ids_metadata_only": 24, "target_draw_rows": 240,
                   "operations": 1440, "primary_operations": 960,
                   "optional_operations": 480, "membership_sets": len(memberships)},
        "outputs": {name: {"sha256": sha256(args.out_dir / name),
                           "bytes": (args.out_dir / name).stat().st_size} for name in output_names},
        "runtime": {"cpu_seconds": time.process_time() - start_cpu,
                    "wall_seconds": time.monotonic() - start_wall,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    (args.out_dir / "PACKET_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "operations": 1440,
                      "protected_accessed": False}, sort_keys=True))


if __name__ == "__main__":
    main()
