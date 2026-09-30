#!/usr/bin/env python3
"""Plan or run the paired S=100 donor comparator over the BNCI2B Review-4 grid."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
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
sys.path.insert(0, str(ROOT / "scripts"))
import run_review4_bnci2b_grid as grid
from expose.review2_controls import fit_representation, train_classifier


SCHEMA = "review4-bnci2b-donors-v1"
SETTINGS = ("legacy_C1_pooled_refit", "tuned_C_source_frozen")
CONDITIONS = ("base", "own", "pooled", "single_0", "single_1", "single_2")


def _write_gzip_csv(path, rows, fields):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _balanced_block(pool, store, total, rng):
    labels = store.labels(pool)
    block = []
    for label in (0, 1):
        values = [trial for trial, value in zip(pool, labels) if value == label]
        if len(values) < total // 2:
            raise ValueError("donor pool has fewer than 30 trials per class outside base S=100")
        block.extend(rng.permutation(values).tolist()[:total // 2])
    return block


def make_donor_plan(config, store, grid_operations, grid_sets, study_id):
    candidates = [op for op in grid_operations
                  if op["study_id"] == study_id and op["method"] == "plain_ts"
                  and op["donor_count"] == "all" and op["source_size"] == 100
                  and op["source_n"] == 100 and op["h_total"] == 60]
    expected = config["draws"] * len({op["target"] for op in candidates})
    if len(candidates) != expected:
        raise ValueError("grid must contain one plain-TS S=100/h=60/all-donor anchor per draw and target")
    sets, operations, identities = {}, [], []
    for anchor in sorted(candidates, key=lambda op: (op["draw"], op["target"])):
        base = list(grid_sets[anchor["memberships"]["source"]])
        own = list(grid_sets[anchor["memberships"]["personal"]])
        evaluation = list(grid_sets[anchor["memberships"]["evaluation"]])
        if len(base) != 100 or len(own) != 60 or set(base) & set(own):
            raise ValueError("donor anchor does not have disjoint B=100 and own h=60")
        donors = list(anchor["donor_subjects"])
        source_pool = [trial for subject in donors
                       for trial in store.by_cell[(subject, 1)]["trial_ids"]]
        outside_base = [trial for trial in source_pool if trial not in set(base)]
        pooled = _balanced_block(
            outside_base, store, 60,
            grid.seeded_rng(config["seed"], "review3_donor_pooled", study_id,
                            anchor["draw"], anchor["target"]),
        )
        eligible = []
        for subject in donors:
            available = [trial for trial in store.by_cell[(subject, 1)]["trial_ids"]
                         if trial not in set(base)]
            counts = {label: int(np.sum(store.labels(available) == label)) for label in (0, 1)}
            if min(counts.values()) >= 30:
                eligible.append(subject)
        if len(eligible) < 3:
            raise ValueError("fewer than three single donors have 30 trials per class outside base")
        chosen = sorted(grid.seeded_rng(
            config["seed"], "review3_donor_choice", study_id,
            anchor["draw"], anchor["target"]).permutation(eligible)[:3].tolist())
        blocks = {"base": [], "own": own, "pooled": pooled}
        for index, subject in enumerate(chosen):
            available = [trial for trial in store.by_cell[(subject, 1)]["trial_ids"]
                         if trial not in set(base)]
            blocks[f"single_{index}"] = _balanced_block(
                available, store, 60,
                grid.seeded_rng(config["seed"], "review3_single_donor_trials", study_id,
                                anchor["draw"], anchor["target"], subject),
            )
        base_ref = grid.add_membership(sets, base)
        eval_ref = grid.add_membership(sets, evaluation)
        for condition in CONDITIONS:
            added = blocks[condition]
            if set(base) & set(added):
                raise ValueError("every added donor block must be trial-disjoint from base S=100")
            added_ref = grid.add_membership(sets, added)
            if condition.startswith("single_"):
                donor_subject = chosen[int(condition[-1])]
            elif condition == "own":
                donor_subject = anchor["target"]
            else:
                donor_subject = None
            base_subjects = store.subjects(base)
            identities.append({
                "study_id": study_id, "draw": anchor["draw"], "target": anchor["target"],
                "condition": condition, "donor_subject": donor_subject,
                "eligible_single_donors": json.dumps(eligible, separators=(",", ":")),
                "chosen_single_donors": json.dumps(chosen, separators=(",", ":")),
                "base_trials_from_donor": ("" if donor_subject is None else
                                             int(np.sum(base_subjects == donor_subject))),
                "added_trial_overlap_with_base": len(set(base) & set(added)),
                "base_subject_counts": json.dumps(
                    {str(value): int(np.sum(base_subjects == value)) for value in donors},
                    sort_keys=True, separators=(",", ":")),
            })
            for setting in SETTINGS:
                operations.append({
                    "operation_id": (f"{study_id}__donor__{setting}__d{anchor['draw']}__"
                                     f"t{anchor['target']}__{condition}"),
                    "study_id": study_id, "dataset": anchor["dataset"],
                    "montage": anchor["montage"], "draw": anchor["draw"],
                    "target": anchor["target"], "condition": condition,
                    "donor_subject": donor_subject, "setting": setting,
                    "source_n": 100, "added_n": len(added),
                    "memberships": {"base": base_ref, "added": added_ref,
                                    "evaluation": eval_ref},
                    "tuning_id": anchor["tuning_id"] if setting == SETTINGS[1] else None,
                })
    if len({op["operation_id"] for op in operations}) != len(operations):
        raise ValueError("duplicate donor operation ID")
    return operations, sets, identities


def plan(config_path, index_path, grid_plan_dir, study_id, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("donor plan output must be a new path")
    index_sha = grid.sha256(index_path)
    config = grid.validate_config(grid.load_json(config_path), index_sha)
    grid_receipt, grid_operations, grid_sets, _ = grid.validate_plan(
        config_path, index_path, grid_plan_dir)
    store = grid.DatasetStore(index_path, index_sha)
    operations, sets, identities = make_donor_plan(
        config, store, grid_operations, grid_sets, study_id)
    output.mkdir(parents=True)
    grid.write_gzip_json(output / "operations.json.gz", operations)
    grid.write_gzip_json(output / "membership_sets.json.gz", sets)
    identity_fields = ["study_id", "draw", "target", "condition", "donor_subject",
                       "eligible_single_donors", "chosen_single_donors",
                       "base_trials_from_donor", "added_trial_overlap_with_base",
                       "base_subject_counts"]
    grid.write_csv(output / "donor_identity.csv", identities, identity_fields)
    receipt = {
        "schema": SCHEMA, "status": "planned_no_fits",
        "created_utc": datetime.now(timezone.utc).isoformat(), "study_id": study_id,
        "config_sha256": grid.sha256(config_path), "index_sha256": index_sha,
        "grid_plan_receipt_sha256": grid.sha256(Path(grid_plan_dir) / "PLAN_RECEIPT.json"),
        "grid_plan_runner_sha256": grid_receipt["runner_sha256"],
        "runner_sha256": grid.sha256(__file__),
        "counts": {"operations": len(operations), "membership_sets": len(sets),
                   "identity_rows": len(identities)},
        "files": {name: grid.sha256(output / name) for name in
                  ("operations.json.gz", "membership_sets.json.gz", "donor_identity.csv")},
        "interpretation": "Descriptive donor concentration conditional on a common B=100 base; donor identities may also contribute other base trials.",
    }
    grid.atomic_json(output / "PLAN_RECEIPT.json", receipt)
    return receipt


def _validate_plan(config_path, index_path, grid_plan_dir, donor_plan_dir):
    donor_plan_dir = Path(donor_plan_dir)
    receipt = grid.load_json(donor_plan_dir / "PLAN_RECEIPT.json")
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "planned_no_fits"
            or receipt.get("config_sha256") != grid.sha256(config_path)
            or receipt.get("index_sha256") != grid.sha256(index_path)
            or receipt.get("grid_plan_receipt_sha256") != grid.sha256(Path(grid_plan_dir) / "PLAN_RECEIPT.json")
            or receipt.get("runner_sha256") != grid.sha256(__file__)):
        raise ValueError("donor plan freeze hash guard failed")
    for name, digest in receipt["files"].items():
        if grid.sha256(donor_plan_dir / name) != digest:
            raise ValueError("donor plan file changed: " + name)
    operations = grid.read_gzip_json(donor_plan_dir / "operations.json.gz")
    sets = grid.read_gzip_json(donor_plan_dir / "membership_sets.json.gz")
    if any(grid.fingerprint(value) != key for key, value in sets.items()):
        raise ValueError("donor membership content hash changed")
    return receipt, operations, sets


def _selected_c(path):
    values = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["tuning_id", "selected_C"]:
            raise ValueError("selected_c.csv schema differs")
        for row in reader:
            if row["tuning_id"] in values:
                raise ValueError("duplicate tuning_id in selected_c.csv")
            value = float(row["selected_C"])
            if value not in (0.1, 1.0, 10.0):
                raise ValueError("selected C lies outside the fixed source-only grid")
            values[row["tuning_id"]] = value
    return values


def run(config_path, index_path, grid_plan_dir, donor_plan_dir, selected_c_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("donor run output must be a new path")
    config = grid.validate_config(grid.load_json(config_path), grid.sha256(index_path))
    plan_receipt, operations, sets = _validate_plan(
        config_path, index_path, grid_plan_dir, donor_plan_dir)
    selected = _selected_c(selected_c_path)
    required = {op["tuning_id"] for op in operations if op["tuning_id"]}
    if not required <= set(selected):
        raise ValueError("selected_c.csv lacks a required frozen source-only tuning context")
    output.mkdir(parents=True)
    limits = config["limits"]
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 10))
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_gib"] * 1024**3,) * 2)
    signal.alarm(limits["wall_seconds"])
    started_wall, started_cpu = time.monotonic(), time.process_time()
    store = grid.DatasetStore(index_path, config["index_sha256"])
    representation_cache, prediction_rows, score_rows, timing_rows = {}, [], [], []
    for number, operation in enumerate(operations):
        before = time.monotonic()
        base_ids = sets[operation["memberships"]["base"]]
        added_ids = sets[operation["memberships"]["added"]]
        evaluation_ids = sets[operation["memberships"]["evaluation"]]
        train_ids = base_ids + added_ids
        cov_train, y_train = store.matrices(train_ids, "cov"), store.labels(train_ids)
        if operation["setting"] == SETTINGS[0]:
            C, mode = 1.0, "pooled_refit"
            representation_key = (mode, operation["memberships"]["base"],
                                  operation["memberships"]["added"])
        else:
            C, mode = selected[operation["tuning_id"]], "source_frozen"
            representation_key = (mode, operation["memberships"]["base"])
        if representation_key not in representation_cache:
            representation_cache[representation_key] = fit_representation(
                cov_train, 100, mode)
        representation = representation_cache[representation_key]
        classifier = train_classifier(
            representation.transform(cov_train), y_train, 100, C, "pooled")
        eval_cov, y_eval = store.matrices(evaluation_ids, "cov"), store.labels(evaluation_ids)
        z_eval = representation.transform(eval_cov)
        prediction, probability = classifier.predict(z_eval), classifier.predict_proba(z_eval)
        for trial_id, truth, pred, probs in zip(evaluation_ids, y_eval, prediction, probability):
            prediction_rows.append({
                "operation_id": operation["operation_id"], "study_id": operation["study_id"],
                "setting": operation["setting"], "condition": operation["condition"],
                "draw": operation["draw"], "target": operation["target"],
                "donor_subject": operation["donor_subject"], "C": C,
                "trial_id": trial_id, "y_true": int(truth), "y_pred": int(pred),
                "p0": float(probs[0]), "p1": float(probs[1]),
                "base_membership": operation["memberships"]["base"],
                "added_membership": operation["memberships"]["added"]})
        score_rows.append({
            "operation_id": operation["operation_id"], "study_id": operation["study_id"],
            "setting": operation["setting"], "condition": operation["condition"],
            "draw": operation["draw"], "target": operation["target"],
            "donor_subject": operation["donor_subject"], "C": C,
            "balanced_accuracy": grid.balanced_accuracy(y_eval, prediction),
            "evaluation_n": len(y_eval)})
        timing_rows.append({"operation_id": operation["operation_id"], "index": number,
                            "wall_seconds": time.monotonic() - before})
    prediction_fields = ["operation_id", "study_id", "setting", "condition", "draw", "target",
                         "donor_subject", "C", "trial_id", "y_true", "y_pred", "p0", "p1",
                         "base_membership", "added_membership"]
    _write_gzip_csv(output / "predictions.csv.gz", prediction_rows, prediction_fields)
    grid.write_csv(output / "operation_scores.csv", score_rows,
                   ["operation_id", "study_id", "setting", "condition", "draw", "target",
                    "donor_subject", "C", "balanced_accuracy", "evaluation_n"])
    grid.write_csv(output / "timing.csv", timing_rows, ["operation_id", "index", "wall_seconds"])
    receipt = {
        "schema": SCHEMA, "status": "completed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "plan_receipt_sha256": grid.sha256(Path(donor_plan_dir) / "PLAN_RECEIPT.json"),
        "selected_c_sha256": grid.sha256(selected_c_path),
        "counts": {"operations": len(operations), "predictions": len(prediction_rows)},
        "outputs": {name: grid.sha256(output / name) for name in
                    ("predictions.csv.gz", "operation_scores.csv", "timing.csv")},
        "no_target_session2_selection": True,
    }
    grid.atomic_json(output / "receipt.json", receipt)
    grid.atomic_json(output / "COMPLETED.json",
                     {"status": "completed", "receipt_sha256": grid.sha256(output / "receipt.json")})
    signal.alarm(0)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        sub = subs.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--index", required=True, type=Path)
        sub.add_argument("--grid-plan-dir", required=True, type=Path)
        sub.add_argument("--donor-plan-dir", type=Path)
        sub.add_argument("--study-id")
        sub.add_argument("--selected-c", type=Path)
        sub.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        if not args.study_id or args.donor_plan_dir or args.selected_c:
            parser.error("plan requires --study-id and forbids run-only arguments")
        result = plan(args.config, args.index, args.grid_plan_dir, args.study_id, args.out_dir)
    else:
        if args.study_id or not args.donor_plan_dir or not args.selected_c:
            parser.error("run requires --donor-plan-dir and --selected-c")
        result = run(args.config, args.index, args.grid_plan_dir, args.donor_plan_dir,
                     args.selected_c, args.out_dir)
    print(json.dumps(result, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
