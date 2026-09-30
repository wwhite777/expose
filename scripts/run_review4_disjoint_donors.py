#!/usr/bin/env python3
"""Plan, run, and descriptively summarize the Review-4 donor-disjoint control."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
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
import run_review3_donors as donors
import run_review3_grid as grid
from expose.review2_controls import fit_representation, train_classifier


SCHEMA = "review4-base-donor-disjoint-v1"
SEED = 2026092411
CONDITIONS = ("own", "single_0", "single_1", "single_2")
LIMITS = {"cpu_seconds": 600, "wall_seconds": 1200, "memory_gib": 16,
          "maximum_output_bytes": 5 * 1024 * 1024}
PREDICTION_FIELDS = ["operation_id", "dataset_key", "target", "draw", "condition",
                     "donor_subject", "C", "trial_id", "y_true", "y_pred", "p0", "p1"]
SCORE_FIELDS = ["operation_id", "dataset_key", "target", "draw", "condition",
                "donor_subject", "C", "balanced_accuracy", "evaluation_n"]


def code_paths():
    return {"runner": ROOT / "scripts/run_review4_disjoint_donors.py",
            "tests": ROOT / "tests/test_review4_disjoint_donors.py",
            "grid_runner": ROOT / "scripts/run_review3_grid.py",
            "donor_runner": ROOT / "scripts/run_review3_donors.py",
            "model_primitives": ROOT / "src/expose/review2_controls.py"}


def dependency_versions():
    import pyriemann
    import scipy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
            "pyriemann": pyriemann.__version__}


def repository_path(value):
    path = Path(value).resolve()
    if ROOT not in path.parents or not path.exists():
        raise ValueError("input must be an existing repository-contained path")
    return path


def identity(path):
    path = repository_path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": grid.sha256(path),
            "bytes": path.stat().st_size}


def resolve_identity(value):
    path = ROOT / value["path"]
    if identity(path) != value:
        raise ValueError("frozen input changed: " + value["path"])
    return path


def load_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def balanced_base(pool, store, dataset_key, target, draw):
    labels = store.labels(pool)
    selected = []
    for label in (0, 1):
        values = [trial for trial, observed in zip(pool, labels) if observed == label]
        if len(values) < 50:
            raise ValueError("remaining donor-disjoint source pool has fewer than 50 trials/class")
        rng = grid.seeded_rng(SEED, "review4_disjoint_base", dataset_key, target, draw, label)
        selected.extend(rng.permutation(values).tolist()[:50])
    if len(selected) != len(set(selected)) or Counter(store.labels(selected).tolist()) != {0: 50, 1: 50}:
        raise ValueError("new base is not a unique balanced 100-trial block")
    return selected


def selected_c_binding(selected_c_path, config_path, index_path, grid_plan_dir):
    selected_c_path = repository_path(selected_c_path)
    receipt_path, completed_path = selected_c_path.parent / "receipt.json", selected_c_path.parent / "COMPLETED.json"
    receipt = grid.load_json(receipt_path); completed = grid.load_json(completed_path)
    if (receipt.get("status") != "completed"
            or receipt.get("config_sha256") != grid.sha256(config_path)
            or receipt.get("index_sha256") != grid.sha256(index_path)
            or receipt.get("plan_receipt_sha256") != grid.sha256(Path(grid_plan_dir) / "PLAN_RECEIPT.json")
            or receipt.get("outputs", {}).get("selected_c.csv") != grid.sha256(selected_c_path)
            or completed.get("status") != "completed"
            or completed.get("receipt_sha256") != grid.sha256(receipt_path)
            or completed.get("completion_type") not in
            (None, "additive_prediction_hash_finalization_recovery")):
        raise ValueError("selected-C completed-run binding differs")
    return donors._selected_c(selected_c_path), identity(receipt_path), identity(completed_path)


def build_dataset_plan(dataset_key, config_path, index_path, grid_plan_dir, c_grid_plan_dir,
                       donor_plan_dir, selected_c_path):
    config_path, index_path = repository_path(config_path), repository_path(index_path)
    grid_plan_dir, c_grid_plan_dir = repository_path(grid_plan_dir), repository_path(c_grid_plan_dir)
    donor_plan_dir = repository_path(donor_plan_dir)
    config = grid.validate_config(grid.load_json(config_path), grid.sha256(index_path))
    old_receipt, old_operations, old_sets = donors._validate_plan(
        config_path, index_path, grid_plan_dir, donor_plan_dir)
    donor_grid_receipt = grid.load_json(grid_plan_dir / "PLAN_RECEIPT.json")
    c_grid_receipt = grid.load_json(c_grid_plan_dir / "PLAN_RECEIPT.json")
    if donor_grid_receipt.get("files") != c_grid_receipt.get("files"):
        raise ValueError("donor and selected-C grid plans do not have identical scientific payloads")
    selected, selected_receipt, selected_completed = selected_c_binding(
        selected_c_path, config_path, index_path, c_grid_plan_dir)
    store = grid.DatasetStore(index_path, config["index_sha256"])
    expected_identity = {"openbmi": ("openbmi_main", "openbmi", "8ch", 12),
                         "bnci": ("bnci_main", "bnci2014", "22ch", 9)}.get(dataset_key)
    if (expected_identity is None or config["draws"] != 10
            or (old_receipt["study_id"], store.dataset, store.montage) != expected_identity[:3]):
        raise ValueError("dataset/study/draw identity differs")
    fields, identity_rows = load_csv(donor_plan_dir / "donor_identity.csv")
    if fields != ["study_id", "draw", "target", "condition", "donor_subject",
                  "eligible_single_donors", "chosen_single_donors",
                  "base_trials_from_donor", "added_trial_overlap_with_base",
                  "base_subject_counts"]:
        raise ValueError("old donor identity schema differs")
    identity_by_coordinate = {(int(row["draw"]), int(row["target"]), row["condition"]): row
                              for row in identity_rows}
    tuned = [op for op in old_operations if op["setting"] == "tuned_C_source_frozen"
             and op["condition"] in CONDITIONS]
    groups = defaultdict(dict)
    for op in tuned:
        key = (int(op["draw"]), int(op["target"]))
        if op["condition"] in groups[key]:
            raise ValueError("duplicate old donor operation coordinate")
        groups[key][op["condition"]] = op
    expected_targets = expected_identity[3]
    targets = {target for _, target in groups}
    if (len(targets) != expected_targets or len(groups) != expected_targets * 10
            or any({draw for draw, observed_target in groups if observed_target == target}
                   != set(range(10)) for target in targets)):
        raise ValueError("old donor target/draw coverage differs")

    sets, operations, audit_rows, c_rows = {}, [], [], []
    for (draw, target), by_condition in sorted(groups.items()):
        if set(by_condition) != set(CONDITIONS):
            raise ValueError("old donor condition coverage differs")
        identity_row = identity_by_coordinate[(draw, target, "own")]
        nominated = [int(value) for value in json.loads(identity_row["chosen_single_donors"])]
        source_subjects = sorted(int(value) for value in json.loads(identity_row["base_subject_counts"]))
        if len(nominated) != 3 or not set(nominated) <= set(source_subjects) or target in source_subjects:
            raise ValueError("old nominated donor/source identity differs")
        remaining = [subject for subject in source_subjects if subject not in nominated]
        pool = [trial for subject in remaining for trial in store.by_cell[(subject, 1)]["trial_ids"]]
        base = balanced_base(pool, store, dataset_key, target, draw)
        if set(store.subjects(base)) & set(nominated):
            raise ValueError("new base contains a nominated donor")
        evaluation_refs = {op["memberships"]["evaluation"] for op in by_condition.values()}
        tuning_ids = {op["tuning_id"] for op in by_condition.values()}
        if len(evaluation_refs) != 1 or len(tuning_ids) != 1:
            raise ValueError("old paired evaluation/tuning context differs")
        evaluation = old_sets[next(iter(evaluation_refs))]
        tuning_id = next(iter(tuning_ids))
        if tuning_id not in selected:
            raise ValueError("selected-C table lacks required source-only context")
        C = selected[tuning_id]
        base_ref = grid.add_membership(sets, base)
        evaluation_ref = grid.add_membership(sets, evaluation)
        c_rows.append({"dataset_key": dataset_key, "target": target, "draw": draw,
                       "tuning_id": tuning_id, "selected_C": C})
        for condition in CONDITIONS:
            old = by_condition[condition]
            added = old_sets[old["memberships"]["added"]]
            donor_subject = old["donor_subject"]
            expected_donor = target if condition == "own" else nominated[int(condition[-1])]
            if (donor_subject != expected_donor or len(added) != 60
                    or Counter(store.labels(added).tolist()) != {0: 30, 1: 30}
                    or set(base) & set(added) or set(base) & set(evaluation)
                    or set(added) & set(evaluation)):
                raise ValueError("new base/addition/evaluation pairing differs")
            added_ref = grid.add_membership(sets, added)
            operation_id = f"review4_disjoint__{dataset_key}__d{draw}__t{target}__{condition}"
            operations.append({"operation_id": operation_id, "dataset_key": dataset_key,
                               "dataset": store.dataset, "montage": store.montage,
                               "target": target, "draw": draw, "condition": condition,
                               "donor_subject": donor_subject, "source_n": 100, "added_n": 60,
                               "C": C, "tuning_id": tuning_id,
                               "memberships": {"base": base_ref, "added": added_ref,
                                               "evaluation": evaluation_ref},
                               "old_operation_id": old["operation_id"]})
            audit_rows.append({"operation_id": operation_id, "dataset_key": dataset_key,
                               "target": target, "draw": draw, "condition": condition,
                               "donor_subject": donor_subject,
                               "nominated_donors": json.dumps(nominated, separators=(",", ":")),
                               "remaining_source_subjects": len(remaining),
                               "base_donor_identity_overlap": len(set(store.subjects(base)) & set(nominated)),
                               "base_added_trial_overlap": len(set(base) & set(added)),
                               "base_class0": int(np.sum(store.labels(base) == 0)),
                               "base_class1": int(np.sum(store.labels(base) == 1)),
                               "added_class0": int(np.sum(store.labels(added) == 0)),
                               "added_class1": int(np.sum(store.labels(added) == 1)),
                               "evaluation_n": len(evaluation)})
    expected_operations = expected_targets * 10 * 4
    if len(operations) != expected_operations or len({op["operation_id"] for op in operations}) != expected_operations:
        raise ValueError("new donor-disjoint operation coverage differs")
    inputs = {
        "config": identity(config_path), "index": identity(index_path),
        "grid_plan_receipt": identity(grid_plan_dir / "PLAN_RECEIPT.json"),
        "selected_c_grid_plan_receipt": identity(c_grid_plan_dir / "PLAN_RECEIPT.json"),
        "donor_plan_receipt": identity(donor_plan_dir / "PLAN_RECEIPT.json"),
        "donor_operations": identity(donor_plan_dir / "operations.json.gz"),
        "donor_memberships": identity(donor_plan_dir / "membership_sets.json.gz"),
        "donor_identity": identity(donor_plan_dir / "donor_identity.csv"),
        "selected_c": identity(selected_c_path), "selected_c_receipt": selected_receipt,
        "selected_c_completed": selected_completed,
    }
    return operations, sets, audit_rows, c_rows, inputs, old_receipt["study_id"]


def plan(specs, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("plan output must be a new path")
    all_operations, all_sets, audit_rows, c_rows, datasets = [], {}, [], [], []
    for spec in specs:
        operations, sets, audit, c_values, inputs, study_id = build_dataset_plan(**spec)
        all_operations.extend(operations); audit_rows.extend(audit); c_rows.extend(c_values)
        for key, values in sets.items():
            if key in all_sets and all_sets[key] != values:
                raise ValueError("membership hash collision")
            all_sets[key] = values
        datasets.append({"dataset_key": spec["dataset_key"], "study_id": study_id,
                         "inputs": inputs})
    if len(all_operations) != 840 or Counter(op["dataset_key"] for op in all_operations) != {
            "openbmi": 480, "bnci": 360}:
        raise ValueError("combined donor-disjoint design must contain exactly 840 fits")
    output.mkdir(parents=True)
    grid.write_gzip_json(output / "operations.json.gz", all_operations)
    grid.write_gzip_json(output / "membership_sets.json.gz", all_sets)
    grid.write_csv(output / "membership_audit.csv", audit_rows,
                   ["operation_id", "dataset_key", "target", "draw", "condition",
                    "donor_subject", "nominated_donors", "remaining_source_subjects",
                    "base_donor_identity_overlap", "base_added_trial_overlap", "base_class0",
                    "base_class1", "added_class0", "added_class1", "evaluation_n"])
    grid.write_csv(output / "c_provenance.csv", c_rows,
                   ["dataset_key", "target", "draw", "tuning_id", "selected_C"])
    receipt = {"schema": SCHEMA, "status": "planned_no_fits", "seed": SEED,
               "created_utc": datetime.now(timezone.utc).isoformat(), "datasets": datasets,
               "command": [sys.executable, *sys.argv],
               "runner_sha256": grid.sha256(__file__), "limits": LIMITS,
               "code": {name: identity(path) for name, path in code_paths().items()},
               "dependencies": dependency_versions(),
               "counts": {"operations": len(all_operations), "membership_sets": len(all_sets),
                          "audit_rows": len(audit_rows), "c_rows": len(c_rows)},
               "outputs": {name: identity(output / name) for name in
                           ("operations.json.gz", "membership_sets.json.gz",
                            "membership_audit.csv", "c_provenance.csv")},
               "scope": "posthoc base-donor-disjoint sensitivity; source tuning could include nominated donors"}
    grid.atomic_json(output / "PLAN_RECEIPT.json", receipt)
    return receipt


def validate_plan(plan_dir):
    plan_dir = repository_path(plan_dir)
    receipt = grid.load_json(plan_dir / "PLAN_RECEIPT.json")
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != "planned_no_fits"
            or receipt.get("seed") != SEED or receipt.get("runner_sha256") != grid.sha256(__file__)
            or receipt.get("limits") != LIMITS
            or receipt.get("dependencies") != dependency_versions()
            or set(receipt.get("code", {})) != set(code_paths())):
        raise ValueError("donor-disjoint plan receipt differs")
    for name, path in code_paths().items():
        if receipt["code"][name] != identity(path):
            raise ValueError("donor-disjoint code changed after plan freeze: " + name)
    for dataset in receipt["datasets"]:
        for value in dataset["inputs"].values():
            resolve_identity(value)
    for name, value in receipt["outputs"].items():
        if identity(plan_dir / name) != value:
            raise ValueError("donor-disjoint plan output changed: " + name)
    operations = grid.read_gzip_json(plan_dir / "operations.json.gz")
    sets = grid.read_gzip_json(plan_dir / "membership_sets.json.gz")
    if (len(operations) != 840 or any(grid.fingerprint(values) != key for key, values in sets.items())):
        raise ValueError("donor-disjoint plan content differs")
    return receipt, operations, sets


def open_prediction_stream(path):
    raw = Path(path).open("xb")
    compressed = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=PREDICTION_FIELDS); writer.writeheader()
    return raw, text, writer


def reconstruct_scores(path, operations, sets, stores=None):
    expected = {op["operation_id"]: op for op in operations}
    seen_trials, counts = defaultdict(set), defaultdict(lambda: [[0, 0], [0, 0]])
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise ValueError("prediction schema differs")
        for row in reader:
            op = expected.get(row["operation_id"])
            if op is None:
                raise ValueError("prediction has unknown operation ID")
            trial = row["trial_id"]
            truth, predicted = int(row["y_true"]), int(row["y_pred"])
            p0, p1 = float(row["p0"]), float(row["p1"])
            if (trial in seen_trials[op["operation_id"]]
                    or trial not in sets[op["memberships"]["evaluation"]]
                    or row["dataset_key"] != op["dataset_key"]
                    or int(row["target"]) != op["target"] or int(row["draw"]) != op["draw"]
                    or row["condition"] != op["condition"]
                    or int(row["donor_subject"]) != op["donor_subject"]
                    or float(row["C"]) != float(op["C"])
                    or truth not in (0, 1) or predicted not in (0, 1)
                    or not np.isfinite([p0, p1]).all() or min(p0, p1) < 0 or max(p0, p1) > 1
                    or abs(p0 + p1 - 1) > 1e-8 or predicted != (0 if p0 >= p1 else 1)):
                raise ValueError("prediction row identity/probability differs")
            if stores is not None and truth != int(stores[op["dataset_key"]].labels([trial])[0]):
                raise ValueError("prediction truth differs from hash-bound dataset index")
            seen_trials[op["operation_id"]].add(trial)
            counts[op["operation_id"]][truth][0] += int(predicted == truth)
            counts[op["operation_id"]][truth][1] += 1
    result = {}
    for operation_id, op in expected.items():
        evaluation = sets[op["memberships"]["evaluation"]]
        if seen_trials[operation_id] != set(evaluation):
            raise ValueError("prediction evaluation coverage differs")
        class_counts = counts[operation_id]
        if any(total == 0 for _, total in class_counts):
            raise ValueError("prediction class coverage differs")
        result[operation_id] = float(np.mean([correct / total for correct, total in class_counts]))
    return result


def summarize(scores, operations):
    coordinate = {(op["dataset_key"], op["target"], op["draw"], op["condition"]):
                  scores[op["operation_id"]] for op in operations}
    draw_rows, person_rows, summaries = [], [], {}
    for dataset_key in ("openbmi", "bnci"):
        targets = sorted({op["target"] for op in operations if op["dataset_key"] == dataset_key})
        for target in targets:
            effects = []
            for draw in range(10):
                own = coordinate[(dataset_key, target, draw, "own")]
                singles = [coordinate[(dataset_key, target, draw, f"single_{index}")]
                           for index in range(3)]
                effect = (own - float(np.mean(singles))) * 100
                effects.append(effect)
                draw_rows.append({"dataset_key": dataset_key, "target": target, "draw": draw,
                                  "own_ba": own, "mean_single_ba": float(np.mean(singles)),
                                  "effect_pp": effect})
            person_rows.append({"dataset_key": dataset_key, "target": target, "draws": 10,
                                "effect_pp": float(np.mean(effects))})
        values = np.asarray([row["effect_pp"] for row in person_rows
                             if row["dataset_key"] == dataset_key])
        rng = np.random.default_rng(SEED)
        boot = np.mean(values[rng.integers(0, len(values), size=(10000, len(values)))], axis=1)
        summaries[dataset_key] = {"n": len(values), "mean_pp": float(np.mean(values)),
                                  "median_pp": float(np.median(values)),
                                  "bootstrap_ci_low_pp": float(np.quantile(boot, .025)),
                                  "bootstrap_ci_high_pp": float(np.quantile(boot, .975)),
                                  "positive": int(np.sum(values > 1e-12)),
                                  "zero": int(np.sum(np.abs(values) <= 1e-12)),
                                  "negative": int(np.sum(values < -1e-12))}
    return {"status": "completed_posthoc_descriptive_sensitivity", "seed": SEED,
            "bootstrap_draws": 10000, "unit": "percentage points",
            "estimand": "own minus mean of three single donors, ten draws averaged within person",
            "scope": "base-donor-disjoint; source-only C tuning may include nominated donors",
            "datasets": summaries}, person_rows, draw_rows


def run(plan_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("run output must be a new path")
    plan_receipt, operations, sets = validate_plan(plan_dir)
    stores = {}
    for dataset in plan_receipt["datasets"]:
        inputs = dataset["inputs"]
        index_path = resolve_identity(inputs["index"])
        config = grid.validate_config(grid.load_json(resolve_identity(inputs["config"])),
                                      grid.sha256(index_path))
        stores[dataset["dataset_key"]] = grid.DatasetStore(index_path, config["index_sha256"])
    output.mkdir(parents=True)
    running = {"schema": SCHEMA, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, *sys.argv], "threads": 1,
               "plan_receipt_sha256": grid.sha256(Path(plan_dir) / "PLAN_RECEIPT.json"),
               "runner_sha256": grid.sha256(__file__), "limits": LIMITS}
    grid.atomic_json(output / "receipt.json", running)
    old_xcpu = signal.signal(signal.SIGXCPU, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU limit")))
    old_alarm = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall limit")))
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_seconds"], LIMITS["cpu_seconds"] + 10))
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS["memory_gib"] * 1024**3,) * 2)
    signal.alarm(LIMITS["wall_seconds"])
    started_wall, started_cpu = time.monotonic(), time.process_time()
    prediction_partial = output / "predictions.csv.gz.partial"
    raw, text, writer = open_prediction_stream(prediction_partial)
    score_rows, timing_rows, representation_cache = [], [], {}
    try:
        for index, op in enumerate(operations):
            before = time.monotonic(); store = stores[op["dataset_key"]]
            base_ids = sets[op["memberships"]["base"]]
            added_ids = sets[op["memberships"]["added"]]
            eval_ids = sets[op["memberships"]["evaluation"]]
            base_cov, base_y = store.matrices(base_ids, "cov"), store.labels(base_ids)
            key = (op["dataset_key"], op["memberships"]["base"])
            if key not in representation_cache:
                representation_cache[key] = fit_representation(base_cov, 100, "source_frozen")
            representation = representation_cache[key]
            train_cov = np.concatenate([base_cov, store.matrices(added_ids, "cov")])
            train_y = np.concatenate([base_y, store.labels(added_ids)])
            classifier = train_classifier(representation.transform(train_cov), train_y, 100,
                                          float(op["C"]), "pooled")
            eval_cov, eval_y = store.matrices(eval_ids, "cov"), store.labels(eval_ids)
            z_eval = representation.transform(eval_cov)
            predicted, probabilities = classifier.predict(z_eval), classifier.predict_proba(z_eval)
            if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                    or np.any(probabilities > 1)
                    or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)):
                raise ValueError("model probability schema differs")
            for trial, truth, prediction, probability in zip(eval_ids, eval_y, predicted, probabilities):
                writer.writerow({"operation_id": op["operation_id"], "dataset_key": op["dataset_key"],
                                 "target": op["target"], "draw": op["draw"],
                                 "condition": op["condition"], "donor_subject": op["donor_subject"],
                                 "C": op["C"], "trial_id": trial, "y_true": int(truth),
                                 "y_pred": int(prediction), "p0": float(probability[0]),
                                 "p1": float(probability[1])})
            score_rows.append({"operation_id": op["operation_id"], "dataset_key": op["dataset_key"],
                               "target": op["target"], "draw": op["draw"],
                               "condition": op["condition"], "donor_subject": op["donor_subject"],
                               "C": op["C"], "balanced_accuracy": grid.balanced_accuracy(eval_y, predicted),
                               "evaluation_n": len(eval_y)})
            timing_rows.append({"operation_id": op["operation_id"], "index": index,
                                "wall_seconds": time.monotonic() - before})
        grid.close_prediction_stream(raw, text)
        os.replace(prediction_partial, output / "predictions.csv.gz")
        reconstructed = reconstruct_scores(output / "predictions.csv.gz", operations, sets, stores)
        if any(abs(float(row["balanced_accuracy"]) - reconstructed[row["operation_id"]]) > 1e-12
               for row in score_rows):
            raise ValueError("saved and independently reconstructed balanced accuracy differ")
        grid.write_csv(output / "operation_scores.csv", score_rows, SCORE_FIELDS)
        grid.write_csv(output / "timing.csv", timing_rows,
                       ["operation_id", "index", "wall_seconds"])
        summary, person_rows, draw_rows = summarize(reconstructed, operations)
        grid.write_csv(output / "person_effects.csv", person_rows,
                       ["dataset_key", "target", "draws", "effect_pp"])
        grid.write_csv(output / "draw_effects.csv", draw_rows,
                       ["dataset_key", "target", "draw", "own_ba", "mean_single_ba", "effect_pp"])
        grid.atomic_json(output / "summary.json", summary)
        output_names = ("predictions.csv.gz", "operation_scores.csv", "timing.csv",
                        "person_effects.csv", "draw_effects.csv", "summary.json")
        output_bytes = sum((output / name).stat().st_size for name in output_names)
        if output_bytes > LIMITS["maximum_output_bytes"]:
            raise RuntimeError("durable outputs exceed 5 MiB")
        receipt = {**running, "status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "cpu_seconds": time.process_time() - started_cpu,
                   "wall_seconds": time.monotonic() - started_wall,
                   "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   "counts": {"operations": len(score_rows),
                              "predictions": sum(row["evaluation_n"] for row in score_rows),
                              "participant_effects": len(person_rows)},
                   "output_bytes_excluding_receipts": output_bytes,
                   "outputs": {name: grid.sha256(output / name) for name in output_names},
                   "balanced_accuracy_reconstructed_from_predictions": True,
                   "confirmation_accessed": False}
        grid.atomic_json(output / "receipt.json", receipt)
        grid.atomic_json(output / "COMPLETED.json",
                         {"status": "completed", "receipt_sha256": grid.sha256(output / "receipt.json")})
        return receipt
    except BaseException as error:
        try:
            if not text.closed:
                grid.close_prediction_stream(raw, text)
        except Exception:
            pass
        running.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat(),
                       cpu_seconds=time.process_time() - started_cpu,
                       wall_seconds=time.monotonic() - started_wall,
                       error={"type": type(error).__name__, "message": str(error)},
                       partial_outputs={path.name: grid.sha256(path) for path in output.iterdir()
                                        if path.is_file() and path.name != "receipt.json"})
        grid.atomic_json(output / "receipt.json", running)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGXCPU, old_xcpu); signal.signal(signal.SIGALRM, old_alarm)


def add_plan_arguments(parser, prefix):
    parser.add_argument(f"--{prefix}-config", required=True, type=Path)
    parser.add_argument(f"--{prefix}-index", required=True, type=Path)
    parser.add_argument(f"--{prefix}-grid-plan", required=True, type=Path)
    parser.add_argument(f"--{prefix}-c-grid-plan", required=True, type=Path)
    parser.add_argument(f"--{prefix}-donor-plan", required=True, type=Path)
    parser.add_argument(f"--{prefix}-selected-c", required=True, type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    add_plan_arguments(plan_parser, "openbmi"); add_plan_arguments(plan_parser, "bnci")
    plan_parser.add_argument("--out-dir", required=True, type=Path)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--plan-dir", required=True, type=Path)
    run_parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        specs = []
        for key in ("openbmi", "bnci"):
            specs.append({"dataset_key": key, "config_path": getattr(args, f"{key}_config"),
                          "index_path": getattr(args, f"{key}_index"),
                          "grid_plan_dir": getattr(args, f"{key}_grid_plan"),
                          "c_grid_plan_dir": getattr(args, f"{key}_c_grid_plan"),
                          "donor_plan_dir": getattr(args, f"{key}_donor_plan"),
                          "selected_c_path": getattr(args, f"{key}_selected_c")})
        result = plan(specs, args.out_dir)
    else:
        result = run(args.plan_dir, args.out_dir)
    print(json.dumps({"status": result["status"], "counts": result.get("counts")}, sort_keys=True))


if __name__ == "__main__":
    main()
