#!/usr/bin/env python3
"""Plan or run the frozen BNCI2014-004 variable-count Review-4 grid."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import resource
import signal
import sys
import time
import warnings

import numpy as np
from pyriemann.classification import MDM

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT
sys.path.insert(0, str(ROOT / "src"))
from expose.review2_controls import fit_representation, train_classifier
from expose.review3_transfer import (
    assemble_recenter_references, fit_mdwm_from_source_means,
    fit_recenter_whitener, mdwm_source_class_means, recenter_source, recenter_target,
)


METHODS = ("plain_ts", "ea_ts", "plain_mdm", "recenter_mdm", "mdwm")
TS_METHODS = {"plain_ts": "plain", "ea_ts": "ea"}
SCHEMA = "newreview4-bnci2b-grid-v1"
EXPECTED_SESSION_TRIAL_COUNTS = {
    1: (120, 120),
    2: (120, 120),
    3: (120, 120),
    4: (120, 140),
    5: (120, 140),
    6: (120, 120),
    7: (120, 120),
    8: (160, 120),
    9: (120, 120),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def write_gzip_json(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write((json.dumps(value, sort_keys=True, separators=(",", ":"),
                                         allow_nan=False) + "\n").encode())
    os.replace(temporary, path)


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def safe_repo_file(relative, expected_sha):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("unsafe repository-relative NPZ path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe repository-relative NPZ path")
    path = DATA_ROOT / pure
    resolved = path.resolve(strict=True)
    if DATA_ROOT.resolve() not in resolved.parents or path.is_symlink() or not path.is_file():
        raise ValueError("NPZ path is not a contained regular file")
    if sha256(path) != expected_sha:
        raise ValueError("NPZ hash differs from dataset index")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config, index_sha):
    required = {"schema", "seed", "draws", "C_grid", "C_tie_tolerance", "methods",
                "mdwm_lambda", "index_sha256", "limits", "studies"}
    if set(config) != required or config["schema"] != SCHEMA:
        raise ValueError("invalid BNCI2B Review-4 configuration schema")
    if (not isinstance(config["seed"], int) or isinstance(config["seed"], bool)
            or config["seed"] != 2026092403
            or not isinstance(config["draws"], int) or isinstance(config["draws"], bool)
            or config["draws"] != 10
            or config["C_grid"] != [0.1, 1.0, 10.0]
            or config["C_tie_tolerance"] != 1e-12 or config["methods"] != list(METHODS)
            or config["mdwm_lambda"] != 0.5 or config["index_sha256"] != index_sha):
        raise ValueError("fixed BNCI2B Review-4 parameters differ")
    limits = config["limits"]
    if (set(limits) != {"cpu_seconds", "wall_seconds", "memory_gib"}
            or any(not isinstance(limits[key], int) or limits[key] <= 0 for key in limits)):
        raise ValueError("invalid positive resource limits")
    seen = set()
    study_fields = {"study_id", "dataset", "montage", "mode", "source_sizes",
                    "history_totals", "donor_counts", "reference_trials_per_person"}
    if len(config["studies"]) != 1:
        raise ValueError("BNCI2B Review-4 requires exactly one study")
    for study in config["studies"]:
        if set(study) != study_fields or study["study_id"] in seen:
            raise ValueError("invalid or duplicate study definition")
        seen.add(study["study_id"])
        if (
            not isinstance(study["study_id"], str)
            or not study["study_id"]
            or (study["dataset"], study["montage"]) != ("bnci2014b", "3ch")
            or study["mode"] != "loso"
            or study["source_sizes"] != [100, "all"]
            or study["history_totals"] != [0, 60]
            or study["donor_counts"] != ["all"]
            or study["reference_trials_per_person"] != "all_available"
        ):
            raise ValueError("invalid frozen BNCI2B study grid")
    return config


class DatasetStore:
    def __init__(self, index_path, expected_sha=None):
        self.index_path = Path(index_path).resolve()
        if expected_sha is not None and sha256(self.index_path) != expected_sha:
            raise ValueError("dataset index hash differs from configuration")
        index = load_json(self.index_path)
        if set(index) != {"dataset", "montage", "records"}:
            raise ValueError("invalid dataset index schema")
        if (index["dataset"], index["montage"]) not in {
                ("openbmi", "8ch"), ("openbmi", "20ch"),
                ("bnci2014", "22ch"), ("bnci2014b", "3ch")}:
            raise ValueError("invalid dataset index identity")
        self.dataset, self.montage = index["dataset"], index["montage"]
        self.records, self.by_cell, self.trials, self.arrays = [], {}, {}, {}
        fields = {"subject", "session", "role", "npz_path", "sha256", "trial_ids"}
        for record_id, raw in enumerate(index["records"]):
            if set(raw) != fields or raw["role"] not in ("source", "development", "external"):
                raise ValueError("invalid dataset record")
            if (not isinstance(raw["subject"], int) or raw["session"] not in (1, 2)
                    or not isinstance(raw["sha256"], str) or len(raw["sha256"]) != 64
                    or not isinstance(raw["trial_ids"], list) or not raw["trial_ids"]
                    or any(not isinstance(value, str) or not value for value in raw["trial_ids"])):
                raise ValueError("invalid dataset record fields")
            cell = (raw["subject"], raw["session"])
            if cell in self.by_cell:
                raise ValueError("duplicate subject/session record")
            path = safe_repo_file(raw["npz_path"], raw["sha256"])
            with np.load(path, allow_pickle=False) as archive:
                names = set(archive.files)
                if names not in ({"cov", "ea_cov", "y"},
                                  {"cov", "ea_cov", "empirical_cov", "y"}):
                    raise ValueError("NPZ must contain cov, ea_cov, y and optional empirical_cov")
                cov, ea_cov, labels = archive["cov"], archive["ea_cov"], archive["y"]
                empirical_cov = archive["empirical_cov"] if "empirical_cov" in names else None
            count = len(raw["trial_ids"])
            if (cov.dtype != np.float64 or ea_cov.dtype != np.float64
                    or cov.shape != ea_cov.shape or cov.ndim != 3 or cov.shape[0] != count
                    or cov.shape[1] != cov.shape[2] or labels.shape != (count,)
                    or labels.dtype.kind not in "biu"
                    or Counter(labels.astype(int).tolist()) != {0: count // 2, 1: count // 2}
                    or count % 2
                    or not np.isfinite(cov).all() or not np.isfinite(ea_cov).all()):
                raise ValueError("invalid saved covariance arrays")
            for name, values in (("cov", cov), ("ea_cov", ea_cov),
                                 ("empirical_cov", empirical_cov)):
                if values is None:
                    continue
                if (values.dtype != np.float64 or values.shape != cov.shape
                        or not np.isfinite(values).all()
                        or not np.allclose(values, values.transpose(0, 2, 1), rtol=0, atol=1e-10)
                        or np.any(np.linalg.eigvalsh(values) <= 0)):
                    raise ValueError(f"invalid {name} covariance arrays")
            if len(set(raw["trial_ids"])) != count:
                raise ValueError("duplicate trial ID within record")
            normalized = dict(raw, record_id=record_id, path=path, y=labels.astype(int))
            self.records.append(normalized)
            self.by_cell[cell] = normalized
            for offset, trial_id in enumerate(raw["trial_ids"]):
                if trial_id in self.trials:
                    raise ValueError("trial ID is not globally unique")
                self.trials[trial_id] = (record_id, offset, int(labels[offset]))

    def load_arrays(self, record_id):
        if record_id not in self.arrays:
            record = self.records[record_id]
            with np.load(record["path"], allow_pickle=False) as archive:
                self.arrays[record_id] = {name: archive[name].copy() for name in ("cov", "ea_cov", "y")}
        return self.arrays[record_id]

    def matrices(self, trial_ids, name):
        return np.stack([self.load_arrays(self.trials[trial_id][0])[name][self.trials[trial_id][1]]
                         for trial_id in trial_ids])

    def labels(self, trial_ids):
        return np.asarray([self.trials[trial_id][2] for trial_id in trial_ids], dtype=int)

    def subjects(self, trial_ids):
        return np.asarray([self.records[self.trials[trial_id][0]]["subject"] for trial_id in trial_ids], dtype=int)


def seeded_rng(seed, *tokens):
    suffix = int(hashlib.sha256(json.dumps(tokens, sort_keys=True, default=str).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(np.random.SeedSequence([seed, suffix & 0xffffffff, suffix >> 32]))


def balanced_prefix(pool, store, total, rng):
    labels = store.labels(pool)
    by_class = {label: [trial for trial, value in zip(pool, labels) if value == label] for label in (0, 1)}
    capacity = 2 * min(map(len, by_class.values()))
    wanted = capacity if total == "all" else min(total, capacity)
    if wanted <= 0 or wanted % 2:
        raise ValueError("balanced source prefix has no positive even capacity")
    return (rng.permutation(by_class[0]).tolist()[:wanted // 2]
            + rng.permutation(by_class[1]).tolist()[:wanted // 2])


def cell_records(store, study):
    if store.dataset != study["dataset"] or store.montage != study["montage"]:
        raise ValueError("study identity differs from dataset index")
    sources = targets = sorted(record["subject"] for record in store.records
                               if record["role"] == "external" and record["session"] == 1)
    if sources != list(EXPECTED_SESSION_TRIAL_COUNTS) or len(store.records) != 18:
        raise ValueError("BNCI2B index must contain exactly nine external people and 18 cells")
    if {record["role"] for record in store.records} != {"external"}:
        raise ValueError("BNCI2B index contains a role other than external")
    for subject in sources:
        for session in (1, 2):
            record = store.by_cell.get((subject, session))
            expected = EXPECTED_SESSION_TRIAL_COUNTS[subject][session - 1]
            if record is None or len(record["trial_ids"]) != expected:
                raise ValueError("subject/session count differs from observed BNCI2B matrix")
    return sources, targets


def add_membership(sets, values):
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError("membership contains duplicate trials")
    identity = fingerprint(values)
    if identity in sets and sets[identity] != values:
        raise ValueError("membership hash collision")
    sets[identity] = values
    return identity


def make_plan(config, store):
    sets, operations, tuning = {}, [], {}
    for study in config["studies"]:
        all_sources, targets = cell_records(store, study)
        for draw in range(config["draws"]):
            for target in targets:
                eligible = ([subject for subject in all_sources if subject != target]
                            if study["mode"] == "loso" else list(all_sources))
                target_key = target if study["mode"] == "loso" else "shared"
                for donor_count in study["donor_counts"]:
                    if donor_count == "all":
                        donors = sorted(eligible)
                    else:
                        if donor_count > len(eligible):
                            raise ValueError("donor count exceeds eligible source people")
                        donors = sorted(seeded_rng(config["seed"], "donors", study["study_id"], draw,
                                                   target_key, donor_count).permutation(eligible)[:donor_count].tolist())
                    source_reference = [trial for subject in donors
                                        for trial in store.by_cell[(subject, 1)]["trial_ids"]]
                    expected_source_reference = sum(
                        EXPECTED_SESSION_TRIAL_COUNTS[subject][0] for subject in donors
                    )
                    if len(source_reference) != expected_source_reference:
                        raise ValueError("source reference allowance count differs")
                    source_pool = list(source_reference)
                    target_reference = list(store.by_cell[(target, 1)]["trial_ids"])
                    evaluation = list(store.by_cell[(target, 2)]["trial_ids"])
                    history_permutations = {}
                    history_labels = store.labels(target_reference)
                    for label in (0, 1):
                        values = [trial for trial, value in zip(target_reference, history_labels) if value == label]
                        history_permutations[label] = seeded_rng(
                            config["seed"], "history", study["study_id"], draw, target, label).permutation(values).tolist()
                    for source_size in study["source_sizes"]:
                        source = balanced_prefix(
                            source_pool, store, source_size,
                            seeded_rng(config["seed"], "source", study["study_id"], draw,
                                       target_key, donor_count),
                        )
                        source_ref = add_membership(sets, source)
                        source_reference_ref = add_membership(sets, source_reference)
                        evaluation_ref = add_membership(sets, evaluation)
                        target_reference_ref = add_membership(sets, target_reference)
                        tune_keys = {}
                        for representation in ("plain", "ea"):
                            tune_id = fingerprint([study["study_id"], draw, target_key, donor_count,
                                                   source_size, representation, donors])
                            if tune_id not in tuning:
                                folds = []
                                for heldout in donors:
                                    train_pool = [trial for subject in donors if subject != heldout
                                                  for trial in store.by_cell[(subject, 1)]["trial_ids"]]
                                    train = balanced_prefix(
                                        train_pool, store, source_size,
                                        seeded_rng(config["seed"], "cv", study["study_id"], draw,
                                                   target_key, donor_count, heldout),
                                    )
                                    heldout_ids = list(store.by_cell[(heldout, 1)]["trial_ids"])
                                    folds.append({"heldout_subject": heldout,
                                                  "train": add_membership(sets, train),
                                                  "evaluation": add_membership(sets, heldout_ids)})
                                tuning[tune_id] = {
                                    "tuning_id": tune_id, "study_id": study["study_id"], "draw": draw,
                                    "target_context": target_key, "donor_count": donor_count,
                                    "donor_subjects": donors, "source_size": source_size,
                                    "representation": representation,
                                    "source_reference": source_reference_ref, "folds": folds,
                                }
                            tune_keys[representation] = tune_id
                        for h_total in study["history_totals"]:
                            if h_total > len(target_reference) or h_total % 2:
                                raise ValueError("history total exceeds available balanced target history")
                            personal = (history_permutations[0][:h_total // 2]
                                        + history_permutations[1][:h_total // 2])
                            personal_ref = add_membership(sets, personal)
                            for method in METHODS:
                                reference_method = method in ("ea_ts", "recenter_mdm")
                                op = {
                                    "operation_id": (f"{study['study_id']}__{method}__d{draw}__t{target}__"
                                                     f"k{donor_count}__S{source_size}__h{h_total}"),
                                    "study_id": study["study_id"], "dataset": study["dataset"],
                                    "montage": study["montage"], "mode": study["mode"],
                                    "method": method, "draw": draw, "target": target,
                                    "donor_count": donor_count, "donor_subjects": donors,
                                    "source_size": source_size, "source_n": len(source), "h_total": h_total,
                                    "reference_trials_per_person": len(target_reference),
                                    "memberships": {
                                        "source": source_ref, "personal": personal_ref,
                                        "evaluation": evaluation_ref,
                                        "source_reference": source_reference_ref if reference_method else None,
                                        "target_reference": target_reference_ref if reference_method else None,
                                    },
                                    "tuning_id": tune_keys.get(TS_METHODS.get(method)),
                                }
                                operations.append(op)
    if len({operation["operation_id"] for operation in operations}) != len(operations):
        raise ValueError("duplicate operation ID")
    return operations, sets, list(tuning.values())


def plan(config_path, index_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("plan output must be a new path")
    index_hash = sha256(index_path)
    config = validate_config(load_json(config_path), index_hash)
    store = DatasetStore(index_path, index_hash)
    operations, sets, tuning = make_plan(config, store)
    output.mkdir(parents=True)
    paths = {"operations": output / "operations.json.gz",
             "memberships": output / "membership_sets.json.gz",
             "tuning": output / "tuning_folds.json.gz"}
    write_gzip_json(paths["operations"], operations)
    write_gzip_json(paths["memberships"], sets)
    write_gzip_json(paths["tuning"], tuning)
    receipt = {
        "status": "planned_no_fits", "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config_path).resolve()), "config_sha256": sha256(config_path),
        "index_path": str(Path(index_path).resolve()), "index_sha256": index_hash,
        "runner_sha256": sha256(__file__),
        "transfer_module_sha256": sha256(ROOT / "src/expose/review3_transfer.py"),
        "counts": {"operations": len(operations), "membership_sets": len(sets),
                   "tuning_contexts": len(tuning),
                   "tuning_folds": sum(len(value["folds"]) for value in tuning)},
        "files": {path.name: sha256(path) for path in paths.values()},
        "no_model_fits": True,
    }
    atomic_json(output / "PLAN_RECEIPT.json", receipt)
    return receipt


def validate_plan(config_path, index_path, plan_dir):
    plan_dir = Path(plan_dir)
    receipt = load_json(plan_dir / "PLAN_RECEIPT.json")
    if (receipt.get("status") != "planned_no_fits" or receipt.get("schema") != SCHEMA
            or receipt.get("config_sha256") != sha256(config_path)
            or receipt.get("index_sha256") != sha256(index_path)
            or receipt.get("runner_sha256") != sha256(__file__)
            or receipt.get("transfer_module_sha256") != sha256(ROOT / "src/expose/review3_transfer.py")):
        raise ValueError("plan freeze hash guard failed")
    for name, digest in receipt.get("files", {}).items():
        path = plan_dir / name
        if not path.is_file() or path.is_symlink() or sha256(path) != digest:
            raise ValueError("frozen plan file changed: " + name)
    operations = read_gzip_json(plan_dir / "operations.json.gz")
    sets = read_gzip_json(plan_dir / "membership_sets.json.gz")
    tuning = read_gzip_json(plan_dir / "tuning_folds.json.gz")
    if receipt["counts"] != {"operations": len(operations), "membership_sets": len(sets),
                             "tuning_contexts": len(tuning),
                             "tuning_folds": sum(len(value["folds"]) for value in tuning)}:
        raise ValueError("plan counts changed")
    if any(fingerprint(value) != key for key, value in sets.items()):
        raise ValueError("membership content hash changed")
    return receipt, operations, sets, tuning


def balanced_accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or set(y_true.tolist()) != {0, 1}:
        raise ValueError("balanced accuracy requires both binary classes")
    return float(np.mean([np.mean(y_pred[y_true == label] == label) for label in (0, 1)]))


def choose_c(scores, C_grid, tolerance):
    best = max(scores.values())
    return min(C for C in C_grid if best - scores[C] <= tolerance)


def tune_c(config, store, sets, tuning, progress=None):
    selected, rows = {}, []
    for context in tuning:
        kind = "cov" if context["representation"] == "plain" else "ea_cov"
        fold_scores = defaultdict(list)
        for fold in context["folds"]:
            train_ids, eval_ids = sets[fold["train"]], sets[fold["evaluation"]]
            train_cov, eval_cov = store.matrices(train_ids, kind), store.matrices(eval_ids, kind)
            y_train, y_eval = store.labels(train_ids), store.labels(eval_ids)
            representation = fit_representation(train_cov, len(train_cov), "source_frozen")
            z_train, z_eval = representation.transform(train_cov), representation.transform(eval_cov)
            for C in config["C_grid"]:
                classifier = train_classifier(z_train, y_train, len(train_cov), C, "pooled")
                score = balanced_accuracy(y_eval, classifier.predict(z_eval))
                fold_scores[float(C)].append(score)
                rows.append({"tuning_id": context["tuning_id"], "representation": context["representation"],
                             "heldout_subject": fold["heldout_subject"], "C": C,
                             "balanced_accuracy": score, "train_n": len(train_ids), "evaluation_n": len(eval_ids)})
        means = {C: float(np.mean(values)) for C, values in fold_scores.items()}
        selected[context["tuning_id"]] = choose_c(means, tuple(config["C_grid"]), config["C_tie_tolerance"])
        if progress is not None:
            progress("tuning_context_completed", len(selected), len(tuning))
    return selected, rows


def fit_mdm(covariances, labels):
    return MDM(metric="riemann", n_jobs=1).fit(covariances, labels)


def write_csv(path, rows, fieldnames, compressed=False):
    temporary = Path(str(path) + ".tmp")
    opener = gzip.open if compressed else open
    with opener(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def close_prediction_stream(raw_prediction, prediction_handle):
    """Close every layer before moving or hashing the deterministic gzip output."""
    try:
        if not prediction_handle.closed:
            prediction_handle.flush()
            prediction_handle.close()
    finally:
        if not raw_prediction.closed:
            raw_prediction.flush()
            raw_prediction.close()


def run(config_path, index_path, plan_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("run output must be a new path")
    config = validate_config(load_json(config_path), sha256(index_path))
    plan_receipt, operations, sets, tuning = validate_plan(config_path, index_path, plan_dir)
    limits = config["limits"]
    output.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "plan_receipt_sha256": sha256(Path(plan_dir) / "PLAN_RECEIPT.json"),
               "config_sha256": sha256(config_path), "index_sha256": sha256(index_path),
               "limits": limits, "confirmation_accessed": False, "GPU_used": False}
    atomic_json(output / "receipt.json", receipt)
    prediction_fields = ["operation_id", "study_id", "method", "draw", "target", "donor_count",
                         "source_size", "source_n", "h_total", "C", "trial_id", "y_true", "y_pred",
                         "p0", "p1", "source_membership", "personal_membership", "evaluation_membership"]
    timing_fields = ["operation_id", "index", "fit_signature", "reused_h0_fit", "wall_seconds"]
    score_fields = ["operation_id", "study_id", "method", "draw", "target", "donor_count",
                    "source_size", "source_n", "h_total", "C", "balanced_accuracy",
                    "evaluation_n", "source_membership", "personal_membership"]
    historical_fields = ["operation_id", "method", "target", "draw", "source_size", "h_total",
                         "correct_count", "history_n", "balanced_accuracy", "history_membership",
                         "h0_fit_signature"]
    counts = {"operations": 0, "predictions": 0, "tuning_rows": 0,
              "selected_C": 0, "historical_decodability_rows": 0, "warnings": 0}
    warning_counts = Counter()
    progress_path = output / "progress.jsonl"

    def elapsed_record(event, completed=None, total=None):
        record = {"event": event, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "wall_seconds": time.monotonic() - started_wall,
                  "cpu_seconds": time.process_time() - started_cpu,
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "counts": dict(counts), "warning_counts": dict(warning_counts)}
        if completed is not None:
            record.update(completed=completed, total=total)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def limit_signal(signum, frame):
        del frame
        raise TimeoutError(f"resource limit signal {signum}")

    old_xcpu = signal.signal(signal.SIGXCPU, limit_signal)
    old_alarm = signal.signal(signal.SIGALRM, limit_signal)
    resource.setrlimit(resource.RLIMIT_CPU,
                       (limits["cpu_seconds"], limits["cpu_seconds"] + 10))
    resource.setrlimit(resource.RLIMIT_AS, (limits["memory_gib"] * 1024**3,) * 2)
    signal.alarm(limits["wall_seconds"])
    partial_paths = [output / "predictions.csv.gz.partial", output / "timing.csv.partial",
                     output / "operation_scores.csv.partial",
                     output / "historical_decodability.csv.partial"]
    raw_prediction = partial_paths[0].open("wb")
    compressed_prediction = gzip.GzipFile(fileobj=raw_prediction, mode="wb", mtime=0)
    prediction_handle = io.TextIOWrapper(compressed_prediction, encoding="utf-8", newline="")
    timing_handle = partial_paths[1].open("w", encoding="utf-8", newline="")
    score_handle = partial_paths[2].open("w", encoding="utf-8", newline="")
    historical_handle = partial_paths[3].open("w", encoding="utf-8", newline="")
    prediction_writer = csv.DictWriter(prediction_handle, fieldnames=prediction_fields)
    timing_writer = csv.DictWriter(timing_handle, fieldnames=timing_fields)
    score_writer = csv.DictWriter(score_handle, fieldnames=score_fields)
    historical_writer = csv.DictWriter(historical_handle, fieldnames=historical_fields)
    for writer in (prediction_writer, timing_writer, score_writer, historical_writer):
        writer.writeheader()
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            store = DatasetStore(index_path, config["index_sha256"])
            selected, tuning_rows = tune_c(config, store, sets, tuning, elapsed_record)
            counts["tuning_rows"], counts["selected_C"] = len(tuning_rows), len(selected)
            write_csv(output / "tuning_scores.csv", tuning_rows,
                      ["tuning_id", "representation", "heldout_subject", "C",
                       "balanced_accuracy", "train_n", "evaluation_n"])
            selected_rows = [{"tuning_id": key, "selected_C": value}
                             for key, value in sorted(selected.items())]
            write_csv(output / "selected_c.csv", selected_rows, ["tuning_id", "selected_C"])

            representation_cache, h0_model_cache, h0_ts_cache, source_means_cache = {}, {}, {}, {}
            recenter_cache, whitener_cache = {}, {}

            def person_whitener(trial_ids):
                record_ids = {store.trials[value][0] for value in trial_ids}
                if len(record_ids) != 1:
                    raise ValueError("one recentering reference must be one person/session record")
                record_id = next(iter(record_ids))
                record = store.records[record_id]
                if list(trial_ids) != list(record["trial_ids"]):
                    raise ValueError("recenter reference must contain the full ordered session-1 record")
                if record_id not in whitener_cache:
                    whitener_cache[record_id] = fit_recenter_whitener(
                        store.matrices(trial_ids, "cov"))
                return record["subject"], whitener_cache[record_id]

            for number, operation in enumerate(operations):
                before = time.monotonic()
                membership = operation["memberships"]
                source_ids, personal_ids = sets[membership["source"]], sets[membership["personal"]]
                eval_ids = sets[membership["evaluation"]]
                y_source, y_personal, y_eval = (store.labels(source_ids), store.labels(personal_ids),
                                                store.labels(eval_ids))
                method = operation["method"]
                C = selected[operation["tuning_id"]] if operation["tuning_id"] else None
                fit_key = (method, membership["source"], membership["personal"], C,
                           membership["source_reference"], membership["target_reference"])
                cache_key = (method, membership["source"], membership["personal"], C)
                reused_h0_fit = bool(not personal_ids and method != "recenter_mdm"
                                     and cache_key in h0_model_cache)
                if reused_h0_fit:
                    model, transform = h0_model_cache[cache_key]
                elif method in TS_METHODS:
                    kind = "cov" if method == "plain_ts" else "ea_cov"
                    source_cov = store.matrices(source_ids, kind)
                    personal_cov = store.matrices(personal_ids, kind) if personal_ids else None
                    representation_key = (method, membership["source"])
                    if representation_key not in representation_cache:
                        representation_cache[representation_key] = fit_representation(
                            source_cov, len(source_cov), "source_frozen")
                    representation = representation_cache[representation_key]
                    train_cov = source_cov if personal_cov is None else np.concatenate([source_cov, personal_cov])
                    train_y = y_source if personal_cov is None else np.concatenate([y_source, y_personal])
                    model = train_classifier(representation.transform(train_cov), train_y,
                                             len(source_cov), C, "pooled")
                    h0_ts_key = (method, membership["source"], C)
                    if h0_ts_key not in h0_ts_cache:
                        h0_ts_cache[h0_ts_key] = (model if not personal_ids else train_classifier(
                            representation.transform(source_cov), y_source, len(source_cov), C, "pooled"))
                    if personal_ids:
                        historical_prediction = h0_ts_cache[h0_ts_key].predict(
                            representation.transform(personal_cov))
                        historical_writer.writerow({
                            "operation_id": operation["operation_id"], "method": method,
                            "target": operation["target"], "draw": operation["draw"],
                            "source_size": operation["source_size"], "h_total": operation["h_total"],
                            "correct_count": int(np.sum(historical_prediction == y_personal)),
                            "history_n": len(y_personal),
                            "balanced_accuracy": balanced_accuracy(y_personal, historical_prediction),
                            "history_membership": membership["personal"],
                            "h0_fit_signature": fingerprint(h0_ts_key)})
                        counts["historical_decodability_rows"] += 1
                    transform = lambda values, representation=representation, kind=kind: representation.transform(
                        store.matrices(values, kind))
                elif method == "plain_mdm":
                    source_cov = store.matrices(source_ids, "cov")
                    train_cov = source_cov if not personal_ids else np.concatenate(
                        [source_cov, store.matrices(personal_ids, "cov")])
                    train_y = y_source if not personal_ids else np.concatenate([y_source, y_personal])
                    model, transform = fit_mdm(train_cov, train_y), lambda values: store.matrices(values, "cov")
                elif method == "mdwm":
                    if membership["source"] not in source_means_cache:
                        source_means_cache[membership["source"]] = mdwm_source_class_means(
                            store.matrices(source_ids, "cov"), y_source)
                    model = fit_mdwm_from_source_means(
                        source_means_cache[membership["source"]],
                        store.matrices(personal_ids, "cov") if personal_ids else None,
                        y_personal if personal_ids else None, config["mdwm_lambda"])
                    transform = lambda values: store.matrices(values, "cov")
                else:
                    ref_key = (membership["source_reference"], membership["target_reference"])
                    if ref_key not in recenter_cache:
                        source_reference_ids = sets[membership["source_reference"]]
                        grouped = defaultdict(list)
                        for trial_id in source_reference_ids:
                            grouped[store.trials[trial_id][0]].append(trial_id)
                        source_whiteners = dict(person_whitener(values)
                                                for _, values in sorted(grouped.items()))
                        target_reference_ids = sets[membership["target_reference"]]
                        _, target_whitener = person_whitener(target_reference_ids)
                        recenter_cache[ref_key] = assemble_recenter_references(
                            source_whiteners, target_whitener,
                            operation["reference_trials_per_person"])
                    references = recenter_cache[ref_key]
                    source_cov = recenter_source(
                        store.matrices(source_ids, "cov"), store.subjects(source_ids), references)
                    personal_cov = (recenter_target(store.matrices(personal_ids, "cov"), references)
                                    if personal_ids else None)
                    train_cov = source_cov if personal_cov is None else np.concatenate([source_cov, personal_cov])
                    train_y = y_source if personal_cov is None else np.concatenate([y_source, y_personal])
                    model = fit_mdm(train_cov, train_y)
                    transform = lambda values, references=references: recenter_target(
                        store.matrices(values, "cov"), references)
                if not personal_ids and method != "recenter_mdm":
                    h0_model_cache[cache_key] = (model, transform)
                eval_cov = transform(eval_ids)
                predicted, probabilities = model.predict(eval_cov), model.predict_proba(eval_cov)
                if (predicted.shape != y_eval.shape or probabilities.shape != (len(y_eval), 2)
                        or not np.isfinite(probabilities).all()
                        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)):
                    raise ValueError("prediction output differs from the fixed binary schema")
                for trial_id, truth, pred, probability in zip(eval_ids, y_eval, predicted, probabilities):
                    prediction_writer.writerow({
                        "operation_id": operation["operation_id"], "study_id": operation["study_id"],
                        "method": method, "draw": operation["draw"], "target": operation["target"],
                        "donor_count": operation["donor_count"], "source_size": operation["source_size"],
                        "source_n": operation["source_n"], "h_total": operation["h_total"], "C": C,
                        "trial_id": trial_id, "y_true": int(truth), "y_pred": int(pred),
                        "p0": float(probability[0]), "p1": float(probability[1]),
                        "source_membership": membership["source"],
                        "personal_membership": membership["personal"],
                        "evaluation_membership": membership["evaluation"]})
                    counts["predictions"] += 1
                score_writer.writerow({
                    "operation_id": operation["operation_id"], "study_id": operation["study_id"],
                    "method": method, "draw": operation["draw"], "target": operation["target"],
                    "donor_count": operation["donor_count"], "source_size": operation["source_size"],
                    "source_n": operation["source_n"], "h_total": operation["h_total"], "C": C,
                    "balanced_accuracy": balanced_accuracy(y_eval, predicted),
                    "evaluation_n": len(y_eval), "source_membership": membership["source"],
                    "personal_membership": membership["personal"]})
                timing_writer.writerow({"operation_id": operation["operation_id"], "index": number,
                                        "fit_signature": fingerprint(fit_key),
                                        "reused_h0_fit": reused_h0_fit,
                                        "wall_seconds": time.monotonic() - before})
                counts["operations"] += 1
                if caught:
                    for item in caught:
                        warning_counts[f"{item.category.__name__}: {item.message}"] += 1
                    counts["warnings"] += len(caught)
                    del caught[:]
                if counts["operations"] % 100 == 0:
                    prediction_handle.flush(); timing_handle.flush(); score_handle.flush(); historical_handle.flush()
                    elapsed_record("operation_checkpoint", counts["operations"], len(operations))
            elapsed_record("operations_completed", counts["operations"], len(operations))
        prediction_handle.flush(); timing_handle.flush(); score_handle.flush(); historical_handle.flush()
        close_prediction_stream(raw_prediction, prediction_handle)
        timing_handle.close(); score_handle.close(); historical_handle.close()
        os.replace(partial_paths[0], output / "predictions.csv.gz")
        os.replace(partial_paths[1], output / "timing.csv")
        os.replace(partial_paths[2], output / "operation_scores.csv")
        os.replace(partial_paths[3], output / "historical_decodability.csv")
        receipt.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - started_wall,
                       cpu_seconds=time.process_time() - started_cpu, counts=counts,
                       warning_counts=dict(warning_counts),
                       outputs={name: sha256(output / name) for name in
                                ("predictions.csv.gz", "timing.csv", "operation_scores.csv",
                                 "historical_decodability.csv", "tuning_scores.csv", "selected_c.csv",
                                 "progress.jsonl")})
        atomic_json(output / "receipt.json", receipt)
        atomic_json(output / "COMPLETED.json", {"receipt_sha256": sha256(output / "receipt.json"),
                                                 "status": "completed"})
        return receipt
    except BaseException as error:
        try:
            close_prediction_stream(raw_prediction, prediction_handle)
        except Exception:
            pass
        for handle in (timing_handle, score_handle, historical_handle):
            try:
                handle.flush(); handle.close()
            except Exception:
                pass
        receipt.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - started_wall,
                       cpu_seconds=time.process_time() - started_cpu, counts=counts,
                       warning_counts=dict(warning_counts),
                       error={"type": type(error).__name__, "message": str(error)},
                       partial_outputs={path.name: sha256(path) for path in partial_paths if path.is_file()})
        atomic_json(output / "receipt.json", receipt)
        elapsed_record("failed")
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGXCPU, old_xcpu)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "run"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--index", required=True, type=Path)
        sub.add_argument("--out-dir", required=True, type=Path)
        if command == "run":
            sub.add_argument("--plan-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args.config, args.index, args.out_dir)
    else:
        result = run(args.config, args.index, args.plan_dir, args.out_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
