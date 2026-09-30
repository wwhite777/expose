#!/usr/bin/env python3
"""Run the prospective source-only-selected OpenBMI8 EEGNet successor."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import resource
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_review3_grid import atomic_json, fingerprint, read_gzip_json, seeded_rng, sha256


SCHEMA_RELATIVE = "result/day3/preparation_r001/schema.json"
EXPECTED_SCHEMA_SHA256 = "4cce8f8f8baf3107df60ee0df23ac650f85cc2639d156d1949f35633050401c3"
OUTPUT_SCHEMA = "review4-eegnet-source-bn-successor-v2"


def contained_file(root, relative, expected_sha):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("unsafe repository-relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe repository-relative path")
    path = root / pure
    resolved = path.resolve(strict=True)
    if root.resolve() not in resolved.parents or path.is_symlink() or not path.is_file():
        raise ValueError("input is not a contained regular file")
    if sha256(path) != expected_sha:
        raise ValueError("input hash differs: " + relative)
    return path


class LazyEpochStore:
    """Index identities eagerly; open epoch/label arrays one declared role at a time."""
    def __init__(self, root, index_path, index_sha256, schema_path):
        self.root = root
        if sha256(index_path) != index_sha256:
            raise ValueError("OpenBMI8 index hash differs")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if set(index) != {"dataset", "montage", "records"} or (
                index["dataset"], index["montage"]) != ("openbmi", "8ch"):
            raise ValueError("successor accepts OpenBMI8 only")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.schema = {(row["subject"], row["session"]): row
                       for row in schema.get("records", [])}
        fields = {"subject", "session", "role", "npz_path", "sha256", "trial_ids"}
        self.records, self.by_trial, self.epochs, self.outcomes = [], {}, {}, {}
        for number, raw in enumerate(index["records"]):
            if (set(raw) != fields or raw["role"] not in ("source", "development")
                    or raw["session"] not in (1, 2)
                    or len(raw["trial_ids"]) != 100
                    or len(set(raw["trial_ids"])) != 100):
                raise ValueError("OpenBMI8 index record differs")
            record = dict(raw, record_id=number)
            self.records.append(record)
            for offset, trial in enumerate(raw["trial_ids"]):
                if trial in self.by_trial:
                    raise ValueError("duplicate trial ID")
                self.by_trial[trial] = (record, offset)
        source = [row for row in self.records if row["role"] == "source"]
        development = [row for row in self.records if row["role"] == "development"]
        if (len(source) != 18 or any(row["session"] != 1 for row in source)
                or len(development) != 24
                or len({row["subject"] for row in development}) != 12
                or Counter(row["session"] for row in development) != {1: 12, 2: 12}):
            raise ValueError("OpenBMI8 role/session contract differs")
        self.loaded_roles = []

    def load_role(self, role):
        if role not in ("source", "development") or role in self.loaded_roles:
            raise ValueError("invalid or repeated epoch-role load")
        from expose.review2_controls import oas_covariances
        for record in self.records:
            if record["role"] != role:
                continue
            old = self.schema.get((record["subject"], record["session"]))
            if old is None or old.get("role") != role:
                raise ValueError("day3 role differs from OpenBMI8 index")
            cache = old.get("cache", {})
            cache_path = contained_file(
                self.root, cache.get("array_path"), cache.get("array_sha256"))
            covariance_path = contained_file(
                self.root, record["npz_path"], record["sha256"])
            with np.load(cache_path, allow_pickle=False) as archive:
                if not {"X", "y"}.issubset(set(archive.files)):
                    raise ValueError("epoch cache lacks X/y")
                x, y = archive["X"], archive["y"]
            with np.load(covariance_path, allow_pickle=False) as archive:
                if not {"cov", "y"}.issubset(set(archive.files)):
                    raise ValueError("OpenBMI8 covariance cache lacks cov/y")
                cov, reference_y = archive["cov"], archive["y"]
            if (x.shape != (100, 8, 750) or y.shape != (100,)
                    or not np.isfinite(x).all() or not np.array_equal(y, reference_y)
                    or Counter(y.astype(int).tolist()) != {0: 50, 1: 50}):
                raise ValueError("epoch/covariance identity differs")
            np.testing.assert_allclose(
                oas_covariances(x[:3]), cov[:3], rtol=1e-10, atol=1e-20)
            for offset, trial in enumerate(record["trial_ids"]):
                self.epochs[trial] = x[offset].astype(np.float32)
                self.outcomes[trial] = int(y[offset])
        self.loaded_roles.append(role)

    def require_loaded(self, trial_ids):
        missing = [trial for trial in trial_ids
                   if trial not in self.epochs or trial not in self.outcomes]
        if missing:
            raise ValueError("trial role has not been loaded: " + missing[0])

    def take(self, trial_ids):
        self.require_loaded(trial_ids)
        return np.stack([self.epochs[trial] for trial in trial_ids])

    def labels(self, trial_ids):
        self.require_loaded(trial_ids)
        return np.asarray([self.outcomes[trial] for trial in trial_ids], dtype=np.int64)

    def subjects(self, trial_ids):
        missing = [trial for trial in trial_ids if trial not in self.by_trial]
        if missing:
            raise ValueError("unknown trial ID")
        return np.asarray([self.by_trial[trial][0]["subject"] for trial in trial_ids],
                          dtype=int)


def balanced_ids(pool, store, wanted, rng):
    if wanted <= 0 or wanted % 60:
        raise ValueError("selection count must be a positive multiple of 60")
    labels = store.labels(pool)
    selected = []
    for label in (0, 1):
        available = [trial for trial, value in zip(pool, labels) if value == label]
        if len(available) < wanted // 2:
            raise ValueError("not enough balanced source-selection trials")
        selected.extend(rng.permutation(available).tolist()[:wanted // 2])
    if len(selected) != wanted or Counter(store.labels(selected).tolist()) != {
            0: wanted // 2, 1: wanted // 2}:
        raise RuntimeError("balanced source sampling failed")
    return selected


def selection_count(final_source_n, available_selection_n, multiple=60):
    usable = multiple * (available_selection_n // multiple)
    chosen = min(final_source_n, usable)
    if chosen <= 0 or chosen % multiple:
        raise ValueError("no valid equal-batch selection count")
    return chosen


def applicable_operations(plan, config):
    selected = [row for row in plan
                if row["method"] == "plain_ts"
                and row["study_id"] == config["study_id"]
                and row["dataset"] == "openbmi" and row["montage"] == "8ch"
                and row["mode"] == "frozen_source" and row["donor_count"] == "all"
                and row["draw"] in config["draws"]
                and row["source_size"] in config["source_sizes"]
                and row["h_total"] in config["history_totals"]]
    if len(selected) != config["expected_scores"]:
        raise ValueError("successor planned score count differs")
    cells = Counter((row["target"], row["draw"], row["source_size"], row["h_total"])
                    for row in selected)
    if (len(cells) != config["expected_scores"] or set(cells.values()) != {1}
            or len({row["target"] for row in selected}) != 12):
        raise ValueError("successor target-cell grid differs")
    return selected


def source_specifications(operations, memberships):
    specs = {}
    for operation in operations:
        refs = operation["memberships"]
        source_ids = memberships[refs["source"]]
        key = fingerprint([refs["source"], operation["draw"]])
        value = {
            "source_key": key,
            "source_membership": refs["source"],
            "source_ids": source_ids,
            "draw": operation["draw"],
            "source_size": operation["source_size"],
        }
        if key in specs and specs[key] != value:
            raise ValueError("source checkpoint specification differs across targets")
        specs[key] = value
    if (len(specs) != 6
            or Counter((value["draw"], value["source_size"])
                       for value in specs.values()) != {
                           (draw, size): 1 for draw in (0, 1, 2)
                           for size in (300, 1800)}):
        raise ValueError("expected three draws by two source sizes")
    return specs


def validate_config(config):
    required = {
        "frozen_files", "seed", "cpu_seconds", "wall_seconds", "memory_gib",
        "index_path", "index_sha256", "operations_path", "memberships_path",
        "study_id", "dataset", "montage", "mode", "draws", "source_sizes",
        "history_totals", "expected_scores", "minimum_source_epochs",
        "maximum_source_epochs", "source_patience", "validation_people",
        "selection_multiple", "fine_epochs",
    }
    if set(config) != required:
        raise ValueError("successor configuration fields differ")
    if (config["seed"] != 2026092410 or config["cpu_seconds"] != 1800
            or config["wall_seconds"] != 3600 or config["memory_gib"] != 16
            or (config["dataset"], config["montage"], config["mode"])
            != ("openbmi", "8ch", "frozen_source")
            or config["draws"] != [0, 1, 2]
            or config["source_sizes"] != [300, 1800]
            or config["history_totals"] != [0, 60]
            or config["expected_scores"] != 144
            or (config["minimum_source_epochs"], config["maximum_source_epochs"],
                config["source_patience"], config["validation_people"],
                config["selection_multiple"], config["fine_epochs"])
            != (15, 50, 10, 5, 60, 20)):
        raise ValueError("successor configuration values differ")
    return config


def run(config_path, output):
    import torch
    from expose.review4_eegnet import (
        adapt_source, normalized_tensor, probabilities, probability_summary,
        select_and_refit_source,
    )

    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    for relative, expected in config["frozen_files"].items():
        contained_file(ROOT, relative, expected)
    if output.exists():
        raise FileExistsError("successor output already exists")
    output.mkdir(parents=True)
    start_wall, start_cpu = time.monotonic(), time.process_time()
    receipt = {
        "status": "running",
        "schema": OUTPUT_SCHEMA,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256(config_path),
        "source_fits": 0,
        "adaptations": 0,
        "scores": 0,
        "predictions": 0,
        "numeric_threads": 1,
        "cuda_used": False,
        "confirmation_data_accessed": False,
        "development_epochs_accessed_before_source_stage_freeze": False,
        "checkpoints": [],
    }
    atomic_json(output / "receipt.json", receipt)
    try:
        resource.setrlimit(resource.RLIMIT_AS,
                           (config["memory_gib"] * 1024**3,
                            config["memory_gib"] * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU,
                           (config["cpu_seconds"], config["cpu_seconds"] + 10))
        def expired(signum, _frame):
            raise TimeoutError("successor resource signal " + str(signum))
        signal.signal(signal.SIGALRM, expired)
        signal.signal(signal.SIGXCPU, expired)
        signal.alarm(config["wall_seconds"])

        schema_path = ROOT / SCHEMA_RELATIVE
        if sha256(schema_path) != EXPECTED_SCHEMA_SHA256:
            raise ValueError("day3 epoch schema differs")
        store = LazyEpochStore(
            ROOT, ROOT / config["index_path"], config["index_sha256"], schema_path)
        plan = read_gzip_json(ROOT / config["operations_path"])
        memberships = read_gzip_json(ROOT / config["memberships_path"])
        operations = applicable_operations(plan, config)
        specifications = source_specifications(operations, memberships)

        # Fit and persist all six source checkpoints before opening any
        # development epoch or outcome array.
        store.load_role("source")
        checkpoints = {}
        source_outputs = {}
        for source_key, spec in sorted(specifications.items()):
            source_ids = spec["source_ids"]
            if len(source_ids) != spec["source_size"]:
                raise ValueError("final source membership size differs")
            source_people = sorted(set(store.subjects(source_ids).tolist()))
            if len(source_people) != 18:
                raise ValueError("final source membership must span 18 SOURCE people")
            source_pool = [trial for record in store.records
                           if record["role"] == "source" and record["session"] == 1
                           and record["subject"] in source_people
                           for trial in record["trial_ids"]]
            rng = seeded_rng(
                config["seed"], "review4_source_validation", spec["draw"],
                spec["source_size"], spec["source_membership"])
            validation_people = sorted(
                rng.permutation(source_people).tolist()[:config["validation_people"]])
            validation_ids = [trial for trial in source_pool
                              if int(store.subjects([trial])[0]) in validation_people]
            selection_pool = [trial for trial in source_pool
                              if int(store.subjects([trial])[0]) not in validation_people]
            if len(validation_ids) != 500 or len(selection_pool) != 1300:
                raise ValueError("five-person source holdout size differs")
            wanted = selection_count(
                len(source_ids), len(selection_pool), config["selection_multiple"])
            selected_ids = balanced_ids(selection_pool, store, wanted, rng)
            unused_ids = sorted(set(selection_pool) - set(selected_ids))
            if (set(selected_ids) & set(validation_ids)
                    or len(unused_ids) != len(selection_pool) - wanted):
                raise ValueError("source selection partition overlaps")
            fit_seed = int(rng.integers(1, 2**30))
            model, normalizer, fitting = select_and_refit_source(
                store.take(selected_ids), store.labels(selected_ids),
                store.take(validation_ids), store.labels(validation_ids),
                store.take(source_ids), store.labels(source_ids), fit_seed,
                minimum_epochs=config["minimum_source_epochs"],
                maximum_epochs=config["maximum_source_epochs"],
                patience=config["source_patience"])
            source_probability = probabilities(
                model, normalized_tensor(store.take(source_ids), normalizer))
            source_metrics = probability_summary(
                source_probability, store.labels(source_ids))
            checkpoint_path = output / f"source_{source_key}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "normalizer_mean": normalizer[0],
                "normalizer_scale": normalizer[1],
                "source_membership": spec["source_membership"],
                "recipe": OUTPUT_SCHEMA,
            }, checkpoint_path)
            fitting.update({
                "source_key": source_key,
                "draw": spec["draw"],
                "source_size": spec["source_size"],
                "source_membership": spec["source_membership"],
                "source_ids": source_ids,
                "selection_ids": selected_ids,
                "validation_ids": validation_ids,
                "unused_selection_pool_ids": unused_ids,
                "validation_people": validation_people,
                "selection_people": sorted(set(store.subjects(selected_ids).tolist())),
                "source_selection_overlap_n": len(set(source_ids) & set(selected_ids)),
                "source_validation_overlap_n": len(set(source_ids) & set(validation_ids)),
                "selection_validation_overlap_n": 0,
                "final_checkpoint_validation_status": "not_held_out_from_final_refit",
                "source_resubstitution_metrics": source_metrics,
                "checkpoint_sha256": sha256(checkpoint_path),
            })
            metadata_path = output / f"source_{source_key}.json"
            atomic_json(metadata_path, fitting)
            checkpoints[source_key] = (model, normalizer, spec["source_membership"])
            source_outputs[checkpoint_path.name] = sha256(checkpoint_path)
            source_outputs[metadata_path.name] = sha256(metadata_path)
            receipt["source_fits"] += 2
            receipt["checkpoints"].append(checkpoint_path.name)
            receipt.update(cpu_seconds=time.process_time() - start_cpu,
                           wall_seconds=time.monotonic() - start_wall)
            atomic_json(output / "receipt.json", receipt)

        source_stage = {
            "status": "complete_before_development_epoch_access",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "source_only": True,
            "source_checkpoints": source_outputs,
            "development_epochs_accessed": False,
            "confirmation_data_accessed": False,
        }
        atomic_json(output / "SOURCE_STAGE_COMPLETED.json", source_stage)

        store.load_role("development")
        receipt["development_epoch_access_started_utc"] = datetime.now(
            timezone.utc).isoformat()
        score_rows = []
        prediction_path = output / "predictions.csv.gz"
        prediction_fields = [
            "operation_id", "target", "draw", "source_size", "h_total",
            "trial_id", "y_true", "y_pred", "p_right", "p_left",
        ]
        with gzip.open(prediction_path, "xt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=prediction_fields)
            writer.writeheader()
            for operation in operations:
                refs = operation["memberships"]
                source_ids, history_ids, evaluation_ids = (
                    memberships[refs[name]] for name in ("source", "personal", "evaluation"))
                source_key = fingerprint([refs["source"], operation["draw"]])
                if source_key not in checkpoints:
                    raise ValueError("operation lacks frozen successor source checkpoint")
                model, normalizer, source_membership = checkpoints[source_key]
                if source_membership != refs["source"] or source_ids != specifications[source_key]["source_ids"]:
                    raise ValueError("operation source membership differs")
                if operation["h_total"] == 0:
                    if history_ids:
                        raise ValueError("h0 operation has history trials")
                    adapted, fine_losses = model, []
                elif operation["h_total"] == 60:
                    if len(history_ids) != 60:
                        raise ValueError("h60 operation history count differs")
                    fine_seed = int(seeded_rng(
                        config["seed"], "review4_fine", source_key,
                        operation["target"]).integers(1, 2**30))
                    adapted, fine_losses = adapt_source(
                        model, normalizer, store.take(history_ids),
                        store.labels(history_ids), fine_seed,
                        epochs=config["fine_epochs"])
                    receipt["adaptations"] += 1
                else:
                    raise ValueError("unexpected history amount")
                if set(history_ids) & set(evaluation_ids):
                    raise ValueError("personal history overlaps evaluation")
                probability = probabilities(
                    adapted, normalized_tensor(store.take(evaluation_ids), normalizer))
                truth = store.labels(evaluation_ids)
                metrics = probability_summary(probability, truth)
                predicted = probability.argmax(axis=1)
                operation_id = "eegnet_r4__" + operation["operation_id"]
                fine_seed_value = (None if operation["h_total"] == 0 else fine_seed)
                score_rows.append({
                    "operation_id": operation_id,
                    "method": "eegnet_r4",
                    "target": operation["target"],
                    "draw": operation["draw"],
                    "source_size": operation["source_size"],
                    "h_total": operation["h_total"],
                    **metrics,
                    "source_checkpoint": source_key,
                    "fine_seed": fine_seed_value,
                    "fine_losses": json.dumps(fine_losses),
                    "source_membership": refs["source"],
                    "history_membership": refs["personal"],
                })
                for index, trial in enumerate(evaluation_ids):
                    writer.writerow({
                        "operation_id": operation_id,
                        "target": operation["target"],
                        "draw": operation["draw"],
                        "source_size": operation["source_size"],
                        "h_total": operation["h_total"],
                        "trial_id": trial,
                        "y_true": int(truth[index]),
                        "y_pred": int(predicted[index]),
                        "p_right": float(probability[index, 0]),
                        "p_left": float(probability[index, 1]),
                    })
                receipt["scores"] += 1
                receipt["predictions"] += len(evaluation_ids)
                if receipt["scores"] % 12 == 0:
                    receipt.update(cpu_seconds=time.process_time() - start_cpu,
                                   wall_seconds=time.monotonic() - start_wall)
                    atomic_json(output / "receipt.json", receipt)
                    print(json.dumps({name: receipt[name] for name in (
                        "scores", "source_fits", "adaptations", "cpu_seconds")}),
                        flush=True)

        score_path = output / "scores.csv"
        with score_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
            writer.writeheader()
            writer.writerows(score_rows)
        if (len(score_rows) != 144 or receipt["scores"] != 144
                or receipt["predictions"] != 14400 or receipt["adaptations"] != 72):
            raise RuntimeError("successor output cardinality differs")
        outputs = {
            **source_outputs,
            "SOURCE_STAGE_COMPLETED.json": sha256(output / "SOURCE_STAGE_COMPLETED.json"),
            "predictions.csv.gz": sha256(prediction_path),
            "scores.csv": sha256(score_path),
        }
        receipt.update({
            "status": "completed",
            "outputs": outputs,
            "output_binding_schema": OUTPUT_SCHEMA,
            "development_epochs_accessed": True,
            "target_recipe_search_performed": False,
        })
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        traceback.print_exc()
    finally:
        signal.alarm(0)
        receipt.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            cpu_seconds=time.process_time() - start_cpu,
            wall_seconds=time.monotonic() - start_wall,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(output / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise SystemExit(1)
    atomic_json(output / "COMPLETED.json", {
        "receipt_sha256": sha256(output / "receipt.json"),
        "predictions_sha256": sha256(output / "predictions.csv.gz"),
        "scores_sha256": sha256(output / "scores.csv"),
        "source_stage_sha256": sha256(output / "SOURCE_STAGE_COMPLETED.json"),
        "outputs": receipt["outputs"],
        "output_binding_schema": OUTPUT_SCHEMA,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config.resolve(), arguments.run_dir.resolve())
