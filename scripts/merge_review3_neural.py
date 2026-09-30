#!/usr/bin/env python3
"""Merge three independently completed Review-3 EEGNet draw partitions."""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import resource
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "review3-neural-partition-merge-v1"
CONFIG_FIELDS = {"frozen_files", "seed", "cpu_seconds", "wall_seconds", "index_path",
                 "index_sha256", "operations_path", "memberships_path", "study_id",
                 "dataset", "montage", "mode", "draws", "source_sizes", "history_totals",
                 "expected_scores", "max_source_epochs", "source_patience", "fine_epochs"}
PREDICTION_FIELDS = ["operation_id", "target", "draw", "source_size", "h_total", "trial_id",
                     "y_true", "y_pred", "p_right", "p_left"]
SCORE_FIELDS = ["operation_id", "method", "target", "draw", "source_size", "h_total",
                "balanced_accuracy", "source_checkpoint", "fine_seed", "fine_losses",
                "source_membership", "history_membership"]
SOURCE_SIZES = ("all",)
HISTORY_TOTALS = (0, 4, 10, 20, 40, 60, 100)
PARTITION_SCORES = 63
EVALUATION_TRIALS = 144


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("paths must be repository-relative strings")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe repository-relative path")
    path = (ROOT / pure).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("path escapes repository")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def validate_manifest(manifest):
    if set(manifest) != {"schema", "parent_config", "parts"} or manifest["schema"] != SCHEMA:
        raise ValueError("invalid neural partition-merge manifest")
    parent = manifest["parent_config"]
    if set(parent) != {"path", "sha256"} or not isinstance(parent["sha256"], str):
        raise ValueError("invalid parent neural configuration binding")
    parts = manifest["parts"]
    if not isinstance(parts, list) or len(parts) != 3:
        raise ValueError("exactly three neural partitions are required")
    seen = set()
    for item in parts:
        if set(item) != {"draw", "config_path", "config_sha256", "directory"}:
            raise ValueError("invalid neural partition entry")
        if item["draw"] not in (0, 1, 2) or item["draw"] in seen:
            raise ValueError("partition draws must be the exact complement 0,1,2")
        seen.add(item["draw"])
        if not isinstance(item["config_sha256"], str) or len(item["config_sha256"]) != 64:
            raise ValueError("invalid partition config hash")
        safe_path(item["config_path"]); safe_path(item["directory"])
    if seen != {0, 1, 2}:
        raise ValueError("partition draws must be the exact complement 0,1,2")
    safe_path(parent["path"])
    return manifest


def validate_parent_config(binding):
    parent_path = safe_path(binding["path"])
    if not parent_path.is_file() or sha256(parent_path) != binding["sha256"]:
        raise ValueError("parent neural configuration hash differs")
    config = load_json(parent_path)
    if set(config) != CONFIG_FIELDS:
        raise ValueError("parent neural config schema differs")
    if (config["draws"] != [0, 1, 2] or config["expected_scores"] != 189
            or [str(value) for value in config["source_sizes"]] != list(SOURCE_SIZES)
            or config["history_totals"] != list(HISTORY_TOTALS)
            or config["study_id"] != "bnci_main" or config["dataset"] != "bnci2014"
            or config["montage"] != "22ch" or config["mode"] != "loso"
            or (config["max_source_epochs"], config["source_patience"], config["fine_epochs"])
               != (50, 10, 20)):
        raise ValueError("parent config differs from the frozen 3-draw BNCI neural design")
    if not isinstance(config["frozen_files"], dict) or not config["frozen_files"]:
        raise ValueError("parent config must bind frozen runner, plan, and data files")
    for relative, expected in config["frozen_files"].items():
        frozen_path = safe_path(relative)
        if not frozen_path.is_file() or sha256(frozen_path) != expected:
            raise ValueError("parent frozen-file binding differs: " + relative)
    return parent_path, config


def validate_part(item, parent):
    config_path = safe_path(item["config_path"])
    directory = safe_path(item["directory"])
    if (not config_path.is_file() or sha256(config_path) != item["config_sha256"]
            or not directory.is_dir() or directory.is_symlink()):
        raise ValueError("partition config or output directory binding differs")
    config = load_json(config_path)
    if set(config) != CONFIG_FIELDS:
        raise ValueError("partition neural config schema differs")
    expected = dict(parent, draws=[item["draw"]], expected_scores=PARTITION_SCORES)
    if config != expected:
        raise ValueError("partition config is not an exact singleton complement of its parent")
    receipt_path, completed_path = directory / "receipt.json", directory / "COMPLETED.json"
    if not receipt_path.is_file() or not completed_path.is_file():
        raise ValueError("partition completion records are missing")
    receipt, completed = load_json(receipt_path), load_json(completed_path)
    prediction_path, score_path = directory / "predictions.csv.gz", directory / "scores.csv"
    output_bindings = receipt.get("outputs")
    if (receipt.get("status") != "completed" or receipt.get("config_sha256") != sha256(config_path)
            or completed.get("receipt_sha256") != sha256(receipt_path)
            or completed.get("predictions_sha256") != sha256(prediction_path)
            or completed.get("scores_sha256") != sha256(score_path)
            or receipt.get("output_binding_schema") != "review3-neural-output-bindings-v2"
            or completed.get("output_binding_schema") != "review3-neural-output-bindings-v2"
            or not isinstance(output_bindings, dict) or completed.get("outputs") != output_bindings
            or output_bindings.get("predictions.csv.gz") != sha256(prediction_path)
            or output_bindings.get("scores.csv") != sha256(score_path)
            or receipt.get("confirmation_data_accessed") is not False
            or receipt.get("cuda_used") is not False
            or receipt.get("numeric_threads") != 1 or receipt.get("interop_threads") != 1):
        raise ValueError("partition completion or output hash binding differs")
    if (receipt.get("scores") != PARTITION_SCORES
            or receipt.get("predictions") != PARTITION_SCORES * EVALUATION_TRIALS
            or receipt.get("source_fits") != 18 or receipt.get("adaptations") != 54):
        raise ValueError("partition receipt counts differ from the frozen design")
    checkpoints = receipt.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 9 or len(set(checkpoints)) != 9:
        raise ValueError("LOSO partition must retain nine target-specific source checkpoints")
    references = []
    for name in checkpoints:
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("source_") or not name.endswith(".pt"):
            raise ValueError("unsafe checkpoint reference")
        checkpoint, metadata = directory / name, directory / (name[:-3] + ".json")
        if not checkpoint.is_file() or not metadata.is_file():
            raise ValueError("checkpoint or its fit metadata is missing")
        fitted = load_json(metadata)
        if (fitted.get("checkpoint_sha256") != sha256(checkpoint)
                or output_bindings.get(checkpoint.name) != sha256(checkpoint)
                or output_bindings.get(metadata.name) != sha256(metadata)):
            raise ValueError("checkpoint or sidecar differs from completed output bindings")
        references.append({"checkpoint": name, "checkpoint_sha256": sha256(checkpoint),
                           "fit_metadata": metadata.name, "fit_metadata_sha256": sha256(metadata),
                           "source_membership": None})
    expected_outputs = {"predictions.csv.gz", "scores.csv"}
    expected_outputs.update(item["checkpoint"] for item in references)
    expected_outputs.update(item["fit_metadata"] for item in references)
    if set(output_bindings) != expected_outputs:
        raise ValueError("partition completed output inventory is not exact")
    return {"draw": item["draw"], "directory": directory, "config": config,
            "config_path": config_path, "receipt": receipt, "completed": completed,
            "receipt_path": receipt_path, "completed_path": completed_path,
            "prediction_path": prediction_path, "score_path": score_path,
            "checkpoint_references": references}


def read_partition_scores(part):
    with part["score_path"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SCORE_FIELDS:
            raise ValueError("partition scores.csv schema differs")
        rows = list(reader)
    if len(rows) != PARTITION_SCORES:
        raise ValueError("partition score row count differs")
    coordinates, operation_ids = set(), set()
    checkpoint_memberships, checkpoint_targets, checkpoint_counts = {}, {}, defaultdict(int)
    for row in rows:
        if row["operation_id"] in operation_ids or row["method"] != "eegnet":
            raise ValueError("duplicate or non-EEGNet partition operation")
        operation_ids.add(row["operation_id"])
        draw, target, history = int(row["draw"]), int(row["target"]), int(row["h_total"])
        source = str(row["source_size"])
        if draw != part["draw"] or source not in SOURCE_SIZES or history not in HISTORY_TOTALS:
            raise ValueError("partition score coordinate differs from its singleton config")
        score = float(row["balanced_accuracy"])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("partition balanced accuracy is invalid")
        coordinates.add((target, source, history))
        valid_checkpoints = {Path(item["checkpoint"]).stem[len("source_"):]
                             for item in part["checkpoint_references"]}
        if row["source_checkpoint"] not in valid_checkpoints:
            raise ValueError("score references an unbound source checkpoint")
        previous = checkpoint_memberships.setdefault(row["source_checkpoint"], row["source_membership"])
        if previous != row["source_membership"]:
            raise ValueError("one source checkpoint refers to multiple source memberships")
        previous_target = checkpoint_targets.setdefault(row["source_checkpoint"], target)
        if previous_target != target:
            raise ValueError("one LOSO source checkpoint is shared across different targets")
        checkpoint_counts[row["source_checkpoint"]] += 1
    targets = sorted({coordinate[0] for coordinate in coordinates})
    required = {(target, source, history) for target in targets
                for source in SOURCE_SIZES for history in HISTORY_TOTALS}
    if len(targets) != 9 or coordinates != required:
        raise ValueError("partition must contain nine targets and the complete all-source x 7-history grid")
    if (set(checkpoint_memberships) != valid_checkpoints
            or len(set(checkpoint_memberships.values())) != 9
            or set(checkpoint_counts.values()) != {7}
            or set(checkpoint_targets.values()) != set(targets)):
        raise ValueError("LOSO checkpoint identities must be one distinct source pool per target and seven histories")
    for reference in part["checkpoint_references"]:
        key = Path(reference["checkpoint"]).stem[len("source_"):]
        reference["source_membership"] = checkpoint_memberships[key]
    return rows, operation_ids, targets


def stream_predictions(part, writer, global_trials):
    per_operation = {}
    with gzip.open(part["prediction_path"], "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise ValueError("partition prediction schema differs")
        count = 0
        for row in reader:
            operation_id, trial_id = row["operation_id"], row["trial_id"]
            if (operation_id, trial_id) in global_trials:
                raise ValueError("duplicate operation/trial prediction across partitions")
            global_trials.add((operation_id, trial_id))
            if int(row["draw"]) != part["draw"]:
                raise ValueError("prediction draw differs from its partition")
            truth, predicted = int(row["y_true"]), int(row["y_pred"])
            probabilities = np.asarray([float(row["p_right"]), float(row["p_left"])])
            if (truth not in (0, 1) or predicted not in (0, 1) or not np.isfinite(probabilities).all()
                    or np.any(probabilities < 0) or np.any(probabilities > 1)
                    or abs(probabilities.sum() - 1) > 1e-6
                    or probabilities.max() - probabilities[predicted] > 1e-12):
                raise ValueError("partition prediction values are invalid")
            per_operation.setdefault(operation_id, set()).add(trial_id)
            writer.writerow(row); count += 1
    if count != PARTITION_SCORES * EVALUATION_TRIALS:
        raise ValueError("partition prediction row count differs")
    if set(per_operation) != part["operation_ids"]:
        raise ValueError("prediction and score operation inventories differ")
    if any(len(values) != EVALUATION_TRIALS for values in per_operation.values()):
        raise ValueError("each neural operation must contain exactly 144 distinct trials")
    return count


def merge(manifest_path, output):
    manifest = validate_manifest(load_json(manifest_path))
    output = Path(output)
    if output.exists():
        raise FileExistsError("aggregate neural output directory must be new")
    parent_path, parent = validate_parent_config(manifest["parent_config"])
    parts = [validate_part(item, parent) for item in sorted(manifest["parts"], key=lambda value: value["draw"])]
    output.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    global_operations, expected_targets, score_count = set(), None, 0
    for part in parts:
        scores, operations, targets = read_partition_scores(part)
        if global_operations & operations:
            raise ValueError("duplicate operation ID across neural partitions")
        global_operations.update(operations)
        if expected_targets is None:
            expected_targets = targets
        elif targets != expected_targets:
            raise ValueError("neural partition target sets differ")
        part["operation_ids"] = operations
        part["score_rows"] = scores
        score_count += len(scores)
    prediction_path = output / "predictions.csv.gz"
    temporary = Path(str(prediction_path) + ".tmp")
    global_trials, prediction_count = set(), 0
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
                writer.writeheader()
                for part in parts:
                    prediction_count += stream_predictions(part, writer, global_trials)
    os.replace(temporary, prediction_path)
    score_path = output / "scores.csv"
    with score_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        for part in parts:
            writer.writerows(part["score_rows"])
    provenance = {
        "type": "partition_merge", "weights_copied": False,
        "parent_config": {"path": manifest["parent_config"]["path"],
                          "sha256": sha256(parent_path)},
        "parts": [{
            "draw": part["draw"], "directory": str(part["directory"].relative_to(ROOT)),
            "config_path": str(part["config_path"].relative_to(ROOT)),
            "config_sha256": sha256(part["config_path"]),
            "receipt_sha256": sha256(part["receipt_path"]),
            "completed_sha256": sha256(part["completed_path"]),
            "receipt": part["receipt"], "completed": part["completed"],
            "checkpoint_references": part["checkpoint_references"],
        } for part in parts],
    }
    provenance_path = output / "PARTITION_PROVENANCE.json"
    atomic_json(provenance_path, provenance)
    receipt = {
        "status": "completed", "type": "partition_merge",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest_path), "merger_sha256": sha256(__file__),
        "parent_config_sha256": sha256(parent_path), "draws": [0, 1, 2],
        "partition_count": 3, "weights_copied": False,
        "confirmation_data_accessed": False, "cuda_used": False,
        "counts": {"scores": score_count, "predictions": prediction_count,
                   "targets": len(expected_targets), "checkpoint_references": 27},
        "summed_partition_cpu_seconds": float(sum(part["receipt"]["cpu_seconds"] for part in parts)),
        "summed_partition_wall_seconds": float(sum(part["receipt"]["wall_seconds"] for part in parts)),
        "merge_cpu_seconds": time.process_time() - started_cpu,
        "merge_wall_seconds": time.monotonic() - started_wall,
        "merge_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "outputs": {"predictions.csv.gz": sha256(prediction_path),
                    "scores.csv": sha256(score_path),
                    "PARTITION_PROVENANCE.json": sha256(provenance_path)},
    }
    receipt_path = output / "receipt.json"
    atomic_json(receipt_path, receipt)
    atomic_json(output / "COMPLETED.json", {
        "status": "completed", "type": "partition_merge",
        "receipt_sha256": sha256(receipt_path),
        "predictions_sha256": sha256(prediction_path), "scores_sha256": sha256(score_path)})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.parts, args.out_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
