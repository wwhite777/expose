#!/usr/bin/env python3
"""Analyze completed secondary fixed-C target-only Review-3 outputs."""
import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

# Set before NumPy imports an underlying BLAS implementation.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_review3 import effect

SCHEMA = "review3-target-only-analysis-inputs-v1"
CORE_FIELDS = {"dataset_variant", "operation_id", "study_id", "method", "target", "draw",
               "source_size", "source_n", "donor_count", "h_total", "evaluation_n", "balanced_accuracy"}
PREDICTION_FIELDS = {"operation_id", "anchor_operation_id", "study_id", "draw", "target", "source_size",
                     "h_total", "C", "trial_id", "y_true", "y_pred", "p0", "p1", "personal_membership", "evaluation_membership"}
SCORE_FIELDS = {"operation_id", "anchor_operation_id", "study_id", "draw", "target", "source_size", "h_total",
                "C", "balanced_accuracy", "evaluation_n", "personal_membership", "evaluation_membership"}
TARGETS = {"openbmi8": 12, "bnci": 9}
FULL = {"openbmi8": "1800", "bnci": "all"}
HISTORY = (4, 10, 20, 40, 60, 100)
C_FIXED = 1.0
MAIN_CORE_STUDY = {"openbmi8": "openbmi_main", "bnci": "bnci_main"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, fields, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        values = list(csv.DictReader(handle))
    if not values or set(values[0]) != fields or any(set(row) != fields for row in values):
        raise ValueError("CSV schema differs: " + str(path))
    return values


def load_inputs(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(value) != {"schema", "core_analysis", "target_only"} or value["schema"] != SCHEMA:
        raise ValueError("invalid target-only analysis input schema")
    core = value["core_analysis"]
    if set(core) != {"directory", "summary_sha256", "reconstructed_scores_sha256"}:
        raise ValueError("invalid core analysis binding")
    if not isinstance(value["target_only"], list) or {item.get("label") for item in value["target_only"]} != set(TARGETS):
        raise ValueError("target-only inputs must contain exactly openbmi8 and bnci")
    if any(set(item) != {"label", "directory"} for item in value["target_only"]):
        raise ValueError("invalid target-only input item")
    return value


def complete_target_directory(path):
    path = Path(path)
    names = ("receipt.json", "COMPLETED.json", "PLAN_BINDING.json", "predictions.csv.gz", "operation_scores.csv")
    if not path.is_dir() or path.is_symlink() or any(not (path / name).is_file() or (path / name).is_symlink() for name in names):
        raise ValueError("target-only directory lacks regular required files")
    receipt, complete = (json.loads((path / name).read_text(encoding="utf-8")) for name in ("receipt.json", "COMPLETED.json"))
    if (receipt.get("schema") != "review3-target-only-v1" or receipt.get("status") != "completed"
            or complete != {"status": "completed", "receipt_sha256": sha(path / "receipt.json")}
            or set(receipt.get("outputs", {})) != {"predictions.csv.gz", "operation_scores.csv", "PLAN_BINDING.json"}
            or any(receipt["outputs"][name] != sha(path / name) for name in receipt["outputs"])):
        raise ValueError("target-only completion/hash guard failed")
    binding = json.loads((path / "PLAN_BINDING.json").read_text(encoding="utf-8"))
    if (binding.get("schema") != "review3-target-only-v1" or binding.get("source_covariances_used_for_fitting") is not False
            or binding.get("source_array_validation_note") != "DatasetStore validates all indexed arrays; source covariances are never passed to target-only fitting."
            or not isinstance(binding.get("operations"), list)):
        raise ValueError("target-only plan binding differs")
    return receipt, binding


def reconstructed_target(path, label):
    receipt, binding = complete_target_directory(path)
    predictions = read_csv(Path(path) / "predictions.csv.gz", PREDICTION_FIELDS, compressed=True)
    scores = read_csv(Path(path) / "operation_scores.csv", SCORE_FIELDS)
    binding_rows = {row["operation_id"]: row for row in binding["operations"]}
    if len(binding_rows) != len(binding["operations"]):
        raise ValueError("duplicate target-only plan binding operation")
    grouped = defaultdict(list)
    for row in predictions:
        operation = row["operation_id"]
        if operation not in binding_rows or row["anchor_operation_id"] != binding_rows[operation]["anchor_operation_id"]:
            raise ValueError("prediction lacks exact target-only plan binding")
        p0, p1 = float(row["p0"]), float(row["p1"])
        truth, pred = int(row["y_true"]), int(row["y_pred"])
        if (not np.isfinite([p0, p1]).all() or not (0 <= p0 <= 1 and 0 <= p1 <= 1)
                or abs(p0 + p1 - 1) > 1e-10 or truth not in (0, 1) or pred not in (0, 1)
                or max(p0, p1) - (p0 if pred == 0 else p1) > 1e-12):
            raise ValueError("invalid saved target-only prediction")
        grouped[operation].append(row)
    score_map = {row["operation_id"]: row for row in scores}
    if len(score_map) != len(scores) or set(score_map) != set(binding_rows) or set(grouped) != set(binding_rows):
        raise ValueError("target-only score/prediction inventory differs from plan binding")
    output = {}
    for operation, bound in binding_rows.items():
        rows, score = grouped[operation], score_map[operation]
        trials = [row["trial_id"] for row in rows]
        if (float(score["C"]) != C_FIXED or int(score["h_total"]) not in HISTORY
                or len(trials) != len(set(trials)) or len(rows) != int(score["evaluation_n"])
                or score["personal_membership"] != bound["personal_membership"]
                or score["evaluation_membership"] != bound["evaluation_membership"]):
            raise ValueError("target-only saved membership/evaluation differs")
        truth = np.asarray([int(row["y_true"]) for row in rows]); pred = np.asarray([int(row["y_pred"]) for row in rows])
        if set(truth.tolist()) != {0, 1}:
            raise ValueError("target-only evaluation lacks a binary class")
        ba = float((np.mean(pred[truth == 0] == 0) + np.mean(pred[truth == 1] == 1)) / 2)
        if abs(ba - float(score["balanced_accuracy"])) > 1e-12:
            raise ValueError("target-only saved score differs from reconstructed balanced accuracy")
        key = (int(score["target"]), int(score["draw"]), int(score["h_total"]))
        if key in output or key != (int(bound["target"]), int(bound["draw"]), int(bound["h_total"])):
            raise ValueError("duplicate or mismatched target-only coordinate")
        output[key] = {"ba": ba, "anchor_operation_id": score["anchor_operation_id"],
                       "personal_membership": score["personal_membership"], "evaluation_membership": score["evaluation_membership"]}
    targets = sorted({key[0] for key in output})
    expected = {(target, draw, h) for target in targets for draw in range(10) for h in HISTORY}
    if len(targets) != TARGETS[label] or set(output) != expected:
        raise ValueError("target-only inventory differs from ten draws by six h doses")
    return output, {"receipt_sha256": sha(Path(path) / "receipt.json"), "binding_sha256": sha(Path(path) / "PLAN_BINDING.json"),
                    "predictions_sha256": sha(Path(path) / "predictions.csv.gz"), "scores_sha256": sha(Path(path) / "operation_scores.csv")}


def core_rows(core):
    directory = Path(core["directory"])
    summary, scores = directory / "summary.json", directory / "reconstructed_scores.csv"
    if sha(summary) != core["summary_sha256"] or sha(scores) != core["reconstructed_scores_sha256"]:
        raise ValueError("core analysis differs from explicit hash binding")
    values = read_csv(scores, CORE_FIELDS)
    lookup = {}
    for row in values:
        # The core analysis also retains source-composition/diversity outputs.
        # They are not a second estimate of this target-only main comparison.
        variant = row["dataset_variant"]
        if (variant not in MAIN_CORE_STUDY or row["study_id"] != MAIN_CORE_STUDY[variant]
                or row["donor_count"] != "all"):
            continue
        key = (row["dataset_variant"], row["method"], str(row["source_size"]), int(row["target"]), int(row["draw"]), int(row["h_total"]))
        if key in lookup: raise ValueError("duplicate core coordinate")
        lookup[key] = float(row["balanced_accuracy"])
    return lookup, {"summary_sha256": sha(summary), "reconstructed_scores_sha256": sha(scores)}


def write_csv(path, fields, values):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists(): raise FileExistsError("analysis output must be new")
    started, cpu = time.monotonic(), time.process_time()
    inputs = load_inputs(args.inputs); core, verification = core_rows(inputs["core_analysis"])
    args.out_dir.mkdir(parents=True)
    participants, contrasts, curves = [], [], []
    for item in sorted(inputs["target_only"], key=lambda x: x["label"]):
        label = item["label"]; target, guarded = reconstructed_target(item["directory"], label); verification[label] = guarded
        grouped = defaultdict(list)
        for (person, draw, h), value in target.items(): grouped[person, h].append((draw, value["ba"]))
        means = {}
        for (person, h), values in sorted(grouped.items()):
            if [draw for draw, _ in sorted(values)] != list(range(10)): raise ValueError("target-only draw inventory differs")
            means[person, h] = float(np.mean([value for _, value in values]))
            participants.append({"dataset_variant": label, "target": person, "h_total": h, "draw_count": 10,
                                 "balanced_accuracy": means[person, h]})
        for h in HISTORY:
            values = [100 * means[person, h] for person in sorted({p for p, _ in means})]
            curves.append({"dataset_variant": label, "h_total": h, "n": len(values), **effect(values)})
        for context in ("100", FULL[label]):
            for person, h in sorted(means):
                paired = []
                for draw in range(10):
                    key = (label, "plain_ts", context, person, draw, h)
                    if key not in core: raise ValueError("core lacks matching source-tuned plain-TS row")
                    paired.append(core[key])
                plain = float(np.mean(paired)); value = 100 * (means[person, h] - plain)
                contrasts.append({"dataset_variant": label, "source_context": context, "target": person, "h_total": h,
                                  "draw_count": 10, "target_only_ba": means[person, h], "plain_ts_ba": plain, "difference_pp": value})
    participant_fields = ["dataset_variant", "target", "h_total", "draw_count", "balanced_accuracy"]
    contrast_fields = ["dataset_variant", "source_context", "target", "h_total", "draw_count", "target_only_ba", "plain_ts_ba", "difference_pp"]
    write_csv(args.out_dir / "participant_means.csv", participant_fields, participants)
    write_csv(args.out_dir / "paired_contrasts.csv", contrast_fields, contrasts)
    summary = {"schema": SCHEMA, "scope": "secondary descriptive fixed-C target-only reference; no h=0 model and no external-family endpoint change",
               "target_only_curves": {f"{r['dataset_variant']}__h{r['h_total']}": r for r in curves},
               "paired_target_only_minus_source_tuned_plain": {f"{label}__S{context}__h{h}": effect([r["difference_pp"] for r in contrasts if r["dataset_variant"] == label and r["source_context"] == context and r["h_total"] == h])
                   for label in TARGETS for context in ("100", FULL[label]) for h in HISTORY},
               "verification": verification, "draw_aggregation": "ten draws averaged within participant before contrasts and intervals",
               "uncertainty": "participant bootstrap 95% interval from scripts/analyze_review3.py; conditional and descriptive"}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    outputs = {name: sha(args.out_dir / name) for name in ("participant_means.csv", "paired_contrasts.csv", "summary.json")}
    receipt = {"schema": SCHEMA, "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "inputs_sha256": sha(args.inputs), "analyzer_sha256": sha(__file__), "outputs": outputs,
               "counts": {"participant_means": len(participants), "paired_contrasts": len(contrasts)},
               "wall_seconds": time.monotonic()-started, "cpu_seconds": time.process_time()-cpu,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "confirmation_accessed": False}
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (args.out_dir / "COMPLETED.json").write_text(json.dumps({"status":"completed", "receipt_sha256":sha(args.out_dir / "receipt.json")}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"completed", "counts":receipt["counts"]}, indent=2))


if __name__ == "__main__":
    main()
