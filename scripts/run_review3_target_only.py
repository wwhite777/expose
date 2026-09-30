#!/usr/bin/env python3
"""Run the secondary fixed-C target-only TS-LR comparator on frozen Review-3 plan rows."""
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
import run_review3_grid as grid
from expose.review2_controls import fit_representation, train_classifier

SCHEMA = "review3-target-only-v1"
C_FIXED = 1.0
HISTORY_TOTALS = (4, 10, 20, 40, 60, 100)
LIMITS = {"cpu_seconds": 600, "wall_seconds": 1800, "memory_gib": 16}


def sha256(path):
    return grid.sha256(path)


def target_only_operation_id(anchor):
    return "target_only__" + anchor["operation_id"]


def expected_source_size(operation):
    return "all" if operation["mode"] == "loso" else "1800"


def select_operations(operations, config):
    """Return the exact full-source, h>0 plain-TS anchors from one main plan."""
    candidates = [op for op in operations if op["method"] == "plain_ts"
                  and (op["study_id"], op["dataset"], op["montage"], op["mode"]) in {
                      ("openbmi_main", "openbmi", "8ch", "frozen_source"),
                      ("bnci_main", "bnci2014", "22ch", "loso")}
                  and str(op["source_size"]) == expected_source_size(op)
                  and op["donor_count"] == "all" and op["h_total"] in HISTORY_TOTALS]
    if not candidates:
        raise ValueError("plan lacks target-only plain-TS h>0 anchors")
    studies = {op["study_id"] for op in candidates}
    if len(studies) != 1:
        raise ValueError("target-only run requires one study per frozen main plan")
    study_id = next(iter(studies))
    configured = [study for study in config["studies"] if study["study_id"] == study_id]
    if len(configured) != 1:
        raise ValueError("anchor study is absent from frozen configuration")
    study = configured[0]
    full = "all" if study["mode"] == "loso" else "1800"
    if str(max((value for value in study["source_sizes"] if value != "all"), default=full)) != full and full != "all":
        raise ValueError("target-only source-size rule differs from the frozen main study")
    targets = sorted({int(op["target"]) for op in candidates})
    required = {(draw, target, h) for draw in range(config["draws"]) for target in targets for h in HISTORY_TOTALS}
    keyed = {(int(op["draw"]), int(op["target"]), int(op["h_total"])): op for op in candidates}
    if (len(keyed) != len(candidates) or set(keyed) != required
            or len(targets) != (12 if study["mode"] == "frozen_source" else 9)):
        raise ValueError("target-only anchors differ from the exact six-dose frozen grid")
    for op in candidates:
        if op["tuning_id"] is None or op["source_n"] != len_hint(op, full):
            raise ValueError("target-only anchor differs from a source-tuned main plain-TS operation")
    return [keyed[key] for key in sorted(keyed)], study_id, targets


def len_hint(operation, full):
    """The plan records full S as 1800 or all available LOSO trials (8*144)."""
    return 1152 if full == "all" else 1800


def validate_memberships(operations, sets, store):
    binding = []
    for op in operations:
        member = op["memberships"]
        if set(member) != {"source", "personal", "evaluation", "source_reference", "target_reference"}:
            raise ValueError("main operation membership fields differ")
        source_ids, personal_ids, evaluation_ids = (sets[member["source"]], sets[member["personal"]],
                                                     sets[member["evaluation"]])
        if (len(personal_ids) != op["h_total"] or len(evaluation_ids) != op["reference_trials_per_person"]
                or set(personal_ids) & set(evaluation_ids) or set(source_ids) & (set(personal_ids) | set(evaluation_ids))):
            raise ValueError("target-only anchor membership is invalid or overlaps source/evaluation")
        if set(store.subjects(personal_ids).tolist()) != {op["target"]} or set(store.subjects(evaluation_ids).tolist()) != {op["target"]}:
            raise ValueError("target-only personal/evaluation membership target differs")
        labels = store.labels(personal_ids)
        if set(labels.tolist()) != {0, 1} or int(np.sum(labels == 0)) != len(labels) // 2:
            raise ValueError("target-only personal membership is not balanced binary")
        binding.append({"operation_id": target_only_operation_id(op), "anchor_operation_id": op["operation_id"],
                        "personal_membership": member["personal"], "evaluation_membership": member["evaluation"],
                        "h_total": op["h_total"], "target": op["target"], "draw": op["draw"]})
    return binding


def fit_target_only(personal_covariances, personal_labels, evaluation_covariances):
    """Fit representation, scaler, and C=1 LR exclusively on the h personal OAS covariances."""
    X = np.asarray(personal_covariances, dtype=np.float64)
    y = np.asarray(personal_labels, dtype=int)
    evaluation = np.asarray(evaluation_covariances, dtype=np.float64)
    if (X.ndim != 3 or len(X) != len(y) or len(X) == 0 or set(y.tolist()) != {0, 1}
            or evaluation.ndim != 3 or evaluation.shape[1:] != X.shape[1:]):
        raise ValueError("invalid target-only covariance/label inputs")
    # source_count=len(X) makes the existing source_frozen helper fit on all (and only) personal rows.
    representation = fit_representation(X, len(X), "source_frozen")
    model = train_classifier(representation.transform(X), y, len(X), C_FIXED, "pooled")
    transformed = representation.transform(evaluation)
    predicted, probabilities = model.predict(transformed), model.predict_proba(transformed)
    if (predicted.shape != (len(evaluation),) or not set(predicted.tolist()) <= {0, 1}
            or probabilities.shape != (len(evaluation), 2)):
        raise ValueError("target-only classifier output violates the fixed binary prediction schema")
    selected_probability = probabilities[np.arange(len(predicted)), predicted]
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-10)
            or np.any(np.max(probabilities, axis=1) - selected_probability > 1e-12)):
        raise ValueError("target-only classifier output violates the fixed binary prediction schema")
    return predicted.astype(int), probabilities


def write_gzip_csv(path, fields, rows):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
    os.replace(temporary, path)


def run(config_path, index_path, plan_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("target-only output must be a new path")
    config = grid.validate_config(grid.load_json(config_path), sha256(index_path))
    plan_receipt, all_operations, sets, _ = grid.validate_plan(config_path, index_path, plan_dir)
    operations, study_id, targets = select_operations(all_operations, config)
    output.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    receipt = {"schema": SCHEMA, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "config_sha256": sha256(config_path), "index_sha256": sha256(index_path),
               "plan_receipt_sha256": sha256(Path(plan_dir) / "PLAN_RECEIPT.json"),
               "plan_files": plan_receipt["files"], "runner_sha256": sha256(__file__),
               "representation": "personal_only_riemannian_tangent_space_plus_standard_scaler",
               "C_fixed": C_FIXED, "comparison_scope": "secondary descriptive fixed-reference comparator; no h=0 model and no target tuning",
               "limits": LIMITS, "confirmation_accessed": False, "GPU_used": False}
    grid.atomic_json(output / "receipt.json", receipt)
    old_xcpu = signal.signal(signal.SIGXCPU, lambda signum, frame: (_ for _ in ()).throw(TimeoutError(f"CPU limit signal {signum}")))
    old_alarm = signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(TimeoutError(f"wall limit signal {signum}")))
    resource.setrlimit(resource.RLIMIT_CPU, (LIMITS["cpu_seconds"], LIMITS["cpu_seconds"] + 10))
    resource.setrlimit(resource.RLIMIT_AS, (LIMITS["memory_gib"] * 1024**3,) * 2)
    signal.alarm(LIMITS["wall_seconds"])
    prediction_fields = ["operation_id", "anchor_operation_id", "study_id", "draw", "target", "source_size", "h_total", "C", "trial_id", "y_true", "y_pred", "p0", "p1", "personal_membership", "evaluation_membership"]
    score_fields = ["operation_id", "anchor_operation_id", "study_id", "draw", "target", "source_size", "h_total", "C", "balanced_accuracy", "evaluation_n", "personal_membership", "evaluation_membership"]
    partial_predictions, partial_scores = output / "predictions.csv.gz.partial", output / "operation_scores.csv.partial"
    count = {"operations": 0, "predictions": 0}
    try:
        store = grid.DatasetStore(index_path, config["index_sha256"])
        binding = validate_memberships(operations, sets, store)
        grid.atomic_json(output / "PLAN_BINDING.json", {"schema": SCHEMA, "study_id": study_id,
                         "draws": config["draws"], "targets": targets, "history_totals": list(HISTORY_TOTALS),
                         "source_covariances_used_for_fitting": False,
                         "source_array_validation_note": "DatasetStore validates all indexed arrays; source covariances are never passed to target-only fitting.",
                         "operations": binding})
        with partial_predictions.open("wb") as raw, partial_scores.open("w", encoding="utf-8", newline="") as score_handle:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as prediction_handle:
                    prediction_writer = csv.DictWriter(prediction_handle, fieldnames=prediction_fields); prediction_writer.writeheader()
                    score_writer = csv.DictWriter(score_handle, fieldnames=score_fields); score_writer.writeheader()
                    for op in operations:
                        membership = op["memberships"]
                        personal_ids, evaluation_ids = sets[membership["personal"]], sets[membership["evaluation"]]
                        # Deliberately no store.matrices call for membership["source"].
                        y_personal, y_eval = store.labels(personal_ids), store.labels(evaluation_ids)
                        predicted, probabilities = fit_target_only(store.matrices(personal_ids, "cov"), y_personal,
                                                                   store.matrices(evaluation_ids, "cov"))
                        operation_id = target_only_operation_id(op)
                        for trial_id, truth, pred, probability in zip(evaluation_ids, y_eval, predicted, probabilities):
                            prediction_writer.writerow({"operation_id": operation_id, "anchor_operation_id": op["operation_id"],
                                "study_id": op["study_id"], "draw": op["draw"], "target": op["target"], "source_size": op["source_size"],
                                "h_total": op["h_total"], "C": C_FIXED, "trial_id": trial_id, "y_true": int(truth), "y_pred": int(pred),
                                "p0": float(probability[0]), "p1": float(probability[1]), "personal_membership": membership["personal"],
                                "evaluation_membership": membership["evaluation"]})
                        score_writer.writerow({"operation_id": operation_id, "anchor_operation_id": op["operation_id"], "study_id": op["study_id"],
                            "draw": op["draw"], "target": op["target"], "source_size": op["source_size"], "h_total": op["h_total"], "C": C_FIXED,
                            "balanced_accuracy": grid.balanced_accuracy(y_eval, predicted), "evaluation_n": len(evaluation_ids),
                            "personal_membership": membership["personal"], "evaluation_membership": membership["evaluation"]})
                        count["operations"] += 1; count["predictions"] += len(evaluation_ids)
        os.replace(partial_predictions, output / "predictions.csv.gz"); os.replace(partial_scores, output / "operation_scores.csv")
        receipt.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(), counts=count,
                       wall_seconds=time.monotonic()-started_wall, cpu_seconds=time.process_time()-started_cpu,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       outputs={name: sha256(output / name) for name in ("predictions.csv.gz", "operation_scores.csv", "PLAN_BINDING.json")})
        grid.atomic_json(output / "receipt.json", receipt)
        grid.atomic_json(output / "COMPLETED.json", {"status": "completed", "receipt_sha256": sha256(output / "receipt.json")})
        return receipt
    except BaseException as error:
        receipt.update(status="failed", finished_utc=datetime.now(timezone.utc).isoformat(), counts=count,
                       wall_seconds=time.monotonic()-started_wall, cpu_seconds=time.process_time()-started_cpu,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       error={"type": type(error).__name__, "message": str(error)},
                       partial_outputs={path.name: sha256(path) for path in (partial_predictions, partial_scores) if path.exists()})
        grid.atomic_json(output / "receipt.json", receipt)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGXCPU, old_xcpu); signal.signal(signal.SIGALRM, old_alarm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.index, args.plan_dir, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
