#!/usr/bin/env python3
"""Separately score current-session EA transforms for frozen OpenBMI8 EA-TS rows."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from expose.review2_controls import fit_representation, train_classifier
from expose.review2_controls import fit_ea_whitener
from run_review3_grid import DatasetStore, atomic_json, balanced_accuracy, read_gzip_json, sha256, validate_plan

HISTORY_TOTALS = (0, 60)
PREFIX = 20
SAMPLES = 750
STUDIES = {
    "openbmi_main": {"dataset": "openbmi", "montage": "8ch", "mode": "frozen_source",
                      "source_sizes": (100, 1800), "targets": 12, "session_trials": 100},
    "bnci_main": {"dataset": "bnci2014", "montage": "22ch", "mode": "loso",
                  "source_sizes": (100, "all"), "targets": 9, "session_trials": 144},
}


def empirical_ml(epochs, channels):
    """Per-trial centered ML covariance, divisor exactly 750."""
    X = np.asarray(epochs, dtype=np.float64)
    if channels not in (8, 22) or X.ndim != 3 or X.shape[1:] != (channels, SAMPLES) or not np.isfinite(X).all():
        raise ValueError("expected finite trials-by-8|22-by-750 epochs")
    centered = X - X.mean(axis=2, keepdims=True)
    result = np.einsum("nct,ndt->ncd", centered, centered) / SAMPLES
    if not np.isfinite(result).all() or not np.allclose(result, result.transpose(0, 2, 1), rtol=0, atol=1e-12):
        raise ValueError("invalid empirical ML covariance")
    return result


def current_whitener(reference_covariances, channels, session_trials):
    values = np.asarray(reference_covariances, dtype=np.float64)
    if (channels not in (8, 22) or session_trials not in (100, 144) or values.ndim != 3
            or values.shape[1:] != (channels, channels) or len(values) not in (PREFIX, session_trials)):
        raise ValueError("current EA reference count or channel dimensions differ")
    # Historical EA uses np.cov(..., ddof=1).  ML / 750 needs this factor before inverse square root.
    mean = (values.mean(axis=0) + values.mean(axis=0).T) * (SAMPLES / (SAMPLES - 1)) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(mean)
    if not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0 or eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise ValueError("current EA empirical reference is singular or ill-conditioned")
    W = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return (W + W.T) / 2


def sklearn_oas_from_ml(empirical, n_samples=SAMPLES):
    """Sklearn OAS formula on centered ML covariance; do not commute OAS and EA."""
    values = np.asarray(empirical, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2] or values.shape[1] not in (8, 22) or n_samples != SAMPLES:
        raise ValueError("OAS requires 8|22 square covariance and exactly 750 time samples")
    features = values.shape[1]
    alpha = np.mean(values ** 2, axis=(1, 2))
    mu = np.trace(values, axis1=1, axis2=2) / features
    denominator = (n_samples + 1) * (alpha - mu ** 2 / features)
    shrinkage = np.where(denominator == 0, 1.0, np.minimum((alpha + mu ** 2) / denominator, 1.0))
    result = (1 - shrinkage)[:, None, None] * values + shrinkage[:, None, None] * mu[:, None, None] * np.eye(features)
    return (result + result.transpose(0, 2, 1)) / 2


def current_ea_oas(empirical, W):
    transformed = W @ np.asarray(empirical, dtype=np.float64) @ W.T
    return sklearn_oas_from_ml((transformed + transformed.transpose(0, 2, 1)) / 2)


def read_selected_c(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8", newline="")))
    if not rows or set(rows[0]) != {"tuning_id", "selected_C"}:
        raise ValueError("selected_c.csv schema differs")
    result = {}
    for row in rows:
        if row["tuning_id"] in result or float(row["selected_C"]) not in (0.1, 1.0, 10.0):
            raise ValueError("invalid or duplicate selected C")
        result[row["tuning_id"]] = float(row["selected_C"])
    return result


def selected_operations(operations, study_id):
    setting = STUDIES[study_id]
    values = [op for op in operations if op["study_id"] == study_id and op["dataset"] == setting["dataset"]
              and op["montage"] == setting["montage"] and op["mode"] == setting["mode"] and op["method"] == "ea_ts"
              and op["donor_count"] == "all" and op["source_size"] in setting["source_sizes"]
              and op["h_total"] in HISTORY_TOTALS]
    keyed = {(op["source_size"], op["h_total"], op["draw"], op["target"]): op for op in values}
    if len(keyed) != len(values) or not values or any(op["tuning_id"] is None for op in values):
        raise ValueError("selected EA-TS operation set is duplicate, empty, or untuned")
    targets = {op["target"] for op in values}
    required = {(source, history, draw, target) for source in setting["source_sizes"] for history in HISTORY_TOTALS
                for draw in range(10) for target in targets}
    if set(keyed) != required or len(targets) != setting["targets"]:
        raise ValueError("current-alignment rows differ from frozen study dimensions")
    return [keyed[key] for key in sorted(keyed, key=lambda k: (str(k[0]), *k[1:]))]


def applicable_operations(operations, dataset, montage):
    studies = [name for name, value in STUDIES.items()
               if value["dataset"] == dataset and value["montage"] == montage]
    if len(studies) != 1:
        raise ValueError("current alignment requires exactly one supported dataset index")
    return {study: selected_operations(operations, study) for study in studies}


def old_cache_records():
    records = json.loads((ROOT / "result/day3/preparation_r001/schema.json").read_text())["records"]
    result = {}
    for record in records:
        if record["role"] == "development" and record["session"] == 2:
            result[int(record["subject"])] = record["cache"]
    if len(result) != 12:
        raise ValueError("need exact 12 development session-2 original-eight cache records")
    return result


def build_empirical_auxiliary(out, store, cache_records):
    """Write only twelve S2 empirical-covariance auxiliaries; old caches are unchanged."""
    auxiliary, guards = {}, {}
    auxdir = out / "empirical_cov_s2"
    auxdir.mkdir()
    for subject, cache in sorted(cache_records.items()):
        record = store.by_cell.get((subject, 2))
        if record is None or record["role"] != "development" or len(record["trial_ids"]) != 100:
            raise ValueError("main index does not provide an allowed development session-2 record")
        source = ROOT / cache["array_path"]
        if sha256(source) != cache["array_sha256"]:
            raise ValueError("frozen original-eight source cache changed")
        with np.load(source, allow_pickle=False) as archive:
            X, y = archive["X"], archive["y"]
        if X.shape != (100, 8, SAMPLES) or not np.array_equal(y.astype(int), store.labels(record["trial_ids"])):
            raise ValueError("auxiliary cache labels/order differ from main index")
        values = empirical_ml(X, 8)
        path = auxdir / f"s{subject:03d}_session2_empirical_cov.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, empirical_cov=values)
        auxiliary[subject] = values
        guards[subject] = {"source_cache": cache["array_path"], "source_cache_sha256": cache["array_sha256"],
                           "auxiliary_path": str(path.relative_to(out)), "auxiliary_sha256": sha256(path),
                           "trial_ids": record["trial_ids"]}
    atomic_json(out / "EMPIRICAL_COVARIANCE_GUARDS.json", guards)
    return auxiliary, guards


def bnci_empirical(store, targets):
    """Use the already-prepared, hash-checked empirical covariance; no OpenBMI cache path."""
    result = {}
    for subject in targets:
        record = store.by_cell.get((subject, 2))
        if record is None or len(record["trial_ids"]) != 144:
            raise ValueError("BNCI target session-2 record differs")
        with np.load(record["path"], allow_pickle=False) as archive:
            if "empirical_cov" not in archive.files:
                raise ValueError("BNCI preparation lacks empirical_cov")
            values = archive["empirical_cov"]
        if values.dtype != np.float64 or values.shape != (144, 22, 22):
            raise ValueError("BNCI empirical covariance dimensions differ")
        result[subject] = values
    return result


def validate_grid_completion(grid_dir, selected_c_path):
    receipt = json.loads((Path(grid_dir) / "receipt.json").read_text(encoding="utf-8"))
    completed = json.loads((Path(grid_dir) / "COMPLETED.json").read_text(encoding="utf-8"))
    if (receipt.get("status") != "completed" or completed.get("status") != "completed"
            or completed.get("receipt_sha256") != sha256(Path(grid_dir) / "receipt.json")
            or receipt.get("outputs", {}).get("selected_c.csv") != sha256(selected_c_path)
            or receipt.get("outputs", {}).get("predictions.csv.gz") != sha256(Path(grid_dir) / "predictions.csv.gz")):
        raise ValueError("main-grid completion or selected-C output binding failed")
    return receipt


def main_prediction_map(grid_dir):
    values = {}
    with gzip.open(Path(grid_dir) / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "ea_ts":
                key = (row["operation_id"], row["trial_id"])
                if key in values:
                    raise ValueError("duplicate main EA-TS prediction")
                values[key] = row
    return values


def write_gzip_csv(path, rows, fields):
    temporary = Path(str(path) + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def run(config, index, plan_dir, grid_dir, out):
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("output directory must be new")
    grid_config = json.loads(Path(config).read_text(encoding="utf-8"))
    plan_receipt, operations, sets, _ = validate_plan(config, index, plan_dir)
    selected_c_path = Path(grid_dir) / "selected_c.csv"
    grid_receipt = validate_grid_completion(grid_dir, selected_c_path)
    selected = read_selected_c(selected_c_path)
    store = DatasetStore(index)
    grouped = applicable_operations(operations, store.dataset, store.montage)
    ops = [op for group in grouped.values() for op in group]
    out.mkdir(parents=True)
    start, cpu = time.monotonic(), time.process_time()
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "threads": 1,
               "config_sha256": sha256(config), "index_sha256": sha256(index),
               "plan_receipt_sha256": sha256(Path(plan_dir) / "PLAN_RECEIPT.json"),
               "selected_c_sha256": sha256(selected_c_path), "runner_sha256": sha256(__file__),
               "grid_receipt_sha256": sha256(Path(grid_dir) / "receipt.json"), "current_session_labels_used_for_whitening": False,
               "constraints": {"threads": 1, "grid_limits": grid_config.get("limits"),
                               "auxiliary_nominal_uncompressed_bytes": 12 * 100 * 8 * 8 * 8}}
    atomic_json(out / "receipt.json", receipt)
    try:
        empirical, bnci = {}, {}
        if "openbmi_main" in grouped:
            openbmi_targets = {op["target"] for op in grouped["openbmi_main"]}
            empirical, empirical_guards = build_empirical_auxiliary(out, store, old_cache_records())
            if set(empirical) != openbmi_targets:
                raise ValueError("OpenBMI empirical auxiliary target set differs")
        else:
            bnci = bnci_empirical(store, {op["target"] for op in grouped["bnci_main"]})
            atomic_json(out / "EMPIRICAL_COVARIANCE_GUARDS.json", {
                "origin": "hash-validated BNCI prepared NPZ empirical_cov",
                "index_sha256": sha256(index), "target_subjects": sorted(bnci)})
        main_predictions = main_prediction_map(grid_dir)
        predictions, scores, references = [], [], []
        for op in ops:
            membership = op["memberships"]
            source_ids, history_ids, evaluation_ids = (sets[membership["source"]], sets[membership["personal"]],
                                                        sets[membership["evaluation"]])
            setting = STUDIES[op["study_id"]]
            channels, session_trials = (8 if setting["montage"] == "8ch" else 22), setting["session_trials"]
            current = empirical[op["target"]] if op["study_id"] == "openbmi_main" else bnci[op["target"]]
            record = store.by_cell.get((op["target"], 2))
            C = selected.get(op["tuning_id"])
            if (C is None or len(evaluation_ids) != session_trials or op["target"] not in (empirical if channels == 8 else bnci)
                    or record is None or evaluation_ids != record["trial_ids"]):
                raise ValueError("missing selected C or current empirical covariance")
            source_cov = store.matrices(source_ids, "ea_cov")
            history_cov = store.matrices(history_ids, "ea_cov") if history_ids else np.empty((0, channels, channels))
            train_cov = np.concatenate([source_cov, history_cov])
            train_y = np.concatenate([store.labels(source_ids), store.labels(history_ids)])
            representation = fit_representation(train_cov, len(source_cov), "source_frozen")
            classifier = train_classifier(representation.transform(train_cov), train_y, len(source_cov), C, "pooled")
            historical = store.matrices(evaluation_ids, "ea_cov")
            W_prefix = current_whitener(current[:PREFIX], channels, session_trials)
            W_all = current_whitener(current, channels, session_trials)
            variants = {"R1_historical_prefix_tail": (historical[PREFIX:], evaluation_ids[PREFIX:]),
                        "R1_historical_all": (historical, evaluation_ids),
                        "R2_current_prefix20_tail": (current_ea_oas(current[PREFIX:], W_prefix), evaluation_ids[PREFIX:]),
                        "R2_current_all_transductive": (current_ea_oas(current, W_all), evaluation_ids)}
            for name, (evaluation_cov, ids) in variants.items():
                probabilities = classifier.predict_proba(representation.transform(evaluation_cov))
                predicted = classifier.predict(representation.transform(evaluation_cov))
                truth = store.labels(ids)  # truth is read only after all transform/prediction work
                score = balanced_accuracy(truth, predicted)
                if name == "R1_historical_all":
                    for trial_id, pred, probability in zip(ids, predicted, probabilities):
                        saved = main_predictions.get((op["operation_id"], trial_id))
                        if (saved is None or int(saved["y_pred"]) != int(pred)
                                or abs(float(saved["p0"]) - float(probability[0])) > 1e-12
                                or abs(float(saved["p1"]) - float(probability[1])) > 1e-12):
                            raise ValueError("R1 historical replay differs from frozen main-grid prediction")
                scores.append({"operation_id": op["operation_id"], "variant": name, "draw": op["draw"], "target": op["target"],
                               "source_size": op["source_size"], "h_total": op["h_total"], "C": C,
                               "evaluation_n": len(ids), "balanced_accuracy": score})
                for trial_id, y_true, y_pred, probability in zip(ids, truth, predicted, probabilities):
                    predictions.append({"operation_id": op["operation_id"], "variant": name, "draw": op["draw"], "target": op["target"],
                                        "source_size": op["source_size"], "h_total": op["h_total"], "C": C, "trial_id": trial_id,
                                        "y_true": int(y_true), "y_pred": int(y_pred), "p0": float(probability[0]), "p1": float(probability[1])})
            references.append({"operation_id": op["operation_id"], "source_membership": membership["source"],
                               "personal_membership": membership["personal"], "evaluation_ids": evaluation_ids,
                               "prefix_reference_ids": evaluation_ids[:PREFIX], "tail_evaluation_ids": evaluation_ids[PREFIX:],
                               "selected_C": C, "historical_covariance": "ea_cov", "current_covariance": "empirical_ml_ddof0_then_ddof1_reference_EA_then_sklearn_OAS",
                               "main_grid_replay_checked": True})
        write_gzip_csv(out / "trial_predictions.csv.gz", predictions, list(predictions[0]))
        with (out / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0])); writer.writeheader(); writer.writerows(scores)
        atomic_json(out / "REFERENCE_AND_EVALUATION_IDS.json", references)
        receipt.update(status="completed", counts={"operations": len(ops), "scores": len(scores), "predictions": len(predictions)},
                       outputs={name: sha256(out / name) for name in ("trial_predictions.csv.gz", "scores.csv", "REFERENCE_AND_EVALUATION_IDS.json", "EMPIRICAL_COVARIANCE_GUARDS.json")})
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        raise
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic() - start,
                       cpu_seconds=time.process_time() - cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    atomic_json(out / "COMPLETED.json", {"status": "completed", "receipt_sha256": sha256(out / "receipt.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--grid-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.index, args.plan_dir, args.grid_dir, args.out_dir)


if __name__ == "__main__":
    main()
