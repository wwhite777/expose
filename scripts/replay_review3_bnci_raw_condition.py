#!/usr/bin/env python3
"""Replay one frozen BNCI raw-to-prediction condition.

This intentionally covers exactly bnci_main/plain_ts/draw0/target1/Sall/h60.
It does not rerun source-only C tuning or claim reproduction of any other cell.
"""

import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
PLAN_DIR = ROOT / "research/review3_20260923/bnci_plan_r002"
GRID_DIR = ROOT / "result/review3_20260923/bnci_grid_r002"
DOWNLOAD_RECEIPT = ROOT / "research/review3_20260923/BNCI_DOWNLOAD_RECEIPT.tsv"
OPERATION_ID = "bnci_main__plain_ts__d0__t1__kall__Sall__h60"
EXPECTED_TUNING_ID = "e8b99bb5b509c1e5458d33525c5aff6553f959f0702f3b82b036ef88c2a84546"
EXPECTED_VERSIONS = {
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "pyriemann": "0.9",
    "threadpoolctl": "3.6.0",
}
PROBABILITY_TOLERANCE = 1e-10


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_versions(get_version=importlib.metadata.version):
    observed = {name: get_version(name) for name in EXPECTED_VERSIONS}
    if observed != EXPECTED_VERSIONS:
        raise ValueError(
            f"installed numerical versions differ: expected {EXPECTED_VERSIONS}, observed {observed}"
        )
    return observed


def validate_plan(plan_dir=PLAN_DIR):
    receipt_path = plan_dir / "PLAN_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "planned_no_fits" or receipt.get("schema") != "review3-grid-v1":
        raise ValueError("BNCI plan receipt is not the frozen no-fit grid plan")
    required = {"operations.json.gz", "membership_sets.json.gz", "tuning_folds.json.gz"}
    if set(receipt.get("files", {})) != required:
        raise ValueError("BNCI plan file inventory differs")
    for name, expected in receipt["files"].items():
        if sha256(plan_dir / name) != expected:
            raise ValueError(f"BNCI plan payload changed: {name}")
    operations = read_gzip_json(plan_dir / "operations.json.gz")
    memberships = read_gzip_json(plan_dir / "membership_sets.json.gz")
    matches = [row for row in operations if row.get("operation_id") == OPERATION_ID]
    if len(matches) != 1:
        raise ValueError("frozen plan must contain exactly one requested operation")
    operation = matches[0]
    expected = {
        "study_id": "bnci_main",
        "dataset": "bnci2014",
        "montage": "22ch",
        "mode": "loso",
        "method": "plain_ts",
        "draw": 0,
        "target": 1,
        "donor_count": "all",
        "source_size": "all",
        "source_n": 1152,
        "h_total": 60,
        "reference_trials_per_person": 144,
        "tuning_id": EXPECTED_TUNING_ID,
        "donor_subjects": list(range(2, 10)),
    }
    if any(operation.get(key) != value for key, value in expected.items()):
        raise ValueError("requested operation metadata differs from the frozen cell")
    if operation.get("memberships", {}).get("source_reference") is not None:
        raise ValueError("plain TS cell unexpectedly has a source reference membership")
    if operation.get("memberships", {}).get("target_reference") is not None:
        raise ValueError("plain TS cell unexpectedly has a target reference membership")
    keys = operation["memberships"]
    source = memberships.get(keys["source"])
    personal = memberships.get(keys["personal"])
    evaluation = memberships.get(keys["evaluation"])
    if not all(isinstance(values, list) for values in (source, personal, evaluation)):
        raise ValueError("requested operation memberships are absent")
    if (len(source), len(personal), len(evaluation)) != (1152, 60, 144):
        raise ValueError("requested operation membership counts differ")
    for name, values in (("source", source), ("personal", personal), ("evaluation", evaluation)):
        if fingerprint(values) != keys[name]:
            raise ValueError(f"{name} membership differs from its content hash")
    if set(source) & set(personal) or set(source) & set(evaluation) or set(personal) & set(evaluation):
        raise ValueError("source, history, and evaluation memberships overlap")
    return receipt_path, receipt, operation, source, personal, evaluation


def validate_completed_grid(grid_dir=GRID_DIR):
    receipt_path = grid_dir / "receipt.json"
    marker_path = grid_dir / "COMPLETED.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed" or marker.get("receipt_sha256") != sha256(receipt_path):
        raise ValueError("corrected BNCI grid is incomplete or changed")
    for name, expected in receipt.get("outputs", {}).items():
        if sha256(grid_dir / name) != expected:
            raise ValueError(f"corrected BNCI grid output changed: {name}")
    selected = [
        row
        for row in csv.DictReader((grid_dir / "selected_c.csv").open(encoding="utf-8", newline=""))
        if row.get("tuning_id") == EXPECTED_TUNING_ID
    ]
    if len(selected) != 1 or float(selected[0]["selected_C"]) not in (0.1, 1.0, 10.0):
        raise ValueError("requested source-only selected C is absent or invalid")
    C = float(selected[0]["selected_C"])
    saved = []
    with gzip.open(grid_dir / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("operation_id") == OPERATION_ID:
                saved.append(row)
    if len(saved) != 144 or len({row["trial_id"] for row in saved}) != 144:
        raise ValueError("saved prediction inventory for requested cell differs")
    scores = [
        row
        for row in csv.DictReader((grid_dir / "operation_scores.csv").open(encoding="utf-8", newline=""))
        if row.get("operation_id") == OPERATION_ID
    ]
    if len(scores) != 1:
        raise ValueError("saved score inventory for requested cell differs")
    return receipt_path, marker_path, receipt, C, saved, scores[0]


def download_manifest(path=DOWNLOAD_RECEIPT):
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline=""), delimiter="\t"))
    if len(rows) != 18 or set(rows[0]) != {"filename", "bytes", "sha256", "url"}:
        raise ValueError("BNCI download manifest schema/count differs")
    result = {row["filename"]: row for row in rows}
    expected_names = {f"A{subject:02d}{session}.mat" for subject in range(1, 10) for session in "TE"}
    if set(result) != expected_names:
        raise ValueError("BNCI download manifest filenames differ")
    return result


def required_raw_files():
    return [f"A{subject:02d}T.mat" for subject in range(2, 10)] + ["A01T.mat", "A01E.mat"]


def validate_raw_files(raw_dir, manifest, official_file_url):
    raw_dir = Path(raw_dir).resolve()
    records = {}
    for filename in required_raw_files():
        path = raw_dir / filename
        row = manifest[filename]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required raw file is absent or not regular: {filename}")
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise ValueError(f"required raw size/hash differs: {filename}")
        subject, session = int(filename[1:3]), filename[3]
        if row["url"] != official_file_url(subject, session):
            raise ValueError(f"required raw URL differs: {filename}")
        records[filename] = {
            "path": path,
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "url": row["url"],
        }
    return records


def load_required_epochs(raw_records, loader):
    epochs = {}
    loaded_files = []
    for filename in required_raw_files():
        subject, session = int(filename[1:3]), filename[3]
        value = loader(raw_records[filename]["path"], subject=subject, session=session)
        if len(value.trial_ids) != 144 or value.X.shape != (144, 22, 750):
            raise ValueError(f"loader output contract differs: {filename}")
        if len(set(value.trial_ids)) != 144 or set(value.y.tolist()) != {0, 1}:
            raise ValueError(f"loader trial identities/labels differ: {filename}")
        for index, trial_id in enumerate(value.trial_ids):
            if trial_id in epochs:
                raise ValueError("raw loader created a duplicate global trial ID")
            epochs[trial_id] = (value.X[index], int(value.y[index]))
        loaded_files.append(filename)
    return epochs, loaded_files


def arrays_for_ids(store, trial_ids):
    missing = [trial_id for trial_id in trial_ids if trial_id not in store]
    if missing:
        raise ValueError(f"raw loader is missing {len(missing)} frozen membership IDs")
    import numpy as np

    X = np.stack([store[trial_id][0] for trial_id in trial_ids]).astype(np.float64, copy=False)
    y = np.asarray([store[trial_id][1] for trial_id in trial_ids], dtype=np.int64)
    return X, y


def oas_covariances(epochs):
    import numpy as np
    from sklearn.covariance import oas

    X = np.asarray(epochs, dtype=np.float64)
    if X.ndim != 3 or X.shape[1:] != (22, 750) or not np.isfinite(X).all():
        raise ValueError("OAS input must be finite trials-by-22-by-750 epochs")
    result = np.stack([oas(trial.T)[0] for trial in X]).astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any(np.linalg.eigvalsh(result) <= 0):
        raise ValueError("OAS produced an invalid SPD covariance")
    return result


def fit_plain_ts(source_cov, source_y, personal_cov, personal_y, evaluation_cov, C):
    import numpy as np
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(source_cov) != 1152 or len(personal_cov) != 60 or len(evaluation_cov) != 144:
        raise ValueError("fixed replay training/evaluation counts differ")
    representation = make_pipeline(
        TangentSpace(metric="riemann", tsupdate=False),
        StandardScaler(),
    )
    representation.fit(source_cov)
    train_cov = np.concatenate([source_cov, personal_cov])
    train_y = np.concatenate([source_y, personal_y])
    classifier = LogisticRegression(
        penalty="l2",
        C=float(C),
        solver="lbfgs",
        max_iter=1000,
        random_state=20260914,
    )
    classifier.fit(representation.transform(train_cov), train_y, sample_weight=np.ones(len(train_y)))
    if classifier.classes_.tolist() != [0, 1]:
        raise ValueError("classifier class order differs")
    transformed = representation.transform(evaluation_cov)
    probabilities = classifier.predict_proba(transformed)
    predicted = classifier.predict(transformed)
    return predicted.astype(int), probabilities.astype(float)


def compare_predictions(evaluation_ids, truth, predicted, probabilities, saved, tolerance=PROBABILITY_TOLERANCE):
    import numpy as np

    by_trial = {row["trial_id"]: row for row in saved}
    if list(evaluation_ids) != [row["trial_id"] for row in saved]:
        raise ValueError("saved prediction order differs from frozen evaluation membership")
    if set(by_trial) != set(evaluation_ids):
        raise ValueError("saved prediction identities differ from frozen evaluation membership")
    saved_truth = np.asarray([int(by_trial[trial]["y_true"]) for trial in evaluation_ids])
    saved_pred = np.asarray([int(by_trial[trial]["y_pred"]) for trial in evaluation_ids])
    saved_probability = np.asarray(
        [[float(by_trial[trial]["p0"]), float(by_trial[trial]["p1"])] for trial in evaluation_ids]
    )
    if not np.array_equal(saved_truth, truth):
        raise ValueError("raw labels differ from saved evaluation labels")
    label_mismatches = int(np.sum(saved_pred != predicted))
    probability_diff = np.abs(saved_probability - probabilities)
    maximum = float(probability_diff.max())
    count_over = int(np.sum(probability_diff > tolerance))
    if label_mismatches or count_over:
        raise ValueError(
            f"raw replay prediction mismatch: labels={label_mismatches}, "
            f"probabilities_over_tolerance={count_over}, max={maximum}"
        )
    return {
        "evaluation_trials": len(evaluation_ids),
        "label_mismatches": label_mismatches,
        "probability_values_compared": int(probability_diff.size),
        "probability_values_over_tolerance": count_over,
        "maximum_absolute_probability_difference": maximum,
        "probability_tolerance": tolerance,
    }


def balanced_accuracy(y_true, y_pred):
    import numpy as np

    return float(np.mean([np.mean(y_pred[y_true == label] == label) for label in (0, 1)]))


def write_predictions(path, trial_ids, truth, predicted, probabilities):
    with gzip.open(path, "xt", encoding="utf-8", newline="") as handle:
        fields = ["operation_id", "trial_id", "y_true", "y_pred", "p0", "p1"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for trial_id, y_true, y_pred, probability in zip(
            trial_ids, truth, predicted, probabilities
        ):
            writer.writerow(
                {
                    "operation_id": OPERATION_ID,
                    "trial_id": trial_id,
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "p0": float(probability[0]),
                    "p1": float(probability[1]),
                }
            )


def run(raw_dir, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("output must be a new exclusive path")
    output.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    receipt = {
        "schema": "review3-bnci-one-cell-raw-replay-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "operation_id": OPERATION_ID,
        "scope": "exactly one raw-to-prediction condition; no C tuning or global-grid claim",
        "limits": {"threads": 1, "memory_gib": 16, "cpu_seconds": 120, "wall_seconds": 300},
        "independence_limit": (
            "Independent fit assembly and sklearn OAS/TS/scaler/LR pipeline, but shared project "
            "BNCI raw loader, frozen memberships, numerical libraries, and saved predictions."
        ),
        "confirmation_accessed": False,
        "GPU_used": False,
    }
    atomic_json(output / "receipt.json", receipt)
    old_xcpu = signal.signal(
        signal.SIGXCPU,
        lambda signum, frame: (_ for _ in ()).throw(TimeoutError(f"CPU limit signal {signum}")),
    )
    old_alarm = signal.signal(
        signal.SIGALRM,
        lambda signum, frame: (_ for _ in ()).throw(TimeoutError(f"wall limit signal {signum}")),
    )
    resource.setrlimit(resource.RLIMIT_CPU, (120, 130))
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    signal.alarm(300)
    try:
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits

        sys.path.insert(0, str(ROOT / "src"))
        from expose.bnci2014 import load_bnci2014_epochs, official_file_url

        versions = validate_versions()
        plan_path, plan_receipt, operation, source_ids, personal_ids, evaluation_ids = validate_plan()
        grid_receipt_path, grid_marker_path, grid_receipt, C, saved, saved_score = validate_completed_grid()
        membership = operation["memberships"]
        if any(row["source_membership"] != membership["source"] for row in saved):
            raise ValueError("saved source membership key differs")
        if any(row["personal_membership"] != membership["personal"] for row in saved):
            raise ValueError("saved personal membership key differs")
        if any(row["evaluation_membership"] != membership["evaluation"] for row in saved):
            raise ValueError("saved evaluation membership key differs")
        if any(abs(float(row["C"]) - C) > 0 for row in saved):
            raise ValueError("saved prediction C differs from selected C")
        manifest = download_manifest()
        raw_records = validate_raw_files(raw_dir, manifest, official_file_url)
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            if any(pool.get("num_threads") != 1 for pool in pools):
                raise ValueError("numerical thread count exceeds one")
            raw_store, loaded_files = load_required_epochs(raw_records, load_bnci2014_epochs)
            X_source, y_source = arrays_for_ids(raw_store, source_ids)
            X_personal, y_personal = arrays_for_ids(raw_store, personal_ids)
            X_evaluation, y_evaluation = arrays_for_ids(raw_store, evaluation_ids)
            if np.bincount(y_source, minlength=2).tolist() != [576, 576]:
                raise ValueError("source labels are not 576 per class")
            if np.bincount(y_personal, minlength=2).tolist() != [30, 30]:
                raise ValueError("history labels are not 30 per class")
            if np.bincount(y_evaluation, minlength=2).tolist() != [72, 72]:
                raise ValueError("evaluation labels are not 72 per class")
            source_cov = oas_covariances(X_source)
            personal_cov = oas_covariances(X_personal)
            evaluation_cov = oas_covariances(X_evaluation)
            predicted, probabilities = fit_plain_ts(
                source_cov,
                y_source,
                personal_cov,
                y_personal,
                evaluation_cov,
                C,
            )
        comparison = compare_predictions(
            evaluation_ids, y_evaluation, predicted, probabilities, saved
        )
        score = balanced_accuracy(y_evaluation, predicted)
        score_difference = abs(score - float(saved_score["balanced_accuracy"]))
        if score_difference > 1e-12:
            raise ValueError("replayed balanced accuracy differs from saved operation score")
        predictions_path = output / "predictions.csv.gz"
        write_predictions(
            predictions_path, evaluation_ids, y_evaluation, predicted, probabilities
        )
        receipt.update(
            status="completed",
            installed_versions=versions,
            selected_C=C,
            counts={
                "raw_files_loaded": len(loaded_files),
                "source_trials": len(source_ids),
                "history_trials": len(personal_ids),
                "evaluation_trials": len(evaluation_ids),
            },
            raw_files={
                name: {
                    "bytes": raw_records[name]["bytes"],
                    "sha256": raw_records[name]["sha256"],
                    "url": raw_records[name]["url"],
                }
                for name in loaded_files
            },
            fixed_inputs={
                str(plan_path.relative_to(ROOT)): sha256(plan_path),
                str(DOWNLOAD_RECEIPT.relative_to(ROOT)): sha256(DOWNLOAD_RECEIPT),
                str(grid_receipt_path.relative_to(ROOT)): sha256(grid_receipt_path),
                str(grid_marker_path.relative_to(ROOT)): sha256(grid_marker_path),
                str((GRID_DIR / "selected_c.csv").relative_to(ROOT)): sha256(
                    GRID_DIR / "selected_c.csv"
                ),
                str((GRID_DIR / "predictions.csv.gz").relative_to(ROOT)): sha256(
                    GRID_DIR / "predictions.csv.gz"
                ),
                "src/expose/bnci2014.py": sha256(ROOT / "src/expose/bnci2014.py"),
                "scripts/replay_review3_bnci_raw_condition.py": sha256(Path(__file__)),
            },
            comparison=comparison,
            balanced_accuracy=score,
            saved_balanced_accuracy=float(saved_score["balanced_accuracy"]),
            balanced_accuracy_absolute_difference=score_difference,
            output_hashes={"predictions.csv.gz": sha256(predictions_path)},
            threadpools=pools,
        )
    except BaseException as error:
        receipt.update(
            status="failed",
            error={"type": type(error).__name__, "message": str(error)},
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGXCPU, old_xcpu)
        signal.signal(signal.SIGALRM, old_alarm)
        receipt.update(
            finished_utc=datetime.now(timezone.utc).isoformat(),
            cpu_seconds=time.process_time() - started_cpu,
            wall_seconds=time.monotonic() - started_wall,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        atomic_json(output / "receipt.json", receipt)
    atomic_json(
        output / "COMPLETED.json",
        {
            "status": "completed",
            "receipt_sha256": sha256(output / "receipt.json"),
            "predictions_sha256": sha256(output / "predictions.csv.gz"),
        },
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.raw_dir, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "operation_id": result["operation_id"],
                "comparison": result["comparison"],
                "cpu_seconds": result["cpu_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
