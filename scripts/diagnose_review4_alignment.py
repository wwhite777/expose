#!/usr/bin/env python3
"""Post-hoc saved-data diagnostic of historical EA and reference drift."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import statistics
import sys
import time

import numpy as np
from pyriemann.utils.mean import mean_covariance
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = 750
BOOTSTRAP_SEED = 2026092402
BOOTSTRAP_DRAWS = 10000
TOL = 1e-12
INDEXES = {
    "openbmi8": ROOT / "research/review3_20260923/OPENBMI8_GRID_INDEX.json",
    "bnci": ROOT / "research/review3_20260923/BNCI_GRID_INDEX.json",
}
SCHEMA = ROOT / "result/day3/preparation_r001/schema.json"
PARTICIPANTS = ROOT / "result/review3_20260923/analysis_complete_r001/participant_means.csv"
DATASETS = {
    "openbmi8": {"dataset": "openbmi", "role": "development", "subjects": 12,
                 "trials": 100, "channels": 8, "full_source": "1800",
                 "study": "openbmi_main"},
    "bnci": {"dataset": "bnci2014", "role": "external", "subjects": 9,
             "trials": 144, "channels": 22, "full_source": "all",
             "study": "bnci_main"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def empirical_ml(epochs):
    values = np.asarray(epochs, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != SAMPLES or not np.isfinite(values).all():
        raise ValueError("epochs must be finite trials-by-channels-by-750")
    centered = values - values.mean(axis=2, keepdims=True)
    result = np.einsum("nct,ndt->ncd", centered, centered) / SAMPLES
    return (result + result.transpose(0, 2, 1)) / 2


def empirical_reference(empirical):
    values = np.asarray(empirical, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2] or not np.isfinite(values).all():
        raise ValueError("invalid empirical covariance array")
    # Saved empirical covariance is centered ML (/750); historical EA used np.cov ddof=1 (/749).
    reference = values.mean(axis=0) * SAMPLES / (SAMPLES - 1)
    return (reference + reference.T) / 2


def inverse_sqrt(matrix):
    value = np.asarray(matrix, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh((value + value.T) / 2)
    if (not np.isfinite(eigenvalues).all() or eigenvalues[-1] <= 0 or
            eigenvalues[0] <= 1e-12 * eigenvalues[-1]):
        raise ValueError("reference is singular or ill-conditioned")
    result = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return (result + result.T) / 2


def affine_distance(left, right):
    """Affine-invariant SPD distance, implemented through a symmetric congruence."""
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("distance matrices differ")
    root = inverse_sqrt(a)
    eigenvalues = np.linalg.eigvalsh((root @ b @ root + root @ b.T @ root) / 2)
    if not np.isfinite(eigenvalues).all() or np.any(eigenvalues <= 0):
        raise ValueError("distance requires SPD matrices")
    return float(np.linalg.norm(np.log(eigenvalues)))


def oas_from_ml(empirical):
    """Exact sklearn OAS formula applied after centering and ML covariance."""
    values = np.asarray(empirical, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("invalid ML covariance array")
    features = values.shape[1]
    alpha = np.mean(values ** 2, axis=(1, 2))
    mu = np.trace(values, axis1=1, axis2=2) / features
    denominator = (SAMPLES + 1) * (alpha - mu ** 2 / features)
    shrinkage = np.where(denominator == 0, 1.0,
                         np.minimum((alpha + mu ** 2) / denominator, 1.0))
    identity = np.eye(features)
    result = ((1 - shrinkage)[:, None, None] * values +
              shrinkage[:, None, None] * mu[:, None, None] * identity)
    return (result + result.transpose(0, 2, 1)) / 2


def ea_before_oas(empirical, whitener):
    values, W = np.asarray(empirical), np.asarray(whitener)
    transformed = W @ values @ W.T
    return oas_from_ml((transformed + transformed.transpose(0, 2, 1)) / 2)


def pearson(left, right):
    x, y = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("invalid correlation vectors")
    x, y = x - x.mean(), y - y.mean()
    denominator = math.sqrt(float(x @ x) * float(y @ y))
    if denominator == 0:
        raise ValueError("correlation is undefined for a constant vector")
    return float((x @ y) / denominator)


def descriptive_correlations(diagnostics, effects):
    rows = []
    for dataset in DATASETS:
        person_rows = sorted((row for row in diagnostics if row["dataset_variant"] == dataset),
                             key=lambda row: row["target"])
        for drift_name in ("empirical_reference_drift", "oas_riemann_mean_drift"):
            drift = [row[drift_name] for row in person_rows]
            for contrast in ("previous_ea_minus_plain_ts", "recenter_mdm_minus_plain_mdm",
                             "recenter_mdm_minus_plain_ts"):
                effect = [effects[(dataset, row["target"], contrast)] for row in person_rows]
                for correlation, left, right in (
                        ("pearson", drift, effect),
                        ("spearman", rankdata(drift), rankdata(effect))):
                    rows.append({"dataset_variant": dataset, "drift_metric": drift_name,
                                 "performance_contrast": contrast,
                                 "effect_direction": "left_minus_right_pp",
                                 "correlation": correlation, "n": len(left),
                                 "coefficient": pearson(left, right)})
    return rows


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower, upper = int(math.floor(position)), int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def effect_summary(values):
    ordered = list(values)
    rng = random.Random(BOOTSTRAP_SEED + len(ordered))
    means = sorted(sum(ordered[rng.randrange(len(ordered))] for _ in ordered) / len(ordered)
                   for _ in range(BOOTSTRAP_DRAWS))
    positive = sum(value > TOL for value in ordered)
    negative = sum(value < -TOL for value in ordered)
    return {"n": len(ordered), "mean_pp": statistics.mean(ordered),
            "median_pp": statistics.median(ordered), "min_pp": min(ordered),
            "max_pp": max(ordered), "ci_low_pp": percentile(means, 0.025),
            "ci_high_pp": percentile(means, 0.975), "positive": positive,
            "zero": len(ordered) - positive - negative, "negative": negative}


def load_records():
    inputs, records = {}, {}
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    open_caches = {(int(row["subject"]), int(row["session"])): row["cache"]
                   for row in schema["records"] if row["role"] == "development"}
    inputs["openbmi_preparation_schema"] = SCHEMA
    if len(open_caches) != 24:
        raise ValueError("OpenBMI development cache coverage differs")

    for dataset, index_path in INDEXES.items():
        spec = DATASETS[dataset]
        index = json.loads(index_path.read_text(encoding="utf-8"))
        inputs[f"{dataset}_index"] = index_path
        if index.get("dataset") != spec["dataset"] or len(index.get("records", [])) < 2 * spec["subjects"]:
            raise ValueError("grid index identity differs")
        selected = [row for row in index["records"] if row["role"] == spec["role"]]
        if len(selected) != 2 * spec["subjects"]:
            raise ValueError("target record coverage differs")
        for row in selected:
            subject, session = int(row["subject"]), int(row["session"])
            key = (dataset, subject, session)
            if key in records or session not in (1, 2) or len(row["trial_ids"]) != spec["trials"]:
                raise ValueError("duplicate or malformed target record")
            path = ROOT / row["npz_path"]
            if sha256(path) != row["sha256"]:
                raise ValueError("indexed covariance archive hash differs")
            metadata = (path.with_suffix(".json") if dataset == "openbmi8"
                        else path.with_suffix(".metadata.json"))
            if not metadata.is_file():
                raise FileNotFoundError(metadata)
            inputs[f"{dataset}_s{subject}_session{session}_npz"] = path
            inputs[f"{dataset}_s{subject}_session{session}_metadata"] = metadata
            cache = None
            if dataset == "openbmi8":
                cache = open_caches[(subject, session)]
                cache_path = ROOT / cache["array_path"]
                if sha256(cache_path) != cache["array_sha256"]:
                    raise ValueError("OpenBMI preparation cache hash differs")
                inputs[f"openbmi8_s{subject}_session{session}_cache"] = cache_path
            records[key] = {"index": row, "path": path, "metadata": metadata,
                            "cache": cache}
    return records, inputs


def diagnose_person(dataset, subject, records):
    spec = DATASETS[dataset]
    sessions, metadata = {}, {}
    for session in (1, 2):
        record = records[(dataset, subject, session)]
        with np.load(record["path"], allow_pickle=False) as archive:
            required = {"cov", "ea_cov", "y"}
            if not required.issubset(archive.files):
                raise ValueError("covariance archive schema differs")
            cov, ea_cov, y = (np.asarray(archive[name]).copy() for name in ("cov", "ea_cov", "y"))
            if dataset == "bnci":
                if "empirical_cov" not in archive.files:
                    raise ValueError("BNCI empirical covariance is absent")
                empirical = np.asarray(archive["empirical_cov"]).copy()
            else:
                cache_path = ROOT / record["cache"]["array_path"]
                with np.load(cache_path, allow_pickle=False) as cache:
                    X, cache_y = np.asarray(cache["X"]), np.asarray(cache["y"])
                if X.shape != (spec["trials"], spec["channels"], SAMPLES) or not np.array_equal(y, cache_y):
                    raise ValueError("OpenBMI cache differs in shape or labels")
                empirical = empirical_ml(X)
            expected_shape = (spec["trials"], spec["channels"], spec["channels"])
            if (cov.shape != expected_shape or ea_cov.shape != expected_shape or
                    empirical.shape != expected_shape or y.shape != (spec["trials"],) or
                    not np.isfinite(cov).all() or not np.isfinite(ea_cov).all() or
                    not np.isfinite(empirical).all()):
                raise ValueError("saved matrix dimensions or values differ")
        metadata[session] = json.loads(record["metadata"].read_text(encoding="utf-8"))
        sessions[session] = {"cov": cov, "ea_cov": ea_cov, "empirical": empirical}

    R1, R2 = (empirical_reference(sessions[s]["empirical"]) for s in (1, 2))
    W = inverse_sqrt(R1)
    identity_error = W @ R1 @ W - np.eye(spec["channels"])
    eigenvalues = np.linalg.eigvalsh(R1)

    if dataset == "openbmi8":
        saved_w1 = np.asarray(metadata[1]["ea_whitener"], dtype=float)
        saved_w2 = np.asarray(metadata[2]["ea_whitener"], dtype=float)
        sidecar_reference_delta = max(float(np.max(np.abs(saved_w1 - saved_w2))),
                                      float(np.max(np.abs(saved_w1 - W))))
        sidecar_reference_relative = max(
            float(np.linalg.norm(saved_w1 - saved_w2) / np.linalg.norm(saved_w1)),
            float(np.linalg.norm(saved_w1 - W) / np.linalg.norm(saved_w1)))
        saved_hash = hashlib.sha256(
            np.ascontiguousarray(saved_w1, dtype=np.float64).tobytes()).hexdigest()
        if metadata[1]["ea_whitener_sha256"] != metadata[2]["ea_whitener_sha256"]:
            raise ValueError("OpenBMI session-2 sidecar did not reuse session-1 whitener")
        if saved_hash != metadata[1]["ea_whitener_sha256"]:
            raise ValueError("OpenBMI saved whitener hash differs")
    else:
        if metadata[1]["ea_fit"] != {**metadata[2]["ea_fit"], "applied_to_session": 1,
                                      "frozen_for_session_2": False}:
            # Compare the scientific fit fields while allowing application-session fields to differ.
            left = {k: v for k, v in metadata[1]["ea_fit"].items()
                    if k not in ("applied_to_session", "frozen_for_session_2")}
            right = {k: v for k, v in metadata[2]["ea_fit"].items()
                     if k not in ("applied_to_session", "frozen_for_session_2")}
            if left != right:
                raise ValueError("BNCI session-2 metadata did not reuse session-1 EA fit")
        sidecar_reference_delta = 0.0
        sidecar_reference_relative = 0.0
        for key, observed in (("eigenvalue_min", eigenvalues[0]),
                              ("eigenvalue_max", eigenvalues[-1])):
            expected = float(metadata[1]["ea_fit"][key])
            if not math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=0.0):
                raise ValueError("BNCI saved session-1 reference spectrum differs")

    maximum_absolute, maximum_relative = 0.0, 0.0
    for session in (1, 2):
        reconstructed = ea_before_oas(sessions[session]["empirical"], W)
        difference = reconstructed - sessions[session]["ea_cov"]
        maximum_absolute = max(maximum_absolute, float(np.max(np.abs(difference))))
        maximum_relative = max(maximum_relative,
                               float(np.linalg.norm(difference) /
                                     np.linalg.norm(sessions[session]["ea_cov"])))
    if (float(np.max(np.abs(identity_error))) > 1e-10 or
            sidecar_reference_relative > 1e-10 or maximum_relative > 1e-10):
        raise ValueError("saved EA implementation consistency exceeds 1e-10 relative tolerance")

    mean1 = mean_covariance(sessions[1]["cov"], metric="riemann")
    mean2 = mean_covariance(sessions[2]["cov"], metric="riemann")
    return {
        "dataset_variant": dataset, "target": subject,
        "session_trials": spec["trials"], "channels": spec["channels"],
        "empirical_reference_drift": affine_distance(R1, R2),
        "oas_riemann_mean_drift": affine_distance(mean1, mean2),
        "s1_reference_eigen_min": float(eigenvalues[0]),
        "s1_reference_eigen_max": float(eigenvalues[-1]),
        "s1_reference_condition": float(eigenvalues[-1] / eigenvalues[0]),
        "whitening_identity_frobenius": float(np.linalg.norm(identity_error)),
        "whitening_identity_max_abs": float(np.max(np.abs(identity_error))),
        "saved_reference_max_abs_difference": sidecar_reference_delta,
        "saved_reference_max_relative_frobenius": sidecar_reference_relative,
        "saved_ea_cov_max_abs_difference": maximum_absolute,
        "saved_ea_cov_max_relative_frobenius": maximum_relative,
        "ea_order": "empirical_centering_then_session1_whitening_then_OAS",
    }


def load_effects(subjects):
    cells = {}
    for row in read_csv(PARTICIPANTS):
        dataset = row["dataset_variant"]
        if dataset not in DATASETS:
            continue
        spec = DATASETS[dataset]
        if (row["study_id"] != spec["study"] or row["source_size"] != spec["full_source"] or
                row["donor_count"] != "all" or int(row["h_total"]) != 0 or int(row["draws"]) != 10):
            continue
        method, target = row["method"], int(row["target"])
        if method not in ("plain_ts", "ea_ts", "plain_mdm", "recenter_mdm"):
            continue
        key = (dataset, target, method)
        if key in cells:
            raise ValueError("duplicate full-source h0 participant mean")
        value = float(row["balanced_accuracy"])
        if not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError("invalid participant balanced accuracy")
        cells[key] = value

    effects, rows = {}, []
    definitions = (
        ("previous_ea_minus_plain_ts", "ea_ts", "plain_ts"),
        ("recenter_mdm_minus_plain_mdm", "recenter_mdm", "plain_mdm"),
        ("recenter_mdm_minus_plain_ts", "recenter_mdm", "plain_ts"),
    )
    for dataset, ids in subjects.items():
        for target in ids:
            for contrast, left, right in definitions:
                value = (cells[(dataset, target, left)] - cells[(dataset, target, right)]) * 100
                effects[(dataset, target, contrast)] = value
                rows.append({"dataset_variant": dataset, "target": target,
                             "source_size": DATASETS[dataset]["full_source"], "h_total": 0,
                             "contrast": contrast, "left_method": left,
                             "right_method": right, "effect_pp": value,
                             "draws_averaged_within_person": 10})
    return effects, rows


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def note_text(summary):
    return f"""# Alignment-reference diagnostic

**Status:** post-hoc diagnostic of approved saved development/external arrays, 2026-09-24. No protected participant, raw download, model fit, model selection, or causal test was used.

The diagnostic covers {summary['people']} people: 12 OpenBMI development participants and 9 external BNCI participants. It separately measures affine-invariant drift between session empirical references and between session OAS Riemannian means. Historical EA was reconstructed in its actual order: temporal centering, empirical session-1 reference with the 750/749 ML-to-ddof-1 conversion, frozen session-1 whitening, and then per-trial OAS. OAS was not commuted with whitening.

Across all records, the largest `W R W - I` residual was {summary['implementation_checks']['maximum_whitening_identity_max_abs']:.3g}; the largest reconstructed-versus-saved EA covariance difference was {summary['implementation_checks']['maximum_saved_ea_cov_abs_difference']:.3g} in saved covariance units, with maximum relative Frobenius difference {summary['implementation_checks']['maximum_saved_ea_cov_relative_difference']:.3g}. These checks support implementation consistency within numerical precision; they do not show that historical EA improves prediction.

At complete source count and h=0, the saved participant scores yield three explicitly separated contrasts: previous-reference EA–TS minus plain TS–LR, recentered MDM minus plain MDM, and recentered MDM minus plain TS–LR. `paired_effects.csv` retains every participant value, while `summary.json` gives conditional descriptive bootstrap intervals. `correlations.csv` reports Pearson and rank correlations between each drift measure and each performance contrast. These are exploratory descriptive associations from overlapping fixed cohorts and have no causal interpretation.

The previously reported 3276.8 µV extreme concerned source participant 5, session 1, CP3, outside the retained epoch support. This diagnostic does not reopen that source signal and it is not attributed to development target 45 or target 2.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--note", type=Path,
                        default=ROOT / "research/review4_20260924/ALIGNMENT_DIAGNOSTIC_NOTE.md")
    args = parser.parse_args()
    args.out_dir, args.note = args.out_dir.resolve(), args.note.resolve()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    if args.out_dir.exists() or args.note.exists():
        raise FileExistsError("output directory and note must be new")

    records, inputs = load_records()
    inputs["participant_means"] = PARTICIPANTS
    before = {name: sha256(path) for name, path in inputs.items()}
    subjects = {dataset: sorted({subject for ds, subject, _ in records if ds == dataset})
                for dataset in DATASETS}
    if any(len(subjects[dataset]) != DATASETS[dataset]["subjects"] for dataset in DATASETS):
        raise ValueError("person coverage differs")

    diagnostics = [diagnose_person(dataset, subject, records)
                   for dataset in DATASETS for subject in subjects[dataset]]
    effects, effect_rows = load_effects(subjects)
    correlations = descriptive_correlations(diagnostics, effects)
    summaries = {}
    for dataset in DATASETS:
        summaries[dataset] = {}
        for contrast in ("previous_ea_minus_plain_ts", "recenter_mdm_minus_plain_mdm",
                         "recenter_mdm_minus_plain_ts"):
            values = [effects[(dataset, subject, contrast)] for subject in subjects[dataset]]
            summaries[dataset][contrast] = effect_summary(values)

    summary = {
        "status": "completed_posthoc_saved_alignment_diagnostic",
        "scope": "approved saved arrays and scores; no fits, protected data, downloads, or causal claims",
        "people": len(diagnostics), "people_by_dataset": {k: len(v) for k, v in subjects.items()},
        "units": {"performance_effect": "percentage points",
                  "affine_invariant_distance": "dimensionless",
                  "covariance_eigenvalues": "signal units squared"},
        "performance_contrasts": summaries,
        "drift_descriptives": {
            dataset: {
                metric: {"mean": statistics.mean(row[metric] for row in diagnostics
                                                   if row["dataset_variant"] == dataset),
                         "median": statistics.median(row[metric] for row in diagnostics
                                                     if row["dataset_variant"] == dataset),
                         "min": min(row[metric] for row in diagnostics
                                    if row["dataset_variant"] == dataset),
                         "max": max(row[metric] for row in diagnostics
                                    if row["dataset_variant"] == dataset)}
                for metric in ("empirical_reference_drift", "oas_riemann_mean_drift")}
            for dataset in DATASETS},
        "implementation_checks": {
            "maximum_whitening_identity_max_abs": max(row["whitening_identity_max_abs"] for row in diagnostics),
            "maximum_saved_reference_abs_difference": max(row["saved_reference_max_abs_difference"] for row in diagnostics),
            "maximum_saved_reference_relative_difference": max(row["saved_reference_max_relative_frobenius"] for row in diagnostics),
            "maximum_saved_ea_cov_abs_difference": max(row["saved_ea_cov_max_abs_difference"] for row in diagnostics),
            "maximum_saved_ea_cov_relative_difference": max(row["saved_ea_cov_max_relative_frobenius"] for row in diagnostics),
            "order": "centered_empirical_ML -> 750/749 session1 reference -> whitening -> OAS",
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "unit": "participant after ten-draw score averaging"},
        "correlations": "descriptive Pearson and average-rank Spearman coefficients; no p-values",
        "limitations": [
            "Post-hoc diagnostic on fixed, overlapping cohorts.",
            "Correlations do not identify a causal effect of drift or alignment.",
            "The diagnostic checks the saved transform implementation, not an optimal alignment rule.",
        ],
    }

    args.out_dir.mkdir(parents=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "person_diagnostics.csv", diagnostics)
    write_csv(args.out_dir / "paired_effects.csv", effect_rows)
    write_csv(args.out_dir / "correlations.csv", correlations)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.note.write_text(note_text(summary), encoding="utf-8")

    after = {name: sha256(path) for name, path in inputs.items()}
    if before != after:
        raise RuntimeError("an input changed during analysis")
    outputs = {}
    for path in (*args.out_dir.iterdir(), args.note):
        outputs[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    receipt = {
        "status": "completed", "analysis_label": "posthoc_saved_alignment_diagnostic",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "threads": 1,
        "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256_before": before[name],
                          "sha256_after": after[name], "bytes": path.stat().st_size}
                   for name, path in inputs.items()},
        "code": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__)},
        "outputs": outputs,
        "runtime": {"cpu_seconds": time.process_time() - start_cpu,
                    "wall_seconds": time.monotonic() - start_wall,
                    "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
    }
    (args.out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    size = sum(path.stat().st_size for path in args.out_dir.iterdir()) + args.note.stat().st_size
    if size > 2 * 1024 * 1024:
        raise RuntimeError("output exceeds 2 MiB")
    print(json.dumps({"status": "completed", "people": len(diagnostics),
                      "cpu_seconds": receipt["runtime"]["cpu_seconds"],
                      "max_rss_kib": receipt["runtime"]["max_rss_kib"]}, sort_keys=True))


if __name__ == "__main__":
    main()
