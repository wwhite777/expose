"""Prepare compact covariance-only BNCI2014-001 external-replication inputs."""

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expose.bnci2014 import load_bnci2014_epochs, official_file_url
from expose.provenance import file_sha256
from expose.review2_controls import oas_covariances


EXPECTED_TOTAL_BYTES = 779_873_919


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing existing output: {path}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing existing output: {path}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _receipt(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 18 or set(rows[0]) != {"filename", "bytes", "sha256", "url"}:
        raise ValueError("download receipt must contain the exact 18-file schema")
    normalized = {}
    for row in rows:
        name = row["filename"]
        if name in normalized or len(row["sha256"]) != 64:
            raise ValueError("duplicate filename or invalid hash in download receipt")
        normalized[name] = {
            "filename": name,
            "bytes": int(row["bytes"]),
            "sha256": row["sha256"],
            "url": row["url"],
        }
    expected = {f"A{subject:02d}{session}.mat" for subject in range(1, 10) for session in ("T", "E")}
    if set(normalized) != expected or sum(row["bytes"] for row in normalized.values()) != EXPECTED_TOTAL_BYTES:
        raise ValueError("download receipt names or aggregate bytes differ")
    return normalized


def _fit_ea_whitener(epochs):
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[0] != 144 or epochs.shape[1:] != (22, 750):
        raise ValueError("external EA requires all 144 session-1 epochs with shape 22x750")
    mean_covariance = np.mean(
        [np.cov(epoch, rowvar=True, ddof=1) for epoch in epochs], axis=0
    )
    mean_covariance = (mean_covariance + mean_covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
    maximum = float(eigenvalues.max())
    if not np.isfinite(eigenvalues).all() or maximum <= 0 or eigenvalues.min() <= 1e-12 * maximum:
        raise ValueError("external EA reference is singular or ill-conditioned; no jitter permitted")
    whitener = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return (whitener + whitener.T) / 2, eigenvalues


def _apply_ea(epochs, whitener):
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[1:] != (22, 750):
        raise ValueError("unexpected external epoch shape")
    aligned = np.einsum("ij,njt->nit", whitener, epochs)
    if not np.isfinite(aligned).all():
        raise ValueError("EA produced nonfinite external signals")
    return aligned


def _validate_covariances(covariance, name, *, require_positive_definite=True):
    covariance = np.asarray(covariance)
    if (
        covariance.ndim != 3
        or covariance.shape[0] == 0
        or covariance.shape[1:] != (22, 22)
        or covariance.dtype != np.float64
    ):
        raise ValueError(f"{name} must have shape (trials,22,22) and dtype float64")
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance, covariance.transpose(0, 2, 1), rtol=0, atol=1e-12
    ):
        raise ValueError(f"{name} contains nonfinite or asymmetric matrices")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if require_positive_definite and np.min(eigenvalues) <= 0:
        raise ValueError(f"{name} contains a non-positive-definite covariance")
    if not require_positive_definite:
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny)
        tolerance = 1e-12 * scale
        if np.min(eigenvalues) < -tolerance:
            raise ValueError(f"{name} contains a materially negative eigenvalue")


def _empirical_covariances(epochs):
    """Per-trial maximum-likelihood covariance after temporal centering."""
    epochs = np.asarray(epochs, dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[1:] != (22, 750):
        raise ValueError("empirical covariance requires trials with shape 22x750")
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    covariance = np.einsum("nct,ndt->ncd", centered, centered) / epochs.shape[-1]
    _validate_covariances(
        covariance, "empirical_cov", require_positive_definite=False
    )
    return covariance


def _prepare_session(
    raw_path, subject, session_letter, whitener, raw_record, output_dir, ea_eigenvalues,
    loaded=None,
):
    if loaded is None:
        loaded = load_bnci2014_epochs(
            raw_path, subject=subject, session=session_letter
        )
    if loaded.X.shape != (144, 22, 750) or loaded.y.shape != (144,):
        raise ValueError("loader output differs from the external data contract")
    if loaded.artifact_flags.shape != (144,) or loaded.metadata["checks"]["artifact_dropped_trials"] != 0:
        raise ValueError("artifact flags are absent or an artifact trial was dropped")
    covariance = oas_covariances(loaded.X)
    ea_covariance = oas_covariances(_apply_ea(loaded.X, whitener))
    empirical_covariance = _empirical_covariances(loaded.X)
    _validate_covariances(covariance, "cov")
    _validate_covariances(ea_covariance, "ea_cov")

    session_number = 1 if session_letter == "T" else 2
    stem = f"subj{subject:02d}_sess{session_number}"
    npz_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.metadata.json"
    _atomic_npz(
        npz_path,
        cov=covariance,
        ea_cov=ea_covariance,
        empirical_cov=empirical_covariance,
        y=loaded.y,
    )
    loader_metadata = dict(loaded.metadata)
    loader_metadata.pop("raw_labels", None)
    loader_metadata.pop("class_counts", None)
    loader_metadata["trial_records"] = [
        {key: value for key, value in record.items() if key not in ("raw_label", "label")}
        for record in loader_metadata["trial_records"]
    ]
    metadata = {
        "dataset": "bnci2014",
        "montage": "22ch",
        "subject": subject,
        "session": session_number,
        "source_session_code": session_letter,
        "role": "external",
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_sha256": raw_record["sha256"],
        "raw_bytes": raw_record["bytes"],
        "raw_url": raw_record["url"],
        "artifact_flags": loaded.artifact_flags.tolist(),
        "artifact_flagged_count": int(loaded.artifact_flags.sum()),
        "artifact_policy": "all left/right trials retained; flags exposed; no outcome exclusion",
        "trial_ids": list(loaded.trial_ids),
        "event_contract": {
            "matlab_trial_indexing": "one-based",
            "trial_onset_to_cue_s": 2.0,
            "independent_filter_input_relative_to_cue_s": [0.0, 4.0],
            "output_window_relative_to_cue_s": [0.5, 3.5],
            "output_window_half_open": True,
        },
        "loader_metadata": loader_metadata,
        "arrays": {
            "cov": "per-trial OAS covariance of the unaligned 22-channel signal",
            "ea_cov": "per-trial OAS covariance after session-1-fitted empirical-centered EA",
            "empirical_cov": "per-trial temporally centered ML covariance (divide by 750)",
            "y": "int64 labels, 0=right hand and 1=left hand",
        },
        "trial_order": "source MAT run order, then increasing trial onset within run",
        "label_storage": "target labels occur only in the NPZ y array; sidecar trial records omit labels",
        "ea_fit": {
            "fit_subject": subject,
            "fit_session": 1,
            "fit_trial_count": 144,
            "labels_used": False,
            "applied_to_session": session_number,
            "frozen_for_session_2": session_number == 2,
            "covariance": "np.cov per trial, rowvar=True, ddof=1; arithmetic mean",
            "eigenvalue_min": float(ea_eigenvalues.min()),
            "eigenvalue_max": float(ea_eigenvalues.max()),
            "relative_minimum": float(ea_eigenvalues.min() / ea_eigenvalues.max()),
            "minimum_threshold": 1e-12,
            "jitter": None,
        },
        "npz_path": str(npz_path.relative_to(ROOT)),
        "npz_sha256": file_sha256(npz_path),
    }
    _atomic_json(metadata_path, metadata)
    del loaded, covariance, ea_covariance, empirical_covariance
    return {
        "subject": subject,
        "session": session_number,
        "role": "external",
        "npz_path": str(npz_path.relative_to(ROOT)),
        "sha256": file_sha256(npz_path),
        "trial_ids": metadata["trial_ids"],
    }, str(metadata_path.relative_to(ROOT)), file_sha256(metadata_path)


def prepare(raw_dir, receipt_path, output_dir):
    raw_dir = Path(raw_dir).resolve()
    receipt_path = Path(receipt_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    rows = _receipt(receipt_path)
    for row in rows.values():
        raw_path = raw_dir / row["filename"]
        if not raw_path.is_file() or raw_path.stat().st_size != row["bytes"]:
            raise ValueError(f"raw size differs: {raw_path}")
        if file_sha256(raw_path) != row["sha256"]:
            raise ValueError(f"raw hash differs: {raw_path}")
        subject = int(row["filename"][1:3])
        session = row["filename"][3]
        if row["url"] != official_file_url(subject, session):
            raise ValueError("raw URL differs from the official loader contract")

    records = []
    sidecars = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("numerical thread count exceeds one")
        for subject in range(1, 10):
            history_name = f"A{subject:02d}T.mat"
            history_path = raw_dir / history_name
            history = load_bnci2014_epochs(history_path, subject=subject, session="T")
            whitener, ea_eigenvalues = _fit_ea_whitener(history.X)
            for session_letter in ("T", "E"):
                filename = f"A{subject:02d}{session_letter}.mat"
                record, sidecar, sidecar_sha = _prepare_session(
                    raw_dir / filename,
                    subject,
                    session_letter,
                    whitener,
                    rows[filename],
                    output_dir,
                    ea_eigenvalues,
                    loaded=history if session_letter == "T" else None,
                )
                records.append(record)
                sidecars.append({"path": sidecar, "sha256": sidecar_sha})
            del history, whitener, ea_eigenvalues

    if len(records) != 18 or any(len(record["trial_ids"]) != 144 for record in records):
        raise ValueError("common index record or trial count differs")
    common_index = {"dataset": "bnci2014", "montage": "22ch", "records": records}
    _atomic_json(output_dir / "common_index.json", common_index)
    receipt = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "raw_download_receipt_path": str(receipt_path.relative_to(ROOT)),
        "raw_download_receipt_sha256": file_sha256(receipt_path),
        "raw_total_bytes": EXPECTED_TOTAL_BYTES,
        "subjects": 9,
        "sessions": 18,
        "trials_per_session": 144,
        "artifact_policy": "retain and expose; no exclusion",
        "common_index_path": str((output_dir / "common_index.json").relative_to(ROOT)),
        "common_index_sha256": file_sha256(output_dir / "common_index.json"),
        "metadata_sidecars": sidecars,
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "threadpools": pools,
        "GPU_used": False,
        "raw_EEG_downloaded": True,
        "raw_EEG_deleted": False,
        "raw_signal_copied_to_derived_cache": False,
        "confirmation_data_accessed": False,
    }
    _atomic_json(output_dir / "PREPARATION_RECEIPT.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir", default=ROOT / "data/raw/bnci2014_001_review3", type=Path
    )
    parser.add_argument(
        "--receipt",
        default=ROOT / "research/review3_20260923/BNCI_DOWNLOAD_RECEIPT.tsv",
        type=Path,
    )
    parser.add_argument(
        "--output", default=ROOT / "data/derived/review3_bnci2014_001", type=Path
    )
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
    signal.alarm(1800)
    print(json.dumps(prepare(args.raw_dir, args.receipt, args.output), allow_nan=False))


if __name__ == "__main__":
    main()
