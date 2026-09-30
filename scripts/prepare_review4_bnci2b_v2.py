"""Prepare v2 compact BNCI2014-004 covariances with observed trial counts."""
# ruff: noqa: E402

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import resource
import signal
import sys
import time
from urllib.request import Request, urlopen

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expose.bnci2b_v2 import (
    EEG_CHANNELS,
    EXPECTED_SESSION_TRIAL_COUNTS,
    load_bnci2b_screening_sessions,
    official_file_url,
)
from expose.provenance import file_sha256
from expose.review2_controls import oas_covariances


MAX_DOWNLOAD_BYTES = 1024**3
MAX_DURABLE_OUTPUT_BYTES = 3_000_000
EXPECTED_SUBJECTS = tuple(range(1, 10))
SCRIPT_PATH = Path(__file__).resolve()
LOADER_PATH = ROOT / "src/expose/bnci2b_v2.py"
SCHEMA_SUMMARY_PATH = (
    ROOT / "result/review4_20260924/bnci2b_schema_diagnostic_r001/schema_summary.json"
)
SCHEMA_SUMMARY_SHA256 = "e8be3b087f6c0e8ea0b5cd07741a5fbef5d19f5b85b5d3400128d4403cb77c7b"
SOURCE_IDENTITIES = {
    1: (34203437, "0da6e77ab0dab5b4aa1d2d5a6a542ac02f6768d3b7a76b0abe896ec1cf259919"),
    2: (33060824, "1c4ace3eee8d72ca184fa6995a9466939b71175b8b3a129bdaa838a85adf6473"),
    3: (35581243, "4ae1b23b4b4359151787a1a61d3acea128719677c49bfaf0113748cffece3b98"),
    4: (35749247, "e7230d6d28a9e81d3afdf7ba5d363979e0186dbbc5a73f66d4b713be020ce960"),
    5: (35047531, "53166e2cf1a97576262b3f2f27395f489af6fda15ade5c3b3cdd2c89f7df0d79"),
    6: (36512412, "d69f679ea33121a9202901d5b62efb50f8ed004fa8ad70d10a59351493ffdad8"),
    7: (33061422, "0b79522d66710a88f852b83802c7bffd015226d765480f60f16b6bedc5ea33c2"),
    8: (38046816, "d96bdfd0c8331f238740c10ff03121dc3e3a20a4b1254840c5cc4878aa149b67"),
    9: (33324824, "4f017991a64ceabb2ca73abf6f3c620b2b8fd7468184f52a8444dfdec1ee2adc"),
}


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


def _inside_root(path):
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("output and contract must remain inside the project root") from error
    return path


def _load_approved_contract(contract_path, approved_contract_sha256, approved_script_sha256):
    contract_path = _inside_root(contract_path)
    if file_sha256(contract_path) != approved_contract_sha256:
        raise ValueError("contract hash differs from the explicit root approval")
    actual_script_sha256 = file_sha256(SCRIPT_PATH)
    if actual_script_sha256 != approved_script_sha256:
        raise ValueError("preparation script hash differs from the explicit root approval")
    contract = json.loads(contract_path.read_text())
    expected_fields = {
        "contract_id", "status", "dataset", "source_files", "selection",
        "preprocessing", "durable_arrays", "index_contract", "resource_bounds",
        "source_evidence", "schema_diagnostic", "observed_session_trial_counts",
        "preparation_script_sha256",
        "loader_sha256",
    }
    if set(contract) != expected_fields:
        raise ValueError("data contract has unexpected top-level fields")
    if contract["contract_id"] != "BNCI2B_DATA_CONTRACT_v2" or contract["status"] != "prefit_pending_root_approval":
        raise ValueError("unexpected contract identity or status")
    if contract["preparation_script_sha256"] != actual_script_sha256:
        raise ValueError("contract does not bind the executing preparation script")
    if contract["loader_sha256"] != file_sha256(LOADER_PATH):
        raise ValueError("BNCI2B loader hash differs from the reviewed contract")
    if contract["dataset"] != {
        "code": "BNCI2014-004", "short_name": "bnci2014b", "montage": "3ch",
        "subjects": list(EXPECTED_SUBJECTS), "sessions": [1, 2],
        "role": "external", "channels": list(EEG_CHANNELS), "sampling_hz": 250,
    }:
        raise ValueError("dataset identity differs from the reviewed contract")
    source_files = contract["source_files"]
    if len(source_files) != 9:
        raise ValueError("contract must bind exactly nine source files")
    for subject, record in zip(EXPECTED_SUBJECTS, source_files):
        content_length, raw_sha256 = SOURCE_IDENTITIES[subject]
        expected = {
            "subject": subject,
            "filename": f"B{subject:02d}T.mat",
            "url": official_file_url(subject),
            "content_length": content_length,
            "raw_sha256": raw_sha256,
        }
        if record != expected:
            raise ValueError("source file contract differs from official identity")
        if not (0 < record["content_length"] <= MAX_DOWNLOAD_BYTES):
            raise ValueError("source file exceeds the one-GiB in-memory limit")
    if contract["selection"] != {
        "mat_struct_indices_zero_based": [0, 1],
        "official_sessions": ["01T", "02T"],
        "chronology": "session1_to_session2",
        "session1_use": "source_person_or_target_history",
        "session2_use": "target_evaluation",
        "feedback": False,
        "trial_policy": "retain_all_observed_balanced_trials",
    }:
        raise ValueError("session selection differs from the reviewed contract")
    if contract["preprocessing"] != {
        "trial_indexing": "MATLAB_one_based_trial_onset",
        "cue_offset_seconds": 3.0,
        "filter_support_relative_to_cue_seconds": [0.0, 4.0],
        "bandpass_hz": [8.0, 30.0],
        "filter": "Butterworth_order4_SOS_forward_backward_per_trial",
        "epoch_relative_to_cue_seconds": [0.5, 3.5],
        "epoch_interval": "half_open",
        "samples": 750,
        "eeg_channels": list(EEG_CHANNELS),
        "artifact_policy": "retain_all_and_expose_flags",
        "ea_reference": "all_available_session1_epochs_label_free_ddof1_mean",
        "ea_application": "same_session1_whitener_to_sessions1_and2_before_OAS",
    }:
        raise ValueError("preprocessing differs from the reviewed contract")
    observed_counts = {
        str(subject): list(EXPECTED_SESSION_TRIAL_COUNTS[subject])
        for subject in EXPECTED_SUBJECTS
    }
    if contract["observed_session_trial_counts"] != observed_counts:
        raise ValueError("observed session trial matrix differs from the reviewed contract")
    if contract["schema_diagnostic"] != {
        "path": str(SCHEMA_SUMMARY_PATH.relative_to(ROOT)),
        "sha256": SCHEMA_SUMMARY_SHA256,
        "status": "completed_schema_diagnostic_only",
        "model_fits": 0,
        "predictions": 0,
    } or file_sha256(SCHEMA_SUMMARY_PATH) != SCHEMA_SUMMARY_SHA256:
        raise ValueError("schema diagnostic identity differs from the reviewed contract")
    if contract["durable_arrays"] != ["cov", "ea_cov", "empirical_cov", "y"]:
        raise ValueError("durable array set differs from the reviewed contract")
    if contract["index_contract"] != {
        "filename": "dataset_index.json", "records": 18,
        "record_fields": ["subject", "session", "role", "npz_path", "sha256", "trial_ids"],
        "role": "external", "sessions_per_subject": [1, 2],
    }:
        raise ValueError("index schema differs from the reviewed contract")
    if contract["resource_bounds"] != {
        "network_bytes": 314587756,
        "maximum_source_file_bytes": 38046816,
        "hard_source_file_limit_bytes": MAX_DOWNLOAD_BYTES,
        "process_address_space_limit_bytes": 16 * 1024**3,
        "numerical_threads": 1,
        "classifier_fits": 0,
        "raw_or_epoch_arrays_persisted": False,
        "maximum_durable_output_bytes": MAX_DURABLE_OUTPUT_BYTES,
    }:
        raise ValueError("resource bounds differ from the reviewed contract")
    return contract, contract_path, actual_script_sha256


def _fetch_bytes(record):
    request = Request(record["url"], headers={"User-Agent": "EXPOSE-BNCI2B/1.0"})
    with urlopen(request, timeout=120) as response:
        if response.geturl() != record["url"]:
            raise ValueError("source URL redirected outside the approved identity")
        header = response.headers.get("Content-Length")
        if header is None or int(header) != record["content_length"]:
            raise ValueError("HTTP content length differs from the approved contract")
        raw = bytearray()
        while True:
            chunk = response.read(1024**2)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_DOWNLOAD_BYTES or len(raw) > record["content_length"]:
                raise ValueError("download exceeded its approved in-memory byte bound")
    if len(raw) != record["content_length"]:
        raise ValueError("downloaded byte count differs from HTTP content length")
    return bytes(raw)


def _validate_covariance(values, name, count):
    values = np.asarray(values)
    if values.dtype != np.float64 or values.shape != (count, 3, 3):
        raise ValueError(f"{name} must have shape (trials,3,3) and dtype float64")
    if not np.isfinite(values).all() or not np.allclose(
        values, values.transpose(0, 2, 1), rtol=0, atol=1e-12
    ):
        raise ValueError(f"{name} contains nonfinite or asymmetric matrices")
    if np.min(np.linalg.eigvalsh(values)) <= 0:
        raise ValueError(f"{name} contains a non-positive-definite covariance")


def _fit_ea_whitener(history_epochs):
    epochs = np.asarray(history_epochs, dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[0] not in (120, 160) or epochs.shape[1:] != (3, 750):
        raise ValueError("EA requires all 120 or 160 session-1 epochs with shape 3x750")
    if not np.isfinite(epochs).all():
        raise ValueError("EA requires finite session-1 epochs")
    mean_covariance = np.mean(
        [np.cov(epoch, rowvar=True, ddof=1) for epoch in epochs], axis=0
    )
    mean_covariance = (mean_covariance + mean_covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
    maximum = float(eigenvalues.max())
    if maximum <= 0 or eigenvalues.min() <= 1e-12 * maximum:
        raise ValueError("session-1 EA reference is singular or ill-conditioned")
    whitener = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return (whitener + whitener.T) / 2, eigenvalues


def _covariance_arrays(epochs, labels, whitener):
    epochs = np.asarray(epochs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    whitener = np.asarray(whitener, dtype=np.float64)
    if epochs.ndim != 3 or epochs.shape[1:] != (3, 750) or labels.shape != (len(epochs),):
        raise ValueError("session epochs must have shape (trials,3,750) with aligned labels")
    if (
        whitener.shape != (3, 3)
        or not np.isfinite(whitener).all()
        or not np.allclose(whitener, whitener.T, rtol=0, atol=1e-12)
    ):
        raise ValueError("EA whitener must be a finite 3x3 matrix")
    centered = epochs - epochs.mean(axis=-1, keepdims=True)
    empirical = np.einsum("nct,ndt->ncd", centered, centered) / epochs.shape[-1]
    covariance = oas_covariances(epochs)
    aligned = np.einsum("ij,njt->nit", whitener, epochs)
    ea_covariance = oas_covariances(aligned)
    _validate_covariance(covariance, "cov", len(epochs))
    _validate_covariance(ea_covariance, "ea_cov", len(epochs))
    _validate_covariance(empirical, "empirical_cov", len(epochs))
    return {
        "cov": covariance,
        "ea_cov": ea_covariance,
        "empirical_cov": empirical,
        "y": labels,
    }


def _record_path(output_dir, subject, session, suffix):
    return output_dir / f"subj{subject:02d}_sess{session}.{suffix}"


def _validate_dataset_index(index):
    if set(index) != {"dataset", "montage", "records"}:
        raise ValueError("dataset index has unexpected top-level fields")
    if index["dataset"] != "bnci2014b" or index["montage"] != "3ch":
        raise ValueError("dataset index identity differs")
    records = index["records"]
    if len(records) != 18:
        raise ValueError("dataset index must contain exactly 18 records")
    expected_fields = {"subject", "session", "role", "npz_path", "sha256", "trial_ids"}
    seen_cells, seen_trials = set(), set()
    for record in records:
        if set(record) != expected_fields or record["role"] != "external":
            raise ValueError("dataset index contains an unexpected field or role")
        cell = (record["subject"], record["session"])
        if cell in seen_cells or record["subject"] not in EXPECTED_SUBJECTS or record["session"] not in (1, 2):
            raise ValueError("dataset index contains an invalid or duplicate subject/session")
        expected_count = EXPECTED_SESSION_TRIAL_COUNTS[cell[0]][cell[1] - 1]
        if len(record["sha256"]) != 64 or len(record["trial_ids"]) != expected_count:
            raise ValueError("dataset index hash or trial count differs")
        if len(set(record["trial_ids"])) != expected_count or seen_trials.intersection(record["trial_ids"]):
            raise ValueError("dataset index trial IDs are not globally unique")
        seen_cells.add(cell)
        seen_trials.update(record["trial_ids"])
    expected_cells = {(subject, session) for subject in EXPECTED_SUBJECTS for session in (1, 2)}
    if seen_cells != expected_cells:
        raise ValueError("dataset index is missing a required subject/session")


def prepare(contract_path, output_dir, approved_contract_sha256, approved_script_sha256):
    contract, contract_path, script_sha256 = _load_approved_contract(
        contract_path, approved_contract_sha256, approved_script_sha256
    )
    output_dir = _inside_root(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.incomplete.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"refusing existing staging directory: {staging}")
    staging.mkdir()

    started_wall, started_cpu = time.monotonic(), time.process_time()
    records, sidecars, raw_inputs = [], [], []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("numerical thread count exceeds one")
        for source_record in contract["source_files"]:
            subject = source_record["subject"]
            raw_bytes = _fetch_bytes(source_record)
            raw_sha256 = sha256(raw_bytes).hexdigest()
            if raw_sha256 != source_record["raw_sha256"]:
                raise ValueError("downloaded source SHA-256 differs from the approved contract")
            loaded = load_bnci2b_screening_sessions(raw_bytes, subject=subject)
            raw_inputs.append({
                "subject": subject,
                "input_url": source_record["url"],
                "content_length": len(raw_bytes),
                "raw_sha256": raw_sha256,
            })
            del raw_bytes
            whitener, ea_eigenvalues = _fit_ea_whitener(loaded.sessions[0].X)
            whitener_sha256 = sha256(
                np.ascontiguousarray(whitener).tobytes(order="C")
            ).hexdigest()
            for session in loaded.sessions:
                session_number = session.metadata["session"]
                arrays = _covariance_arrays(session.X, session.y, whitener)
                intended_npz = _record_path(output_dir, subject, session_number, "npz")
                staging_npz = _record_path(staging, subject, session_number, "npz")
                _atomic_npz(staging_npz, **arrays)
                npz_sha256 = file_sha256(staging_npz)
                metadata = {
                    "dataset": "bnci2014b",
                    "dataset_full_name": "BNCI2014-004 / BCI Competition IV 2b",
                    "montage": "3ch",
                    "channel_names": list(EEG_CHANNELS),
                    "subject": subject,
                    "session": session_number,
                    "official_session_name": session.metadata["official_session_name"],
                    "role": "external",
                    "allowed_study_use": session.metadata["session_role"],
                    "input_url": source_record["url"],
                    "content_length": source_record["content_length"],
                    "raw_sha256": raw_sha256,
                    "artifact_flags": session.artifact_flags.tolist(),
                    "artifact_flagged_count": int(session.artifact_flags.sum()),
                    "artifact_policy": "all labeled trials retained; flags exposed; no exclusion",
                    "trial_count": len(session.trial_ids),
                    "class_counts": session.metadata["class_counts"],
                    "trial_ids": list(session.trial_ids),
                    "trial_order": session.metadata["trial_order"],
                    "arrays": {
                        "cov": "per-trial OAS covariance after temporal centering by sklearn OAS",
                        "ea_cov": "per-trial OAS covariance after the frozen session-1 EA transform",
                        "empirical_cov": "per-trial temporally centered ML covariance; divisor 750",
                        "y": "int64; 0=right hand and 1=left hand",
                    },
                    "ea_fit": {
                        "fit_subject": subject,
                        "fit_session": 1,
                        "fit_trial_count": len(loaded.sessions[0].trial_ids),
                        "labels_used": False,
                        "applied_to_session": session_number,
                        "frozen_for_session_2": session_number == 2,
                        "covariance": "np.cov per trial, rowvar=True, ddof=1; arithmetic mean",
                        "whitener": whitener.tolist(),
                        "whitener_sha256": whitener_sha256,
                        "eigenvalues": ea_eigenvalues.tolist(),
                        "minimum_relative_eigenvalue": float(
                            ea_eigenvalues.min() / ea_eigenvalues.max()
                        ),
                        "minimum_threshold": 1e-12,
                        "jitter": None,
                    },
                    "loader_metadata": session.metadata,
                    "npz_path": str(intended_npz.relative_to(ROOT)),
                    "npz_sha256": npz_sha256,
                }
                staging_metadata = _record_path(staging, subject, session_number, "metadata.json")
                _atomic_json(staging_metadata, metadata)
                records.append({
                    "subject": subject,
                    "session": session_number,
                    "role": "external",
                    "npz_path": str(intended_npz.relative_to(ROOT)),
                    "sha256": npz_sha256,
                    "trial_ids": list(session.trial_ids),
                })
                sidecars.append({
                    "path": str(_record_path(output_dir, subject, session_number, "metadata.json").relative_to(ROOT)),
                    "sha256": file_sha256(staging_metadata),
                })
                del arrays
            del loaded, whitener, ea_eigenvalues

    index = {"dataset": "bnci2014b", "montage": "3ch", "records": records}
    _validate_dataset_index(index)
    _atomic_json(staging / "dataset_index.json", index)
    receipt = {
        "status": "completed_covariance_preparation_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpreter": sys.executable,
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": approved_contract_sha256,
        "preparation_script_sha256": script_sha256,
        "raw_inputs": raw_inputs,
        "subjects": 9,
        "sessions": 18,
        "observed_session_trial_counts": {
            str(subject): list(EXPECTED_SESSION_TRIAL_COUNTS[subject])
            for subject in EXPECTED_SUBJECTS
        },
        "role": "external",
        "channels": list(EEG_CHANNELS),
        "artifact_policy": "retain and expose; no exclusion",
        "dataset_index_path": str((output_dir / "dataset_index.json").relative_to(ROOT)),
        "dataset_index_sha256": file_sha256(staging / "dataset_index.json"),
        "metadata_sidecars": sidecars,
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "threadpools": pools,
        "classifier_fits": 0,
        "GPU_used": False,
        "raw_EEG_persisted": False,
        "epoch_arrays_persisted": False,
        "confirmation_data_accessed": False,
    }
    _atomic_json(staging / "PREPARATION_RECEIPT.json", receipt)
    durable_bytes = sum(path.stat().st_size for path in staging.iterdir())
    if durable_bytes > MAX_DURABLE_OUTPUT_BYTES:
        raise ValueError("durable covariance preparation exceeds its three-MB bound")
    os.replace(staging, output_dir)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/review4_20260924/BNCI2B_DATA_CONTRACT_v2.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/derived/review4_bnci2014_004_v2"
    )
    parser.add_argument("--approved-contract-sha256", required=True)
    parser.add_argument("--approved-script-sha256", required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (900, 910))
    signal.alarm(1800)
    print(json.dumps(prepare(
        args.contract,
        args.output,
        args.approved_contract_sha256,
        args.approved_script_sha256,
    ), allow_nan=False))


if __name__ == "__main__":
    main()
