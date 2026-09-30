"""Inspect BNCI2014-004 session schema without preparing arrays or fitting models."""
# ruff: noqa: E402

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import resource
import signal
import sys
from urllib.request import Request, urlopen

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expose.bnci2014 import _integer_vector, _runs
from expose.provenance import file_sha256


MAX_DOWNLOAD_BYTES = 1024**3
CUE_OFFSET_SAMPLES = 750
SUPPORT_SAMPLES = 1000
EXPECTED_CONTRACT_SHA256 = "d3f57674ae94e6530e5de61349d263b910282b1faf816a6638d289f9fbdc4015"


def _inside_root(path):
    path = Path(path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("diagnostic paths must remain inside the project root") from error
    return path


def _contract(path, expected_sha256):
    path = _inside_root(path)
    if expected_sha256 != EXPECTED_CONTRACT_SHA256 or file_sha256(path) != expected_sha256:
        raise ValueError("data contract differs from the authorized identity")
    value = json.loads(path.read_text())
    files = value.get("source_files")
    if not isinstance(files, list) or len(files) != 9:
        raise ValueError("authorized contract must contain nine source files")
    normalized = {}
    for subject, record in enumerate(files, start=1):
        if record.get("subject") != subject or record.get("filename") != f"B{subject:02d}T.mat":
            raise ValueError("source-file order or identity differs")
        if not isinstance(record.get("content_length"), int) or not 0 < record["content_length"] <= MAX_DOWNLOAD_BYTES:
            raise ValueError("invalid contracted content length")
        normalized[subject] = record
    return normalized, path


def _fetch(record):
    request = Request(record["url"], headers={"User-Agent": "EXPOSE-BNCI2B-schema/1.0"})
    with urlopen(request, timeout=120) as response:
        if response.geturl() != record["url"]:
            raise ValueError("source URL redirected outside the authorized identity")
        header = response.headers.get("Content-Length")
        if header is None or int(header) != record["content_length"]:
            raise ValueError("HTTP content length differs from the authorized contract")
        raw = bytearray()
        while True:
            chunk = response.read(1024**2)
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > record["content_length"] or len(raw) > MAX_DOWNLOAD_BYTES:
                raise ValueError("download exceeded its authorized in-memory bound")
    if len(raw) != record["content_length"]:
        raise ValueError("downloaded byte count differs from Content-Length")
    return bytes(raw)


def _session_summary(run, session):
    if not isinstance(run, dict):
        raise ValueError(f"session {session} is not a MATLAB struct")
    missing = sorted({"X", "trial", "y", "fs", "classes", "artifacts"} - run.keys())
    if missing:
        raise ValueError(f"session {session} missing fields: {missing}")
    signal_array = np.asarray(run["X"])
    if signal_array.ndim != 2 or signal_array.dtype.kind not in "iuf":
        raise ValueError(f"session {session} X is not a numeric samples-by-channels matrix")
    trials = _integer_vector(run["trial"], f"session {session} trial")
    labels = _integer_vector(run["y"], f"session {session} y")
    artifacts = _integer_vector(run["artifacts"], f"session {session} artifacts")
    fs = _integer_vector(run["fs"], f"session {session} fs")
    if labels.size != trials.size or artifacts.size != trials.size or fs.size != 1:
        raise ValueError(f"session {session} trial/y/artifact/fs lengths are inconsistent")
    if not np.all(np.isin(labels, (1, 2))):
        raise ValueError(f"session {session} contains labels outside classes 1 and 2")
    increasing_one_based = bool(trials.size and np.all(trials >= 1) and np.all(np.diff(trials) > 0))
    cue_onsets = trials - 1 + CUE_OFFSET_SAMPLES
    support_in_bounds = bool(
        increasing_one_based
        and cue_onsets[0] >= 0
        and cue_onsets[-1] + SUPPORT_SAMPLES <= signal_array.shape[0]
    )
    support_nonoverlap = bool(
        increasing_one_based and np.all(np.diff(cue_onsets) >= SUPPORT_SAMPLES)
    )
    return {
        "session": session,
        "official_session_name": f"0{session}T",
        "trial_count": int(trials.size),
        "class_counts": {
            "left_raw_1": int(np.sum(labels == 1)),
            "right_raw_2": int(np.sum(labels == 2)),
        },
        "artifact_flag_count": int(np.sum(artifacts == 1)),
        "sampling_hz": int(fs[0]),
        "raw_channel_count": int(signal_array.shape[1]),
        "selected_eeg_channel_count": 3,
        "index_support": {
            "matlab_one_based_strictly_increasing": increasing_one_based,
            "cue_offset_samples": CUE_OFFSET_SAMPLES,
            "support_samples": SUPPORT_SAMPLES,
            "all_supports_in_bounds": support_in_bounds,
            "supports_nonoverlapping": support_nonoverlap,
        },
    }


def _known_subject(subject, staging, source_record):
    sessions = []
    raw_hashes = set()
    lengths = set()
    for session in (1, 2):
        path = staging / f"subj{subject:02d}_sess{session}.metadata.json"
        value = json.loads(path.read_text())
        loader = value["loader_metadata"]
        if value["subject"] != subject or value["session"] != session:
            raise ValueError("preserved sidecar subject/session differs")
        raw_hashes.add(value["raw_sha256"])
        lengths.add(value["content_length"])
        class_counts = loader["class_counts"]
        sessions.append({
            "session": session,
            "official_session_name": loader["official_session_name"],
            "trial_count": int(loader["shape"][0]),
            "class_counts": {
                "left_raw_1": int(class_counts["left_hand"]),
                "right_raw_2": int(class_counts["right_hand"]),
            },
            "artifact_flag_count": int(loader["artifact_flagged_count"]),
            "sampling_hz": int(loader["raw_fs"]),
            "raw_channel_count": len(loader["raw_channel_names"]),
            "selected_eeg_channel_count": len(loader["channel_names"]),
            "index_support": {
                "matlab_one_based_strictly_increasing": True,
                "cue_offset_samples": CUE_OFFSET_SAMPLES,
                "support_samples": SUPPORT_SAMPLES,
                "all_supports_in_bounds": bool(loader["checks"]["segments_in_bounds_and_nonoverlapping"]),
                "supports_nonoverlapping": bool(loader["checks"]["segments_in_bounds_and_nonoverlapping"]),
            },
        })
    if len(raw_hashes) != 1 or lengths != {source_record["content_length"]}:
        raise ValueError("preserved sidecar raw identity differs from contract")
    return {
        "subject": subject,
        "schema_source": "preserved_failed_r001_sidecars",
        "input_url": source_record["url"],
        "content_length": source_record["content_length"],
        "raw_sha256": raw_hashes.pop(),
        "aggregate_session_struct_count": 3,
        "sessions": sessions,
    }


def inspect(contract_path, contract_sha256, known_staging, output_path):
    sources, contract_path = _contract(contract_path, contract_sha256)
    known_staging = _inside_root(known_staging)
    output_path = _inside_root(output_path)
    if output_path.exists() or output_path.with_name(output_path.name + ".tmp").exists():
        raise FileExistsError(f"refusing existing diagnostic output: {output_path}")
    records = [_known_subject(subject, known_staging, sources[subject]) for subject in (1, 2, 3)]
    for subject in range(4, 10):
        raw_bytes = _fetch(sources[subject])
        raw_sha256 = sha256(raw_bytes).hexdigest()
        mat = loadmat(BytesIO(raw_bytes), variable_names=["data"], simplify_cells=True)
        aggregate = _runs(mat.get("data")) if mat.get("data") is not None else []
        if len(aggregate) < 2:
            raise ValueError(f"subject {subject} aggregate has fewer than two session structs")
        records.append({
            "subject": subject,
            "schema_source": "authorized_in_memory_stream",
            "input_url": sources[subject]["url"],
            "content_length": len(raw_bytes),
            "raw_sha256": raw_sha256,
            "aggregate_session_struct_count": len(aggregate),
            "sessions": [
                _session_summary(aggregate[index], index + 1) for index in range(2)
            ],
        })
        del raw_bytes, mat, aggregate
    session_counts = [session["trial_count"] for row in records for session in row["sessions"]]
    value = {
        "status": "completed_schema_diagnostic_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path.relative_to(ROOT)),
        "contract_sha256": contract_sha256,
        "diagnostic_script_sha256": file_sha256(Path(__file__).resolve()),
        "subjects": records,
        "summary": {
            "subject_count": len(records),
            "selected_session_count": len(session_counts),
            "trial_counts_observed": sorted(set(session_counts)),
            "all_sampling_hz_250": all(
                session["sampling_hz"] == 250 for row in records for session in row["sessions"]
            ),
            "all_raw_channel_counts_6": all(
                session["raw_channel_count"] == 6 for row in records for session in row["sessions"]
            ),
            "all_selected_eeg_channel_counts_3": all(
                session["selected_eeg_channel_count"] == 3
                for row in records for session in row["sessions"]
            ),
            "all_index_support_valid": all(
                session["index_support"]["all_supports_in_bounds"]
                and session["index_support"]["supports_nonoverlapping"]
                for row in records for session in row["sessions"]
            ),
            "model_fits": 0,
            "predictions": 0,
            "raw_eeg_persisted": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, output_path)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "research/review4_20260924/BNCI2B_DATA_CONTRACT_v1.json",
    )
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument(
        "--known-staging",
        type=Path,
        default=ROOT / "data/derived/.review4_bnci2014_004.incomplete.84912",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 31))
    signal.alarm(600)
    print(json.dumps(inspect(
        args.contract,
        args.contract_sha256,
        args.known_staging,
        args.output,
    ), allow_nan=False))


if __name__ == "__main__":
    main()
