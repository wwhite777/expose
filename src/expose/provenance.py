"""Bind Day1 cached arrays to the verified raw-file and schema receipts."""
import hashlib
from pathlib import Path
import numpy as np


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_cache(root, subject, session, role, X, y, meta, schema, download):
    root = Path(root).resolve()
    if download.get("status") != "completed" or download.get("exit_code") != 0:
        raise ValueError("download did not complete successfully")
    records = [r for r in schema["records"] if (r["subject"], r["session"]) == (subject, session)]
    if len(records) != 1 or records[0]["role"] != role or records[0]["status"] != "pass_with_documented_smt_one_sample_offset":
        raise ValueError("missing or invalid schema record")
    record = records[0]
    expected_raw = f"data/raw/lee2019/session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
    if Path(meta["path"]).resolve() != (root / expected_raw).resolve():
        raise ValueError("raw subject/session lineage differs")
    raw = [r for r in download["files"] if r["path"] == expected_raw]
    if len(raw) != 1 or raw[0]["subject_role"] != role or raw[0]["md5"] != raw[0]["published_md5"] or raw[0]["actual_bytes"] != raw[0]["bytes"]:
        raise ValueError("raw file lacks a successful matching integrity receipt")
    cache = record["cache"]
    for kind, suffix in [("array", ".npz"), ("metadata", ".json")]:
        expected = f"data/derived/day1/subj{subject:02d}_sess{session:02d}_offline{suffix}"
        if cache[kind + "_path"] != expected or file_sha256(root / expected) != cache[kind + "_sha256"]:
            raise ValueError("cached file hash or identity differs from producer")
    if meta != record["metadata"]:
        raise ValueError("sidecar differs from producer metadata")
    required = {"units": "V", "raw_units": "uV", "raw_fs": 1000, "fs": 250,
                "class_names": ["right_hand", "left_hand"], "label_mapping": {"1": 0, "2": 1},
                "raw_class_mapping": {"1": "right", "2": "left"}, "loaded_variables": ["EEG_MI_train"]}
    if any(meta.get(k) != v for k, v in required.items()):
        raise ValueError("unit, rate, class or loaded-variable contract differs")
    if not np.array_equal(y, np.asarray(meta["raw_labels"]) - 1):
        raise ValueError("labels differ from verified raw mapping")
    hashes = [hashlib.sha256(np.asarray(epoch, dtype="<f8").tobytes(order="C")).hexdigest() for epoch in X]
    if hashes != meta["trial_epoch_sha256"]:
        raise ValueError("epoch content differs from verified source conversion")
