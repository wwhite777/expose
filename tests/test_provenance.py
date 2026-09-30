"""Destroy only temporary known fixtures, never the real frozen files."""
import copy
import hashlib
import json
import numpy as np
import pytest
from expose.provenance import file_sha256, verify_cache


def fixture_cache(root):
    X = np.arange(12, dtype="<f8").reshape(2, 2, 3)
    y = np.array([0, 1])
    raw = "data/raw/lee2019/session1/s5/sess01_subj05_EEG_MI.mat"
    meta = {"path": str(root / raw), "units": "V", "raw_units": "uV", "raw_fs": 1000,
            "fs": 250, "class_names": ["right_hand", "left_hand"], "label_mapping": {"1": 0, "2": 1},
            "raw_class_mapping": {"1": "right", "2": "left"}, "loaded_variables": ["EEG_MI_train"],
            "raw_labels": [1, 2], "trial_epoch_sha256": [hashlib.sha256(x.tobytes()).hexdigest() for x in X]}
    base = root / "data/derived/day1/subj05_sess01_offline"
    base.parent.mkdir(parents=True)
    np.savez(base.with_suffix(".npz"), X=X, y=y)
    base.with_suffix(".json").write_text(json.dumps(meta))
    cache = {}
    for kind, suffix in [("array", ".npz"), ("metadata", ".json")]:
        path = base.with_suffix(suffix)
        cache[kind+"_path"] = str(path.relative_to(root))
        cache[kind+"_sha256"] = file_sha256(path)
    schema = {"records": [{"subject": 5, "session": 1, "role": "source", "status": "pass_with_documented_smt_one_sample_offset", "metadata": copy.deepcopy(meta), "cache": cache}]}
    download = {"status": "completed", "exit_code": 0, "files": [{"path": raw, "subject_role": "source", "md5": "known", "published_md5": "known", "actual_bytes": 1, "bytes": 1}]}
    return X, y, meta, schema, download


def test_valid_cache_identity(tmp_path):
    verify_cache(tmp_path, 5, 1, "source", *fixture_cache(tmp_path))


@pytest.mark.parametrize("defect", ["session_swap", "labels", "epochs", "bytes", "failed_download", "failed_schema", "unit"])
def test_cache_detector_rejects_relevant_mutations(tmp_path, defect):
    X, y, meta, schema, download = fixture_cache(tmp_path)
    if defect == "session_swap": meta["path"] = str(tmp_path / "data/raw/lee2019/session2/s5/sess02_subj05_EEG_MI.mat")
    elif defect == "labels": y = y[::-1]
    elif defect == "epochs": X = X + 1
    elif defect == "bytes": (tmp_path/schema["records"][0]["cache"]["array_path"]).write_bytes(b"corrupt")
    elif defect == "failed_download": download["status"] = "failed"
    elif defect == "failed_schema": schema["records"][0]["status"] = "failed"
    elif defect == "unit": meta["units"] = "uV"
    with pytest.raises(ValueError):
        verify_cache(tmp_path, 5, 1, "source", X, y, meta, schema, download)
