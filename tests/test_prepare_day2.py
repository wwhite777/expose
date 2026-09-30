"""Synthetic byte-stream and manifest integrity tests; no raw EEG or network."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_day2.py"
spec = importlib.util.spec_from_file_location("prepare_day2", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture
def selection():
    roles = {1: "confirmation", 2: "development", 5: "source"}
    plan, inventory, checksums = [], {}, {}
    for subject, session in ((5, 1), (2, 1), (2, 2)):
        relative = f"session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat"
        url = "https://example.test/data/" + relative
        plan.append({"key": "live/pub/100542/" + relative, "url": url, "bytes": 12, "etag": '"frozen"'})
        inventory[relative] = {"subject_id": str(subject), "session_id": str(session), "relative_path": relative,
                               "url": url, "bytes": "12", "etag": "frozen", "archive_md5": "a" * 32}
        checksums[relative] = "a" * 32
    return plan, roles, inventory, checksums


def test_frozen_selection_has_exact_source_and_development_sessions(selection):
    plan, roles, inventory, checksums = selection
    items = prepare.validate_plan(plan, roles, inventory, checksums, [5], [2])
    assert [(item["subject"], item["session"], item["role"]) for item in items] == [(5, 1, "source"), (2, 1, "development"), (2, 2, "development")]
    assert all(item["published_md5"] == "a" * 32 for item in items)


@pytest.mark.parametrize("change", ["missing", "duplicate", "url", "bytes", "etag", "published", "inventory_md5", "confirmation", "path_disagreement"])
def test_manifest_drift_and_confirmation_are_rejected(selection, change):
    plan, roles, inventory, checksums = selection
    if change == "missing":
        plan.pop()
    elif change == "duplicate":
        plan.append(copy.deepcopy(plan[0]))
    elif change == "url":
        plan[0]["url"] = "https://other.test/mat"
    elif change == "bytes":
        plan[0]["bytes"] = 13
    elif change == "etag":
        plan[0]["etag"] = "changed"
    elif change == "published":
        plan[0]["published_md5"] = "b" * 32
    elif change == "inventory_md5":
        next(iter(inventory.values()))["archive_md5"] = "b" * 32
    elif change == "confirmation":
        roles[2] = "confirmation"
    elif change == "path_disagreement":
        plan[0]["key"] = plan[0]["key"].replace("subj05", "subj06")
    with pytest.raises(ValueError):
        prepare.validate_plan(plan, roles, inventory, checksums, [5], [2])


@pytest.fixture
def raw_bytes():
    payload = b"synthetic offline bytes\x00\x01"
    hashes = {"actual_bytes": len(payload), "bytes": len(payload), "md5": hashlib.md5(payload).hexdigest(),
              "published_md5": hashlib.md5(payload).hexdigest(), "sha256": hashlib.sha256(payload).hexdigest()}
    item = {"url": "https://example.test/raw.mat", "bytes": len(payload), "published_md5": hashes["md5"], "etag": '"fixed"'}
    return payload, item, hashes


def test_verified_existing_raw_is_reused_without_modification(tmp_path, raw_bytes):
    payload, item, prior = raw_bytes
    path = tmp_path / "raw.mat"
    path.write_bytes(payload)
    before = path.stat().st_mtime_ns
    actual = prepare.verify_existing_raw(path, item, prior)
    assert actual["sha256"] == prior["sha256"]
    assert path.stat().st_mtime_ns == before
    assert path.read_bytes() == payload


@pytest.mark.parametrize("change", ["missing_receipt", "wrong_sha", "wrong_md5", "wrong_size", "corrupt_local"])
def test_existing_raw_requires_matching_prior_and_physical_hashes(tmp_path, raw_bytes, change):
    payload, item, prior = raw_bytes
    path = tmp_path / "raw.mat"
    path.write_bytes(payload)
    if change == "missing_receipt":
        prior = None
    elif change == "wrong_sha":
        prior["sha256"] = "0" * 64
    elif change == "wrong_md5":
        prior["md5"] = "0" * 32
    elif change == "wrong_size":
        prior["actual_bytes"] += 1
    elif change == "corrupt_local":
        path.write_bytes(payload[:-1] + b"x")
    with pytest.raises(ValueError):
        prepare.verify_existing_raw(path, item, prior)


class Response:
    def __init__(self, chunks, etag='"fixed"'):
        self.chunks = chunks
        self.headers = {"ETag": etag}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk


def test_download_is_published_only_after_full_hash_validation(tmp_path, raw_bytes):
    payload, item, prior = raw_bytes
    path = tmp_path / "raw.mat"
    storage_requests = []
    actual = prepare.stream_download(path, item, get=lambda *args, **kwargs: Response([payload[:5], b"", payload[5:]]), check_storage=storage_requests.append)
    assert path.read_bytes() == payload
    assert not path.with_suffix(".mat.part").exists()
    assert actual["sha256"] == prior["sha256"]
    assert storage_requests == [len(payload)]


@pytest.mark.parametrize("failure", ["truncated", "wrong_md5", "overflow", "timeout", "etag"])
def test_failed_download_never_publishes_raw_and_preserves_received_partial(tmp_path, raw_bytes, failure):
    payload, item, prior = raw_bytes
    path = tmp_path / "raw.mat"
    chunks, etag = [payload], '"fixed"'
    if failure == "truncated":
        chunks = [payload[:-1]]
    elif failure == "wrong_md5":
        chunks = [payload[:-1] + b"x"]
    elif failure == "overflow":
        chunks = [payload, b"extra"]
    elif failure == "timeout":
        chunks = [payload[:5], TimeoutError("synthetic network timeout")]
    elif failure == "etag":
        etag = "changed"
    with pytest.raises((ValueError, TimeoutError)):
        prepare.stream_download(path, item, get=lambda *args, **kwargs: Response(chunks, etag), check_storage=lambda reserve: None)
    assert not path.exists()
    if failure != "etag":
        assert path.with_suffix(".mat.part").exists()


@pytest.mark.parametrize("suffix", [".mat", ".mat.part"])
def test_download_refuses_overwriting_original_or_failed_partial(tmp_path, raw_bytes, suffix):
    _, item, _ = raw_bytes
    path = tmp_path / "raw.mat"
    existing = path.with_suffix(suffix)
    existing.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        prepare.stream_download(path, item, get=lambda *args, **kwargs: pytest.fail("must not request network"), check_storage=lambda reserve: None)
    assert existing.read_bytes() == b"preserved"


def test_storage_rejection_precedes_network(tmp_path, raw_bytes):
    _, item, _ = raw_bytes
    def refuse(reserve):
        raise ValueError("storage cap")
    with pytest.raises(ValueError, match="storage cap"):
        prepare.stream_download(tmp_path / "raw.mat", item, get=lambda *args, **kwargs: pytest.fail("network reached"), check_storage=refuse)


def records():
    return [{"metadata": {"raw_labels": [1, 2], "trial_raw_support_sha256": ["a" * 64, "b" * 64], "trial_epoch_sha256": ["c" * 64, "d" * 64]}}]


def test_unique_raw_and_processed_trials_are_both_counted():
    counts = prepare.assert_unique_trials(records())
    assert counts["trial_raw_support_sha256"]["trials"] == 2
    assert counts["trial_epoch_sha256"]["unique_hashes"] == 2


@pytest.mark.parametrize("key", ["trial_raw_support_sha256", "trial_epoch_sha256"])
def test_duplicate_content_across_records_is_rejected(key):
    data = records()
    duplicate = copy.deepcopy(data[0])
    other_key = "trial_epoch_sha256" if key == "trial_raw_support_sha256" else "trial_raw_support_sha256"
    duplicate["metadata"][other_key] = ["e" * 64, "f" * 64]
    data.append(duplicate)
    with pytest.raises(ValueError, match="duplicate selected trial"):
        prepare.assert_unique_trials(data)


@pytest.mark.parametrize("hashes", [["broken", "a" * 64], ["a" * 64]])
def test_invalid_or_misaligned_trial_hashes_raise(hashes):
    data = records()
    data[0]["metadata"]["trial_raw_support_sha256"] = hashes
    with pytest.raises(ValueError):
        prepare.assert_unique_trials(data)


def test_atomic_json_serialization_failure_keeps_existing_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    prepare.write_json(path, {"status": "running"})
    with pytest.raises(ValueError):
        prepare.write_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text()) == {"status": "running"}


def test_existing_run_directory_is_never_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    out = tmp_path / "result/day2/preparation_r001"
    out.mkdir(parents=True)
    (out / "receipt.json").write_text("preserved")
    with pytest.raises(FileExistsError):
        prepare.main([])
    assert (out / "receipt.json").read_text() == "preserved"
