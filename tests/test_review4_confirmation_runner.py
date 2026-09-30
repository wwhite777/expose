"""Synthetic and metadata-only checks for the guarded confirmation runner."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from expose.lee import MOTOR_CHANNELS, load_offline_epochs
from expose.lee_memory import download_verified, load_offline_epochs_from_bytes


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_review4_confirmation.py"
SPEC = importlib.util.spec_from_file_location("review4_confirmation_runner", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
ANALYZER_PATH = ROOT / "scripts/analyze_review4_confirmation.py"
ANALYZER_SPEC = importlib.util.spec_from_file_location("review4_confirmation_analyzer", ANALYZER_PATH)
ANALYZER = importlib.util.module_from_spec(ANALYZER_SPEC)
ANALYZER_SPEC.loader.exec_module(ANALYZER)


def synthetic_mat(tmp_path):
    time = np.arange(9000) / 1000
    frequencies = np.array([12, 15, 18, 20, 22, 25, 10, 16])
    training = {
        "x": np.sin(2 * np.pi * time[:, None] * frequencies) * np.arange(10, 18),
        "t": np.array([1, 5001]),
        "fs": 1000,
        "chan": np.array(MOTOR_CHANNELS, dtype=object),
        "y_dec": np.array([2, 1]),
        "y_class": np.array(["left", "right"], dtype=object),
        "y_logic": np.array([[0, 1], [1, 0]], dtype=np.uint8),
        "class": np.array([["1", "right"], ["2", "left"]], dtype=object),
    }
    path = tmp_path / "synthetic.mat"
    savemat(path, {"EEG_MI_train": training, "EEG_MI_test": {"x": np.nan}})
    return path


def test_in_memory_adapter_is_numerically_identical_to_unchanged_loader(tmp_path):
    path = synthetic_mat(tmp_path)
    expected = load_offline_epochs(path)
    observed = load_offline_epochs_from_bytes(
        path.read_bytes(), "https://public.example/synthetic.mat", subject=1, session=2)
    np.testing.assert_array_equal(observed.X, expected.X)
    np.testing.assert_array_equal(observed.y, expected.y)
    for key in ("trial_raw_support_sha256", "trial_epoch_sha256", "config", "filter",
                "resample", "class_counts", "event_samples_matlab"):
        assert observed.metadata[key] == expected.metadata[key]
    assert observed.metadata["path"] == "memory://openbmi/s01/session2"
    assert observed.metadata["public_uri"] == "https://public.example/synthetic.mat"
    assert observed.metadata["raw_persisted_to_disk"] is False


class FakeResponse:
    def __init__(self, value, declared=None):
        self.value = value
        self.offset = 0
        self.headers = {} if declared is None else {"Content-Length": str(declared)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size):
        block = self.value[self.offset:self.offset + size]
        self.offset += len(block)
        return block


def test_public_download_enforces_frozen_size_and_md5(monkeypatch):
    value = b"public synthetic object"
    md5 = hashlib.md5(value).hexdigest()
    monkeypatch.setattr("expose.lee_memory.urllib.request.urlopen",
                        lambda request, timeout: FakeResponse(value, len(value)))
    observed, identity = download_verified("https://public.example/object.mat", len(value), md5)
    assert observed == value
    assert identity == {"bytes": len(value), "md5": md5,
                        "sha256": hashlib.sha256(value).hexdigest()}
    with pytest.raises(OSError, match="Content-Length"):
        download_verified("https://public.example/object.mat", len(value) + 1, md5)


def test_history_selection_is_balanced_deterministic_and_draw_specific():
    labels = np.repeat([0, 1], 50)
    first = RUNNER.history_indices(labels, draw=0, participant=1)
    assert first == RUNNER.history_indices(labels, draw=0, participant=1)
    assert first != RUNNER.history_indices(labels, draw=1, participant=1)
    assert len(first) == len(set(first)) == 60
    assert {label: sum(labels[index] == label for index in first) for label in (0, 1)} == {0: 30, 1: 30}


def test_authorization_must_bind_exact_packet_and_code_freeze(tmp_path):
    packet = tmp_path / "PACKET_RECEIPT.json"; packet.write_text("packet\n")
    freeze = tmp_path / "CODE_FREEZE.json"; freeze.write_text("freeze\n")
    authorization = tmp_path / "AUTHORIZATION.json"
    record = ROOT / "research/review4_20260924/CONFIRMATION_DECISION_PACKET_v2.md"
    value = {"schema": "review4-confirmation-authorization-v1", "author_adopted": True,
             "protected_access_authorized": True,
             "code_freeze_sha256": RUNNER.sha256(freeze),
             "packet_receipt_sha256": RUNNER.sha256(packet),
             "adoption_record": str(record.relative_to(ROOT)),
             "adoption_record_sha256": RUNNER.sha256(record),
             "decision": "adopt_and_authorize_protected_confirmation_execution"}
    authorization.write_text(json.dumps(value))
    assert RUNNER.validate_authorization(authorization, freeze, packet) == value
    value["code_freeze_sha256"] = "0" * 64
    authorization.write_text(json.dumps(value))
    with pytest.raises(PermissionError, match="authorization is absent"):
        RUNNER.validate_authorization(authorization, freeze, packet)


def test_frozen_role_and_public_inventory_cover_exact_confirmation_people():
    packet = ROOT / "research/review4_20260924/confirmation_source_packet_r001/MACHINE_CONFIG.json"
    machine = json.loads(packet.read_text())
    roles = RUNNER.validate_roles(machine["protected_participant_ids"])
    inventory = RUNNER.validate_public_inventory(machine["protected_participant_ids"])
    assert {subject for subject, role in roles.items() if role == "confirmation"} == set(
        machine["protected_participant_ids"])
    assert set(inventory) == {(subject, session)
                              for subject in machine["protected_participant_ids"]
                              for session in (1, 2)}


def test_existing_source_packet_passes_full_operation_and_membership_validation():
    packet = ROOT / "research/review4_20260924/confirmation_source_packet_r001"
    _, receipt, machine, operations, memberships = RUNNER.validate_packet(packet)
    assert receipt["status"] == "completed_source_only_packet"
    assert len(machine["protected_participant_ids"]) == 24
    assert len(operations) == 1440 and len(memberships) == 971


def test_complete_code_freeze_binds_full_import_closure(tmp_path):
    packet = ROOT / "research/review4_20260924/confirmation_source_packet_r001"
    output = tmp_path / "CODE_FREEZE.json"
    value = RUNNER.freeze(packet, output)
    assert value["status"] == "complete_code_freeze_requires_author_adoption"
    assert value["protected_access_authorized"] is False
    assert set(value["code"]) == set(RUNNER.code_files())
    assert {"package_init", "baseline_helpers"} <= set(value["code"])
    assert value["code"]["analyzer"]["sha256"] == RUNNER.sha256(ANALYZER_PATH)
    authorization = {"code_freeze_sha256": RUNNER.sha256(output)}
    assert ANALYZER.validate_code_freeze(
        output, packet / "PACKET_RECEIPT.json", authorization) == value


def test_analyzer_requires_completed_runner_output_hashes(tmp_path):
    packet = tmp_path / "PACKET_RECEIPT.json"; packet.write_text("packet\n")
    authorization = tmp_path / "AUTHORIZATION.json"; authorization.write_text("authorization\n")
    code_freeze = tmp_path / "CODE_FREEZE.json"
    code_freeze.write_text(json.dumps({"code": {"runner": {"sha256": "runner-code"}}}))
    output = tmp_path / "run"; output.mkdir()
    files = {
        "scores.csv": "header\n",
        "trial_predictions.csv.gz": "predictions-placeholder\n",
        "person_status.csv": "status-placeholder\n",
        "protected_covariance_guards.json": "{}\n",
        "protected_memberships.json": "{}\n",
        "access_log.jsonl": "",
    }
    for name, value in files.items():
        (output / name).write_text(value)
    staged = {"status": "staged_inventory_locked_before_analysis",
              "complete_people": 0, "failed_people": 24,
              "scores_written": 0, "predictions_written": 0,
              "scores_partial_sha256": ANALYZER.sha256(output / "scores.csv")}
    (output / "STAGED_INTEGRITY.json").write_text(json.dumps(staged))
    output_names = set(files) | {"STAGED_INTEGRITY.json"}
    receipt = {"status": "completed_score_inventory_no_analysis",
               "aggregate_analysis_called": False, "raw_MAT_persisted": False,
               "complete_people": 0, "failed_people": 24, "operations": 0, "predictions": 0,
               "packet_receipt_sha256": ANALYZER.sha256(packet),
               "authorization_sha256": ANALYZER.sha256(authorization),
               "code_freeze_sha256": ANALYZER.sha256(code_freeze),
               "runner_sha256": "runner-code",
               "outputs": {name: ANALYZER.sha256(output / name) for name in output_names}}
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    completed = output / "COMPLETED.json"
    completed.write_text(json.dumps({"status": receipt["status"],
                                     "receipt_sha256": ANALYZER.sha256(receipt_path)}))
    assert ANALYZER.validate_runner_binding(receipt_path, completed, output / "scores.csv",
                                            packet, authorization, code_freeze) == receipt
    (output / "scores.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="binding differs|changed after completion"):
        ANALYZER.validate_runner_binding(receipt_path, completed, output / "scores.csv",
                                         packet, authorization, code_freeze)
