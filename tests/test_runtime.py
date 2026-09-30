"""Tiny subprocess fixtures; never read EEG data or launch numerical jobs."""
import csv
import json
import signal
import sys

import pytest

from expose.runtime import atomic_write_csv, atomic_write_json, supervise


def child_command(run_dir, defect="good"):
    script = """
import hashlib, json, os, pathlib, signal, sys, time
out = pathlib.Path(sys.argv[1])
defect = sys.argv[2]
if defect == 'exit_error': sys.exit(7)
if defect == 'killed': os.kill(os.getpid(), signal.SIGKILL)
if defect == 'timeout': time.sleep(60)
print('child stdout', flush=True)
print('child stderr', file=sys.stderr, flush=True)
receipt = {'status': 'completed', 'exit_code': 0, 'pid': os.getpid(), 'run_id': 'fixture'}
if defect == 'bad_status': receipt['status'] = 'running'
if defect == 'bad_pid': receipt['pid'] = -1
(out / 'receipt.json').write_text(json.dumps(receipt))
(out / 'values.csv').write_text('value\\n1\\n')
(out / 'environment.json').write_text(json.dumps({key: os.environ.get(key) for key in
    ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES']}))
sha = lambda name: hashlib.sha256((out / name).read_bytes()).hexdigest()
marker = {'run_id': 'fixture', 'receipt_sha256': sha('receipt.json'),
          'outputs': {name: sha(name) for name in ['receipt.json', 'values.csv', 'environment.json']}}
if defect == 'missing_completion': sys.exit(0)
if defect == 'bad_hash': marker['receipt_sha256'] = 'wrong'
if defect == 'bad_output': (out / 'values.csv').write_text('corrupted')
if defect == 'bad_run_id': marker['run_id'] = 'other'
if defect == 'missing_outputs': marker['outputs'] = {}
(out / 'COMPLETED.json').write_text('{' if defect == 'invalid_json' else json.dumps(marker))
"""
    return [sys.executable, "-c", script, str(run_dir), defect]


@pytest.mark.parametrize("defect,status,exit_code", [
    ("good", "completed", 0), ("exit_error", "failed", 7),
    ("killed", "killed", -signal.SIGKILL), ("timeout", "timeout", -signal.SIGKILL),
    *[(defect, "invalid_completion", 0) for defect in ["missing_completion", "invalid_json",
      "bad_hash", "bad_output", "bad_status", "bad_pid", "bad_run_id", "missing_outputs"]],
])
def test_child_outcomes_are_finalized(tmp_path, monkeypatch, defect, status, exit_code):
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        monkeypatch.setenv(key, "8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    run_dir = tmp_path / "run"
    argv = child_command(run_dir, defect)
    receipt = supervise(argv, tmp_path, run_dir, 0.2 if defect == "timeout" else 5)
    assert receipt == json.loads((run_dir / "supervisor_receipt.json").read_text())
    assert receipt["status"] == status
    assert receipt["exit_code"] == exit_code
    assert receipt["argv"] == argv
    assert receipt["child_pid"] > 0
    assert receipt["started_utc"] <= receipt["finished_utc"]
    if status == "completed":
        assert receipt["run_id"] == "fixture"
        assert len(receipt["completion_sha256"]) == 64
        assert (run_dir / "stdout.log").read_text() == "child stdout\n"
        assert (run_dir / "stderr.log").read_text() == "child stderr\n"
        assert json.loads((run_dir / "environment.json").read_text()) == {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


def test_launch_error_is_finalized(tmp_path):
    run_dir = tmp_path / "run"
    receipt = supervise([str(tmp_path / "missing_program")], tmp_path, run_dir, 5)
    assert receipt["status"] == "launch_error"
    assert receipt["child_pid"] is None and receipt["exit_code"] is None
    assert receipt == json.loads((run_dir / "supervisor_receipt.json").read_text())


def test_existing_run_is_untouched(tmp_path):
    run_dir = tmp_path / "run"
    argv = child_command(run_dir)
    supervise(argv, tmp_path, run_dir, 5)
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    with pytest.raises(FileExistsError):
        supervise(argv, tmp_path, run_dir, 5)
    assert before == {p.name: p.read_bytes() for p in run_dir.iterdir()}


def test_atomic_json_and_csv(tmp_path):
    atomic_write_json(tmp_path / "output.json", {"value": 2})
    atomic_write_csv(tmp_path / "output.csv", ["value"], [{"value": "comma, newline\n"}])
    assert json.loads((tmp_path / "output.json").read_text()) == {"value": 2}
    with (tmp_path / "output.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == [{"value": "comma, newline\n"}]
    assert (tmp_path / "output.json").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("format", ["json", "csv"])
@pytest.mark.parametrize("failure", ["serialize", "fsync", "replace"])
def test_atomic_failure_preserves_destination(tmp_path, monkeypatch, format, failure):
    destination = tmp_path / f"output.{format}"
    destination.write_bytes(b"previous evidence\n")
    def fail(*args):
        raise OSError("fixture I/O failure")
    if failure != "serialize":
        monkeypatch.setattr(f"expose.runtime.os.{failure}", fail)
    with pytest.raises((TypeError, ValueError, OSError)):
        if format == "json":
            atomic_write_json(destination, {"value": object() if failure == "serialize" else 2})
        else:
            rows = [{"value": 1}, {"invalid" if failure == "serialize" else "value": 2}]
            atomic_write_csv(destination, ["value"], rows)
    assert destination.read_bytes() == b"previous evidence\n"
    assert list(tmp_path.iterdir()) == [destination]
