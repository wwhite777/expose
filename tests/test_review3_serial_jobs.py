import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]


def load_supervisor():
    spec = importlib.util.spec_from_file_location("review3_serial_jobs", ROOT / "scripts/run_review3_serial_jobs.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def job(name, code, wall=5):
    return {"name": name, "argv": [sys.executable, "-c", code], "cpu_seconds": 30, "wall_seconds": wall}


def test_execute_job_records_running_pid_then_finished_success(tmp_path):
    runner = load_supervisor(); launch = tmp_path / "launch"; launch.mkdir(); result = {}
    thread = threading.Thread(target=lambda: result.setdefault("record", runner.execute_job(
        0, job("success", "import time; time.sleep(.25)"), launch, dict(runner.THREAD_ENV))))
    thread.start()
    deadline = time.monotonic() + 3
    while not (launch / "CURRENT_JOB.json").exists() and time.monotonic() < deadline: time.sleep(.01)
    current = json.loads((launch / "CURRENT_JOB.json").read_text())
    assert current["state"] == "running" and current["pid"] > 0
    assert current["argv"] == job("success", "import time; time.sleep(.25)")["argv"]
    thread.join(timeout=5); assert not thread.is_alive()
    finished = json.loads((launch / "CURRENT_JOB.json").read_text())
    assert result["record"]["returncode"] == 0 and finished["state"] == "finished"
    assert finished["log_sha256"] == runner.sha256(launch / finished["log"])


def test_execute_job_preserves_nonzero_and_timeout_records(tmp_path):
    runner = load_supervisor(); launch = tmp_path / "launch"; launch.mkdir(); env = dict(runner.THREAD_ENV)
    failed = runner.execute_job(0, job("nonzero", "raise SystemExit(7)"), launch, env)
    timeout = runner.execute_job(1, job("timeout", "import time; time.sleep(30)", wall=1), launch, env)
    assert failed["returncode"] == 7 and failed["timed_out"] is False
    assert timeout["timed_out"] is True and timeout["returncode"] != 0
    assert json.loads((launch / "CURRENT_JOB.json").read_text())["name"] == "timeout"


def test_run_stops_after_first_failure(tmp_path, monkeypatch):
    runner = load_supervisor(); monkeypatch.setattr(runner, "ROOT", tmp_path)
    frozen = tmp_path / "frozen.txt"; frozen.write_text("fixed")
    prereq = tmp_path / "result" / "pre"; prereq.mkdir(parents=True); output = prereq / "out.txt"; output.write_text("ok")
    receipt = {"status": "completed", "outputs": {"out.txt": hashlib.sha256(output.read_bytes()).hexdigest()}}
    (prereq / "receipt.json").write_text(json.dumps(receipt))
    (prereq / "COMPLETED.json").write_text(json.dumps({"status": "completed", "receipt_sha256": hashlib.sha256((prereq / "receipt.json").read_bytes()).hexdigest()}))
    marker = tmp_path / "must_not_exist.txt"
    manifest = {"prerequisite": "result/pre/COMPLETED.json", "one_numerical_job": True, "budget_scope": "fixture",
                "files": {"frozen.txt": hashlib.sha256(frozen.read_bytes()).hexdigest()}, "jobs": [
                    job("first_failure", "raise SystemExit(3)"),
                    job("must_not_run", f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')")]} 
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    status, ledger = runner.run(path, tmp_path / "launch")
    assert status == 1 and ledger["status"] == "failed_stopped_on_first_failure"
    assert [row["name"] for row in ledger["jobs"]] == ["first_failure"] and not marker.exists()
