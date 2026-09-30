"""Atomic output commits and supervision of one directly owned child process."""
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


_CHILD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}


@contextmanager
def _atomic_file(path):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, prefix=f".{path.name}.",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path, value):
    """Replace path only after successful serialization, flush, and fsync."""
    with _atomic_file(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_write_csv(path, fieldnames, rows):
    with _atomic_file(path) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_completion(run_dir, pid):
    receipt_path = run_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    marker = json.loads((run_dir / "COMPLETED.json").read_text())
    if not isinstance(receipt, dict) or not isinstance(marker, dict):
        raise ValueError("receipt and completion marker must be JSON objects")
    if (receipt.get("status") != "completed" or type(receipt.get("exit_code")) is not int
            or receipt["exit_code"] != 0 or receipt.get("pid") != pid):
        raise ValueError("child receipt status, exit code, or PID differs")
    run_id = receipt.get("run_id")
    if not isinstance(run_id, str) or not run_id or marker.get("run_id") != run_id:
        raise ValueError("missing or mismatched run identity")
    receipt_hash = _sha256(receipt_path)
    if marker.get("receipt_sha256") != receipt_hash:
        raise ValueError("receipt hash differs")
    outputs = marker.get("outputs")
    if not isinstance(outputs, dict) or len(outputs) < 2 or outputs.get("receipt.json") != receipt_hash:
        raise ValueError("marker must bind the receipt and at least one output")
    for name, expected in outputs.items():
        if (Path(name).name != name or name in {".", "..", "COMPLETED.json", "supervisor_receipt.json"}
                or (run_dir / name).is_symlink()):
            raise ValueError("output must be a direct child file")
        if _sha256(run_dir / name) != expected:
            raise ValueError(f"output hash differs: {name}")
    return {"run_id": run_id, "receipt_sha256": receipt_hash,
            "completion_sha256": _sha256(run_dir / "COMPLETED.json")}


def supervise(argv, cwd, run_dir, wall_seconds):
    """Run one child in a fresh directory; return its terminal supervisor receipt.

    The caller passes run_dir to the child through argv. No shell or descendant
    process management is supplied. A supervisor SIGKILL cannot be finalized.
    """
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("argv must be a nonempty argument sequence")
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive and finite")
    argv = [os.fspath(arg) for arg in argv]
    cwd, run_dir = Path(cwd).resolve(), Path(run_dir).resolve()
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    now = lambda: datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    receipt = {"status": "launching", "supervisor_pid": os.getpid(), "child_pid": None,
               "started_utc": now(), "argv": argv, "cwd": str(cwd), "run_dir": str(run_dir),
               "wall_seconds": wall_seconds, "exit_code": None, "environment_overrides": _CHILD_ENV.copy()}
    destination = run_dir / "supervisor_receipt.json"
    atomic_write_json(destination, receipt)
    child = None
    try:
        with (run_dir / "stdout.log").open("x") as stdout, (run_dir / "stderr.log").open("x") as stderr:
            child = subprocess.Popen(argv, cwd=cwd, stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                                     env={**os.environ, **_CHILD_ENV})
            receipt.update(status="running", child_pid=child.pid, child_started_utc=now())
            atomic_write_json(destination, receipt)
            try:
                code = child.wait(timeout=max(0, wall_seconds - (time.monotonic() - start)))
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait()
                receipt["status"] = "timeout"
            else:
                receipt["status"] = "killed" if code < 0 else "failed" if code else "invalid_completion"
                if code == 0:
                    receipt.update(_validate_completion(run_dir, child.pid))
                    receipt["status"] = "completed"
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        if child is None:
            receipt["status"] = "launch_error"
        elif receipt["status"] == "running":
            receipt["status"] = "supervisor_error"
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            receipt["exit_code"] = child.wait()
        receipt.update(finished_utc=now(), elapsed_seconds=time.monotonic() - start)
        atomic_write_json(destination, receipt)
    return receipt
