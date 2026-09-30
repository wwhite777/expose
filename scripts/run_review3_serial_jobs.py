#!/usr/bin/env python3
"""Run a frozen Review-3 job manifest sequentially under per-child limits."""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import resource
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
MEMORY_BYTES = 16 * 1024**3
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
              "CUDA_VISIBLE_DEVICES": ""}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_repo_path(relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("manifest paths must be repository-relative strings")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe manifest path")
    path = (ROOT / pure).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("manifest path escapes repository")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def validate_manifest(manifest):
    fields = {"prerequisite", "jobs", "one_numerical_job", "budget_scope", "files"}
    optional = {"supersedes"}
    if not fields <= set(manifest) or set(manifest) - fields - optional or manifest["one_numerical_job"] is not True:
        raise ValueError("invalid serial-job manifest schema")
    if "supersedes" in manifest:
        prior = manifest["supersedes"]
        if set(prior) != {"path", "sha256", "note"} or not isinstance(prior["note"], str) or not prior["note"]:
            raise ValueError("invalid manifest predecessor link")
        prior_path = safe_repo_path(prior["path"])
        if (not isinstance(prior["sha256"], str) or len(prior["sha256"]) != 64
                or not prior_path.is_file() or prior_path.is_symlink() or sha256(prior_path) != prior["sha256"]):
            raise ValueError("invalid or changed manifest predecessor hash")
    if not isinstance(manifest["budget_scope"], str) or not manifest["budget_scope"]:
        raise ValueError("manifest budget scope is missing")
    safe_repo_path(manifest["prerequisite"])
    if not isinstance(manifest["files"], dict) or not manifest["files"]:
        raise ValueError("manifest frozen-file inventory is empty")
    for relative, digest in manifest["files"].items():
        safe_repo_path(relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("invalid frozen-file digest")
    if not isinstance(manifest["jobs"], list) or not manifest["jobs"]:
        raise ValueError("manifest has no jobs")
    names = set()
    for job in manifest["jobs"]:
        if set(job) != {"name", "argv", "cpu_seconds", "wall_seconds"}:
            raise ValueError("invalid job schema")
        name = job["name"]
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name)
                or name in names):
            raise ValueError("invalid or duplicate job name")
        names.add(name)
        if (not isinstance(job["argv"], list) or not job["argv"]
                or any(not isinstance(value, str) or not value for value in job["argv"])):
            raise ValueError("job argv must be a nonempty string list")
        if (not Path(job["argv"][0]).is_absolute()
                or not Path(job["argv"][0]).is_file()
                or not os.access(job["argv"][0], os.X_OK)):
            raise ValueError("job executable must be an existing absolute file")
        for key in ("cpu_seconds", "wall_seconds"):
            if isinstance(job[key], bool) or not isinstance(job[key], int) or job[key] <= 0:
                raise ValueError("job limits must be positive integers")
    return manifest


def verify_frozen_files(manifest):
    mismatches = []
    for relative, expected in manifest["files"].items():
        path = safe_repo_path(relative)
        observed = sha256(path) if path.is_file() and not path.is_symlink() else None
        if observed != expected:
            mismatches.append({"path": relative, "expected_sha256": expected,
                               "observed_sha256": observed})
    if mismatches:
        raise ValueError("frozen file hash mismatch: " + json.dumps(mismatches, sort_keys=True))


def verify_prerequisite(relative):
    completed_path = safe_repo_path(relative)
    if not completed_path.is_file() or completed_path.is_symlink():
        raise ValueError("prerequisite COMPLETED.json is absent")
    completed = load_json(completed_path)
    receipt_path = completed_path.parent / "receipt.json"
    if (completed.get("status") != "completed" or not receipt_path.is_file()
            or completed.get("receipt_sha256") != sha256(receipt_path)):
        raise ValueError("prerequisite completion or receipt binding differs")
    receipt = load_json(receipt_path)
    if receipt.get("status") != "completed":
        raise ValueError("prerequisite receipt is not completed")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("prerequisite receipt has no output hash inventory")
    for name, expected in outputs.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("unsafe prerequisite output binding")
        path = completed_path.parent / name
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError("prerequisite output hash differs: " + name)
    return {"completed_path": relative, "completed_sha256": sha256(completed_path),
            "receipt_sha256": sha256(receipt_path), "outputs": outputs}


def verify_new_outputs(jobs):
    seen = set()
    for job in jobs:
        argv = job["argv"]
        for flag in ("--out-dir", "--run-dir"):
            positions = [index for index, value in enumerate(argv) if value == flag]
            if len(positions) > 1 or (positions and positions[0] + 1 >= len(argv)):
                raise ValueError("invalid output flag in job " + job["name"])
            if positions:
                value = argv[positions[0] + 1]
                path = safe_repo_path(value)
                identity = str(path)
                if identity in seen:
                    raise ValueError("duplicate planned output path")
                seen.add(identity)
                if path.exists():
                    raise FileExistsError("planned child output already exists: " + value)


def child_limits(cpu_seconds):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 10))
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    return apply


def children_cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def execute_job(index, job, launch_dir, environment):
    log_path = launch_dir / f"{index:02d}_{job['name']}.log"
    current_path = launch_dir / "CURRENT_JOB.json"
    started_utc = datetime.now(timezone.utc).isoformat()
    started_wall, before_cpu = time.monotonic(), children_cpu_seconds()
    timed_out, returncode, error = False, None, None
    process = None
    with log_path.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"event": "child_start", "name": job["name"],
                              "argv": job["argv"], "started_utc": started_utc,
                              "cpu_seconds_limit": job["cpu_seconds"],
                              "wall_seconds_limit": job["wall_seconds"],
                              "memory_bytes_limit": MEMORY_BYTES}, sort_keys=True) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(job["argv"], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       text=True, preexec_fn=child_limits(job["cpu_seconds"]))
            atomic_json(current_path, {"state": "running", "index": index, "name": job["name"],
                                       "pid": process.pid, "argv": job["argv"], "started_utc": started_utc,
                                       "log": log_path.name, "cpu_seconds_limit": job["cpu_seconds"],
                                       "wall_seconds_limit": job["wall_seconds"],
                                       "memory_bytes_limit": MEMORY_BYTES})
            try:
                returncode = process.wait(timeout=job["wall_seconds"])
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
        except BaseException as caught:
            error = {"type": type(caught).__name__, "message": str(caught)}
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            returncode = process.returncode if process is not None else None
        finally:
            ended_utc = datetime.now(timezone.utc).isoformat()
            record = {"index": index, "name": job["name"], "argv": job["argv"],
                      "started_utc": started_utc, "ended_utc": ended_utc,
                      "wall_seconds": time.monotonic() - started_wall,
                      "child_cpu_seconds": children_cpu_seconds() - before_cpu,
                      "cpu_seconds_limit": job["cpu_seconds"],
                      "wall_seconds_limit": job["wall_seconds"],
                      "memory_bytes_limit": MEMORY_BYTES, "returncode": returncode,
                      "timed_out": timed_out, "error": error, "pid": None if process is None else process.pid,
                      "state": "finished", "log": log_path.name, "log_sha256": None}
            log.write(json.dumps({"event": "child_end", **record}, sort_keys=True) + "\n")
    record["log_sha256"] = sha256(log_path)
    atomic_json(current_path, record)
    return record


def run(manifest_path, launch_dir):
    manifest_path = Path(manifest_path).resolve()
    manifest = validate_manifest(load_json(manifest_path))
    verify_frozen_files(manifest)
    prerequisite = verify_prerequisite(manifest["prerequisite"])
    verify_new_outputs(manifest["jobs"])
    launch_dir = Path(launch_dir).resolve()
    if ROOT.resolve() not in launch_dir.parents:
        raise ValueError("launch ledger directory must be inside the repository")
    if launch_dir.exists():
        raise FileExistsError("launch ledger directory must be new")
    launch_dir.mkdir(parents=True)
    environment = dict(os.environ)
    environment.update(THREAD_ENV)
    ledger = {"status": "running", "type": "strict_serial_execution",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
              "supervisor_sha256": sha256(__file__), "prerequisite": prerequisite,
              "thread_environment": THREAD_ENV, "memory_bytes_per_child": MEMORY_BYTES,
              "planned_jobs": [job["name"] for job in manifest["jobs"]], "jobs": []}
    ledger_path = launch_dir / "ledger.json"
    atomic_json(ledger_path, ledger)
    try:
        for index, job in enumerate(manifest["jobs"]):
            verify_frozen_files(manifest)
            record = execute_job(index, job, launch_dir, environment)
            ledger["jobs"].append(record)
            atomic_json(ledger_path, ledger)
            if record["returncode"] != 0 or record["timed_out"] or record["error"] is not None:
                ledger.update(status="failed_stopped_on_first_failure",
                              stopped_after=job["name"],
                              finished_utc=datetime.now(timezone.utc).isoformat())
                atomic_json(ledger_path, ledger)
                return 1, ledger
        ledger.update(status="completed", finished_utc=datetime.now(timezone.utc).isoformat(),
                      summed_child_cpu_seconds=sum(job["child_cpu_seconds"] for job in ledger["jobs"]),
                      summed_child_wall_seconds=sum(job["wall_seconds"] for job in ledger["jobs"]))
        atomic_json(ledger_path, ledger)
        return 0, ledger
    except BaseException as error:
        ledger.update(status="supervisor_error_stopped", finished_utc=datetime.now(timezone.utc).isoformat(),
                      supervisor_error={"type": type(error).__name__, "message": str(error)})
        atomic_json(ledger_path, ledger)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--launch-dir", required=True, type=Path)
    args = parser.parse_args()
    status, ledger = run(args.manifest, args.launch_dir)
    print(json.dumps({"status": ledger["status"], "ledger": str(args.launch_dir / "ledger.json")},
                     indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
