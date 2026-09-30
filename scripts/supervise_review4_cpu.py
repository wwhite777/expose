#!/usr/bin/env python3
"""Run one single-process Review-4 command under an external CPU-time ceiling."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_cpu_seconds(pid: int) -> float:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    return (int(fields[13]) + int(fields[14])) / os.sysconf("SC_CLK_TCK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-seconds", type=int, required=True)
    parser.add_argument("--grace-seconds", type=int, default=10)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cpu_seconds <= 0 or args.grace_seconds <= 0:
        parser.error("CPU and grace limits must be positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")

    child = subprocess.Popen(command)

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    old_handlers = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    limit_sent_at = None
    try:
        while child.poll() is None:
            try:
                used = process_cpu_seconds(child.pid)
            except FileNotFoundError:
                break
            if used >= args.cpu_seconds and limit_sent_at is None:
                child.send_signal(signal.SIGXCPU)
                limit_sent_at = time.monotonic()
            elif (
                limit_sent_at is not None
                and time.monotonic() - limit_sent_at >= args.grace_seconds
            ):
                child.kill()
            time.sleep(0.25)
        return child.wait()
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=args.grace_seconds)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
