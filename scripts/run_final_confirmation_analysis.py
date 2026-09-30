#!/usr/bin/env python3
"""Run the unchanged frozen analyzer with its omitted stdlib gzip import supplied.

The failed original CLI attempt and this mechanical repair are recorded in
research/final_draft_20260924/ANALYSIS_ENTRYPOINT_REPAIR_v1.json. No validation,
authorization, scoring, aggregation, interval or decision code is replaced.
"""

import gzip
import hashlib
import json
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "research/review4_20260924/CONFIRMATION_CODE_FREEZE_v2.json"
FROZEN_SHA256 = "80d5b26e42cd64e758607ae70d8e6e5dc2c389a01ef80001516ecc2cd867bdff"


def main():
    assert hashlib.sha256(FREEZE.read_bytes()).hexdigest() == FROZEN_SHA256
    freeze = json.loads(FREEZE.read_text())
    original = ROOT / freeze["code"]["analyzer"]["path"]
    assert hashlib.sha256(original.read_bytes()).hexdigest() == freeze["code"]["analyzer"]["sha256"]
    sys.argv[0] = str(original)
    # Supplying the missing name is equivalent to adding `import gzip`; the
    # original analyzer still runs all of its freeze and authorization guards.
    runpy.run_path(str(original), init_globals={"gzip": gzip}, run_name="__main__")


if __name__ == "__main__":
    main()
