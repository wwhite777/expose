"""Externally supervise the separate Day5 raw verifier; reuse only process/receipt utilities."""
import argparse
import os
from pathlib import Path
import re
import sys

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from expose.runtime import supervise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    run_id = parser.parse_args().run_id
    if not re.fullmatch(r'raw_reference_r[0-9]{3}', run_id):
        raise ValueError('expected versioned raw-reference run ID')
    out = ROOT / 'result/day5' / run_id
    receipt = supervise([sys.executable, str(ROOT / 'scripts/verify_day5_raw_reference.py'),
                         '--run-dir', str(out)], ROOT, out, 1800)
    print(receipt)
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
