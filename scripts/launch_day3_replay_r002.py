"""Supervise the fixed r002 model-replay consumer in a fresh evidence directory."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from expose.runtime import supervise


def main():
    out = ROOT/'result/day3/model_replay_r002'
    receipt = supervise([sys.executable, str(ROOT/'scripts/verify_day3_r002.py')], ROOT, out, 3600)
    print(receipt)
    return 0 if receipt['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
