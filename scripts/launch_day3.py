"""One supervised, exclusive Day3 stage per invocation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from expose.runtime import supervise

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=['preparation', 'grid', 'analysis'])
parser.add_argument('--run-id', required=True)
args = parser.parse_args()
if Path(args.run_id).name != args.run_id or not args.run_id.startswith(args.stage + '_r'):
    raise ValueError('invalid run ID')
script = {'preparation': 'prepare_day3.py', 'grid': 'run_day3.py', 'analysis': 'analyze_day3.py'}[args.stage]
out = ROOT / 'result/day3' / args.run_id
receipt = supervise([sys.executable, str(ROOT / 'scripts' / script), '--run-dir', str(out)],
                    ROOT, out, 14400 if args.stage == 'preparation' else 3600)
print(receipt)
raise SystemExit(0 if receipt['status'] == 'completed' else 1)
