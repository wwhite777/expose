"""Launch exactly one frozen development-control process with durable supervision."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from expose.runtime import supervise

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',required=True,type=Path)
args=parser.parse_args()
out=args.run_dir.resolve()
receipt=supervise([sys.executable,ROOT/'scripts/run_review2_controls.py','--run-dir',out],
                  ROOT,out,14400)
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if receipt['status']=='completed' else 1)
