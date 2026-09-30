"""Record the required existing and new tests before any grid outcomes."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from datetime import datetime, timezone
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from expose.provenance import file_sha256
from expose.runtime import atomic_write_json

out = ROOT/'result/day3/prefit_checks_r001'
out.mkdir(exist_ok=False)
inputs = {str(p.relative_to(ROOT)):file_sha256(p) for pattern in ['src/expose/*.py', 'scripts/*day3*.py', 'tests/*.py'] for p in ROOT.glob(pattern)}
atomic_write_json(out/'INPUT_MANIFEST.json', inputs)
start = time.monotonic()
cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
command = [sys.executable, '-m', 'pytest', '-q']
with (out/'pytest.log').open('x') as log:
    completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
after = resource.getrusage(resource.RUSAGE_CHILDREN)
receipt = dict(command=command, exit_code=completed.returncode, finished_utc=datetime.now(timezone.utc).isoformat(),
               wall_seconds=time.monotonic()-start, child_cpu_seconds=after.ru_utime+after.ru_stime-cpu.ru_utime-cpu.ru_stime,
               child_peak_rss_kib=after.ru_maxrss, log_sha256=file_sha256(out/'pytest.log'),
               inputs_unchanged=all(file_sha256(ROOT/p)==h for p, h in inputs.items()),
               scope='all historical tests plus new data-selection, policy, calibration and failure-fixture tests before grid fits')
atomic_write_json(out/'receipt.json', receipt)
print((out/'pytest.log').read_text())
print(receipt)
raise SystemExit(completed.returncode or (0 if receipt['inputs_unchanged'] else 1))
