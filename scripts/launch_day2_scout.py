"""Supervise the frozen Day2 numerical process; preserve exit/completion evidence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import pathlib,sys
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from expose.runtime import supervise
from expose.scout import load_config
cfg=load_config(ROOT);out=ROOT/'result/day2/scout_r001'
r=supervise([sys.executable,str(ROOT/'scripts/run_day2_scout.py'),'--run-dir',str(out)],ROOT,out,cfg['resource']['scout_wall_seconds'])
print(r)
raise SystemExit(0 if r['status']=='completed' else 1)
