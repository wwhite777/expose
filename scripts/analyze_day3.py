"""Analyze the completed frozen grid; keep descriptive development scope explicit."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from expose.development import load_config, operations
from expose.development_analysis import summarize
from expose.development_validation import completed_stage, validate_grid
from expose.provenance import file_sha256 as sha
from expose.runtime import atomic_write_csv, atomic_write_json


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def tree_bytes(paths):
    seen, total = set(), 0
    for root in paths:
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink():
                stat = p.stat()
                identity = stat.st_dev, stat.st_ino
                if identity not in seen:
                    seen.add(identity); total += stat.st_size
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    out = parser.parse_args().run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'receipt.json').exists():
        raise FileExistsError('existing analysis evidence')
    started, cpu = time.monotonic(), time.process_time()
    rec = dict(run_id=out.name, pid=os.getpid(), status='running', started_utc=datetime.now(timezone.utc).isoformat())
    atomic_write_json(out/'receipt.json', rec)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
        def expired(*_):
            raise TimeoutError('analysis resource limit')
        signal.signal(signal.SIGXCPU, expired); signal.signal(signal.SIGALRM, expired); signal.alarm(3600)
        cfg = load_config(ROOT)
        grid, prep = ROOT/'result/day3/grid_r001', ROOT/'result/day3/preparation_r001'
        required = {'receipt.json', 'INPUT_MANIFEST.json', 'predictions.csv', 'operation_membership.csv', 'timing.csv'}
        for op in operations(cfg):
            required.update(op['operation_id']+suffix for suffix in ('.json', '.joblib', '.predictions.csv', '.membership.csv'))
        marker = completed_stage(ROOT, grid, required)
        if set(marker['outputs']) != required:
            raise ValueError('grid output set differs')
        paths = ['research/protocols/DAY3_CONFIG_v1.json', 'research/protocols/DAY3_CONFIG_v1.json.sha256',
                 'research/day3/PROTOCOL_v1.md', 'research/day3/ARTIFACT_CONTRACT_v1.md',
                 'src/expose/development.py', 'src/expose/development_validation.py', 'src/expose/development_analysis.py',
                 'src/expose/runtime.py', 'scripts/analyze_day3.py', 'scripts/launch_day3.py',
                 'result/day3/preparation_r001/SPLIT_MANIFEST.csv']
        paths += [str((grid/p).relative_to(ROOT)) for p in ('COMPLETED.json', 'supervisor_receipt.json', *sorted(required & {'receipt.json', 'INPUT_MANIFEST.json', 'predictions.csv', 'operation_membership.csv', 'timing.csv'}))]
        manifest = {p: sha(ROOT/p) for p in paths}
        atomic_write_json(out/'INPUT_MANIFEST.json', manifest)
        predictions, members, timings = [read_csv(grid/p) for p in ('predictions.csv', 'operation_membership.csv', 'timing.csv')]
        split = read_csv(prep/'SPLIT_MANIFEST.csv')
        checks = validate_grid(predictions, members, timings, split, cfg)
        metrics, summary, controls = summarize(predictions, timings, cfg)
        preparation = json.loads((prep/'receipt.json').read_text())
        run = json.loads((grid/'receipt.json').read_text())
        measured = dict(preparation_cpu_seconds=preparation['cpu_seconds'], grid_cpu_seconds=run['cpu_seconds'],
                        stage_cpu_core_hours_so_far=(preparation['cpu_seconds']+run['cpu_seconds']+time.process_time()-cpu)/3600,
                        project_plus_environment_bytes=tree_bytes([ROOT, Path(sys.executable).parent.parent]),
                        peak_rss_kib=max(preparation['peak_rss_kib'], run['peak_rss_kib'], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                        gpu_hours=0, scope='stage receipts, not complete shell/test/audit session CPU accounting')
        measured['within_caps'] = (measured['stage_cpu_core_hours_so_far'] <= 20
                                  and measured['project_plus_environment_bytes'] <= 30_000_000_000
                                  and measured['peak_rss_kib'] <= 16*1024**2)
        if not measured['within_caps']:
            summary['disposition'] = 'ERROR'
        decision = dict(disposition=summary['disposition'], primary_mixed_gains=summary['primary_mixed_gains'],
                        diagnostic_contrasts=summary['diagnostic_contrasts'], controls_passed=controls['passed'],
                        resources_within_caps=measured['within_caps'], scientific_gate_passed=False,
                        confirmation_accessed=False, scope='frozen exploratory development routing; not scientific GO/KILL')
        for name, value in [('summary.json', summary), ('controls.json', controls), ('decision.json', decision),
                            ('integrity.json', checks), ('resources.json', measured)]:
            atomic_write_json(out/name, value)
        atomic_write_csv(out/'subject_metrics.csv', list(metrics[0]), metrics)
        choices = [{**r, 'training_subjects': json.dumps(r['training_subjects']), 'dose_specific': json.dumps(r['dose_specific'])}
                   for r in summary['lopo_choices']]
        atomic_write_csv(out/'lopo_choices.csv', list(choices[0]), choices)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout='constrained')
        colors = ['#2563eb', '#d97706', '#64748b']
        for arm, ax in [('practical', axes[0]), ('fixed_total', axes[1])]:
            for model, color in zip(cfg['model_order'], colors):
                rows = summary['curves'][arm+'/'+model]
                ax.plot(cfg['history_doses_per_class'], [r['mean'] for r in rows], 'o-', label=model.upper().replace('_', '–'), color=color)
            ax.axhline(.5, color='gray', linestyle=':')
            ax.set(xlabel='History trials per class', ylabel='Mean balanced accuracy', ylim=(.45, 1),
                   xticks=cfg['history_doses_per_class'], title='Full source + history' if arm=='practical' else '100-trial replacement control')
            ax.legend(fontsize=8)
        selected = summary['curves']['selected_calibration']
        axes[0].plot(cfg['history_doses_per_class'], [r['mean'] for r in selected], 's--', color='#7c3aed', label='LOPO calibration')
        axes[0].legend(fontsize=8)
        labels = ['a0', 'a_uniform', 'simple_calibration']
        values = [summary['primary_mixed_gains'][k] for k in labels]
        means = np.array([r['mean'] for r in values])*100
        intervals = np.array([r['descriptive_ci95'] for r in values]).T*100
        axes[2].errorbar(means, range(3), xerr=np.abs(intervals-means), fmt='o', color='#334155', capsize=4)
        axes[2].axvline(0, color='gray', linestyle=':'); axes[2].axvline(2, color='#b91c1c', linestyle='--', label='2 pp point threshold')
        axes[2].set(yticks=range(3), yticklabels=['source-selected', 'uniform-selected', 'calibration'],
                    xlabel='Mixed BA gain (percentage points)', title='LOPO aπ versus required comparators')
        axes[2].legend(fontsize=8)
        fig.suptitle('EXPOSE development · 12 people · 2 history draws · descriptive intervals · '+summary['disposition'], fontsize=11)
        for ext in ('png', 'pdf', 'svg'):
            fig.savefig(out/('development_summary.'+ext), dpi=160)
        plt.close(fig)
        if any(sha(ROOT/p) != h for p, h in manifest.items()):
            raise ValueError('analysis input changed')
        rec.update(status='completed', exit_code=0, disposition=summary['disposition'])
    except BaseException as error:
        rec.update(status='failed', exit_code=1, error=repr(error))
        traceback.print_exc()
    finally:
        rec.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
                   cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_write_json(out/'receipt.json', rec)
        signal.alarm(0)
    if rec['status']=='completed':
        outputs = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json', '.csv', '.png', '.pdf', '.svg')
                   and p.name not in ('COMPLETED.json', 'supervisor_receipt.json')}
        atomic_write_json(out/'COMPLETED.json', dict(run_id=out.name, receipt_sha256=sha(out/'receipt.json'), outputs=outputs))
    print(json.dumps(rec, indent=2))
    return rec['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
