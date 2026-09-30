#!/usr/bin/env python3
"""Post-audit descriptive neural-output check; no fitting or endpoint changes."""
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def main():
    start_cpu, start_wall = time.process_time(), time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    out = ROOT / 'result/review3_20260923/neural_output_diagnostic_r001'
    out.mkdir(exist_ok=False)
    rows, groups, sources, inputs = [], {}, [], {}
    for dataset, dirname in [('openbmi8', 'openbmi8_neural_r001'), ('bnci', 'bnci_neural_merged_r001')]:
        directory = ROOT / 'result/review3_20260923' / dirname
        receipt = json.loads((directory / 'receipt.json').read_text())
        if receipt['status'] != 'completed':
            raise ValueError('Incomplete neural job')
        pred = directory / 'predictions.csv.gz'
        if sha(pred) != receipt['outputs'][pred.name]:
            raise ValueError('Prediction hash changed')
        inputs[str(pred.relative_to(ROOT))] = sha(pred)
        by_op = {}
        with gzip.open(pred, 'rt') as f:
            for row in csv.DictReader(f):
                if row['h_total'] in ('0', '60'):
                    by_op.setdefault(row['operation_id'], []).append(row)
        for operation, trials in sorted(by_op.items()):
            expected = 100 if dataset == 'openbmi8' else 144
            if len(trials) != expected or len({r['trial_id'] for r in trials}) != expected:
                raise ValueError('Trial inventory mismatch')
            probabilities = [float(r['p_right']) for r in trials]
            item = {'dataset': dataset, 'operation_id': operation, 'source_size': trials[0]['source_size'],
                    'h_total': int(trials[0]['h_total']), 'target': int(trials[0]['target']), 'draw': int(trials[0]['draw']),
                    'n_trials': len(trials), 'p_right_mean': statistics.mean(probabilities),
                    'p_right_sample_sd': statistics.stdev(probabilities), 'p_right_min': min(probabilities),
                    'p_right_max': max(probabilities), 'single_predicted_class': len({r['y_pred'] for r in trials}) == 1}
            rows.append(item)
            groups.setdefault(f"{dataset}/S{item['source_size']}/h{item['h_total']}", []).append(item)
        if dataset == 'openbmi8':
            for f in sorted(directory.glob('source_*.json')):
                if sha(f) != receipt['outputs'][f.name]:
                    raise ValueError('Source metadata hash changed')
                inputs[str(f.relative_to(ROOT))] = sha(f)
                d = json.loads(f.read_text())
                sources.append({'metadata': str(f.relative_to(ROOT)), 'sha256': sha(f),
                                'final_source_trials': d['final_source_trials'], 'selected_epochs': d['selected_epochs']})
    summary = {key: {'operations': len(values), 'single_class_operations': sum(v['single_predicted_class'] for v in values),
                     'median_within_operation_probability_sd': statistics.median(v['p_right_sample_sd'] for v in values),
                     'operation_mean_probability_range': [min(v['p_right_mean'] for v in values), max(v['p_right_mean'] for v in values)]}
               for key, values in groups.items()}
    with (out / 'operation_diagnostics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    record = {'status': 'completed', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'trigger': 'Auditor D1 after completed neural results; post hoc descriptive diagnostic',
              'method': 'Sample SD of p_right within each target × draw evaluation; count operations with one predicted class',
              'groups': summary, 'openbmi_source_checkpoints': sources,
              'scope': 'An operation is an evaluation, not an independently trained source model. Three source models per OpenBMI source count are reused across twelve targets at h=0. No refits, retuning, endpoint edits, or new confirmation data.',
              'inputs': inputs, 'script_sha256': sha(__file__), 'cpu_seconds': time.process_time() - start_cpu,
              'wall_seconds': time.monotonic() - start_wall, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (out / 'summary.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'groups': summary, 'cpu_seconds': record['cpu_seconds']}))


if __name__ == '__main__':
    main()
