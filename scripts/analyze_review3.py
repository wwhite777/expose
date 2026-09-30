"""Reconstruct Review3 balanced accuracies from saved trial predictions."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as handle:
        yield from csv.DictReader(handle)


def save_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()


def read_gzip_json(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def frozen_plan(plan, grid_receipt):
    receipt_path = plan / 'PLAN_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('status') != 'planned_no_fits'
            or grid_receipt.get('plan_receipt_sha256') != sha(receipt_path)):
        raise ValueError('grid receipt does not bind the frozen plan receipt')
    files = receipt.get('files', {})
    required = {'operations.json.gz', 'membership_sets.json.gz'}
    if not required <= set(files):
        raise ValueError('plan receipt lacks operations or membership hash')
    for name, digest in files.items():
        path = plan / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('frozen plan payload changed: ' + name)
    operations_list = read_gzip_json(plan / 'operations.json.gz')
    operations = {row['operation_id']: row for row in operations_list}
    if len(operations) != len(operations_list):
        raise ValueError('duplicate operation ID in frozen plan')
    memberships = read_gzip_json(plan / 'membership_sets.json.gz')
    if any(fingerprint(value) != key for key, value in memberships.items()):
        raise ValueError('frozen membership payload differs from its key')
    for operation in operations.values():
        key = operation.get('memberships', {}).get('evaluation')
        if key not in memberships:
            raise ValueError('operation evaluation membership is absent')
    return operations, memberships


def saved_score_rows(path, operations):
    fields = {'operation_id', 'study_id', 'method', 'draw', 'target', 'donor_count',
              'source_size', 'source_n', 'h_total', 'C', 'balanced_accuracy',
              'evaluation_n', 'source_membership', 'personal_membership'}
    rows = list(csv.DictReader(path.open('r', newline='')))
    if not rows or set(rows[0]) != fields:
        raise ValueError('saved score CSV schema differs')
    result = {}
    for row in rows:
        opid = row['operation_id']
        if opid not in operations or opid in result:
            raise ValueError('saved score inventory has unexpected or duplicate operation')
        result[opid] = row
    if set(result) != set(operations):
        raise ValueError('saved score inventory is incomplete')
    return result


def selected_c_rows(path, operations):
    rows = list(csv.DictReader(path.open('r', newline='')))
    if not rows or set(rows[0]) != {'tuning_id', 'selected_C'}:
        raise ValueError('selected C CSV schema differs')
    values = {}
    for row in rows:
        if row['tuning_id'] in values or float(row['selected_C']) not in (0.1, 1.0, 10.0):
            raise ValueError('invalid selected C inventory')
        values[row['tuning_id']] = float(row['selected_C'])
    needed = {op['tuning_id'] for op in operations.values() if op.get('tuning_id')}
    if not needed <= set(values):
        raise ValueError('selected C inventory misses frozen tuning contexts')
    return values


def effect(values, seed=2026092304):
    """Participant-average estimand; ties preserve their theoretical score precision."""
    values = np.round(np.asarray(values, dtype=float), 12)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('effect requires complete finite participant values')
    draws = np.random.default_rng(seed).integers(0, len(values), size=(10000, len(values)))
    means = values[draws].mean(axis=1)
    signs = np.asarray(list(itertools.product([-1., 1.], repeat=len(values))))
    p = float(np.mean(np.abs(signs @ values / len(values)) >= abs(values.mean()) - 1e-10))
    return {'n': len(values), 'mean_pp': float(values.mean()), 'median_pp': float(np.median(values)),
            'ci_low_pp': float(np.quantile(means, .025)), 'ci_high_pp': float(np.quantile(means, .975)),
            'positive': int(np.sum(values > 1e-10)), 'zero': int(np.sum(np.abs(values) <= 1e-10)),
            'negative': int(np.sum(values < -1e-10)), 'signflip_p': p,
            'values_pp': values.tolist()}


def holm(pvalues):
    ordered = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0.
    for i, key in enumerate(ordered):
        previous = max(previous, (len(ordered) - i) * pvalues[key])
        adjusted[key] = min(1., previous)
    return adjusted


def reconstruct_grid(label, directory, plan):
    receipt = json.loads((directory/'receipt.json').read_text())
    complete = json.loads((directory/'COMPLETED.json').read_text())
    if receipt['status'] != 'completed' or complete['receipt_sha256'] != sha(directory/'receipt.json'):
        raise ValueError('grid is incomplete or changed')
    prediction, score_path, selected_path = (directory/'predictions.csv.gz', directory/'operation_scores.csv',
                                              directory/'selected_c.csv')
    if (receipt['outputs'].get('predictions.csv.gz') != sha(prediction)
            or receipt['outputs'].get('operation_scores.csv') != sha(score_path)
            or receipt['outputs'].get('selected_c.csv') != sha(selected_path)):
        raise ValueError('prediction hash changed')
    operations, memberships = frozen_plan(plan, receipt)
    saved_scores = saved_score_rows(score_path, operations)
    selected_c = selected_c_rows(selected_path, operations)
    actual, seen, n_predictions = [], set(), 0
    for opid, group in itertools.groupby(read_rows(prediction), key=lambda r:r['operation_id']):
        rows = list(group)
        if opid in seen or opid not in operations:
            raise ValueError('unexpected or noncontiguous duplicate operation')
        seen.add(opid); op=operations[opid]
        ids=[r['trial_id'] for r in rows]
        expected_ids = memberships[op['memberships']['evaluation']]
        if ids != expected_ids:
            raise ValueError('prediction evaluation IDs differ from frozen membership order')
        truth=np.array([int(r['y_true']) for r in rows]);pred=np.array([int(r['y_pred']) for r in rows])
        p=np.array([[float(r['p0']),float(r['p1'])] for r in rows])
        if (set(truth.tolist())!={0,1} or not np.isfinite(p).all() or np.any(p < -1e-12)
                or np.any(p > 1 + 1e-12) or not np.allclose(p.sum(1),1,rtol=0,atol=1e-10)):
            raise ValueError('invalid predicted probabilities or evaluation labels')
        if not set(pred.tolist()) <= {0, 1} or np.any(p.max(1)-p[np.arange(len(pred)),pred] > 1e-12):
            raise ValueError('saved labels disagree with class-probability maxima beyond tie tolerance')
        ba=float(np.mean([np.mean(pred[truth==c]==c) for c in (0,1)]))
        saved = saved_scores[opid]
        if (int(saved['evaluation_n']) != len(rows)
                or abs(float(saved['balanced_accuracy']) - ba) > 1e-12):
            raise ValueError('saved score differs from reconstruction')
        for field in ('study_id', 'method', 'draw', 'target', 'donor_count', 'source_size',
                      'source_n', 'h_total', 'source_membership', 'personal_membership'):
            expected = op[field] if field in op else op['memberships'][field.removesuffix('_membership')]
            if str(saved[field]) != str(expected):
                raise ValueError('saved score coordinate differs from frozen operation')
        expected_c = selected_c[op['tuning_id']] if op.get('tuning_id') else None
        if ((expected_c is None and saved['C'] != '')
                or (expected_c is not None and abs(float(saved['C']) - expected_c) > 1e-12)):
            raise ValueError('saved score C differs from frozen selected C')
        actual.append({'dataset_variant':label,'operation_id':opid,'study_id':op['study_id'],
                       'method':op['method'],'target':op['target'],'draw':op['draw'],
                       'source_size':str(op['source_size']),'source_n':op['source_n'],
                       'donor_count':str(op['donor_count']),'h_total':op['h_total'],
                       'evaluation_n':len(rows),'balanced_accuracy':ba})
        n_predictions+=len(rows)
    if seen!=set(operations):
        raise ValueError('operation inventory incomplete')
    return actual, {'operations':len(actual),'predictions':n_predictions,
                    'prediction_sha256':sha(prediction),'receipt_sha256':sha(directory/'receipt.json')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=False)
    start,cpu=time.monotonic(),time.process_time()
    spec=json.loads(args.inputs.read_text());all_rows=[];verification={}
    for item in spec['grids']:
        rows,verified=reconstruct_grid(item['label'],ROOT/item['directory'],ROOT/item['plan'])
        all_rows+=rows;verification[item['label']]=verified
    save_csv(args.out_dir/'reconstructed_scores.csv',all_rows)
    grouped=defaultdict(list)
    key_fields=['dataset_variant','study_id','method','source_size','source_n','donor_count','h_total','target']
    for row in all_rows:
        grouped[tuple(row[k] for k in key_fields)].append(row)
    people=[]
    for key,rows in grouped.items():
        if len(rows)!=10 or len({r['draw'] for r in rows})!=10:
            raise ValueError('each classical participant cell requires ten complete draws')
        people.append(dict(zip(key_fields,key),draws=10,balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in rows]))))
    save_csv(args.out_dir/'participant_means.csv',people)
    by_curve=defaultdict(list)
    for row in people:
        by_curve[tuple(row[k] for k in key_fields[:-1])].append(row)
    curves=[]
    for key,rows in by_curve.items():
        values=100*np.asarray([r['balanced_accuracy'] for r in rows]);stats=effect(values)
        curves.append(dict(zip(key_fields[:-1],key),n=len(rows),mean_ba=stats['mean_pp'],
                           ci_low=stats['ci_low_pp'],ci_high=stats['ci_high_pp']))
    save_csv(args.out_dir/'curves.csv',curves)
    endpoints={}
    for variant,study,full in [('openbmi8','openbmi_main','1800'),('bnci','bnci_main','all')]:
        selected=[r for r in people if r['dataset_variant']==variant and r['study_id']==study]
        if not selected:continue
        targets=sorted({r['target'] for r in selected});lookup={(r['method'],r['source_size'],r['h_total'],r['target']):r['balanced_accuracy'] for r in selected}
        def v(method,source,h):return 100*np.asarray([lookup[method,source,h,t] for t in targets])
        small=v('plain_ts','100',60)-v('plain_ts','100',0)
        large=v('plain_ts',full,60)-v('plain_ts',full,0)
        contrasts={'plain_label_gain_S100':small,'plain_label_gain_Sall':large,
                   'source_context_interaction':small-large,
                   'mdwm_minus_plain_Sall_h60':v('mdwm',full,60)-v('plain_ts',full,60),
                   'historical_EA_minus_plain_Sall_h0':v('ea_ts',full,0)-v('plain_ts',full,0),
                   'alignment_label_interaction_Sall':v('ea_ts',full,60)-v('ea_ts',full,0)-large}
        endpoints[variant]={'targets':targets,'contrasts':{k:effect(z) for k,z in contrasts.items()},
                            'multiplicity':'Six-endpoint replication family requires donor endpoint before Holm; Sall plain gain is descriptive, excluded from that family.'}
    payload={'endpoint_results':endpoints,'verification':verification,
             'unit':'person after averaging draws','bootstrap_draws':10000,'seed':2026092304,
             'CI_status':'conditional on observed source cohort and frozen study definition; development results descriptive',
             'signflip_assumption':'sign exchangeability under paired null; overlapping external training folds limit independence',
             'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.monotonic()-start,
             'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    (args.out_dir/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'verification':verification,'endpoints':endpoints,'cpu_seconds':payload['cpu_seconds']}))


if __name__=='__main__':main()
