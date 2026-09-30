"""Replay every saved model and independently reconstruct training identities."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from expose.provenance import file_sha256 as sha
from expose.runtime import atomic_write_json
from expose.development import load_config, operations
from expose.development_validation import completed_stage


def rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def validate_replay_inputs(root, grid, prep, cfg, expected_operations=None):
    expected_operations = operations(cfg) if expected_operations is None else expected_operations
    grid_required = {'receipt.json', 'INPUT_MANIFEST.json', 'predictions.csv',
                     'operation_membership.csv', 'timing.csv'}
    for op in expected_operations:
        grid_required.update(op['operation_id']+suffix
                             for suffix in ('.json', '.joblib', '.predictions.csv', '.membership.csv'))
    grid_marker = completed_stage(root, grid, grid_required)
    if set(grid_marker['outputs']) != grid_required:
        raise ValueError('grid output set differs')
    prep_required = {'receipt.json', 'INPUT_MANIFEST.json', 'DATA_MANIFEST.csv',
                     'SPLIT_MANIFEST.csv', 'schema.json'}
    prep_marker = completed_stage(root, prep, prep_required)
    if set(prep_marker['outputs']) != prep_required:
        raise ValueError('preparation output set differs')
    return grid_marker, prep_marker


def initialize_replay_output(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'receipt.json').exists():
        raise FileExistsError('existing replay evidence')


def write_replay_completion(out):
    required = ('receipt.json', 'INPUT_MANIFEST.json', 'checks.json')
    outputs = {name: sha(out/name) for name in required}
    marker = dict(run_id=out.name, receipt_sha256=outputs['receipt.json'], outputs=outputs)
    atomic_write_json(out/'COMPLETED.json', marker)
    return marker


def main():
    out = ROOT/'result/day3/model_replay_r002'
    initialize_replay_output(out)
    start, cpu = time.monotonic(), time.process_time()
    receipt = dict(run_id=out.name, status='running', started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                   scope='Builder QA, shared prepared arrays/serialized models; no independent raw reprocessing or refit')
    atomic_write_json(out/'receipt.json', receipt)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,)*2)
        resource.setrlimit(resource.RLIMIT_CPU, (600, 610))
        import joblib
        import numpy as np
        from threadpoolctl import threadpool_limits
        cfg = load_config(ROOT)
        grid, prep = ROOT/'result/day3/grid_r002', ROOT/'result/day3/preparation_r001'
        validate_replay_inputs(ROOT, grid, prep, cfg)
        split = {r['trial_id']: r for r in rows(prep/'SPLIT_MANIFEST.csv')}
        schema = json.loads((prep/'schema.json').read_text())
        epochs = {}
        for item in schema['records']:
            path = ROOT/item['cache']['array_path']
            assert sha(path) == item['cache']['array_sha256']
            with np.load(path, allow_pickle=False) as z:
                for i, x in enumerate(z['X']):
                    tid = f"lee2019:s{item['subject']:03d}:session{item['session']}:offline:t{i:03d}"
                    epochs[tid] = x.copy()
                    assert int(z['y'][i]) == int(split[tid]['label'])
        inputs = {str((grid/p).relative_to(ROOT)):sha(grid/p) for p in ['supervisor_receipt.json', 'COMPLETED.json', 'receipt.json', 'INPUT_MANIFEST.json', 'timing.csv']}
        inputs.update({str((prep/p).relative_to(ROOT)):sha(prep/p) for p in ['supervisor_receipt.json', 'COMPLETED.json', 'receipt.json', 'INPUT_MANIFEST.json', 'schema.json', 'SPLIT_MANIFEST.csv']})
        inputs['scripts/verify_day3_r002.py'] = sha(Path(__file__))
        inputs['scripts/launch_day3_replay_r002.py'] = sha(ROOT/'scripts/launch_day3_replay_r002.py')
        for name in ['research/protocols/DAY3_CONFIG_v1.json', 'research/protocols/DAY3_CONFIG_v1.json.sha256',
                     'src/expose/development.py', 'src/expose/development_validation.py',
                     'src/expose/provenance.py', 'src/expose/runtime.py']:
            inputs[name] = sha(ROOT/name)
        previous = ROOT/'result/day2/scout_r001/predictions.csv'
        inputs[str(previous.relative_to(ROOT))] = sha(previous)
        atomic_write_json(out/'INPUT_MANIFEST.json', inputs)
        all_source = sorted(t for t, r in split.items() if r['role']=='source')
        maximum, checked, membership_checks = 0., [], 0
        with threadpool_limits(limits=1):
            for number, timing in enumerate(rows(grid/'timing.csv'), 1):
                oid, arm = timing['operation_id'], timing['arm']
                dose = int(timing['dose_per_class'])
                draw = int(timing['draw']) if timing['draw'] else None
                target = int(timing['target_subject']) if timing['target_subject'] else None
                source = list(all_source)
                if arm in ['fixed_total', 'removal']:
                    rng = np.random.default_rng(cfg['source_draw_seeds'][draw])
                    source = []
                    for c in [0, 1]:
                        pool = [t for t in all_source if int(split[t]['label'])==c]
                        source += list(np.asarray(pool)[rng.permutation(len(pool))[:50-dose]])
                if arm=='target_only':
                    source = []
                history = []
                if target:
                    rng = np.random.default_rng(np.random.SeedSequence([cfg['history_draw_seeds'][draw], target]))
                    for c in [0, 1]:
                        pool = sorted(t for t, r in split.items() if int(r['subject_id'])==target
                                      and int(r['session'])==1 and int(r['label'])==c)
                        history += list(np.asarray(pool)[rng.permutation(len(pool))[:dose]])
                query_ids = sorted(t for t, r in split.items() if int(r['session'])==2
                                   and (target is None or int(r['subject_id'])==target))
                members = rows(grid/(oid+'.membership.csv'))
                source_usage = 'source_representation' if arm=='calibration' and dose else 'source_training'
                history_usage = 'target_update' if arm=='calibration' else 'history_training'
                expected = {(source_usage, t) for t in source} | {(history_usage, t) for t in history} | {('evaluation', t) for t in query_ids}
                assert {(r['usage'], r['trial_id']) for r in members} == expected
                assert len(members)==len(expected) and not set(source+history)&set(query_ids)
                membership_checks += len(members)
                model = joblib.load(grid/(oid+'.joblib'))
                fingerprint = joblib.hash(model)
                probs = model.predict_proba(np.stack([epochs[t] for t in query_ids]))
                assert joblib.hash(model)==fingerprint
                pred = rows(grid/(oid+'.predictions.csv'))
                positions = {t:i for i, t in enumerate(query_ids)}
                values = np.array([[float(r['p_class0_right']), float(r['p_class1_left'])] for r in pred])
                replay = probs[[positions[r['trial_id']] for r in pred]]
                difference = float(np.abs(values-replay).max())
                maximum = max(maximum, difference)
                assert difference <= 1e-12
                assert all(int(r['y_pred'])==int(replay[i].argmax()) and int(r['y_true'])==int(split[r['trial_id']]['label']) for i, r in enumerate(pred))
                checked.append(dict(operation_id=oid, prediction_rows=len(pred), max_probability_difference=difference))
                if number % 100==0:
                    print('replayed', number, 'of 808', flush=True)
        assert len(checked)==808 and sum(r['prediction_rows'] for r in checked)==110400
        prior = {(r['model'], r['trial_id']):r for r in rows(previous) if int(r['dose_per_class'])==0}
        prior_differences = []
        for r in rows(grid/'predictions.csv'):
            key = r['model'], r['trial_id']
            if r['arm']=='practical' and r['draw']=='0' and r['dose_per_class']=='0' and key in prior:
                old = prior[key]
                assert r['y_true']==old['y_true'] and r['y_pred']==old['y_pred']
                prior_differences.append(max(abs(float(r[k])-float(old[k])) for k in ['p_class0_right', 'p_class1_left']))
        assert len(prior_differences)==1600 and max(prior_differences)<=1e-10
        if any(sha(ROOT/p)!=h for p, h in inputs.items()):
            raise ValueError('replay input changed')
        atomic_write_json(out/'checks.json', dict(operations=checked, membership_rows=membership_checks,
                                                 max_probability_difference=maximum,
                                                 prior_day2_source_only_rows=1600,
                                                 prior_day2_max_probability_difference=max(prior_differences)))
        receipt.update(status='completed', exit_code=0, operations=808, prediction_rows=110400,
                       membership_rows=membership_checks, max_probability_difference=maximum)
    except BaseException as error:
        receipt.update(status='failed', exit_code=1, error=repr(error))
        traceback.print_exc()
    finally:
        receipt.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_write_json(out/'receipt.json', receipt)
    if receipt['status']=='completed':
        write_replay_completion(out)
    print(json.dumps(receipt, indent=2))
    return receipt['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
