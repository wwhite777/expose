"""Execute the frozen development grid, preserving each completed operation."""
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
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from expose.development import SourceCentroid, load_config, logical_cells, make_model, membership, operations
from expose.development_validation import completed_stage, membership_rows, validate_grid, validate_operation, validate_split
from expose.provenance import file_sha256 as sha
from expose.runtime import atomic_write_csv, atomic_write_json
from expose.scout import trial_id, validate_prepared_array


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        raise FileExistsError('existing grid evidence')
    started, cpu = time.monotonic(), time.process_time()
    rec = dict(run_id=out.name, pid=os.getpid(), status='running', operations=[],
               started_utc=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable)
    atomic_write_json(out/'receipt.json', rec)
    try:
        cfg = load_config(ROOT)
        limits = cfg['resource']
        resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (limits['grid_cpu_soft_seconds'], limits['grid_cpu_hard_seconds']))
        def expired(*_):
            raise TimeoutError('grid wall/CPU limit')
        signal.signal(signal.SIGALRM, expired)
        signal.signal(signal.SIGXCPU, expired)
        signal.alarm(limits['grid_wall_seconds'])
        import joblib
        import mne
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        mne.set_log_level('WARNING')
        prep = ROOT/'result/day3/preparation_r001'
        completed_stage(ROOT, prep, ['schema.json', 'DATA_MANIFEST.csv', 'SPLIT_MANIFEST.csv', 'INPUT_MANIFEST.json'])
        records = json.loads((prep/'schema.json').read_text())['records']
        split_rows = read_csv(prep/'SPLIT_MANIFEST.csv')
        split = validate_split(split_rows, cfg)
        paths = ['research/protocols/DAY3_CONFIG_v1.json', 'research/protocols/DAY3_CONFIG_v1.json.sha256',
                 'research/day3/PROTOCOL_v1.md', 'research/day3/ARTIFACT_CONTRACT_v1.md',
                 cfg['role_manifest'], cfg['loader_file'], cfg['model_code'],
                 'src/expose/development.py', 'src/expose/development_validation.py', 'src/expose/scout.py',
                 'src/expose/provenance.py', 'src/expose/runtime.py', 'scripts/run_day3.py', 'scripts/launch_day3.py',
                 'research/environment/requirements.lock.txt', 'research/day3/PREFIT_MANIFEST.json',
                 'research/day3/CODE_FREEZE.json']
        paths += [str(p.relative_to(ROOT)) for p in prep.iterdir() if p.suffix in ('.json', '.csv')]
        arrays, labels, identifiers = [], [], []
        for record in sorted(records, key=lambda r: (r['subject'], r['session'])):
            paths += [record['cache']['array_path'], record['cache']['metadata_path']]
            with np.load(ROOT/record['cache']['array_path'], allow_pickle=False) as z:
                X, y = z['X'], z['y']
            validate_prepared_array(ROOT, record, X, y, cfg)
            ids = [trial_id(record['subject'], record['session'], i) for i in range(100)]
            if any(int(split[t]['label']) != int(y[i]) or split[t]['epoch_sha256'] != record['metadata']['trial_epoch_sha256'][i]
                   for i, t in enumerate(ids)):
                raise ValueError('split labels/content differ from prepared producer')
            arrays.append(X); labels.append(y); identifiers.extend(ids)
        Xall, yall = np.concatenate(arrays), np.concatenate(labels)
        del arrays, labels, X, y
        index = {tid: i for i, tid in enumerate(identifiers)}
        if len(index) != 4200:
            raise ValueError('array identity set differs')
        manifest = {p: sha(ROOT/p) for p in paths}
        frozen = json.loads((ROOT/'research/day3/PREFIT_MANIFEST.json').read_text())
        if any(sha(ROOT/p) != h for p, h in frozen['files'].items()):
            raise ValueError('prefit snapshot changed')
        atomic_write_json(out/'INPUT_MANIFEST.json', manifest)
        rec['config_sha256'] = sha(ROOT/'research/protocols/DAY3_CONFIG_v1.json')
        rec['input_manifest_sha256'] = sha(out/'INPUT_MANIFEST.json')
        predictions, members, timings = [], [], []
        base = None
        with threadpool_limits(limits=1):
            rec['threadpools'] = threadpool_info()
            if any(p['num_threads'] != 1 for p in rec['threadpools']):
                raise ValueError('numerical thread limit differs')
            for number, op in enumerate(operations(cfg), 1):
                oid = op['operation_id']
                source, history, evaluation = membership(op, split, cfg)
                use_ids = history if op['arm'] == 'calibration' and op['dose'] else source + history
                ix = [index[t] for t in use_ids]
                train_X, train_y = Xall[ix], yall[ix]
                fw, fc = time.perf_counter(), time.process_time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    if op['arm'] == 'calibration':
                        if op['dose'] == 0:
                            model = SourceCentroid().fit(train_X, train_y)
                            base = model
                        else:
                            before_base = joblib.hash(base)
                            model = base.adapted(train_X, train_y, op['prior'])
                            if joblib.hash(base) != before_base:
                                raise ValueError('calibration changed source model')
                    else:
                        model = make_model(op['model']).fit(train_X, train_y)
                fit_cpu, fit_wall = time.process_time()-fc, time.perf_counter()-fw
                if any('converg' in str(w.message).lower() for w in caught):
                    raise ValueError('fit convergence warning: '+oid)
                if model.classes_.tolist() != [0, 1]:
                    raise ValueError('model class order differs')
                fingerprint = joblib.hash(model)
                qx = [index[t] for t in evaluation]
                query = Xall[qx]
                pw, pc = time.perf_counter(), time.process_time()
                probability = model.predict_proba(query)
                predict_cpu, predict_wall = time.process_time()-pc, time.perf_counter()-pw
                if joblib.hash(model) != fingerprint:
                    raise ValueError('evaluation modified fitted model')
                if probability.shape != (len(evaluation), 2):
                    raise ValueError('probability shape differs')
                predicted = probability.argmax(axis=1)
                pred = [dict(operation_id=oid, arm=op['arm'], model=op['model'], draw=draw,
                             dose_per_class=op['dose'], prior_weight='' if prior is None else str(prior),
                             subject_id=int(split[tid]['subject_id']), trial_id=tid, y_true=int(yall[qx[i]]),
                             y_pred=int(predicted[i]), p_class0_right=float(probability[i, 0]),
                             p_class1_left=float(probability[i, 1]))
                        for draw, prior in logical_cells(op, cfg) for i, tid in enumerate(evaluation)]
                mem = membership_rows(op, source, history, evaluation)
                validate_operation(op, pred, mem, split, cfg)
                timing = dict(operation_id=oid, arm=op['arm'], model=op['model'], draw=op['draw'],
                              dose_per_class=op['dose'], target_subject=op['target'], prior_weight=op['prior'],
                              operation_kind='target_update' if op['arm']=='calibration' and op['dose'] else 'fit',
                              source_trials=len(source), history_trials=len(history), evaluation_trials=len(evaluation),
                              fit_cpu_seconds=fit_cpu, fit_wall_seconds=fit_wall,
                              predict_cpu_seconds=predict_cpu, predict_wall_seconds=predict_wall)
                atomic_write_csv(out/(oid+'.predictions.csv'), list(pred[0]), pred)
                atomic_write_csv(out/(oid+'.membership.csv'), list(mem[0]), mem)
                joblib.dump(model, out/(oid+'.joblib'))
                artifact_hashes = {oid+suffix: sha(out/(oid+suffix)) for suffix in ('.predictions.csv', '.membership.csv', '.joblib')}
                atomic_write_json(out/(oid+'.json'), dict(**timing, warnings=[str(w.message) for w in caught],
                                  fitted_model_hash_before_and_after_evaluation=fingerprint, model_classes=[0, 1], artifacts=artifact_hashes))
                predictions.extend(pred); members.extend(mem); timings.append(timing)
                rec['operations'].append(dict(operation_id=oid, operation_kind=timing['operation_kind'],
                                              receipt_sha256=sha(out/(oid+'.json'))))
                atomic_write_json(out/'receipt.json', rec)
                if number % 20 == 0 or number == 808:
                    print(f'operation {number}/808 {oid}; elapsed {time.monotonic()-started:.1f}s', flush=True)
                del train_X, train_y, query, model, pred, mem
            rec['checks'] = validate_grid(predictions, members, timings, split_rows, cfg)
            for name, rows in [('predictions.csv', predictions), ('operation_membership.csv', members), ('timing.csv', timings)]:
                atomic_write_csv(out/name, list(rows[0]), rows)
        counts = {k: sum(t['operation_kind']==k for t in timings) for k in ('fit', 'target_update')}
        if counts != {'fit': 520, 'target_update': 288}:
            raise ValueError('fit/update counts differ')
        rec['counts'] = counts
        if any(sha(ROOT/p) != h for p, h in manifest.items()):
            raise ValueError('input mutation during grid')
        rec.update(status='completed', exit_code=0)
    except BaseException as error:
        rec.update(status='failed', exit_code=1, error=repr(error))
        traceback.print_exc()
    finally:
        rec.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-started,
                   cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_write_json(out/'receipt.json', rec)
        signal.alarm(0)
    if rec['status'] == 'completed':
        outputs = {p.name: sha(p) for p in out.iterdir() if p.suffix in ('.json', '.csv', '.joblib')
                   and p.name not in ('supervisor_receipt.json', 'COMPLETED.json')}
        atomic_write_json(out/'COMPLETED.json', dict(run_id=out.name, receipt_sha256=sha(out/'receipt.json'), outputs=outputs))
    return rec['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
