"""Extend verified Day2 preparation to all frozen development people; no fits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from expose.provenance import file_sha256 as sha
from expose.runtime import atomic_write_json, atomic_write_csv
from expose.development import load_config
from expose.scout import validate_prepared_array

spec = importlib.util.spec_from_file_location('day2_preparation_helpers', ROOT / 'scripts/prepare_day2.py')
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if (out / 'receipt.json').exists():
        raise FileExistsError('existing preparation evidence')
    out.mkdir(parents=True, exist_ok=True)
    start, cpu = time.monotonic(), time.process_time()
    rec = dict(run_id=out.name, pid=os.getpid(), status='running', started_utc=datetime.now(timezone.utc).isoformat(),
               records=[], new_raw_files=[], reused_day2_files=34, interpreter=sys.executable)
    atomic_write_json(out / 'receipt.json', rec)
    try:
        helpers.enforce_resources()
        import numpy as np
        import requests
        from threadpoolctl import threadpool_limits, threadpool_info
        cfg = load_config(ROOT)
        old = ROOT / 'result/day2/preparation_r001'
        marker = json.loads((old / 'COMPLETED.json').read_text())
        for name, expected in marker['outputs'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Day2 preparation changed')
        records = json.loads((old / 'schema.json').read_text())['records']
        with (old / 'DATA_MANIFEST.csv').open() as f:
            data_rows = list(csv.DictReader(f))
        with (old / 'SPLIT_MANIFEST.csv').open() as f:
            split_rows = list(csv.DictReader(f))
        inputs = ['research/protocols/DAY3_CONFIG_v1.json', 'research/protocols/DAY3_CONFIG_v1.json.sha256',
                  cfg['role_manifest'], cfg['loader_file'], 'scripts/prepare_day2.py', 'scripts/prepare_day3.py',
                  'src/expose/development.py', 'src/expose/scout.py', 'src/expose/provenance.py', 'src/expose/baselines.py',
                  'research/day1/sources/lee2019_mi_remote_inventory.csv', 'research/day1/sources/100542.md5',
                  'research/environment/requirements.lock.txt', 'result/day2/preparation_r001/schema.json',
                  'result/day2/preparation_r001/COMPLETED.json', 'result/day2/preparation_r001/receipt.json',
                  'result/day2/preparation_r001/DATA_MANIFEST.csv', 'result/day2/preparation_r001/SPLIT_MANIFEST.csv',
                  'result/day2/revalidation_20260916T055024Z/raw_checks_v2.json']
        for row in records:
            inputs.extend(row['cache'][key] for key in ('array_path', 'metadata_path'))
        manifest = {p: sha(ROOT / p) for p in inputs}
        atomic_write_json(out / 'INPUT_MANIFEST.json', manifest)
        with (ROOT / cfg['role_manifest']).open() as f:
            roles = {int(r['subject_id']): r['role'] for r in csv.DictReader(f)}
        with (ROOT / 'research/day1/sources/lee2019_mi_remote_inventory.csv').open() as f:
            inventory = {r['relative_path']: r for r in csv.DictReader(f)}
        checksums = {s.split()[1].removeprefix('./'): s.split()[0] for s in
                     (ROOT / 'research/day1/sources/100542.md5').read_text().splitlines() if len(s.split()) == 2}
        expected = {(s, 1) for s in cfg['source_subjects']} | {(s, t) for s in cfg['development_subjects'] for t in (1, 2)}
        plan = [dict(key='live/pub/100542/'+p, url=r['url'], bytes=int(r['bytes']), etag=r['etag'])
                for p, r in inventory.items() if (int(r['subject_id']), int(r['session_id'])) in expected]
        items = helpers.validate_plan(plan, roles, inventory, checksums, cfg['source_subjects'], cfg['development_subjects'])
        present = {(r['subject'], r['session']) for r in records}
        new = [r for r in items if (r['subject'], r['session']) not in present]
        assert len(new) == 8 and len(items) == 42
        def storage(reserve=0):
            used = helpers.storage_bytes([ROOT, Path(sys.executable).parent.parent])
            if used + reserve > 30_000_000_000 or shutil.disk_usage(ROOT).free < reserve + 256*1024**2:
                raise ValueError('storage cap/reserve exceeded')
            return used
        rec['initial_storage_bytes'] = storage(sum(r['bytes'] for r in new) + 800_000_000)
        with threadpool_limits(limits=1), requests.Session() as client:
            rec['threadpools'] = threadpool_info()
            assert all(p['num_threads'] == 1 for p in rec['threadpools'])
            for row in records:
                with np.load(ROOT / row['cache']['array_path'], allow_pickle=False) as z:
                    validate_prepared_array(ROOT, row, z['X'], z['y'], cfg)
                if (ROOT / row['raw']['path']).stat().st_size != row['raw']['bytes']:
                    raise ValueError('old raw size changed')
            rec['records'] = records
            for number, item in enumerate(new, 1):
                began = time.monotonic()
                observed = helpers.stream_download(ROOT/item['path'], item, get=client.get, check_storage=storage)
                raw = {**item, **observed, 'reused': False}
                rec['new_raw_files'].append(raw)
                atomic_write_json(out/'receipt.json', rec)
                X, y, metadata, schema = helpers.load_and_audit(ROOT/item['path'])
                helpers.validate_epochs(X, y, metadata, cfg['preprocessing'])
                p = ROOT/'data/derived/day3'/f"subj{item['subject']:02d}_sess{item['session']:02d}_offline.npz"
                with p.open('xb') as f:
                    np.savez_compressed(f, X=X, y=y)
                atomic_write_json(p.with_suffix('.json'), metadata)
                cache = dict(array_path=str(p.relative_to(ROOT)), array_sha256=sha(p),
                             metadata_path=str(p.with_suffix('.json').relative_to(ROOT)), metadata_sha256=sha(p.with_suffix('.json')), reused=False)
                record = dict(subject=item['subject'], session=item['session'], role=item['role'], raw=raw,
                              cache=cache, metadata=metadata, schema=schema, status='pass_with_documented_smt_one_sample_offset')
                validate_prepared_array(ROOT, record, X, y, cfg)
                rec['records'].append(record)
                data_rows.append(dict(dataset='Lee2019_MI', subject_id=item['subject'], role=item['role'], session=item['session'],
                    raw_path=item['path'], source_url=item['url'], bytes=observed['actual_bytes'], published_md5=item['published_md5'],
                    local_sha256=observed['sha256'], license='CC0-1.0', used_variable='EEG_MI_train', trials=100, right_trials=50,
                    left_trials=50, raw_fs=1000, output_fs=250, epoch_shape='100x8x750', derived_path=cache['array_path'],
                    derived_sha256=cache['array_sha256'], metadata_path=cache['metadata_path'], metadata_sha256=cache['metadata_sha256'],
                    reused_raw=False, reused_cache=False, calendar_acquisition_date='unverified'))
                for i in range(100):
                    row = dict(trial_id=f"lee2019:s{item['subject']:03d}:session{item['session']}:offline:t{i:03d}",
                               subject_id=item['subject'], role=item['role'], session=item['session'], mat_variable='EEG_MI_train',
                               trial_index_zero_based=i, label=int(y[i]), class_name=metadata['class_names'][int(y[i])],
                               usage='development_history' if item['session']==1 else 'development_evaluation')
                    mapping = {'event_sample_matlab':'event_samples_matlab', 'event_sample_zero_based':'event_samples_zero_based',
                               'filter_start_sample':'filter_input_start_samples_zero_based', 'filter_stop_sample_exclusive':'filter_input_stop_samples_zero_based_exclusive',
                               'epoch_start_sample':'epoch_start_samples_zero_based', 'epoch_stop_sample_exclusive':'epoch_stop_samples_zero_based_exclusive',
                               'raw_support_sha256':'trial_raw_support_sha256','epoch_sha256':'trial_epoch_sha256'}
                    row.update({k:metadata[v][i] for k,v in mapping.items()})
                    split_rows.append(row)
                helpers.assert_unique_trials(rec['records'])
                atomic_write_json(out/'receipt.json', rec)
                print(json.dumps(dict(new_files=number, required_new_files=8, subject=item['subject'], session=item['session'], wall_seconds=time.monotonic()-began)), flush=True)
                del X, y
        assert len(data_rows)==42 and len(split_rows)==len({r['trial_id'] for r in split_rows})==4200
        assert {(r['subject'],r['session']) for r in rec['records']}==expected
        if any(sha(ROOT/p)!=h for p,h in manifest.items()):
            raise ValueError('preparation dependency changed during run')
        atomic_write_json(out/'schema.json', dict(records=rec['records'], duplicate_checks=helpers.assert_unique_trials(rec['records']),
                                                confirmation_data_accessed=False, model_fits=0))
        for name, rows in [('DATA_MANIFEST.csv',data_rows),('SPLIT_MANIFEST.csv',split_rows)]:
            atomic_write_csv(out/name, list(rows[0]), rows)
        rec.update(status='completed', exit_code=0, files=42, trials=4200, final_storage_bytes=storage())
    except BaseException as exc:
        rec.update(status='failed', exit_code=1, error=repr(exc))
        traceback.print_exc()
    finally:
        rec.update(finished_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=time.monotonic()-start,
                   cpu_seconds=time.process_time()-cpu, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_write_json(out/'receipt.json', rec)
        signal.alarm(0)
    if rec['status']=='completed':
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv') and p.name!='supervisor_receipt.json'}
        atomic_write_json(out/'COMPLETED.json',dict(run_id=out.name,receipt_sha256=sha(out/'receipt.json'),outputs=outputs))
    return rec['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
