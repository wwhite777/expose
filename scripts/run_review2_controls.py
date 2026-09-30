"""Execute the frozen, development-only Review-2 controls with saved predictions."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
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
from expose.baselines import nested_history_indices
from expose.development import membership as old_membership
from expose.development_validation import validate_split
from expose.provenance import file_sha256 as sha
from expose.review2_controls import (operations, membership, oas_covariances,
    fit_representation, train_classifier, fit_ea_whitener, apply_ea,
    fit_source_ea, apply_source_ea)
from expose.runtime import atomic_write_csv, atomic_write_json, _validate_completion
from expose.scout import trial_id, validate_prepared_array

STUDY = ROOT / 'research/review2_execution_20260923'
CONFIG = STUDY / 'CONFIG_v1.json'


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def config_and_split():
    cfg = json.loads(CONFIG.read_text())
    oldpath = ROOT / cfg['old_config_path']
    if sha(oldpath) != cfg['old_config_sha256']:
        raise ValueError('original configuration changed')
    oldcfg = json.loads(oldpath.read_text())
    if any(cfg[k] != oldcfg[k] for k in ('source_subjects', 'development_subjects')):
        raise ValueError('participant roles changed')
    prep = ROOT / cfg['preparation_dir']
    return cfg, oldcfg, validate_split(read_csv(prep / 'SPLIT_MANIFEST.csv'), oldcfg)


def old_reference(op):
    draw, target = op['draw'], op['target']
    if op['module'] == 'A' and op['C'] == 1:
        if op['source_n'] == 40:
            return f'removal__ts_lr__r{draw}_n30'
        if op['source_n'] == 100:
            return f'fixed_total__ts_lr__r{draw}_n0'
    if (op['module'] == 'B' and op['source_n'] == 1800 and op['C'] == 1
            and op['lambda_personal'] == 'pooled' and op['representation'] == 'pooled_refit'):
        return f'practical__ts_lr__r{draw}_n30_s{target:03d}'
    if op['module'] == 'C' and op['source_n'] == 40 and op['donor_kind'] == 'own':
        return f'fixed_total__ts_lr__r{draw}_n30_s{target:03d}'
    return None


def make_plan(destination):
    """Read only role/trial metadata; freeze memberships without opening EEG arrays."""
    cfg, oldcfg, split = config_and_split()
    destination.mkdir(parents=True, exist_ok=False)
    planned, sets, memo, donor_rows = [], {}, {}, []
    for op in operations(cfg):
        key = tuple(op[k] for k in ('module', 'source_n', 'h_total', 'draw', 'target', 'donor_kind'))
        if key not in memo:
            memo[key] = membership(op, split, oldcfg, cfg)
        source, added, evaluation, unlabeled, donors = memo[key]
        refs = {}
        for usage, identifiers in [('source', source), ('added', added),
                                   ('evaluation', evaluation), ('unlabeled_history', unlabeled)]:
            digest = fingerprint(identifiers)
            sets[digest] = identifiers
            refs[usage] = digest
        if op['module'] == 'C':
            oldop = dict(arm='fixed_total', dose=30 if op['source_n'] == 40 else 0,
                         draw=op['draw'], target=op['target'])
            expected_source, expected_history, _ = old_membership(oldop, split, oldcfg)
            if source != expected_source:
                raise ValueError('common donor base differs from original prefix')
            if op['source_n'] == 40 and op['donor_kind'] == 'own' and added != expected_history:
                raise ValueError('own donor block differs from original history')
            if op['source_n'] == 40 and op['donor_kind'] == 'pooled':
                oldop['dose'] = 0
                full, _, _ = old_membership(oldop, split, oldcfg)
                if set(source + added) != set(full):
                    raise ValueError('pooled donor bridge does not reconstruct original100')
            for donor in donors:
                donor_rows.append(dict(operation_id=op['operation_id'], base_n=op['source_n'],
                    target=op['target'], draw=op['draw'], donor_kind=op['donor_kind'], donor=donor,
                    donor_trials_in_base=sum(int(split[t]['subject_id']) == donor for t in source),
                    donor_trials_added=sum(int(split[t]['subject_id']) == donor for t in added)))
        planned.append(dict(operation=op, membership=refs, donor_subjects=donors,
                            original_reference=old_reference(op)))
    anchors = [
        lambda o: o['module']=='A' and o['source_n']==40 and o['C']==1 and o['draw']==0,
        lambda o: o['module']=='B' and o['source_n']==1800 and o['C']==1 and o['draw']==0
            and o['target']==2 and o['representation']=='pooled_refit' and o['lambda_personal']=='pooled',
        lambda o: o['module']=='C' and o['source_n']==40 and o['draw']==0 and o['target']==2
            and o['donor_kind']=='own',
        lambda o: o['module']=='D' and o['h_total']==0 and o['draw']==0 and o['target']==2]
    first = []
    for predicate in anchors:
        matches = [row for row in planned if predicate(row['operation'])]
        if len(matches) != 1:
            raise ValueError('timing anchor not unique')
        first += matches
    planned = first + [row for row in planned if row not in first]
    counts = Counter(row['operation']['module'] for row in planned)
    prediction_rows = sum(len(sets[row['membership']['evaluation']]) for row in planned)
    if dict(counts) != cfg['expected_by_module'] or prediction_rows != cfg['expected_prediction_rows']:
        raise ValueError('planned operation/prediction counts differ')
    atomic_write_json(destination / 'operations.json', planned)
    atomic_write_json(destination / 'membership_sets.json', sets)
    atomic_write_csv(destination / 'donor_base_overlap.csv', list(donor_rows[0]), donor_rows)
    atomic_write_json(destination / 'PLAN_RECEIPT.json', dict(
        created_utc=datetime.now(timezone.utc).isoformat(), config_sha256=sha(CONFIG),
        split_sha256=sha(ROOT/cfg['preparation_dir']/'SPLIT_MANIFEST.csv'),
        operations=len(planned), by_module=dict(counts), prediction_rows=prediction_rows,
        membership_sets=len(sets), signal_arrays_opened=0, model_fits=0,
        files={p.name: sha(p) for p in destination.iterdir()}))
    print(json.dumps(dict(operations=len(planned), by_module=dict(counts),
                          prediction_rows=prediction_rows, signal_arrays_opened=0)))


def run(out, plan_dir, freeze_path):
    import joblib
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    from threadpoolctl import threadpool_limits, threadpool_info

    out.mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        raise FileExistsError('existing experimental evidence')
    started, cpu = time.monotonic(), time.process_time()
    rec = dict(run_id=out.name, pid=os.getpid(), status='running', operations=[],
        started_utc=datetime.now(timezone.utc).isoformat(), interpreter=sys.executable,
        confirmation_data_accessed=False, GPU_used=False)
    atomic_write_json(out/'receipt.json', rec)
    try:
        cfg, oldcfg, split = config_and_split()
        limits = cfg['resource']
        resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (limits['run_cpu_soft_seconds'], limits['run_cpu_hard_seconds']))
        def expired(*_):
            raise TimeoutError('development control wall/CPU cap')
        signal.signal(signal.SIGALRM, expired)
        signal.signal(signal.SIGXCPU, expired)
        signal.alarm(limits['job_wall_seconds'])
        frozen = json.loads(freeze_path.read_text())
        code_freeze=json.loads((STUDY/'CODE_FREEZE.json').read_text())
        if code_freeze['prefit_manifest_sha256']!=sha(freeze_path):
            raise ValueError('private pre-fit commit does not bind this input manifest')
        if any(sha(ROOT/name) != value for name, value in frozen['files'].items()):
            raise ValueError('frozen inputs changed before execution')
        atomic_write_json(out/'INPUT_MANIFEST.json', frozen['files'])
        rec['freeze_sha256'] = sha(freeze_path)
        rec['private_prefit_commit'] = code_freeze['commit']
        planned = json.loads((plan_dir/'operations.json').read_text())
        sets = json.loads((plan_dir/'membership_sets.json').read_text())
        prep = ROOT/cfg['preparation_dir']
        supervisor = json.loads((prep/'supervisor_receipt.json').read_text())
        _validate_completion(prep, supervisor['child_pid'])
        records = json.loads((prep/'schema.json').read_text())['records']
        allowed = {(s,1):'source' for s in cfg['source_subjects']}
        allowed.update({(s,t):'development' for s in cfg['development_subjects'] for t in (1,2)})
        if len(records)!=42 or {(r['subject'],r['session']) for r in records} != set(allowed):
            raise ValueError('prepared file allowlist differs before array loading')
        arrays, labels, identifiers = [], [], []
        for record in sorted(records, key=lambda r:(r['subject'],r['session'])):
            s,t = record['subject'],record['session']
            if record['role'] != allowed[s,t] or record['cache']['array_path'] not in frozen['files']:
                raise ValueError('cache role or frozen path differs before array loading')
            with np.load(ROOT/record['cache']['array_path'], allow_pickle=False) as archive:
                X,y = archive['X'],archive['y']
            validate_prepared_array(ROOT,record,X,y,oldcfg)
            ids = [trial_id(s,t,i) for i in range(100)]
            if any(int(split[tid]['label'])!=int(y[i]) or split[tid]['epoch_sha256']!=record['metadata']['trial_epoch_sha256'][i]
                   for i,tid in enumerate(ids)):
                raise ValueError('cache labels/content differ from split')
            arrays.append(X); labels.append(y); identifiers.extend(ids)
        Xall,yall = np.concatenate(arrays),np.concatenate(labels)
        del arrays,labels,X,y
        index={tid:i for i,tid in enumerate(identifiers)}
        if len(index)!=4200:
            raise ValueError('allowed array identity count differs')
        idx=lambda ids:np.asarray([index[t] for t in ids],dtype=int)
        predictions,metrics,timings,reference_checks=[],[],[],[]
        training_signatures=set()
        representations={}
        ea_covariances={}
        ea_whiteners={}
        with threadpool_limits(limits=1):
            rec['threadpools']=threadpool_info()
            if any(p['num_threads']!=1 for p in rec['threadpools']):
                raise ValueError('numerical threads exceed1')
            pc,pw=time.process_time(),time.monotonic()
            covariances=oas_covariances(Xall)
            rec['covariance_preparation']=dict(cpu_seconds=time.process_time()-pc,
                                               wall_seconds=time.monotonic()-pw,shape=list(covariances.shape))
            for number,item in enumerate(planned,1):
                oc,ow=time.process_time(),time.monotonic()
                op=item['operation']; oid=op['operation_id']
                source,added,evaluation,unlabeled=[sets[item['membership'][usage]] for usage in
                                                  ('source','added','evaluation','unlabeled_history')]
                fit_ids=source+added
                ix,qx=idx(fit_ids),idx(evaluation)
                alignment_key='unaligned'
                ac=time.process_time()
                if op['module']=='D':
                    if 'source' not in ea_covariances:
                        sx=idx(source)
                        subjects=np.asarray([int(split[t]['subject_id']) for t in source])
                        ea_whiteners['source']=fit_source_ea(Xall[sx],subjects)
                        ea_covariances['source']={tid:cov for tid,cov in zip(source,
                            oas_covariances(apply_source_ea(Xall[sx],subjects,ea_whiteners['source'])))}
                        joblib.dump(ea_whiteners['source'],out/'ea_source_whiteners.joblib')
                    target=op['target']
                    if target not in ea_covariances:
                        ea_whiteners[target]=fit_ea_whitener(Xall[idx(unlabeled)])
                        target_ids=unlabeled+evaluation
                        ea_covariances[target]={tid:cov for tid,cov in zip(target_ids,
                            oas_covariances(apply_ea(Xall[idx(target_ids)],ea_whiteners[target])))}
                        joblib.dump(ea_whiteners[target],out/f'ea_target_s{target:03d}.joblib')
                    aligned={**ea_covariances['source'],**ea_covariances[target]}
                    train_cov=np.asarray([aligned[t] for t in fit_ids])
                    eval_cov=np.asarray([aligned[t] for t in evaluation])
                    alignment_key=f'EA_source_and_s{target:03d}_session1'
                else:
                    train_cov,eval_cov=covariances[ix],covariances[qx]
                alignment_cpu=time.process_time()-ac
                repr_ids=source if op['representation']=='source_frozen' else fit_ids
                # A target-specific alignment is conservatively part of this key even at h=0.
                repr_key=fingerprint([op['representation'],alignment_key,repr_ids])
                fc,fw=time.process_time(),time.monotonic()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    new_repr=repr_key not in representations
                    if new_repr:
                        representation=fit_representation(train_cov,len(source),op['representation'])
                        representations[repr_key]=representation
                        joblib.dump(representation,out/f'representation_{repr_key}.joblib')
                    else:
                        representation=representations[repr_key]
                    repr_before=joblib.hash(representation)
                    classifier=train_classifier(representation.transform(train_cov),yall[ix],len(source),
                                                op['C'],op['lambda_personal'])
                training_signature=fingerprint([fit_ids,op['C'],op['lambda_personal'],
                                                repr_before,joblib.hash(train_cov)])
                training_signatures.add(training_signature)
                fit_cpu,fit_wall=time.process_time()-fc,time.monotonic()-fw
                if any('converg' in str(w.message).lower() for w in caught):
                    raise ValueError('convergence warning: '+oid)
                model_before=joblib.hash(classifier)
                pc,pw=time.process_time(),time.monotonic()
                probability=classifier.predict_proba(representation.transform(eval_cov))
                predict_cpu,predict_wall=time.process_time()-pc,time.monotonic()-pw
                if joblib.hash(classifier)!=model_before or joblib.hash(representation)!=repr_before:
                    raise ValueError('prediction altered fitted parameters')
                if (classifier.classes_.tolist()!=[0,1] or probability.shape!=(len(evaluation),2)
                        or not np.isfinite(probability).all() or (probability<0).any()
                        or (probability>1).any() or not np.allclose(probability.sum(1),1,rtol=0,atol=1e-12)):
                    raise ValueError('class/probability contract differs')
                predicted=probability.argmax(1)
                pred=[dict(operation_id=oid,subject_id=int(split[tid]['subject_id']),trial_id=tid,
                    y_true=int(yall[qx[i]]),y_pred=int(predicted[i]),p_class0_right=float(probability[i,0]),
                    p_class1_left=float(probability[i,1])) for i,tid in enumerate(evaluation)]
                for target in sorted({p['subject_id'] for p in pred}):
                    mask=np.asarray([p['subject_id']==target for p in pred])
                    if mask.sum()!=100 or Counter(yall[qx][mask])!={0:50,1:50}:
                        raise ValueError('evaluation participant/class count differs')
                    metrics.append(dict(operation_id=oid,module=op['module'],source_n=op['source_n'],
                        h_total=op['h_total'],C=op['C'],lambda_personal=op['lambda_personal'],
                        representation=op['representation'],draw=op['draw'],target_subject=target,
                        donor_kind=op['donor_kind'] or '',
                        balanced_accuracy=float(balanced_accuracy_score(yall[qx][mask],predicted[mask]))))
                reference=item['original_reference']
                if reference:
                    original=read_csv(ROOT/'result/day3/grid_r002'/f'{reference}.predictions.csv')
                    expected={r['trial_id']:r for r in original if int(r['draw'])==op['draw']}
                    if set(expected)!=set(evaluation):
                        raise ValueError('original reference evaluation identity differs')
                    discrepancy=max(abs(p[k]-float(expected[p['trial_id']][k])) for p in pred
                                    for k in ('p_class0_right','p_class1_left'))
                    if discrepancy>cfg['analysis']['reference_prediction_tolerance']:
                        raise ValueError(f'original prediction discrepancy {discrepancy}: {oid}')
                    if any(p['y_pred']!=int(expected[p['trial_id']]['y_pred']) for p in pred):
                        raise ValueError('original anchor labels/BA differ')
                    reference_checks.append(dict(operation_id=oid,original_operation=reference,
                                                  max_probability_difference=discrepancy,identical_predictions=True))
                atomic_write_csv(out/f'{oid}.predictions.csv',list(pred[0]),pred)
                joblib.dump(classifier,out/f'{oid}.joblib')
                timing=dict(operation_id=oid,module=op['module'],fit_cpu_seconds=fit_cpu,
                    fit_wall_seconds=fit_wall,predict_cpu_seconds=predict_cpu,predict_wall_seconds=predict_wall,
                    alignment_cpu_seconds=alignment_cpu,representation_newly_fitted=new_repr,
                    operation_cpu_seconds=time.process_time()-oc,operation_wall_seconds=time.monotonic()-ow)
                atomic_write_json(out/f'{oid}.json',dict(operation=op,membership=item['membership'],
                    donor_subjects=item['donor_subjects'],representation_key=repr_key,alignment=alignment_key,
                    training_signature=training_signature,
                    model_hash_before_and_after_prediction=model_before,
                    representation_hash_before_and_after_prediction=repr_before,
                    timing=timing,warnings=[str(w.message) for w in caught],
                    artifacts={oid+suffix:sha(out/(oid+suffix)) for suffix in ('.predictions.csv','.joblib')},
                    representation_sha256=sha(out/f'representation_{repr_key}.joblib')))
                predictions.extend(pred);timings.append(timing)
                rec['operations'].append(dict(operation_id=oid,receipt_sha256=sha(out/f'{oid}.json')))
                atomic_write_json(out/'receipt.json',rec)
                if number==4:
                    forecast={m:4*next(t['operation_cpu_seconds'] for t in timings if t['module']==m)
                              * (cfg['expected_by_module'][m]-1) for m in cfg['expected_by_module']}
                    projected_cpu=time.process_time()-cpu+sum(forecast.values())
                    bytes_so_far=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
                    projected_bytes=bytes_so_far/4*cfg['expected_operations']
                    atomic_write_json(out/'TIMING_CHECKPOINT.json',dict(
                        first_four=[t['operation_id'] for t in timings],method='4x per-module first-operation CPU plus elapsed',
                        estimated_total_cpu_seconds=projected_cpu,estimated_output_bytes=projected_bytes,
                        remaining_module_cpu_forecast=forecast,performance_not_used_to_adapt_design=True))
                    if projected_cpu>limits['run_cpu_soft_seconds'] or projected_bytes>limits['new_output_forecast_bytes_cap']:
                        raise RuntimeError('timing/output forecast exceeds frozen cap')
                if number%50==0 or number in (4,len(planned)):
                    print(f'operation {number}/{len(planned)}; CPU {time.process_time()-cpu:.1f}s; wall {time.monotonic()-started:.1f}s',flush=True)
            # Historical-only predictor: reuse the original source-only fit; charge every label scored.
            source_model_path=ROOT/cfg['predictor']['source_model_path']
            if sha(source_model_path)!=cfg['predictor']['source_model_sha256']:
                raise ValueError('historical predictor source model changed')
            source_model=joblib.load(source_model_path)
            model_hash=joblib.hash(source_model)
            predictor_rows=[]
            for target in cfg['development_subjects']:
                pool=sorted(t for t,r in split.items() if int(r['subject_id'])==target and int(r['session'])==1)
                pool_ix=idx(pool)
                for draw in range(cfg['draws']):
                    for h in cfg['predictor']['history_label_budgets']:
                        seed=np.random.SeedSequence([oldcfg['history_draw_seeds'][draw],target])
                        chosen=nested_history_indices(yall[pool_ix],h//2,seed)
                        historical_ids=[pool[i] for i in chosen]
                        hx=idx(historical_ids)
                        predicted=source_model.predict(Xall[hx])
                        predictor_rows.append(dict(subject_id=target,draw=draw,h_total=h,
                            balanced_accuracy=float(balanced_accuracy_score(yall[hx],predicted)),
                            trial_ids_json=json.dumps(historical_ids)))
            if joblib.hash(source_model)!=model_hash:
                raise ValueError('historical scoring mutated source model')
            for name,rows in [('predictions.csv',predictions),('subject_metrics.csv',metrics),
                              ('timing.csv',timings),('reference_checks.csv',reference_checks),
                              ('predictor_scores.csv',predictor_rows)]:
                atomic_write_csv(out/name,list(rows[0]),rows)
        if (len(timings)!=1464 or len(predictions)!=172800 or len(metrics)!=1728
                or len(reference_checks)!=52 or len(predictor_rows)!=72):
            raise ValueError('final operation, reference or score counts differ')
        if len({(p['operation_id'],p['trial_id']) for p in predictions})!=172800:
            raise ValueError('duplicate operation/trial prediction')
        if any(sha(ROOT/name)!=value for name,value in frozen['files'].items()):
            raise ValueError('frozen input mutation during execution')
        output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        if output_bytes>limits['new_output_forecast_bytes_cap']:
            raise ValueError('actual output exceeds cap')
        rec.update(status='completed',exit_code=0,counts=dict(training_operations=len(timings),
            predictions=len(predictions),subject_scores=len(metrics),reference_checks=len(reference_checks),
            predictor_scores=len(predictor_rows),fitted_representations=len(representations)),
            distinct_training_signatures=len(training_signatures),
            output_bytes=output_bytes,allowed_cache_arrays_loaded=42)
    except BaseException as error:
        rec.update(status='failed',exit_code=1,error=repr(error))
        traceback.print_exc()
    finally:
        rec.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-started,
                   cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_write_json(out/'receipt.json',rec)
        signal.alarm(0)
    if rec['status']=='completed':
        outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.json','.csv','.joblib')
                 and p.name not in ('supervisor_receipt.json','COMPLETED.json')}
        atomic_write_json(out/'COMPLETED.json',dict(run_id=out.name,receipt_sha256=sha(out/'receipt.json'),outputs=outputs))
    return rec['exit_code']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-plan',type=Path)
    parser.add_argument('--plan-dir',type=Path,default=STUDY/'frozen_plan')
    parser.add_argument('--freeze',type=Path,default=STUDY/'PREFIT_MANIFEST.json')
    parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.prepare_plan:
        make_plan(args.prepare_plan.resolve())
        return 0
    if not args.run_dir:
        parser.error('--run-dir or --prepare-plan is required')
    return run(args.run_dir.resolve(),args.plan_dir.resolve(),args.freeze.resolve())


if __name__=='__main__':
    raise SystemExit(main())
