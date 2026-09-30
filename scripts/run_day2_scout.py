"""Execute the frozen18-fit development scout; all outputs stay under a new run ID."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,datetime,hashlib,json,pathlib,resource,signal,sys,time,traceback,warnings
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from expose.runtime import atomic_write_json,atomic_write_csv
from expose.provenance import file_sha256 as sha
from expose.scout import load_config,trial_id,expected_fit_id,validate_prepared_array,validate_predictions
p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);args=p.parse_args()
out=pathlib.Path(args.run_dir).resolve();out.mkdir(exist_ok=True)
if (out/'receipt.json').exists():raise FileExistsError('preserve existing scientific run; use a documented successor')
start=time.perf_counter();cpu=time.process_time()
rec={'run_id':out.name,'status':'running','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fits':[],'scope':'development-only exploratory18-fit scout; no confirmation','interpreter':sys.executable}
atomic_write_json(out/'receipt.json',rec)

def write_csv(name,rows):
    if not rows:raise ValueError('required output has zero rows: '+name)
    atomic_write_csv(out/name,list(rows[0]),rows)

def transform_hashes(model):
    return {name+':'+attr:hashlib.sha256(getattr(step,attr).tobytes()).hexdigest()
            for name,step in model.named_steps.items() for attr in ['reference_','mean_','scale_','filters_'] if hasattr(step,attr)}

try:
    cfg=load_config(ROOT);limits=cfg['resource']
    resource.setrlimit(resource.RLIMIT_AS,(limits['process_memory_gib']*1024**3,)*2)
    resource.setrlimit(resource.RLIMIT_CPU,(limits['scout_cpu_soft_seconds'],limits['scout_cpu_hard_seconds']))
    def timeout(signum,frame):raise TimeoutError('scout CPU/wall limit reached')
    signal.signal(signal.SIGXCPU,timeout);signal.signal(signal.SIGALRM,timeout);signal.alarm(limits['scout_wall_seconds'])
    import numpy as np
    import mne,joblib
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_info,threadpool_limits
    from expose.baselines import make_baseline,nested_history_indices,assert_disjoint_trial_ids
    mne.set_log_level('WARNING')
    prep=ROOT/'result/day2/preparation_r001'
    preparation=json.loads((prep/'receipt.json').read_text());marker=json.loads((prep/'COMPLETED.json').read_text())
    if preparation['status']!='completed' or preparation['exit_code']!=0 or marker['receipt_sha256']!=sha(prep/'receipt.json'):
        raise ValueError('data preparation did not complete with valid receipt')
    for name,digest in marker['outputs'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT):raise ValueError('preparation output escapes project')
        if sha(path)!=digest:raise ValueError('preparation output mismatch: '+name)
    schema=json.loads((prep/'schema.json').read_text())
    with (ROOT/'research/day2/SPLIT_MANIFEST.csv').open() as f:split_rows=list(csv.DictReader(f))
    needed={(s,1) for s in cfg['source_subjects']}|{(s,d) for s in cfg['development_subjects'] for d in [1,2]}
    if {(r['subject'],r['session']) for r in schema['records']}!=needed or len(schema['records'])!=34:raise ValueError('preparation set differs')
    paths=[ROOT/'research/protocols/DAY2_CONFIG_v1.json',ROOT/cfg['role_manifest'],ROOT/cfg['loader_file'],ROOT/cfg['model_code'],ROOT/'src/expose/scout.py',ROOT/'src/expose/runtime.py',ROOT/'src/expose/provenance.py',pathlib.Path(__file__),ROOT/'research/environment/requirements.lock.txt',prep/'receipt.json',prep/'schema.json',prep/'COMPLETED.json',ROOT/'research/day2/SPLIT_MANIFEST.csv',ROOT/'research/day2/DATA_MANIFEST.csv']
    arrays={};all_hashes=[]
    for row in schema['records']:
        path=ROOT/row['cache']['array_path'];mp=ROOT/row['cache']['metadata_path'];paths.extend([path,mp])
        with np.load(path,allow_pickle=False) as z:X,y=z['X'],z['y']
        validate_prepared_array(ROOT,row,X,y,cfg)
        arrays[row['subject'],row['session']]=(X,y)
        all_hashes.extend(row['metadata']['trial_epoch_sha256'])
    if len(set(all_hashes))!=3400:raise ValueError('duplicate prepared EEG content')
    manifest={str(path.relative_to(ROOT)):sha(path) for path in paths}
    atomic_write_json(out/'INPUT_MANIFEST.json',manifest)
    rec['config_sha256']=sha(ROOT/'research/protocols/DAY2_CONFIG_v1.json');rec['input_manifest_sha256']=sha(out/'INPUT_MANIFEST.json')
    Xs=np.concatenate([arrays[s,1][0] for s in cfg['source_subjects']]);ys=np.concatenate([arrays[s,1][1] for s in cfg['source_subjects']])
    src_ids=[trial_id(s,1,i) for s in cfg['source_subjects'] for i in range(100)]
    if len(Xs)!=cfg['source_trials']:raise ValueError('source count differs')
    predictions=[];members=[];times=[]
    with threadpool_limits(limits=limits['numerical_threads']):
        rec['threadpools']=threadpool_info()
        if limits['numerical_threads']!=1 or any(x['num_threads']!=1 for x in rec['threadpools']):raise ValueError('thread contract differs')
        for model_name in cfg['model_order']:
            for dose in cfg['history_doses_per_class']:
                targets=[None] if dose==0 else cfg['development_subjects']
                for target in targets:
                    fit_id=expected_fit_id(model_name,dose,target)
                    train_ids=list(src_ids);Xtr,ytr=Xs,ys
                    if dose:
                        Xh,yh=arrays[target,1];take=nested_history_indices(yh,dose,cfg['history_draw_seed'])
                        Xtr=np.concatenate([Xs,Xh[take]]);ytr=np.concatenate([ys,yh[take]])
                        train_ids.extend(trial_id(target,1,int(i)) for i in take)
                    fit_wall=time.perf_counter();fit_cpu=time.process_time()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');model=make_baseline(model_name).fit(Xtr,ytr)
                    fit_wall=time.perf_counter()-fit_wall;fit_cpu=time.process_time()-fit_cpu
                    if any('converge' in str(w.message).lower() for w in caught):raise RuntimeError('required fit convergence warning: '+fit_id)
                    if model.classes_.tolist()!=[0,1]:raise ValueError('unexpected model class order')
                    frozen=transform_hashes(model)
                    fitpred=[];fitmembers=[{'fit_id':fit_id,'membership':'training','trial_id':tid} for tid in train_ids]
                    predict_wall=0.;predict_cpu=0.
                    for subject in (cfg['development_subjects'] if target is None else [target]):
                        Xq,yq=arrays[subject,2];qids=[trial_id(subject,2,i) for i in range(100)]
                        assert_disjoint_trial_ids(train_ids,qids)
                        pw=time.perf_counter();pc=time.process_time();prob=model.predict_proba(Xq)
                        predict_wall+=time.perf_counter()-pw;predict_cpu+=time.process_time()-pc
                        pred=model.classes_[np.argmax(prob,axis=1)]
                        if not np.all(np.isfinite(prob)) or prob.shape!=(100,2):raise ValueError('invalid probabilities')
                        fitpred.extend({'fit_id':fit_id,'model':model_name,'dose_per_class':dose,'subject_id':subject,'trial_id':qids[i],'y_true':int(yq[i]),'y_pred':int(pred[i]),'p_class0_right':float(prob[i,0]),'p_class1_left':float(prob[i,1])} for i in range(100))
                        fitmembers.extend({'fit_id':fit_id,'membership':'evaluation','trial_id':tid} for tid in qids)
                    if transform_hashes(model)!=frozen:raise ValueError('evaluation modified fitted transforms')
                    predictions.extend(fitpred);members.extend(fitmembers)
                    row={'fit_id':fit_id,'model':model_name,'dose_per_class':dose,'target_subject':target,'training_trials':len(Xtr),'evaluation_trials':len(fitpred),'fit_wall_seconds':fit_wall,'fit_cpu_seconds':fit_cpu,'predict_wall_seconds':predict_wall,'predict_cpu_seconds':predict_cpu,'warnings':[str(w.message) for w in caught]}
                    times.append({k:v for k,v in row.items() if k!='warnings'})
                    # Each finished fit is independently recoverable even if a later fit dies.
                    write_csv(fit_id+'.predictions.csv',fitpred);write_csv(fit_id+'.membership.csv',fitmembers)
                    joblib.dump(model,out/(fit_id+'.joblib'))
                    atomic_write_json(out/(fit_id+'.json'),{**row,'class_names':cfg['probability_schema']['class_names'],'model_classes':[0,1],'transform_hashes':frozen,'predictions_sha256':sha(out/(fit_id+'.predictions.csv')),'membership_sha256':sha(out/(fit_id+'.membership.csv')),'model_sha256':sha(out/(fit_id+'.joblib'))})
                    rec['fits'].append(row)
                    atomic_write_json(out/'receipt.json',rec)
                    print(f'fit {len(rec["fits"]):02d}/18 {fit_id}: {len(Xtr)} training, CPU {fit_cpu:.3f}s',flush=True)
        validate_predictions(predictions,cfg,split_rows,members)
        write_csv('predictions.csv',predictions);write_csv('fit_membership.csv',members);write_csv('timing.csv',times)
        # Additional full-source timing probes, separate from scout scores/comparisons.
        cw=time.perf_counter();cc=time.process_time();cov=Covariances(estimator='oas');Cs=cov.transform(Xs);mdm=MDM(metric='riemann',n_jobs=1).fit(Cs,ys)
        mdm_fit_wall=time.perf_counter()-cw;mdm_fit_cpu=time.process_time()-cc
        pw=time.perf_counter();pc=time.process_time();mp=mdm.predict(cov.transform(Xs[:100]));mdm_pred_wall=time.perf_counter()-pw;mdm_pred_cpu=time.process_time()-pc
        cw=time.perf_counter();cc=time.process_time();cov_cal=Covariances(estimator='oas');ts=TangentSpace(metric='riemann',tsupdate=False);Z=ts.fit_transform(cov_cal.transform(Xs));scaler=StandardScaler().fit(Z);Z=scaler.transform(Z);centers=np.stack([Z[ys==c].mean(axis=0) for c in [0,1]])
        cal_fit_wall=time.perf_counter()-cw;cal_fit_cpu=time.process_time()-cc
        subject=cfg['timing_only_probes']['history_subject'];Xh,yh=arrays[subject,1];take=nested_history_indices(yh,15,cfg['history_draw_seed'])
        uc=time.process_time();Zh=scaler.transform(ts.transform(cov_cal.transform(Xh[take])));weight=cfg['timing_only_probes']['calibration_prior_weight_per_class'];adapted=np.stack([(weight*centers[c]+Zh[yh[take]==c].sum(axis=0))/(weight+15) for c in [0,1]]);update_cpu=time.process_time()-uc
        pw=time.perf_counter();pc=time.process_time();Zp=scaler.transform(ts.transform(cov_cal.transform(Xs[:100])));cal_pred=((Zp[:,None,:]-adapted[None,:,:])**2).sum(axis=2).argmin(axis=1);cal_pred_wall=time.perf_counter()-pw;cal_pred_cpu=time.process_time()-pc
        if mp.shape!=(100,) or cal_pred.shape!=(100,) or not np.all(np.isfinite(adapted)):raise ValueError('timing probe output invalid')
        atomic_write_json(out/'timing_probes.json',{'scope':'runtime only, prediction input=first100 source training trials; no accuracy computed, not scout evidence','mdm':{'training_trials':1800,'fit_wall_seconds':mdm_fit_wall,'fit_cpu_seconds':mdm_fit_cpu,'predict100_wall_seconds':mdm_pred_wall,'predict100_cpu_seconds':mdm_pred_cpu},'source_fixed_ts_centroid_calibration':{'training_trials':1800,'fit_wall_seconds':cal_fit_wall,'fit_cpu_seconds':cal_fit_cpu,'target_history_per_class':15,'source_prior_weight_per_class':weight,'target_update_cpu_seconds':update_cpu,'predict100_wall_seconds':cal_pred_wall,'predict100_cpu_seconds':cal_pred_cpu,'tuning':'none; fixed engineering probe, scientific lambda still not selected'}})
    if len(rec['fits'])!=cfg['expected_fits'] or len(predictions)!=cfg['expected_prediction_rows']:raise ValueError('required run count differs')
    if any(sha(ROOT/name)!=digest for name,digest in manifest.items()):raise ValueError('input mutation during scout')
    rec['checks']={'required_fits':len(rec['fits']),'prediction_rows':len(predictions),'membership_rows':len(members),'unique_prepared_trials':3400,'no_confirmation':True,'scorer_structure_passed':True,'fitted_transforms_unchanged_by_evaluation':True,'inputs_unchanged':True}
    rec.update(status='completed',exit_code=0)
except Exception as error:
    rec.update(status='failed',exit_code=1,error=repr(error));traceback.print_exc()
finally:
    rec.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_write_json(out/'receipt.json',rec)
if rec['status']=='completed':
    outputs={f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.suffix in ['.json','.csv','.joblib'] and f.name not in ['supervisor_receipt.json','COMPLETED.json']}
    atomic_write_json(out/'COMPLETED.json',{'run_id':rec['run_id'],'receipt_sha256':sha(out/'receipt.json'),'outputs':outputs})
signal.alarm(0)
raise SystemExit(rec['exit_code'])
