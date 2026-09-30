"""Run the frozen, four-fit Day1 CPU smoke; preserve every failed attempt."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import csv, datetime, hashlib, json, pathlib, resource, signal, sys, time, traceback, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,16*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600,3610))
def timed_out(signum, frame):
    raise TimeoutError(f'owned Day1 process exceeded time cap (signal {signum})')
signal.signal(signal.SIGALRM, timed_out)
signal.signal(signal.SIGXCPU, timed_out)
signal.alarm(1800)
import numpy as np
import mne
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_info, threadpool_limits
from expose.baselines import MODEL_ORDER, assert_disjoint_trial_ids, make_baseline, nested_history_indices, validate_epochs
from expose.provenance import verify_cache
mne.set_log_level('WARNING')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024**2), b''):h.update(chunk)
    return h.hexdigest()
def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv_write(path,rows):
    if not rows:raise ValueError('zero required rows')
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def trial_ids(subject,session):return [f'lee2019:s{subject:03d}:session{session}:offline:t{i:03d}' for i in range(100)]
def transform_hashes(model):
    arrays=[]
    for step in model.named_steps.values():
        for attr in ('reference_','mean_','scale_','filters_'):
            if hasattr(step,attr): arrays.append((type(step).__name__+':'+attr, hashlib.sha256(np.asarray(getattr(step,attr)).tobytes()).hexdigest()))
    return arrays
out=ROOT/'result/day1/smoke_r001';out.mkdir(exist_ok=True)
if (out/'receipt.json').exists():raise RuntimeError('r001 already exists; preserve original, use a successor for an authorized retry')
start=time.perf_counter();cpu_start=time.process_time()
receipt={'status':'running','run_id':'day1_smoke_r001','pid':os.getpid(),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'interpreter':sys.executable,'scope':'engineering only; four fits, one source and one development subject','fits':[]}
write_json(out/'receipt.json',receipt)
try:
    config_path=ROOT/'research/protocols/DAY1_CONFIG_v1.json'
    if sha(config_path)!=(config_path.with_suffix('.json.sha256')).read_text().split()[0]:raise ValueError('config identity mismatch')
    cfg=json.loads(config_path.read_text())
    if cfg['model_order']!=list(MODEL_ORDER) or [cfg[k] for k in ['source_session','history_session','evaluation_session']]!=[1,1,2]:raise ValueError('unsupported model or session configuration')
    if [cfg[k] for k in ['numerical_threads','memory_limit_gib','process_cpu_limit_seconds','job_wall_limit_seconds']]!=[1,16,3600,1800]:raise ValueError('resource contract differs from enforced limits')
    expected_models={'csp_lda':{'csp_components':4,'csp_reg':'ledoit_wolf','csp_cov_est':'concat','csp_log':True,'csp_norm_trace':False,'lda_solver':'lsqr','lda_shrinkage':'auto'},'ts_lr':{'covariance':'oas','tangent_metric':'riemann','tsupdate':False,'scaler':'StandardScaler fitted only on training','lr_C':1.0,'lr_max_iter':1000,'lr_solver':'lbfgs'}}
    if cfg['models']!=expected_models:raise ValueError('frozen model settings differ from fixed baseline implementation')
    role_path=ROOT/'research/PARTICIPANT_ROLES_v1.csv'
    if sha(role_path)!=(role_path.with_suffix('.csv.sha256')).read_text().split()[0]:raise ValueError('role identity mismatch')
    roles={int(x['subject_id']):x['role'] for x in csv.DictReader(role_path.open())}
    if roles[cfg['source_subject']]!='source' or roles[cfg['development_subject']]!='development':raise ValueError('subject role violation')
    selected=json.loads((ROOT/cfg['selection_file']).read_text())
    if sha(ROOT/cfg['selection_file'])!=cfg['selection_sha256']:raise ValueError('selection identity mismatch')
    if (cfg['source_subject'],cfg['development_subject'])!=(selected['day1_source_subject'],selected['day1_development_subject']):raise ValueError('Day1 selection differs')
    paths=[config_path,role_path,ROOT/cfg['selection_file'],ROOT/'src/expose/lee.py',ROOT/'src/expose/baselines.py',ROOT/'src/expose/provenance.py',pathlib.Path(__file__),ROOT/'research/environment/requirements.lock.txt',ROOT/'result/day1/download_receipt.json',ROOT/'result/day1/schema_validation.json']
    schema=json.loads((ROOT/'result/day1/schema_validation.json').read_text())
    download=json.loads((ROOT/'result/day1/download_receipt.json').read_text())
    arrays={};metas={};split_rows=[];seen=set()
    for subject in [cfg['source_subject'],cfg['development_subject']]:
        for session in [1,2]:
            path=ROOT/f'data/derived/day1/subj{subject:02d}_sess{session:02d}_offline.npz'
            meta_path=path.with_suffix('.json');paths += [path,meta_path]
            with np.load(path,allow_pickle=False) as loaded:X,y=loaded['X'],loaded['y']
            meta=json.loads(meta_path.read_text())
            verify_cache(ROOT,subject,session,roles[subject],X,y,meta,schema,download)
            validate_epochs(X,y)
            if list(X.shape)!=cfg['preprocessing']['expected_shape'] or np.bincount(y).tolist()!=[50,50]:raise ValueError('unexpected epoch count or balance')
            if meta['channel_names']!=cfg['preprocessing']['channels'] or meta['mat_variable']!='EEG_MI_train':raise ValueError('preprocessing identity differs')
            if any(meta['config'][k]!=cfg['preprocessing'][k] for k in ['l_freq','h_freq','tmin','tmax','target_fs','filter_order']):raise ValueError('epoch settings differ from frozen config')
            ids=trial_ids(subject,session)
            for i,tid in enumerate(ids):
                epoch_hash=hashlib.sha256(X[i].tobytes()).hexdigest()
                if epoch_hash in seen:raise ValueError('exact processed EEG duplicate across Day1 trials')
                seen.add(epoch_hash)
                split_rows.append({'trial_id':tid,'subject_id':subject,'role':roles[subject],'session':session,'mat_variable':'EEG_MI_train','trial_index_zero_based':i,'label':int(y[i]),'event_sample_zero_based':meta['event_samples_zero_based'][i],'filter_start_sample':meta['filter_input_start_samples_zero_based'][i],'filter_stop_sample_exclusive':meta['filter_input_stop_samples_zero_based_exclusive'][i],'epoch_sha256':epoch_hash,'usage':'evaluation_smoke' if subject==cfg['development_subject'] and session==2 else ('schema_only' if subject==cfg['source_subject'] and session==2 else 'training_or_history')})
            arrays[subject,session]=(X,y);metas[subject,session]=meta
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write_json(out/'INPUT_MANIFEST.json',manifest)
    csv_write(ROOT/'research/day1/SPLIT_MANIFEST.csv',split_rows)
    receipt['config_sha256']=sha(config_path);receipt['input_manifest_sha256']=sha(out/'INPUT_MANIFEST.json')
    Xs,ys=arrays[cfg['source_subject'],1];Xh,yh=arrays[cfg['development_subject'],1];Xq,yq=arrays[cfg['development_subject'],2]
    src_ids=trial_ids(cfg['source_subject'],1);history_ids=trial_ids(cfg['development_subject'],1);eval_ids=trial_ids(cfg['development_subject'],2)
    timings=[];predictions=[];fit_membership=[]
    with threadpool_limits(limits=1):
        receipt['threadpools']=threadpool_info()
        if any(x['num_threads']!=1 for x in receipt['threadpools']):raise RuntimeError('numerical thread cap violated')
        for condition in cfg['conditions']:
            take=nested_history_indices(yh,condition['n_per_class'],cfg['history_draw_seed'])
            Xtr=np.concatenate([Xs,Xh[take]]);ytr=np.concatenate([ys,yh[take]])
            train_ids=src_ids+[history_ids[i] for i in take]
            assert_disjoint_trial_ids(train_ids,eval_ids);validate_epochs(Xtr,ytr)
            if len(Xtr)!=condition['total_training_trials']:raise ValueError('training count differs')
            for name in MODEL_ORDER:
                fit_id=condition['name']+'__'+name
                model=make_baseline(name)
                fitstart=time.perf_counter();cpustart=time.process_time()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(Xtr,ytr)
                fitwall=time.perf_counter()-fitstart;fitcpu=time.process_time()-cpustart
                before=transform_hashes(model)
                predstart=time.perf_counter();predcpu=time.process_time()
                probs=model.predict_proba(Xq);pred=model.classes_[np.argmax(probs,axis=1)]
                predwall=time.perf_counter()-predstart;predcpu=time.process_time()-predcpu
                if before!=transform_hashes(model):raise ValueError('evaluation changed fitted transforms')
                if probs.shape!=(100,2) or not np.all(np.isfinite(probs)) or not np.allclose(probs.sum(axis=1),1):raise ValueError('invalid probability output')
                cm=confusion_matrix(yq,pred,labels=[0,1]);manual_ba=float(np.mean(cm.diagonal()/cm.sum(axis=1)));ba=float(balanced_accuracy_score(yq,pred))
                if abs(manual_ba-ba)>1e-12:raise ValueError('manual confusion BA and library disagree')
                row={'fit_id':fit_id,'model':name,'condition':condition['name'],'n_per_class':condition['n_per_class'],'training_trials':len(Xtr),'evaluation_trials':len(Xq),'fit_wall_seconds':fitwall,'fit_cpu_seconds':fitcpu,'predict_wall_seconds':predwall,'predict_cpu_seconds':predcpu,'balanced_accuracy_smoke_only':ba,'confusion_matrix':cm.tolist(),'warnings':[str(x.message) for x in caught],'fitted_transform_hashes':before}
                receipt['fits'].append(row)
                timings.append({k:v for k,v in row.items() if k not in ['warnings','confusion_matrix','fitted_transform_hashes']})
                predictions.extend({'fit_id':fit_id,'trial_id':tid,'y_true':int(yq[i]),'y_pred':int(pred[i]),'p_right':float(probs[i,0]),'p_left':float(probs[i,1])} for i,tid in enumerate(eval_ids))
                fit_membership.extend({'fit_id':fit_id,'trial_id':tid,'membership':'training'} for tid in train_ids)
                fit_membership.extend({'fit_id':fit_id,'trial_id':tid,'membership':'evaluation'} for tid in eval_ids)
                csv_write(out/'timing.csv',timings);csv_write(out/'predictions.csv',predictions);csv_write(out/'fit_membership.csv',fit_membership)
                write_json(out/'receipt.json',receipt)
                print(f'{fit_id}: fit CPU {fitcpu:.4f}s, predict CPU {predcpu:.4f}s, smoke BA {ba:.3f}',flush=True)
    if len(receipt['fits'])!=cfg['expected_fits']:raise ValueError('required fit missing')
    if any(sha(ROOT/k)!=v for k,v in manifest.items()):raise ValueError('input mutation during run')
    csv_write(out/'timing.csv',timings);csv_write(out/'predictions.csv',predictions);csv_write(out/'fit_membership.csv',fit_membership)
    receipt['checks']={'required_fits':4,'unique_schema_trials':len(seen),'prediction_rows':len(predictions),'train_evaluation_ids_disjoint':True,'fitted_transforms_unchanged_by_evaluation':True,'manual_BA_matches_library':True,'input_hashes_unchanged':True,'confirmation_accessed':False}
    receipt['status']='completed';receipt['exit_code']=0
except Exception as e:
    receipt['status']='failed';receipt['exit_code']=1;receipt['error']=repr(e);traceback.print_exc()
finally:
    receipt['elapsed_seconds']=time.perf_counter()-start;receipt['cpu_seconds']=time.process_time()-cpu_start
    receipt['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    receipt['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(out/'receipt.json',receipt)
if receipt['status']=='completed':
    write_json(out/'COMPLETED.json',{'run_id':receipt['run_id'],'receipt_sha256':sha(out/'receipt.json'),'outputs':{f.name:sha(f) for f in sorted(out.iterdir()) if f.is_file() and f.suffix in ['.csv','.json']}})
signal.alarm(0)
raise SystemExit(receipt['exit_code'])
