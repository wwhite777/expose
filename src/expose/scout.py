"""Frozen Day2 definitions and checks; exploration on development people only."""
import hashlib
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .baselines import assert_disjoint_trial_ids, validate_epochs, nested_history_indices
from .provenance import file_sha256


def load_config(root):
    root=Path(root);path=root/'research/protocols/DAY2_CONFIG_v1.json'
    if file_sha256(path)!=path.with_suffix('.json.sha256').read_text().split()[0]:
        raise ValueError('frozen Day2 config changed')
    cfg=json.loads(path.read_text())
    for p,k in [(cfg['role_manifest'],'role_manifest_sha256'),(cfg['loader_file'],'loader_sha256'),(cfg['model_code'],'model_code_sha256')]:
        if file_sha256(root/p)!=cfg[k]:raise ValueError('frozen dependency changed: '+p)
    required={'draws':1,'expected_fits':18,'expected_prediction_rows':3200,
              'expected_evaluation_trials_per_subject':100,'source_trials':1800,
              'positive_dose_training_trials':1830,'history_trials_at_positive_dose':30,
              'history_session':1,'evaluation_session':2}
    if any(cfg.get(k)!=v for k,v in required.items()):raise ValueError('unsupported scout count/session configuration')
    if cfg['history_doses_per_class']!=[0,15] or cfg['model_order']!=['csp_lda','ts_lr']:
        raise ValueError('unsupported model/dose configuration')
    if len(cfg['source_subjects'])!=18 or len(cfg['development_subjects'])!=8:
        raise ValueError('unexpected participant count')
    if len(set(cfg['source_subjects']+cfg['development_subjects']))!=26:
        raise ValueError('role IDs overlap or repeat')
    roles=list(csv.DictReader((root/cfg['role_manifest']).open()))
    if cfg['source_subjects']!=sorted(int(r['subject_id']) for r in roles if r['role']=='source') or cfg['development_subjects']!=sorted(int(r['subject_id']) for r in roles if r['role']=='development')[:8]:
        raise ValueError('selected IDs differ from frozen role rule')
    # The library implementation was frozen on Day1; forbid descriptive settings drifting.
    old=json.loads((root/'research/protocols/DAY1_CONFIG_v1.json').read_text())
    if cfg['models']!=old['models'] or cfg['preprocessing']!=old['preprocessing']:
        raise ValueError('frozen preprocessing/model implementation differs')
    return cfg


def expected_fit_id(model,dose,subject=None):
    if dose==0:return f'{model}__source_n0'
    return f'{model}__s{subject:03d}_n{dose}'


def trial_id(subject,session,index):
    return f'lee2019:s{subject:03d}:session{session}:offline:t{index:03d}'


def validate_prepared_array(root,record,X,y,cfg):
    subject,session=record['subject'],record['session'];meta=record['metadata']
    expected_role='source' if subject in cfg['source_subjects'] else 'development' if subject in cfg['development_subjects'] else None
    if expected_role is None or record['role']!=expected_role or (expected_role=='source' and session!=1) or session not in [1,2]:
        raise ValueError('unallowed participant/session in prepared data')
    validate_epochs(X,y)
    if list(X.shape)!=cfg['preprocessing']['expected_shape'] or np.bincount(y).tolist()!=[50,50]:
        raise ValueError('unbalanced or incorrect epoch dimensions')
    for kind in ['array','metadata']:
        path=Path(root)/record['cache'][kind+'_path']
        if file_sha256(path)!=record['cache'][kind+'_sha256']:raise ValueError('prepared cache hash differs')
    stored=json.loads((Path(root)/record['cache']['metadata_path']).read_text())
    if stored!=meta:raise ValueError('metadata receipt mismatch')
    expected_raw=Path(root)/f'data/raw/lee2019/session{session}/s{subject}/sess{session:02d}_subj{subject:02d}_EEG_MI.mat'
    if Path(meta['path']).resolve()!=expected_raw.resolve():raise ValueError('raw subject/session lineage differs')
    if meta['mat_variable']!='EEG_MI_train' or meta['loaded_variables']!=['EEG_MI_train']:
        raise ValueError('unexpected loaded variable')
    if meta['label_mapping']!={'1':0,'2':1} or meta['class_names']!=['right_hand','left_hand'] or not np.array_equal(y,np.asarray(meta['raw_labels'])-1):
        raise ValueError('raw label contract differs')
    if meta['units']!='V' or meta['raw_fs']!=1000 or meta['fs']!=250 or meta['channel_names']!=cfg['preprocessing']['channels']:
        raise ValueError('units/rate/channel contract differs')
    if any(meta['config'][k]!=cfg['preprocessing'][k] for k in ['l_freq','h_freq','tmin','tmax','target_fs','filter_order']):
        raise ValueError('epoch settings differ')
    hashes=[hashlib.sha256(epoch.tobytes(order='C')).hexdigest() for epoch in X]
    if hashes!=meta['trial_epoch_sha256']:raise ValueError('epoch content differs')


def validate_predictions(rows,cfg,split_rows,membership_rows):
    if len(rows)!=cfg['expected_prediction_rows']:raise ValueError('missing/extra prediction rows')
    split={r['trial_id']:r for r in split_rows}
    if len(split)!=len(split_rows) or len(split)!=cfg['expected_preparation_trials']:
        raise ValueError('duplicate or missing split rows')
    expected_split={trial_id(s,d,i) for s in cfg['source_subjects']+cfg['development_subjects'] for d in ([1] if s in cfg['source_subjects'] else [1,2]) for i in range(100)}
    if set(split)!=expected_split:raise ValueError('split trial identity set differs')
    expected={(m,n,s) for m in cfg['model_order'] for n in cfg['history_doses_per_class'] for s in cfg['development_subjects']}
    grouped=defaultdict(list);members=defaultdict(lambda:defaultdict(list))
    for row in membership_rows:members[row['fit_id']][row['membership']].append(row['trial_id'])
    expected_fits={expected_fit_id(m,n,s) for m,n,s in expected}
    if set(members)!=expected_fits:raise ValueError('missing/extra fit membership')
    for row in rows:
        key=(row['model'],int(row['dose_per_class']),int(row['subject_id']))
        if key not in expected or row['fit_id']!=expected_fit_id(*key[:2],key[2]):raise ValueError('unexpected prediction identity')
        tid=row['trial_id'];s=split.get(tid)
        if not s or int(s['subject_id'])!=key[2] or int(s['session'])!=2 or s['role']!='development':
            raise ValueError('evaluation subject/session/role violation')
        truth,pred=int(row['y_true']),int(row['y_pred'])
        if truth not in [0,1] or pred not in [0,1] or truth!=int(s['label']):raise ValueError('label contract differs')
        probs=[float(row['p_class0_right']),float(row['p_class1_left'])]
        if any(not math.isfinite(p) or p<0 or p>1 for p in probs) or abs(sum(probs)-1)>1e-10:
            raise ValueError('invalid probability simplex')
        if pred!=(0 if probs[0]>=probs[1] else 1):raise ValueError('class mapping/argmax differs')
        grouped[key].append(row)
    if set(grouped)!=expected:raise ValueError('missing required subject/model/dose evaluation')
    for (m,n,s),group in grouped.items():
        ids=[r['trial_id'] for r in group]
        if len(ids)!=100 or len(set(ids))!=100:raise ValueError('duplicate/missing eval trials')
        if [sum(int(r['y_true'])==c for r in group) for c in [0,1]]!=[50,50]:raise ValueError('evaluation class count differs')
    for fit_id,parts in members.items():
        if set(parts)!={'training','evaluation'}:raise ValueError('missing/unknown membership type')
        train,evaluation=parts['training'],parts['evaluation']
        assert_disjoint_trial_ids(train,evaluation)
        related=[r for r in rows if r['fit_id']==fit_id]
        if set(evaluation)!={r['trial_id'] for r in related}:raise ValueError('prediction/membership evaluation mismatch')
        dose=int(related[0]['dose_per_class']);subject=int(related[0]['subject_id'])
        if len(train)!=(1800 if dose==0 else 1830):raise ValueError('training count differs')
        source_count=0;history_counts=[0,0]
        for tid in train:
            r=split.get(tid)
            if r is None or int(r['session'])!=1:raise ValueError('future/unknown training trial')
            if int(r['subject_id']) in cfg['source_subjects'] and r['role']=='source':source_count+=1
            elif dose==15 and int(r['subject_id'])==subject and r['role']=='development':history_counts[int(r['label'])]+=1
            else:raise ValueError('wrong target/confirmation in training')
        if source_count!=1800 or history_counts!=([0,0] if dose==0 else [15,15]):raise ValueError('source/history allocation differs')
        if dose:
            history_ids=[trial_id(subject,1,i) for i in range(100)]
            labels=np.array([int(split[tid]['label']) for tid in history_ids])
            selected=nested_history_indices(labels,dose,cfg['history_draw_seed'])
            actual={tid for tid in train if int(split[tid]['subject_id'])==subject}
            if actual!={history_ids[i] for i in selected}:raise ValueError('history selection differs from frozen seed')
    return grouped


def balanced_accuracy(truth,pred):
    truth=np.asarray(truth);pred=np.asarray(pred)
    if truth.shape!=pred.shape or truth.ndim!=1 or not len(truth) or set(truth.tolist())!={0,1} or not np.all(np.isin(pred,[0,1])):
        raise ValueError('invalid balanced-accuracy input')
    return float(np.mean([np.mean(pred[truth==c]==c) for c in [0,1]]))


def decision(cfg,delta,winner_change,*,engineering_ok,resource_ok,literature_reviewed,direct_overlap):
    if not engineering_ok or not math.isfinite(delta):return 'ERROR'
    if direct_overlap or resource_ok is False:return 'KILL'
    if resource_ok is None or not literature_reviewed:return 'INCONCLUSIVE'
    return 'GO_TO_DEVELOP' if abs(delta)>=cfg['scout_decision']['meaningful_change_abs_D'] or winner_change else 'INCONCLUSIVE'
