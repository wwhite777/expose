import copy
import math
import numpy as np
import pytest
from expose.baselines import nested_history_indices
from expose.scout import balanced_accuracy,decision,expected_fit_id,trial_id,validate_predictions


def fixture_tables():
    cfg={'model_order':['csp_lda','ts_lr'],'history_doses_per_class':[0,15],
         'source_subjects':list(range(10,28)),'development_subjects':list(range(2,10)),
         'history_draw_seed':20260915,'expected_prediction_rows':3200,'expected_preparation_trials':3400,
         'scout_decision':{'meaningful_change_abs_D':.02}}
    split=[]
    for s in cfg['source_subjects']+cfg['development_subjects']:
        for session in ([1] if s in cfg['source_subjects'] else [1,2]):
            split.extend({'trial_id':trial_id(s,session,i),'subject_id':s,'session':session,
                          'role':'source' if s>=10 else 'development','label':i%2} for i in range(100))
    src=[trial_id(s,1,i) for s in cfg['source_subjects'] for i in range(100)]
    rows=[];members=[]
    for m in cfg['model_order']:
        for n in [0,15]:
            for target in ([None] if n==0 else cfg['development_subjects']):
                f=expected_fit_id(m,n,target)
                train=src+([] if target is None else [trial_id(target,1,int(i)) for i in nested_history_indices(np.arange(100)%2,15,cfg['history_draw_seed'])])
                members.extend({'fit_id':f,'membership':'training','trial_id':tid} for tid in train)
                for s in (cfg['development_subjects'] if target is None else [target]):
                    for i in range(100):
                        tid=trial_id(s,2,i);y=i%2
                        rows.append({'fit_id':f,'model':m,'dose_per_class':n,'subject_id':s,'trial_id':tid,
                                     'y_true':y,'y_pred':y,'p_class0_right':.9 if y==0 else .1,
                                     'p_class1_left':.1 if y==0 else .9})
                        members.append({'fit_id':f,'membership':'evaluation','trial_id':tid})
    return cfg,rows,split,members


def test_known_full_contract_and_balanced_accuracy():
    cfg,rows,split,members=fixture_tables()
    assert len(validate_predictions(rows,cfg,split,members))==32
    assert balanced_accuracy([0,0,1,1],[0,0,1,1])==1
    assert balanced_accuracy([0,0,1,1],[0,0,0,0])==.5
    assert balanced_accuracy([0,0,1,1],[1,1,0,0])==0


@pytest.mark.parametrize('defect',['empty','missing','duplicate','nan','inverted','truth','future_training','wrong_target','missing_fit'])
def test_actual_contract_rejects_separated_defect_fixtures(defect):
    cfg,rows,split,members=fixture_tables()
    if defect=='empty':rows=[]
    elif defect=='missing':rows.pop()
    elif defect=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif defect=='nan':rows[0]['p_class0_right']=math.nan
    elif defect=='inverted':rows[0]['p_class0_right'],rows[0]['p_class1_left']=.1,.9
    elif defect=='truth':rows[0]['y_true']=1
    elif defect=='future_training':members[0]['trial_id']=trial_id(2,2,0)
    elif defect=='wrong_target':
        row=next(r for r in members if r['membership']=='training' and '_s002_n15' in r['fit_id'] and ':s002:' in r['trial_id'])
        row['trial_id']=trial_id(3,1,0)
    elif defect=='missing_fit':members=[r for r in members if r['fit_id']!='ts_lr__source_n0']
    with pytest.raises(ValueError):validate_predictions(rows,cfg,split,members)


@pytest.mark.parametrize('delta,cross,eng,res,review,overlap,expected',[
    (.02,False,True,True,True,False,'GO_TO_DEVELOP'),
    (-.02,False,True,True,True,False,'GO_TO_DEVELOP'),
    (0,True,True,True,True,False,'GO_TO_DEVELOP'),
    (.019,False,True,True,True,False,'INCONCLUSIVE'),
    (.03,False,False,True,True,False,'ERROR'),
    (math.nan,False,True,True,True,False,'ERROR'),
    (.03,False,True,False,True,False,'KILL'),
    (.03,False,True,True,True,True,'KILL'),
    (.03,False,True,None,True,False,'INCONCLUSIVE'),
    (.03,False,True,True,False,False,'INCONCLUSIVE')])
def test_frozen_scout_judge_branches(delta,cross,eng,res,review,overlap,expected):
    cfg={'scout_decision':{'meaningful_change_abs_D':.02}}
    assert decision(cfg,delta,cross,engineering_ok=eng,resource_ok=res,literature_reviewed=review,direct_overlap=overlap)==expected


def test_balanced_but_wrong_frozen_history_rejected():
    cfg,rows,split,members=fixture_tables()
    chosen=[r for r in members if r['fit_id']=='ts_lr__s002_n15' and r['membership']=='training' and ':s002:' in r['trial_id']]
    original=chosen[0]['trial_id']; original_label=int(original[-3:])%2
    used={r['trial_id'] for r in chosen}
    chosen[0]['trial_id']=next(trial_id(2,1,i) for i in range(100) if i%2==original_label and trial_id(2,1,i) not in used)
    with pytest.raises(ValueError,match='frozen seed'):validate_predictions(rows,cfg,split,members)
