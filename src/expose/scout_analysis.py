"""Descriptive paired-subject analysis; never a confirmatory significance test."""
import numpy as np
from .scout import balanced_accuracy


def summarize(grouped, cfg):
    subjects=cfg['development_subjects']
    cells=[(m,n) for m in cfg['model_order'] for n in cfg['history_doses_per_class']]
    scores=np.empty((len(subjects),len(cells))); subject_rows=[]; predictions=[]; truths=[]
    for i,s in enumerate(subjects):
        current=[]; truth=None
        for j,(m,n) in enumerate(cells):
            rows=sorted(grouped[m,n,s],key=lambda r:r['trial_id'])
            y=np.array([int(r['y_true']) for r in rows]); pred=np.array([int(r['y_pred']) for r in rows])
            if truth is not None and not np.array_equal(truth,y):raise ValueError('unpaired evaluation labels')
            truth=y;current.append(pred);scores[i,j]=balanced_accuracy(y,pred)
            subject_rows.append({'subject_id':s,'model':m,'dose_per_class':n,'balanced_accuracy':scores[i,j],
                                 'correct_class0':int(np.sum(pred[y==0]==0)),'total_class0':int(np.sum(y==0)),
                                 'correct_class1':int(np.sum(pred[y==1]==1)),'total_class1':int(np.sum(y==1))})
        truths.append(truth);predictions.append(np.stack(current,axis=1))
    # Column order CSP0,CSP15,TS0,TS15 is frozen by config validation.
    rng=np.random.default_rng(cfg['bootstrap']['seed'])
    ix=rng.integers(0,len(subjects),size=(cfg['bootstrap']['draws'],len(subjects)))
    means=scores[ix].mean(axis=1)
    measures={f'{m}_n{n}':(scores[:,j],means[:,j]) for j,(m,n) in enumerate(cells)}
    measures.update(csp_history_gain=(scores[:,1]-scores[:,0],means[:,1]-means[:,0]),
                    ts_history_gain=(scores[:,3]-scores[:,2],means[:,3]-means[:,2]),
                    relative_at_n0=(scores[:,0]-scores[:,2],means[:,0]-means[:,2]),
                    relative_at_n15=(scores[:,1]-scores[:,3],means[:,1]-means[:,3]),
                    D=(scores[:,1]-scores[:,3]-scores[:,0]+scores[:,2],means[:,1]-means[:,3]-means[:,0]+means[:,2]))
    summary={name:{'mean':float(values.mean()),'descriptive_ci95':np.quantile(boots,[.025,.975]).tolist(),
                   'subject_values':values.tolist()} for name,(values,boots) in measures.items()}
    a,b=summary['relative_at_n0']['mean'],summary['relative_at_n15']['mean']
    summary['winner_change']=bool(abs(a)>1e-12 and abs(b)>1e-12 and np.sign(a)!=np.sign(b))
    summary['scope']='8 development people, fixed source, one history draw; paired percentile CIs are descriptive'
    # A within-person label shuffle is shared across all four paired predictions.
    rng=np.random.default_rng(cfg['null_control']['seed'])
    count=cfg['null_control']['label_permutations'];null=np.zeros((count,len(cells)))
    majority=[]
    for y,pred in zip(truths,predictions):
        shuffled=np.stack([rng.permutation(y) for _ in range(count)])
        recalls=[]
        for label in (0,1):
            mask=shuffled==label
            recalls.append(((pred[None,:,:]==label)&mask[:,:,None]).sum(axis=1)/mask.sum(axis=1)[:,None])
        null+=(recalls[0]+recalls[1])/2/len(subjects)
        majority.append(balanced_accuracy(y,np.zeros_like(y)))
    control={'seed':cfg['null_control']['seed'],'permutations':count,'refits':0,
             'mean_BA':{f'{m}_n{n}':float(null[:,j].mean()) for j,(m,n) in enumerate(cells)},
             'permutation_percentile95':{f'{m}_n{n}':np.quantile(null[:,j],[.025,.975]).tolist() for j,(m,n) in enumerate(cells)},
             'majority_BA_by_subject':majority,
             'interpretation':'scorer/label sanity check; no p-values, not proof against all leakage'}
    control['passed']=bool(np.all(np.abs(null.mean(axis=0)-.5)<=cfg['null_control']['mean_BA_tolerance']) and all(v==.5 for v in majority))
    return subject_rows,summary,control
