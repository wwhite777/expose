import copy
from test_scout import fixture_tables
from expose.scout import validate_predictions
from expose.scout_analysis import summarize


def test_perfect_paired_predictions_have_zero_history_contrast_and_valid_null():
    cfg,rows,split,members=fixture_tables()
    cfg.update(bootstrap={'seed':20260916,'draws':10000},null_control={'seed':20260917,'label_permutations':1000,'mean_BA_tolerance':.015})
    _,summary,null=summarize(validate_predictions(rows,cfg,split,members),cfg)
    assert summary['csp_lda_n0']['descriptive_ci95']==[1.,1.]
    assert summary['D']['mean']==0 and summary['D']['descriptive_ci95']==[0.,0.]
    assert not summary['winner_change'] and null['passed']


def test_known_opposite_history_effect_keeps_pairing_and_contrast_sign():
    cfg,rows,split,members=fixture_tables()
    cfg.update(bootstrap={'seed':20260916,'draws':10000},null_control={'seed':20260917,'label_permutations':1000,'mean_BA_tolerance':.015})
    for row in rows:
        if (row['model']=='csp_lda' and row['dose_per_class']==0) or (row['model']=='ts_lr' and row['dose_per_class']==15):
            row['y_pred']=1-row['y_true'];row['p_class0_right'],row['p_class1_left']=row['p_class1_left'],row['p_class0_right']
    _,summary,null=summarize(validate_predictions(rows,cfg,split,members),cfg)
    assert summary['D']['mean']==2 and summary['D']['descriptive_ci95']==[2.,2.]
    assert summary['winner_change'] and null['passed']
