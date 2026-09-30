"""Person-level exploratory selection analysis; never a confirmation judge."""
from collections import defaultdict

import numpy as np


def ba(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (truth.shape != prediction.shape or truth.ndim != 1 or set(truth.tolist()) != {0, 1}
            or not set(prediction.tolist()) <= {0, 1}):
        raise ValueError('invalid balanced-accuracy inputs')
    return float(np.mean([np.mean(prediction[truth == c] == c) for c in (0, 1)]))


def choose(scores, costs=None, order=None):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError('invalid selection scores')
    candidates = np.flatnonzero(scores >= scores.max() - 1e-12).tolist()
    if costs is not None:
        return min(candidates, key=lambda i: (costs[i], i))
    return next(i for i in (order if order is not None else range(len(scores))) if i in candidates)


def policies(classical, calibration, costs, cfg):
    classical, calibration = np.asarray(classical), np.asarray(calibration)
    n = len(cfg['development_subjects'])
    if classical.shape != (n, 3, 4) or calibration.shape != (n, 4, 4):
        raise ValueError('policy tensor dimensions differ')
    if not np.isfinite(classical).all() or not np.isfinite(calibration).all():
        raise ValueError('nonfinite policy scores')
    weights = np.asarray(cfg['mixed_weights'])
    prior_order = [cfg['calibration_prior_grid'].index(w) for w in cfg['calibration_tie_order']]
    selected = {key: np.empty((n, 4)) for key in ('a_pi', 'a0', 'a_uniform', 'simple_calibration', 'dose_specific')}
    choices = []
    for heldout in range(n):
        train = np.arange(n) != heldout
        cm, km = classical[train].mean(0), calibration[train].mean(0)
        indices = dict(a_pi=choose(cm @ weights, costs), a0=choose(cm[:, 0], costs),
                       a_uniform=choose(cm.mean(1), costs),
                       simple_calibration=choose(km @ weights, order=prior_order))
        dose_choice = [choose(cm[:, d], costs) for d in range(4)]
        for key in ('a_pi', 'a0', 'a_uniform'):
            selected[key][heldout] = classical[heldout, indices[key]]
        selected['simple_calibration'][heldout] = calibration[heldout, indices['simple_calibration']]
        selected['dose_specific'][heldout] = classical[heldout, dose_choice, np.arange(4)]
        choices.append(dict(subject_id=cfg['development_subjects'][heldout],
                            training_subjects=[s for i, s in enumerate(cfg['development_subjects']) if i != heldout],
                            **{key: cfg['model_order'][indices[key]] for key in ('a_pi', 'a0', 'a_uniform')},
                            calibration_prior=cfg['calibration_prior_grid'][indices['simple_calibration']],
                            dose_specific=[cfg['model_order'][i] for i in dose_choice]))
    cm, km = classical.mean(0), calibration.mean(0)
    pooled = dict(a_pi=cfg['model_order'][choose(cm @ weights, costs)],
                  a0=cfg['model_order'][choose(cm[:, 0], costs)],
                  a_uniform=cfg['model_order'][choose(cm.mean(1), costs)],
                  calibration_prior=cfg['calibration_prior_grid'][choose(km @ weights, order=prior_order)],
                  scope='all12 resubstitution candidate only; no frozen confirmation selector')
    return selected, choices, pooled


def disposition(primary_gains, calibration_gain, excess_gain, controls_passed=True):
    values = np.asarray([*primary_gains, calibration_gain, excess_gain], dtype=float)
    if len(primary_gains) != 3 or not controls_passed or not np.isfinite(values).all():
        return 'ERROR'
    if min(primary_gains) >= .02:
        return 'SELECTION_SIGNAL'
    if calibration_gain >= .02 and excess_gain >= .02:
        return 'CALIBRATION_ONLY'
    return 'INCONCLUSIVE'


def summarize(predictions, timings, cfg):
    grouped = defaultdict(list)
    for r in predictions:
        key = (r['arm'], r['model'], str(r['prior_weight']), int(r['draw']),
               int(r['dose_per_class']), int(r['subject_id']))
        grouped[key].append(r)
    scores, metrics = {}, []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r['trial_id'])
        score = ba([int(r['y_true']) for r in rows], [int(r['y_pred']) for r in rows])
        scores[key] = score
        metrics.append(dict(arm=key[0], model=key[1], prior_weight=key[2], draw=key[3],
                            dose_per_class=key[4], subject_id=key[5], balanced_accuracy=score))
    subjects, models, doses = cfg['development_subjects'], cfg['model_order'], cfg['history_doses_per_class']
    def curve(arm, model, prior='', ds=None):
        ds = doses if ds is None else ds
        return np.array([[np.mean([scores[arm, model, str(prior), d, dose, s] for d in range(2)])
                          for dose in ds] for s in subjects])
    classical = np.stack([curve('practical', m) for m in models], axis=1)
    calibration = np.stack([curve('calibration', 'centroid', w) for w in cfg['calibration_prior_grid']], axis=1)
    costs = [np.mean([(float(t['fit_cpu_seconds'])+float(t['predict_cpu_seconds']))/(int(t['evaluation_trials'])/100)
                      for t in timings if t['arm']=='practical' and t['model']==m]) for m in models]
    selected, choices, pooled = policies(classical, calibration, costs, cfg)
    indices = np.random.default_rng(cfg['bootstrap']['seed']).integers(0, len(subjects),
                size=(cfg['bootstrap']['draws'], len(subjects)))
    def estimate(values):
        values = np.asarray(values, dtype=float)
        quantiles = np.quantile(values[indices].mean(axis=1), [.025, .975])
        return dict(mean=float(values.mean()), descriptive_ci95=quantiles.tolist(), subject_values=values.tolist())
    primary = {key: estimate((selected['a_pi']-selected[key]) @ cfg['mixed_weights'])
               for key in ('a0', 'a_uniform', 'simple_calibration')}
    gains = selected['simple_calibration'][:, 3] - selected['simple_calibration'][:, 0]
    ts_gains = classical[:, 1, 3] - classical[:, 1, 0]
    contrasts = dict(calibration_history_gain_n30=estimate(gains), pooled_ts_history_gain_n30=estimate(ts_gains),
                     calibration_gain_minus_pooled_ts_gain=estimate(gains-ts_gains),
                     mixed_gain_vs_dose_specific=estimate((selected['a_pi']-selected['dose_specific']) @ cfg['mixed_weights']))
    scenarios = {name: {key: estimate(value @ weights) for key, value in selected.items()}
                 for name, weights in cfg['scenarios'].items()}
    curves, decomposition = {}, {}
    for m in models:
        practical, fixed = curve('practical', m), curve('fixed_total', m)
        removal = curve('removal', m, ds=[30])[:, 0]
        curves['practical/'+m] = [estimate(practical[:, d]) for d in range(4)]
        curves['fixed_total/'+m] = [estimate(fixed[:, d]) for d in range(4)]
        decomposition[m] = dict(source_removal=estimate(removal-fixed[:, 0]),
            adding_history_to_removed_source=estimate(fixed[:, 3]-removal),
            net_fixed_total_change=estimate(fixed[:, 3]-fixed[:, 0]))
    for w in cfg['calibration_prior_grid']:
        curves['calibration/w'+str(w)] = [estimate(curve('calibration', 'centroid', w)[:, d]) for d in range(4)]
    curves['target_only/ts_lr'] = [estimate(curve('target_only', 'ts_lr', ds=doses[1:])[:, d]) for d in range(3)]
    curves['selected_calibration'] = [estimate(selected['simple_calibration'][:, d]) for d in range(4)]
    group_description = {}
    for name, subset in [('original_eight', cfg['prior_exposed_development_subjects']), ('additional_four', cfg['new_development_subjects'])]:
        ii = [subjects.index(s) for s in subset]
        group_description[name] = dict(subjects=subset,
             primary_mixed_gains={k: float(np.array(v['subject_values'])[ii].mean()) for k, v in primary.items()},
             calibration_history_gain_n30=float(gains[ii].mean()),
             scope='descriptive post-scout subgroup; not independent confirmation')
    conditions = sorted({k[:5] for k in grouped})
    rng = np.random.default_rng(cfg['null_control']['seed'])
    aggregate_null = np.zeros((cfg['null_control']['label_permutations'], len(conditions)))
    for s in subjects:
        truth = np.array([int(r['y_true']) for r in grouped[conditions[0]+(s,)]])
        pred = np.array([[int(r['y_pred']) for r in grouped[c+(s,)]] for c in conditions])
        if len(truth)!=100 or np.bincount(truth).tolist()!=[50, 50]:
            raise ValueError('null requires balanced evaluation cells')
        permuted = np.stack([rng.permutation(truth) for _ in range(cfg['null_control']['label_permutations'])])
        # Every shuffle has 50 examples/class, so accuracy equals mean class recall.
        aggregate_null += (permuted[:, None, :] == pred[None, :, :]).mean(axis=2)/len(subjects)
        if ba(truth, np.zeros(100, dtype=int)) != .5 or ba(truth, np.ones(100, dtype=int)) != .5:
            raise ValueError('constant-label scorer control failed')
    null_rows = [dict(arm=c[0], model=c[1], prior_weight=c[2], draw=c[3], dose_per_class=c[4],
                     shuffled_mean_ba=float(aggregate_null[:, i].mean()),
                     permutation_sd=float(aggregate_null[:, i].std())) for i, c in enumerate(conditions)]
    maximum = max(abs(r['shuffled_mean_ba']-.5) for r in null_rows)
    controls = dict(passed=maximum <= cfg['null_control']['tolerance'], conditions=len(conditions),
                    permutations=cfg['null_control']['label_permutations'], max_absolute_deviation_from_half=maximum,
                    constant_class_ba=.5, scope=cfg['null_control']['scope'], rows=null_rows)
    label = disposition([v['mean'] for v in primary.values()], float(gains.mean()), float((gains-ts_gains).mean()), controls['passed'])
    summary = dict(subjects=subjects, draw_aggregation='mean within each person',
                   primary_mixed_gains=primary, diagnostic_contrasts=contrasts, scenarios=scenarios,
                   curves=curves, source_removal_decomposition=decomposition, groups=group_description,
                   lopo_choices=choices, pooled_resubstitution_choices=pooled,
                   practical_mean_operation_cpu_per_evaluated_person=dict(zip(models, map(float, costs))),
                   disposition=label, scientific_gate_passed=False,
                   limitations=['Exploratory post-Day2 development', 'Shared source pool and two correlated draws',
                                'LOPO training sets overlap; fixed-selector paired intervals are descriptive',
                                'Common-pipeline deployment constraint and human prior data exposure unresolved',
                                'No untouched-confirmation, causal dilution mechanism, novelty or publication claim'])
    return metrics, summary, controls
