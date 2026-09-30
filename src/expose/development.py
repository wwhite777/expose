"""Day3 development definitions; historical Day1/2 modules remain unchanged."""
import copy
import csv
import json
from pathlib import Path

import numpy as np
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import make_baseline, nested_history_indices
from .provenance import file_sha256


def load_config(root):
    root = Path(root)
    path = root / 'research/protocols/DAY3_CONFIG_v1.json'
    if file_sha256(path) != path.with_suffix('.json.sha256').read_text().split()[0]:
        raise ValueError('Day3 config checksum differs')
    cfg = json.loads(path.read_text())
    for name, key in [('role_manifest', 'role_manifest_sha256'), ('loader_file', 'loader_sha256'),
                      ('model_code', 'model_code_sha256')]:
        if file_sha256(root / cfg[name]) != cfg[key]:
            raise ValueError('frozen dependency differs: ' + name)
    with (root / cfg['role_manifest']).open() as f:
        roles = list(csv.DictReader(f))
    for role, field in [('source', 'source_subjects'), ('development', 'development_subjects')]:
        if cfg[field] != sorted(int(r['subject_id']) for r in roles if r['role'] == role):
            raise ValueError('participant role drift')
    if len(cfg['source_subjects']) != 18 or len(cfg['development_subjects']) != 12:
        raise ValueError('unexpected cohort size')
    if cfg['history_doses_per_class'] != [0, 5, 15, 30] or cfg['model_order'] != ['csp_lda', 'ts_lr', 'mdm']:
        raise ValueError('unsupported model/dose grid')
    old = json.loads((root / 'research/protocols/DAY2_CONFIG_v1.json').read_text())
    if cfg['preprocessing'] != old['preprocessing'] or cfg['models'] != old['models']:
        raise ValueError('original preprocessing or baselines changed')
    return cfg


def operations(cfg):
    result = []

    def add(arm, model, draw, dose, target=None, prior=None):
        name = f'{arm}__{model}__r{draw if draw is not None else "shared"}_n{dose}'
        if target is not None:
            name += f'_s{target:03d}'
        if prior is not None:
            name += f'_w{prior}'
        result.append(dict(operation_id=name, arm=arm, model=model, draw=draw,
                           dose=dose, target=target, prior=prior))

    for arm in cfg['arms']:
        for model in cfg['model_order']:
            for draw in ([None] if arm == 'practical' else range(cfg['draws'])):
                add(arm, model, draw, 0)
            for draw in range(cfg['draws']):
                for dose in cfg['history_doses_per_class'][1:]:
                    for subject in cfg['development_subjects']:
                        add(arm, model, draw, dose, subject)
    for model in cfg['model_order']:
        for draw in range(cfg['draws']):
            add('removal', model, draw, 30)
    for draw in range(cfg['draws']):
        for dose in cfg['history_doses_per_class'][1:]:
            for subject in cfg['development_subjects']:
                add('target_only', cfg['target_only_model'], draw, dose, subject)
    add('calibration', 'centroid', None, 0)
    for draw in range(cfg['draws']):
        for dose in cfg['history_doses_per_class'][1:]:
            for subject in cfg['development_subjects']:
                for prior in cfg['calibration_prior_grid']:
                    add('calibration', 'centroid', draw, dose, subject, prior)
    return result


def membership(op, split, cfg):
    source = [r['trial_id'] for r in split.values() if r['role'] == 'source' and int(r['session']) == 1]
    source.sort()
    if op['arm'] in ('fixed_total', 'removal'):
        rng = np.random.default_rng(cfg['source_draw_seeds'][op['draw']])
        source = [tid for c in (0, 1)
                  for tid in rng.permutation([t for t in source if int(split[t]['label']) == c])[:50-op['dose']]]
    elif op['arm'] == 'target_only':
        source = []
    history = []
    if op['target'] is not None:
        pool = sorted([r['trial_id'] for r in split.values()
                       if int(r['subject_id']) == op['target'] and int(r['session']) == 1])
        y = np.array([int(split[t]['label']) for t in pool])
        seed = np.random.SeedSequence([cfg['history_draw_seeds'][op['draw']], op['target']])
        history = [pool[i] for i in nested_history_indices(y, op['dose'], seed)]
    subjects = [op['target']] if op['target'] is not None else cfg['development_subjects']
    evaluation = sorted([r['trial_id'] for r in split.values()
                         if int(r['subject_id']) in subjects and int(r['session']) == 2])
    if not evaluation or set(source + history) & set(evaluation):
        raise ValueError('empty/overlapping evaluation')
    if any(split[t]['role'] not in ('source', 'development') for t in source + history + evaluation):
        raise ValueError('confirmation role in operation')
    return source, history, evaluation


def logical_cells(op, cfg):
    draws = range(cfg['draws']) if op['draw'] is None else [op['draw']]
    priors = cfg['calibration_prior_grid'] if op['arm'] == 'calibration' and op['dose'] == 0 else [op['prior']]
    return [(d, w) for d in draws for w in priors]


def make_model(name):
    if name == 'mdm':
        return make_pipeline(Covariances(estimator='oas'), MDM(metric='riemann', n_jobs=1))
    return make_baseline(name)


class SourceCentroid:
    def fit(self, X, y):
        self.representation_ = make_pipeline(Covariances(estimator='oas'),
                                           TangentSpace(metric='riemann', tsupdate=False), StandardScaler())
        z = self.representation_.fit_transform(X)
        self.classes_ = np.array([0, 1])
        self.source_centroids_ = np.stack([z[y == c].mean(axis=0) for c in self.classes_])
        self.centroids_ = self.source_centroids_.copy()
        return self

    def adapted(self, X, y, prior):
        if prior < 0 or len(y) == 0 or set(y.tolist()) != {0, 1}:
            raise ValueError('invalid calibration history/prior')
        result = copy.deepcopy(self)
        z = self.representation_.transform(X)
        result.centroids_ = np.stack([(prior * self.source_centroids_[c] + z[y == c].sum(axis=0)) /
                                     (prior + np.count_nonzero(y == c)) for c in self.classes_])
        return result

    def predict_proba(self, X):
        z = self.representation_.transform(X)
        score = -np.sum((z[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        score -= score.max(axis=1, keepdims=True)
        probability = np.exp(score)
        return probability / probability.sum(axis=1, keepdims=True)
