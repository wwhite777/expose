from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pytest

from expose.development import (SourceCentroid, load_config, logical_cells,
                                make_model, membership, operations)
from expose.scout import trial_id

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(ROOT)


@pytest.fixture
def split(cfg):
    rows = {}
    for role, subjects in [('source', cfg['source_subjects']), ('development', cfg['development_subjects'])]:
        for subject in subjects:
            for session in ([1] if role == 'source' else [1, 2]):
                for i in range(100):
                    tid = trial_id(subject, session, i)
                    rows[tid] = dict(trial_id=tid, role=role, subject_id=subject, session=session, label=i % 2)
    return rows


def test_full_operation_contract(cfg):
    ops = operations(cfg)
    assert len(ops) == len({o['operation_id'] for o in ops}) == 808
    assert Counter(o['arm'] for o in ops) == dict(practical=219, fixed_total=222, removal=6,
                                                target_only=72, calibration=289)
    assert sum((100 if o['target'] else 1200) * len(logical_cells(o, cfg)) for o in ops) == 110400


def test_memberships_nested_balanced_and_matched(cfg, split):
    ops = operations(cfg)
    seen_history = {}
    for op in ops:
        source, history, query = membership(op, split, cfg)
        assert len(source + history + query) == len(set(source + history + query))
        assert len(query) == (100 if op['target'] else 1200)
        assert all(split[t]['session'] == 2 and split[t]['role'] == 'development' for t in query)
        assert all(split[t]['session'] == 1 and split[t]['role'] == 'source' for t in source)
        if op['target']:
            assert Counter(split[t]['label'] for t in history) == {0: op['dose'], 1: op['dose']}
            key = (op['draw'], op['dose'], op['target'])
            assert seen_history.setdefault(key, history) == history
        if op['arm'] == 'fixed_total':
            assert len(source + history) == 100
        elif op['arm'] == 'removal':
            assert len(source) == 40 and not history
        elif op['arm'] == 'target_only':
            assert not source
        else:
            assert len(source) == 1800
    for draw in range(2):
        for subject in cfg['development_subjects']:
            assert set(seen_history[draw, 5, subject]) < set(seen_history[draw, 15, subject]) < set(seen_history[draw, 30, subject])
    a = [t.split(':')[-1] for t in seen_history[0, 5, 2]]
    b = [t.split(':')[-1] for t in seen_history[0, 5, 3]]
    assert a != b


def test_centroid_update_formula_and_prediction_does_not_fit():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 3, 120))
    y = np.arange(24) % 2
    base = SourceCentroid().fit(X, y)
    fingerprint = joblib.hash(base)
    z = base.representation_.transform(X[:8])
    for w in [0, 5, 20, 100]:
        adapted = base.adapted(X[:8], y[:8], w)
        expected = np.stack([(w * base.source_centroids_[c] + z[y[:8] == c].sum(0)) / (w+4) for c in (0, 1)])
        np.testing.assert_allclose(adapted.centroids_, expected)
        assert joblib.hash(base) == fingerprint
        before = joblib.hash(adapted)
        p = adapted.predict_proba(X[8:])
        np.testing.assert_allclose(p.sum(1), 1)
        assert np.isfinite(p).all() and (p >= 0).all()
        assert joblib.hash(adapted) == before
        np.testing.assert_array_equal(p.argmax(1), ((adapted.representation_.transform(X[8:])[:, None] - expected) ** 2).sum(2).argmin(1))
    with pytest.raises(ValueError):
        base.adapted(X[:1], y[:1], 5)


def test_mdm_two_class_probabilities_are_fixed():
    rng = np.random.default_rng(43)
    X = rng.normal(size=(20, 3, 80))
    model = make_model('mdm').fit(X, np.arange(20) % 2)
    before = joblib.hash(model)
    p = model.predict_proba(X[:4])
    assert model.classes_.tolist() == [0, 1]
    np.testing.assert_allclose(p.sum(1), 1)
    assert np.isfinite(p).all() and joblib.hash(model) == before
