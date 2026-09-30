import copy
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest

from expose.baselines import make_baseline
from expose.development import load_config, membership as old_membership
from expose.review2_controls import (apply_ea, apply_source_ea,
    classifier_sample_weights, fit_ea_whitener, fit_representation,
    fit_source_ea, membership, oas_covariances, operations, train_classifier)
from expose.scout import trial_id


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def oldcfg():
    return load_config(ROOT)


@pytest.fixture
def cfg():
    return json.loads((ROOT / "research/review2_execution_20260923/CONFIG_v1.json").read_text())


@pytest.fixture
def split(oldcfg):
    rows = {}
    for role, subjects in (("source", oldcfg["source_subjects"]),
                           ("development", oldcfg["development_subjects"])):
        sessions = (1,) if role == "source" else (1, 2)
        for subject in subjects:
            for session in sessions:
                for index in range(100):
                    identity = trial_id(subject, session, index)
                    rows[identity] = {"trial_id": identity, "role": role, "subject_id": subject,
                                      "session": session, "label": index % 2}
    assert len(rows) == 4200
    return rows


def find_op(rows, **conditions):
    matches = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    assert len(matches) == 1
    return matches[0]


def test_operation_grid_exact_fields_counts_and_module_c_label_semantics(cfg):
    rows = operations(cfg)
    assert len(rows) == len({row["operation_id"] for row in rows}) == 1464
    assert Counter(row["module"] for row in rows) == {"A": 24, "B": 1152, "C": 240, "D": 48}
    required = {"operation_id", "module", "source_n", "h_total", "C", "lambda_personal",
                "representation", "draw", "target", "donor_kind"}
    assert all(set(row) == required for row in rows)
    assert all(row["target"] is None and row["h_total"] == 0 for row in rows if row["module"] == "A")
    assert all(row["h_total"] == (60 if row["donor_kind"] == "own" else 0)
               for row in rows if row["module"] == "C")
    assert all(row["representation"] == "source_frozen" for row in rows if row["module"] == "D")


def test_membership_reuses_frozen_source_and_history_prefixes(cfg, oldcfg, split):
    rows = operations(cfg)
    own = find_op(rows, module="B", source_n=100, h_total=60, C=1.0,
                  lambda_personal="pooled", representation="pooled_refit", draw=0,
                  target=2, donor_kind="own")
    source, history, evaluation, unlabeled, donors = membership(own, split, oldcfg, cfg)
    old_source, _, _ = old_membership(
        {"arm": "fixed_total", "draw": 0, "dose": 0, "target": 2}, split, oldcfg)
    _, old_history, old_evaluation = old_membership(
        {"arm": "fixed_total", "draw": 0, "dose": 30, "target": 2}, split, oldcfg)
    assert source == old_source
    assert history == old_history
    assert evaluation == old_evaluation
    assert unlabeled == [] and donors == [2]
    assert Counter(split[trial]["label"] for trial in source) == {0: 50, 1: 50}
    assert Counter(split[trial]["label"] for trial in history) == {0: 30, 1: 30}


def test_pooled_and_single_donor_blocks_are_matched_and_reused(cfg, oldcfg, split):
    rows = operations(cfg)
    pooled40 = find_op(rows, module="C", source_n=40, h_total=0, C=1.0,
                       lambda_personal="pooled", representation="pooled_refit", draw=1,
                       target=3, donor_kind="pooled")
    source40, block40, _, _, pooled_donors = membership(pooled40, split, oldcfg, cfg)
    original100, _, _ = old_membership(
        {"arm": "fixed_total", "draw": 1, "dose": 0, "target": 3}, split, oldcfg)
    assert set(source40 + block40) == set(original100)
    assert set(source40).isdisjoint(block40) and len(block40) == 60
    assert pooled_donors == sorted(set(split[trial]["subject_id"] for trial in block40))

    for donor_kind in ("single_0", "single_1", "single_2"):
        low = find_op(rows, module="C", source_n=40, h_total=0, C=1.0,
                      lambda_personal="pooled", representation="pooled_refit", draw=1,
                      target=3, donor_kind=donor_kind)
        high = find_op(rows, module="C", source_n=100, h_total=0, C=1.0,
                       lambda_personal="pooled", representation="pooled_refit", draw=1,
                       target=3, donor_kind=donor_kind)
        low_source, low_block, _, _, low_donor = membership(low, split, oldcfg, cfg)
        high_source, high_block, _, _, high_donor = membership(high, split, oldcfg, cfg)
        assert low_block == high_block and low_donor == high_donor and len(low_donor) == 1
        assert set(low_block).isdisjoint(high_source)
        assert all(split[trial]["subject_id"] == low_donor[0] for trial in low_block)
        assert Counter(split[trial]["label"] for trial in low_block) == {0: 30, 1: 30}
        # The added trials, rather than the donor identity, are excluded from B=100.
        # Other trials from this donor may remain in the shared base by design.
        same_donor_base = {trial for trial in high_source
                           if split[trial]["subject_id"] == low_donor[0]}
        assert same_donor_base <= set(high_source)
        assert same_donor_base.isdisjoint(low_block)
    chosen = [membership(find_op(rows, module="C", source_n=40, h_total=0, C=1.0,
                                 lambda_personal="pooled", representation="pooled_refit", draw=1,
                                 target=3, donor_kind=f"single_{index}"), split, oldcfg, cfg)[4][0]
              for index in range(3)]
    assert len(set(chosen)) == 3


def test_ea_membership_is_label_blind_and_evaluation_never_learned(cfg, oldcfg, split):
    op = find_op(operations(cfg), module="D", source_n=1800, h_total=0, C=1.0,
                 lambda_personal="pooled", representation="source_frozen", draw=0,
                 target=8, donor_kind=None)
    first = membership(op, split, oldcfg, cfg)
    assert len(first[0]) == 1800 and first[1] == [] and len(first[2]) == 100
    assert len(first[3]) == 100 and first[4] == []
    assert set(first[3]).isdisjoint(first[2])
    changed = copy.deepcopy(split)
    evaluation = first[2]
    zero = next(trial for trial in evaluation if changed[trial]["label"] == 0)
    one = next(trial for trial in evaluation if changed[trial]["label"] == 1)
    changed[zero]["label"], changed[one]["label"] = 1, 0
    assert membership(op, changed, oldcfg, cfg) == first


@pytest.mark.parametrize("defect", ["role", "extra", "missing", "unbalanced"])
def test_membership_rejects_any_old_split_drift(cfg, oldcfg, split, defect):
    changed = copy.deepcopy(split)
    if defect == "role":
        changed[next(iter(changed))]["role"] = "confirmation"
    elif defect == "extra":
        changed["extra"] = {"trial_id": "extra", "role": "source", "subject_id": 999,
                            "session": 1, "label": 0}
    elif defect == "missing":
        changed.pop(next(iter(changed)))
    else:
        rows = list(changed.values())
        zero = next(row for row in rows if row["role"] == "source" and row["label"] == 0)
        zero["label"] = 1
    op = operations(cfg)[0]
    with pytest.raises(ValueError):
        membership(op, changed, oldcfg, cfg)


def test_normalized_classifier_weights_and_pooled_unit_weights():
    np.testing.assert_array_equal(classifier_sample_weights(160, 100, "pooled"), np.ones(160))
    weights = classifier_sample_weights(160, 100, .25)
    np.testing.assert_allclose(weights[:100], 1.2)
    np.testing.assert_allclose(weights[100:], 2 / 3)
    assert weights.sum() == pytest.approx(160, abs=1e-12)
    with pytest.raises(ValueError):
        classifier_sample_weights(100, 100, .5)


def test_separated_pooled_helpers_match_original_ts_lr_pipeline():
    rng = np.random.default_rng(20260923)
    epochs = rng.normal(size=(48, 4, 80))
    labels = np.tile([0, 1], 24)
    epochs[labels == 1, 0] *= 1.4
    test_epochs = rng.normal(size=(8, 4, 80))
    original = make_baseline("ts_lr").fit(epochs, labels)
    covariances = oas_covariances(epochs)
    representation = fit_representation(covariances, len(epochs), "pooled_refit")
    classifier = train_classifier(representation.transform(covariances), labels,
                                  len(epochs), 1.0, "pooled")
    expected = original.predict_proba(test_epochs)
    observed = classifier.predict_proba(representation.transform(oas_covariances(test_epochs)))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-10)


def test_source_frozen_representation_is_invariant_to_personal_change():
    rng = np.random.default_rng(7)
    source_epochs = rng.normal(size=(20, 3, 60))
    personal_a = rng.normal(size=(8, 3, 60))
    personal_b = rng.normal(size=(8, 3, 60)) * 3
    source_cov = oas_covariances(source_epochs)
    first = fit_representation(np.concatenate([source_cov, oas_covariances(personal_a)]), 20,
                               "source_frozen")
    second = fit_representation(np.concatenate([source_cov, oas_covariances(personal_b)]), 20,
                                "source_frozen")
    np.testing.assert_allclose(first.transform(source_cov), second.transform(source_cov),
                               rtol=0, atol=1e-12)


def test_ea_is_label_blind_frozen_and_rejects_singular_covariance():
    rng = np.random.default_rng(8)
    historical = rng.normal(size=(100, 3, 80))
    evaluation = rng.normal(size=(6, 3, 80))
    labels = np.tile([0, 1], 50)
    whitener = fit_ea_whitener(historical)
    before = apply_ea(evaluation, whitener)
    rng.shuffle(labels)
    np.testing.assert_allclose(fit_ea_whitener(historical), whitener, rtol=0, atol=0)
    changed_evaluation = evaluation.copy()
    changed_evaluation[0] += 100
    after = apply_ea(changed_evaluation, whitener)
    np.testing.assert_allclose(after[1:], before[1:])
    singular = historical.copy()
    singular[:, 1] = singular[:, 0]
    with pytest.raises(ValueError, match="no jitter"):
        fit_ea_whitener(singular)


def test_source_ea_fits_and_applies_separate_person_transforms():
    rng = np.random.default_rng(9)
    source_ids = [5, 6, 10, 12, 14, 18, 20, 21, 25, 26, 29, 31, 32, 36, 39, 42, 47, 52]
    epochs = rng.normal(size=(1800, 3, 40))
    for index in range(18):
        epochs[index * 100:(index + 1) * 100] *= 1 + index / 20
    subjects = np.repeat(source_ids, 100)
    whiteners = fit_source_ea(epochs, subjects)
    assert set(whiteners) == set(source_ids)
    assert not np.allclose(whiteners[5], whiteners[6])
    transformed = apply_source_ea(epochs, subjects, whiteners)
    np.testing.assert_allclose(transformed[:100], apply_ea(epochs[:100], whiteners[5]))
    np.testing.assert_allclose(transformed[100:200], apply_ea(epochs[100:200], whiteners[6]))
