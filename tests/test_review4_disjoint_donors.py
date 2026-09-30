"""Focused synthetic checks for the base-donor-disjoint sensitivity."""

from collections import Counter
import csv
import gzip
import importlib.util
import io
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_review4_disjoint_donors.py"
SPEC = importlib.util.spec_from_file_location("review4_disjoint_donors", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FakeStore:
    def __init__(self, trial_labels, trial_subjects=None):
        self.trial_labels = trial_labels
        self.trial_subjects = trial_subjects or {trial: 0 for trial in trial_labels}

    def labels(self, trials):
        return np.asarray([self.trial_labels[trial] for trial in trials], dtype=int)

    def subjects(self, trials):
        return np.asarray([self.trial_subjects[trial] for trial in trials], dtype=int)


def test_new_base_is_balanced_deterministic_and_uses_only_remaining_people():
    labels, subjects, pool = {}, {}, []
    for subject in (1, 2, 3, 4):
        for label in (0, 1):
            for index in range(55):
                trial = f"s{subject}-y{label}-t{index}"
                labels[trial] = label; subjects[trial] = subject; pool.append(trial)
    store = FakeStore(labels, subjects)
    first = MOD.balanced_base(pool, store, "fixture", target=90, draw=2)
    second = MOD.balanced_base(pool, store, "fixture", target=90, draw=2)
    assert first == second and len(first) == len(set(first)) == 100
    assert Counter(store.labels(first).tolist()) == {0: 50, 1: 50}
    assert set(store.subjects(first)) <= {1, 2, 3, 4}


def write_predictions(path, operation, evaluation, truths, tamper=None):
    raw = path.open("wb")
    compressed = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=MOD.PREDICTION_FIELDS); writer.writeheader()
    for index, (trial, truth) in enumerate(zip(evaluation, truths)):
        predicted = truth if index != 1 else 1 - truth
        row = {"operation_id": operation["operation_id"], "dataset_key": "openbmi",
               "target": 2, "draw": 0, "condition": "own", "donor_subject": 2,
               "C": 0.1, "trial_id": trial, "y_true": truth, "y_pred": predicted,
               "p0": 0.8 if predicted == 0 else 0.2,
               "p1": 0.8 if predicted == 1 else 0.2}
        if tamper and index == 0:
            row.update(tamper)
        writer.writerow(row)
    text.close(); raw.close()


def test_prediction_reconstruction_checks_coordinates_truth_and_balanced_accuracy(tmp_path):
    evaluation = ["e0", "e1", "e2", "e3"]
    truths = [0, 0, 1, 1]
    operation = {"operation_id": "op", "dataset_key": "openbmi", "target": 2,
                 "draw": 0, "condition": "own", "donor_subject": 2, "C": 0.1,
                 "memberships": {"evaluation": "eval"}}
    sets = {"eval": evaluation}; stores = {"openbmi": FakeStore(dict(zip(evaluation, truths)))}
    path = tmp_path / "predictions.csv.gz"
    write_predictions(path, operation, evaluation, truths)
    assert MOD.reconstruct_scores(path, [operation], sets, stores) == {"op": 0.75}
    bad = tmp_path / "bad.csv.gz"
    write_predictions(bad, operation, evaluation, truths, {"target": 3})
    with pytest.raises(ValueError, match="identity"):
        MOD.reconstruct_scores(bad, [operation], sets, stores)


def test_summary_averages_draws_within_person_and_three_donors_with_fixed_bootstrap():
    operations, scores = [], {}
    for dataset, targets in (("openbmi", range(1, 13)), ("bnci", range(1, 10))):
        for target in targets:
            for draw in range(10):
                for condition in MOD.CONDITIONS:
                    operation_id = f"{dataset}-{target}-{draw}-{condition}"
                    operations.append({"operation_id": operation_id, "dataset_key": dataset,
                                       "target": target, "draw": draw, "condition": condition})
                    scores[operation_id] = 0.70 if condition == "own" else 0.65
    summary, people, draws = MOD.summarize(scores, operations)
    assert len(operations) == 840 and len(people) == 21 and len(draws) == 210
    assert all(row["effect_pp"] == pytest.approx(5.0) for row in people)
    for dataset, count in (("openbmi", 12), ("bnci", 9)):
        result = summary["datasets"][dataset]
        assert result["n"] == count and result["mean_pp"] == pytest.approx(5.0)
        assert result["positive"] == count and result["zero"] == result["negative"] == 0
        assert result["bootstrap_ci_low_pp"] == pytest.approx(5.0)
        assert result["bootstrap_ci_high_pp"] == pytest.approx(5.0)

