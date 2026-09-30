import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "replay_review3_bnci_raw_condition",
        ROOT / "scripts/replay_review3_bnci_raw_condition.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_rows(module, trial_ids, truth, prediction, probabilities):
    return [
        {
            "operation_id": module.OPERATION_ID,
            "trial_id": trial_id,
            "y_true": str(int(y_true)),
            "y_pred": str(int(y_pred)),
            "p0": repr(float(probability[0])),
            "p1": repr(float(probability[1])),
        }
        for trial_id, y_true, y_pred, probability in zip(
            trial_ids, truth, prediction, probabilities
        )
    ]


def test_scope_and_required_raw_files_are_exactly_one_bnci_cell():
    module = load_module()
    assert module.OPERATION_ID == "bnci_main__plain_ts__d0__t1__kall__Sall__h60"
    assert module.PROBABILITY_TOLERANCE == 1e-10
    assert module.required_raw_files() == [
        "A02T.mat",
        "A03T.mat",
        "A04T.mat",
        "A05T.mat",
        "A06T.mat",
        "A07T.mat",
        "A08T.mat",
        "A09T.mat",
        "A01T.mat",
        "A01E.mat",
    ]


def test_version_guard_requires_the_complete_pinned_mapping():
    module = load_module()
    assert module.validate_versions(lambda name: module.EXPECTED_VERSIONS[name]) == (
        module.EXPECTED_VERSIONS
    )
    with pytest.raises(ValueError, match="versions differ"):
        module.validate_versions(
            lambda name: "0.0" if name == "pyriemann" else module.EXPECTED_VERSIONS[name]
        )


def test_prediction_comparison_checks_order_labels_predictions_and_probabilities():
    module = load_module()
    trial_ids = ["trial-a", "trial-b"]
    truth = np.array([0, 1])
    prediction = np.array([0, 1])
    probability = np.array([[0.75, 0.25], [0.2, 0.8]])
    saved = saved_rows(module, trial_ids, truth, prediction, probability)
    result = module.compare_predictions(
        trial_ids, truth, prediction, probability, saved, tolerance=1e-10
    )
    assert result == {
        "evaluation_trials": 2,
        "label_mismatches": 0,
        "probability_values_compared": 4,
        "probability_values_over_tolerance": 0,
        "maximum_absolute_probability_difference": 0.0,
        "probability_tolerance": 1e-10,
    }

    with pytest.raises(ValueError, match="order differs"):
        module.compare_predictions(
            trial_ids[::-1], truth, prediction, probability, saved, tolerance=1e-10
        )
    wrong_truth = truth.copy()
    wrong_truth[0] = 1
    with pytest.raises(ValueError, match="raw labels differ"):
        module.compare_predictions(
            trial_ids, wrong_truth, prediction, probability, saved, tolerance=1e-10
        )
    changed = probability.copy()
    changed[1, 0] += 2e-10
    with pytest.raises(ValueError, match="prediction mismatch"):
        module.compare_predictions(
            trial_ids, truth, prediction, changed, saved, tolerance=1e-10
        )


def test_oas_and_fit_entry_points_do_not_import_review3_fit_helpers():
    source = (ROOT / "scripts/replay_review3_bnci_raw_condition.py").read_text(
        encoding="utf-8"
    )
    assert "from expose.bnci2014 import" in source
    assert "from expose.review2_controls import" not in source
    assert "from expose.review3_transfer import" not in source
    assert "from run_review3_grid import" not in source
    assert "from sklearn.covariance import oas" in source
    assert "TangentSpace(metric=\"riemann\", tsupdate=False)" in source
    assert "random_state=20260914" in source
