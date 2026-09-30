import numpy as np
import pytest
import torch
from torch import nn

import expose.review4_eegnet as successor
from expose.review4_eegnet import (
    EEGNet, early_stopping_decision, probability_summary, recalibrate_batchnorm,
    stratified_calibration_batches,
)
from scripts.run_review4_eegnet import (
    applicable_operations, selection_count, source_specifications,
)


def test_architecture_and_batchnorm_momentum_are_unchanged():
    model = EEGNet(8, 750)
    assert model.classifier.in_features == 16 * 23
    layers = [model.bn1, model.bn2, model.bn3]
    assert all(layer.momentum == 0.01 and layer.eps == 1e-3 for layer in layers)


def test_source_only_recalibration_uses_equal_cumulative_batches_on_copy():
    torch.manual_seed(1)
    model = EEGNet(1, 32, dropout=0.9)
    original = {name: value.detach().clone()
                for name, value in model.state_dict().items()}
    source = torch.randn(120, 1, 32)
    labels = torch.tensor([0] * 60 + [1] * 60)
    calibrated = recalibrate_batchnorm(model, source, labels, batch_size=60)
    for name in ("bn1", "bn2", "bn3"):
        layer = getattr(calibrated, name)
        assert layer.momentum is None
        assert int(layer.num_batches_tracked) == 2
        assert not layer.training
    assert not calibrated.drop.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match="source/label"):
        recalibrate_batchnorm(model, source[:119], labels, batch_size=60)


def test_calibration_batches_are_stable_and_each_exactly_30_per_class():
    labels = np.array(([0, 1, 0, 1, 1, 0] * 20), dtype=int)
    batches = stratified_calibration_batches(labels, batch_size=60)
    assert len(batches) == 2
    zeros = np.flatnonzero(labels == 0)
    ones = np.flatnonzero(labels == 1)
    for number, indexes in enumerate(batches):
        assert np.count_nonzero(labels[indexes] == 0) == 30
        assert np.count_nonzero(labels[indexes] == 1) == 30
        np.testing.assert_array_equal(indexes[:30], zeros[number * 30:(number + 1) * 30])
        np.testing.assert_array_equal(indexes[30:], ones[number * 30:(number + 1) * 30])


def test_minimum_epoch_patience_and_smallest_exact_tie():
    best_epoch, best_loss = 0, float("inf")
    best_epoch, best_loss, stop = early_stopping_decision(
        15, 0.5, best_epoch, best_loss)
    assert (best_epoch, best_loss, stop) == (15, 0.5, False)
    for epoch in range(16, 25):
        best_epoch, best_loss, stop = early_stopping_decision(
            epoch, 0.5, best_epoch, best_loss)
        assert best_epoch == 15 and not stop
    best_epoch, best_loss, stop = early_stopping_decision(
        25, 0.5, best_epoch, best_loss)
    assert best_epoch == 15 and stop
    with pytest.raises(ValueError, match="minimum"):
        early_stopping_decision(14, 0.4, 0, float("inf"))


def test_epoch_selection_recalibrates_only_selection_training_tensor(monkeypatch):
    class Tiny(nn.Module):
        def __init__(self, channels, samples):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))
        def forward(self, x):
            return torch.stack((self.weight.expand(len(x)),
                                self.weight.expand(len(x))), dim=1)

    seen_calibration = []
    seen_validation = []
    monkeypatch.setattr(successor, "EEGNet", Tiny)
    monkeypatch.setattr(successor, "train_epoch", lambda *args, **kwargs: 0.7)
    def fake_recalibration(model, x, y):
        seen_calibration.append((x, y))
        return model
    def fake_loss(model, x, y):
        seen_validation.append((x, y))
        return 0.5
    monkeypatch.setattr(successor, "recalibrate_batchnorm", fake_recalibration)
    monkeypatch.setattr(successor, "model_log_loss", fake_loss)
    train = np.random.default_rng(2).normal(size=(60, 1, 32))
    validation = np.random.default_rng(3).normal(size=(20, 1, 32))
    labels = np.tile([0, 1], 30)
    validation_labels = np.tile([0, 1], 10)
    _, _, log, selected, loss = successor.fit_source_selection(
        train, labels, validation, validation_labels, seed=5,
        minimum_epochs=15, maximum_epochs=50, patience=10)
    assert selected == 15 and loss == 0.5 and log[-1]["epoch"] == 25
    assert len(seen_calibration) == len(seen_validation) == 11
    assert all(len(value[0]) == len(value[1]) == 60 for value in seen_calibration)
    assert all(len(value[0]) == 20 for value in seen_validation)
    assert all(value[0].data_ptr() != seen_validation[0][0].data_ptr()
               for value in seen_calibration)


def test_selection_count_uses_300_or_1260_equal_blocks():
    assert selection_count(300, 1300) == 300
    assert selection_count(1800, 1300) == 1260


def test_selection_count_rejects_when_no_complete_block():
    with pytest.raises(ValueError, match="no valid"):
        selection_count(30, 59)


def test_exact_144_cell_operation_contract_and_six_sources():
    plan, memberships = [], {}
    for draw in range(3):
        for source_size in (300, 1800):
            source_ref = f"source-{draw}-{source_size}"
            memberships[source_ref] = [f"s-{draw}-{source_size}-{n}"
                                       for n in range(source_size)]
            for target in range(12):
                for h_total in (0, 60):
                    personal_ref = f"personal-{target}-{h_total}"
                    evaluation_ref = f"evaluation-{target}"
                    memberships.setdefault(personal_ref, [f"h-{target}-{n}"
                                                          for n in range(h_total)])
                    memberships.setdefault(evaluation_ref, [f"e-{target}-{n}"
                                                            for n in range(100)])
                    plan.append({
                        "method": "plain_ts", "study_id": "openbmi_main",
                        "dataset": "openbmi", "montage": "8ch",
                        "mode": "frozen_source", "donor_count": "all",
                        "draw": draw, "source_size": source_size,
                        "h_total": h_total, "target": target,
                        "memberships": {"source": source_ref,
                                        "personal": personal_ref,
                                        "evaluation": evaluation_ref},
                    })
    config = {"study_id": "openbmi_main", "draws": [0, 1, 2],
              "source_sizes": [300, 1800], "history_totals": [0, 60],
              "expected_scores": 144}
    selected = applicable_operations(plan, config)
    specifications = source_specifications(selected, memberships)
    assert len(selected) == 144 and len(specifications) == 6


def test_probability_summary_reports_loss_spread_and_collapse():
    probability = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    result = probability_summary(probability, np.array([0, 1, 0, 1]))
    assert result["single_predicted_class"]
    assert result["predicted_right_n"] == 4 and result["predicted_left_n"] == 0
    assert result["p_left_sd"] > 0 and result["log_loss"] > 0
