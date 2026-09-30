import numpy as np
import pytest
import torch

from expose.review3_eegnet import EEGNet
from scripts.diagnose_review4_eegnet_bn import (
    batchnorm_layers, comparison_metrics, layer_summary, probability_metrics,
    predict, recalibrate_batchnorm, validate_sidecar,
)


def sidecar(source, selection, validation, epochs=2):
    return {
        "checkpoint_sha256": "a" * 64,
        "final_source_trials": len(source),
        "refit_log": [{"epoch": n} for n in range(1, epochs + 1)],
        "refit_seed": 11,
        "selected_epochs": epochs,
        "selection_ids": selection,
        "selection_log": [],
        "selection_model_discarded": True,
        "selection_seed": 10,
        "selection_training_trials": len(selection),
        "selection_validation_trials": len(validation),
        "source_ids": source,
        "target_evaluation_used": False,
        "validation_ids": validation,
        "validation_people": [1],
    }


def test_membership_reports_final_refit_overlap_without_calling_it_held_out():
    value = sidecar(["a", "b", "c", "d"], ["a", "c"], ["b", "x"])
    result = validate_sidecar(value, {"a", "b", "c", "d", "x"})
    assert result == {
        "source_selection_overlap_n": 2,
        "source_validation_overlap_n": 1,
        "selection_validation_overlap_n": 0,
        "final_checkpoint_validation_status": "not_held_out_from_final_refit",
    }


def test_membership_fails_closed_on_selection_validation_overlap_or_non_source():
    with pytest.raises(ValueError, match="overlap"):
        validate_sidecar(sidecar(["a", "b"], ["a"], ["a"]), {"a", "b"})
    with pytest.raises(ValueError, match="SOURCE-session1"):
        validate_sidecar(sidecar(["a", "target"], ["a"], ["b"]), {"a", "b"})


def test_layer_summary_reports_momentum_residual_from_tracked_batches():
    layer = torch.nn.BatchNorm2d(3, momentum=0.01)
    layer.num_batches_tracked.fill_(7)
    layer.running_mean.copy_(torch.tensor([-1.0, 0.0, 2.0]))
    layer.running_var.copy_(torch.tensor([0.5, 1.0, 2.0]))
    result = layer_summary(layer)
    assert result["num_batches_tracked"] == 7
    assert result["initial_stat_fraction"] == pytest.approx(0.99 ** 7)
    assert result["running_var_min"] == 0.5
    assert result["running_var_max"] == 2.0


def test_batch_stat_copy_disables_dropout_and_preserves_original_state():
    torch.manual_seed(3)
    model = EEGNet(channels=1, samples=32, dropout=0.9)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x = torch.randn(6, 1, 32)
    first = predict(model, x, batch_size=3, batch_statistics=True)
    second = predict(model, x, batch_size=3, batch_statistics=True)
    np.testing.assert_array_equal(first, second)
    # The diagnostic copy uses evaluation-mode dropout; the original retains
    # both its parameters and its pre-call module mode.
    assert model.training and model.drop.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_source_recalibration_uses_cumulative_stats_on_copy_only():
    torch.manual_seed(4)
    model = EEGNet(channels=1, samples=32, dropout=0.5)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x = torch.randn(6, 1, 32)
    recalibrated = recalibrate_batchnorm(model, x, batch_size=3)
    for _, layer in batchnorm_layers(recalibrated):
        assert layer.momentum is None
        assert int(layer.num_batches_tracked) == 2
        assert not layer.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_prediction_summaries_label_source_loss_as_resubstitution():
    y = np.array([0, 1, 0])
    reference = np.array([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4]])
    candidate = np.array([[0.4, 0.6], [0.4, 0.6], [0.55, 0.45]])
    metrics = probability_metrics(reference, y)
    assert "source_resubstitution_log_loss" in metrics
    assert metrics["predicted_right_n"] == 2
    comparison = comparison_metrics(reference, candidate)
    assert comparison["changed_predictions_n"] == 1
    assert comparison["max_abs_probability_change"] == pytest.approx(0.4)
