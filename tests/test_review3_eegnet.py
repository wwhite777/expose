import numpy as np
import pytest
import torch

import expose.review3_eegnet as eegnet
from expose.review3_eegnet import (EEGNet, adapt_source, fit_source, normalized_tensor,
    probabilities, seed_cpu, source_normalizer, train_epoch, validate_probabilities)


def test_shapes_probabilities_and_max_norm():
    seed_cpu(12)
    model = EEGNet(8, 750)
    model.constrain()
    x = torch.randn(6, 8, 750)
    p = probabilities(model, x)
    assert p.shape == (6, 2)
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
    assert model.spatial.weight.flatten(1).norm(dim=1).max() <= 1.000001
    assert model.classifier.weight.norm(dim=1).max() <= 0.250001


def test_finetune_preserves_source_and_bn_statistics():
    seed_cpu(15)
    model = EEGNet(8, 750)
    source = np.random.default_rng(1).normal(size=(8, 8, 750))
    norm = source_normalizer(source)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    adapted, losses = adapt_source(model, norm, source[:4], np.array([0, 1, 0, 1]), 20, epochs=1)
    assert len(losses) == 1
    for key, value in original.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        if "running_" in key or "num_batches_tracked" in key:
            torch.testing.assert_close(adapted.state_dict()[key], value, rtol=0, atol=0)
    assert not torch.equal(adapted.classifier.weight, model.classifier.weight)
    np.testing.assert_allclose(norm[0], source.mean(axis=(0, 2), keepdims=True))
    assert normalized_tensor(source, norm).dtype == torch.float32


def test_h0_adaptation_is_a_distinct_nonmutating_checkpoint_copy():
    seed_cpu(16)
    model = EEGNet(8, 750)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    adapted, losses = adapt_source(model, (np.zeros((1, 8, 1)), np.ones((1, 8, 1))),
                                   np.empty((0, 8, 750)), np.empty(0, dtype=int), 21)
    assert adapted is not model and losses == []
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        torch.testing.assert_close(adapted.state_dict()[key], value, rtol=0, atol=0)


def test_post_optimizer_constraints_and_750_feature_length():
    seed_cpu(17)
    model = EEGNet(8, 750)
    assert model.classifier.in_features == 16 * 23
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(4, 8, 750)
    y = torch.tensor([0, 1, 0, 1])
    train_epoch(model, x, y, optimizer, np.random.default_rng(17), batch_size=4)
    assert model.spatial.weight.flatten(1).norm(dim=1).max() <= 1.000001
    assert model.classifier.weight.norm(dim=1).max() <= 0.250001


def test_one_epoch_source_fit_is_deterministic_and_normalizer_excludes_validation():
    source = np.random.default_rng(18).normal(size=(4, 8, 750))
    labels = np.array([0, 1, 0, 1])
    validation = np.random.default_rng(19).normal(size=(2, 8, 750))
    first = fit_source(source, labels, 22, 1, validation=(validation, np.array([0, 1])))
    second = fit_source(source, labels, 22, 1, validation=(validation + 100, np.array([0, 1])))
    for key, value in first[0].state_dict().items():
        torch.testing.assert_close(value, second[0].state_dict()[key], rtol=0, atol=0)
    np.testing.assert_allclose(first[1][0], source.mean(axis=(0, 2), keepdims=True), rtol=0, atol=0)
    np.testing.assert_allclose(first[1][0], second[1][0], rtol=0, atol=0)


def test_selection_model_is_discarded_and_final_refit_contract_is_mechanical(monkeypatch):
    calls = []
    def fake_fit(x, y, seed, epochs, validation=None, patience=10):
        calls.append((x, y, seed, epochs, validation, patience))
        return object(), ("normalizer", seed), [{"epoch": n} for n in range(1, epochs + 1)], 3 if validation else epochs
    monkeypatch.setattr(eegnet, "fit_source", fake_fit)
    model, normalizer, receipt = eegnet.select_and_refit_source("select", "sy", "valid", "vy", "final", "fy", 30)
    assert len(calls) == 2 and calls[0][2:5] == (30, 50, ("valid", "vy"))
    assert calls[1][2:5] == (31, 3, None)
    assert normalizer == ("normalizer", 31) and receipt["selection_model_discarded"]
    assert receipt["selected_epochs"] == len(receipt["refit_log"]) == 3


def test_probability_validation_rejects_nonfinite_and_unnormalized_rows():
    valid = validate_probabilities(np.array([[0.25, 0.75], [1.0, 0.0]]), 2)
    assert valid.shape == (2, 2)
    with pytest.raises(FloatingPointError, match="probabilities"):
        validate_probabilities(np.array([[np.nan, np.nan]]), 1)
    with pytest.raises(FloatingPointError, match="probabilities"):
        validate_probabilities(np.array([[0.25, 0.70]]), 1)


def test_train_epoch_rejects_nonfinite_loss_before_backward(monkeypatch):
    seed_cpu(31)
    model = EEGNet(1, 32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    monkeypatch.setattr(eegnet.F, "cross_entropy",
                        lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(FloatingPointError, match="training loss"):
        train_epoch(model, torch.zeros(2, 1, 32), torch.tensor([0, 1]), optimizer,
                    np.random.default_rng(31), batch_size=2)


def test_source_fit_rejects_nonfinite_validation_loss(monkeypatch):
    monkeypatch.setattr(eegnet, "train_epoch", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(eegnet.F, "cross_entropy",
                        lambda *_args, **_kwargs: torch.tensor(float("nan")))
    source = np.random.default_rng(32).normal(size=(4, 1, 32))
    labels = np.array([0, 1, 0, 1])
    validation = np.random.default_rng(33).normal(size=(2, 1, 32))
    with pytest.raises(FloatingPointError, match="source-validation loss"):
        fit_source(source, labels, 34, 1, validation=(validation, np.array([0, 1])))


def test_selection_rejects_zero_epoch_before_refit(monkeypatch):
    calls = []
    def fake_fit(*args, **kwargs):
        calls.append((args, kwargs))
        return object(), object(), [], 0
    monkeypatch.setattr(eegnet, "fit_source", fake_fit)
    with pytest.raises(RuntimeError, match="selected source epoch"):
        eegnet.select_and_refit_source("select", "sy", "valid", "vy", "final", "fy", 35)
    assert len(calls) == 1
