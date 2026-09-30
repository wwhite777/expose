"""Prospective source-selected EEGNet successor with source-only BN recalibration.

The architecture and personal-history adaptation are inherited unchanged from
``review3_eegnet``.  This module changes only the source-training recipe that
was selected from the source-only BatchNorm diagnostic.
"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .review3_eegnet import (
    EEGNet, adapt_source, normalized_tensor, probabilities, seed_cpu,
    source_normalizer, train_epoch, validate_probabilities,
)


BN_RECALIBRATION_BATCH_SIZE = 60
SOURCE_TRAIN_BATCH_SIZE = 64


def batchnorm_layers(model):
    layers = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, nn.BatchNorm2d)]
    if [name for name, _ in layers] != ["bn1", "bn2", "bn3"]:
        raise ValueError("EEGNet BatchNorm layer contract differs")
    if any(layer.momentum != 0.01 for _, layer in layers):
        raise ValueError("EEGNet BatchNorm momentum differs")
    return layers


def stratified_calibration_batches(source_labels,
                                   batch_size=BN_RECALIBRATION_BATCH_SIZE):
    """Deterministic 50/50 source-label batches, stable within each class."""
    labels = (source_labels.detach().cpu().numpy() if isinstance(source_labels, torch.Tensor)
              else np.asarray(source_labels))
    if (labels.ndim != 1 or len(labels) == 0 or len(labels) % batch_size
            or batch_size % 2 or not np.array_equal(labels, labels.astype(np.int64))):
        raise ValueError("BN calibration labels/count do not form equal blocks")
    labels = labels.astype(np.int64)
    zeros = np.flatnonzero(labels == 0)
    ones = np.flatnonzero(labels == 1)
    if len(zeros) != len(ones) or len(zeros) + len(ones) != len(labels):
        raise ValueError("BN calibration requires balanced binary source labels")
    half = batch_size // 2
    if len(zeros) % half:
        raise ValueError("class counts do not divide into balanced calibration blocks")
    return [np.concatenate((zeros[start:start + half], ones[start:start + half]))
            for start in range(0, len(zeros), half)]


@torch.no_grad()
def recalibrate_batchnorm(model, normalized_source, source_labels,
                          batch_size=BN_RECALIBRATION_BATCH_SIZE):
    """Return a copy with BN statistics recalibrated on source samples only.

    Stable equal-size batches and ``momentum=None`` give every calibration
    batch equal weight.  The caller must supply a sample count divisible by the
    fixed batch size; validation or target samples are never accepted here.
    """
    if (not isinstance(normalized_source, torch.Tensor)
            or normalized_source.ndim != 3 or len(normalized_source) == 0
            or len(normalized_source) != len(source_labels)):
        raise ValueError("BN calibration source/label arrays differ")
    batches = stratified_calibration_batches(source_labels, batch_size=batch_size)
    calibrated = deepcopy(model)
    calibrated.eval()
    layers = batchnorm_layers(calibrated)
    for _, layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    for indexes in batches:
        calibrated(normalized_source[torch.as_tensor(indexes, dtype=torch.long)])
    expected = len(batches)
    if any(int(layer.num_batches_tracked) != expected for _, layer in layers):
        raise RuntimeError("source-only BN recalibration pass count differs")
    calibrated.eval()
    return calibrated


@torch.no_grad()
def model_log_loss(model, x, y, batch_size=128):
    model.eval()
    total = 0.0
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        loss = F.cross_entropy(model(x[start:stop]), y[start:stop], reduction="sum")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite source-validation loss")
        total += float(loss)
    value = total / len(x)
    if not np.isfinite(value):
        raise FloatingPointError("nonfinite source-validation mean loss")
    return value


def early_stopping_decision(epoch, loss, best_epoch, best_loss,
                            minimum_epochs=15, patience=10):
    """Update source-only early stopping, preserving the first exact tie."""
    if epoch < minimum_epochs:
        raise ValueError("validation is not eligible before the minimum epoch")
    if not np.isfinite(loss):
        raise FloatingPointError("nonfinite source-validation loss")
    if best_epoch == 0 or loss < best_loss:
        best_epoch, best_loss = epoch, float(loss)
    stop = epoch - best_epoch >= patience
    return best_epoch, best_loss, stop


def _validated_xy(x, y, name):
    normalizer = source_normalizer(x)
    tensor = normalized_tensor(x, normalizer)
    labels = torch.as_tensor(y, dtype=torch.long)
    if len(tensor) != len(labels) or set(labels.tolist()) != {0, 1}:
        raise ValueError(name + " must contain valid binary source trials")
    return tensor, labels, normalizer


def fit_source_selection(train_x, train_y, validation_x, validation_y, seed,
                         minimum_epochs=15, maximum_epochs=50, patience=10):
    """Select an epoch using five held-out SOURCE people only.

    At every eligible epoch, a discarded copy is BN-recalibrated using the
    selection-training trials. Validation signals never enter recalibration.
    """
    if not (1 <= minimum_epochs <= maximum_epochs and patience >= 1):
        raise ValueError("invalid source epoch-selection limits")
    seed_cpu(seed)
    x, y, normalizer = _validated_xy(train_x, train_y, "selection training")
    vx = normalized_tensor(validation_x, normalizer)
    vy = torch.as_tensor(validation_y, dtype=torch.long)
    if len(vx) != len(vy) or set(vy.tolist()) != {0, 1}:
        raise ValueError("source validation must contain both classes")
    if len(x) % BN_RECALIBRATION_BATCH_SIZE:
        raise ValueError("selection training count must be a multiple of 60")

    model = EEGNet(x.shape[1], x.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=maximum_epochs)
    rng = np.random.default_rng(seed)
    log, best_epoch, best_loss = [], 0, float("inf")
    for epoch in range(1, maximum_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = train_epoch(
            model, x, y, optimizer, rng, batch_size=SOURCE_TRAIN_BATCH_SIZE)
        scheduler.step()
        row = {"epoch": epoch, "learning_rate": learning_rate,
               "selection_train_loss": train_loss, "eligible": epoch >= minimum_epochs}
        if epoch >= minimum_epochs:
            calibrated = recalibrate_batchnorm(model, x, y)
            validation_loss = model_log_loss(calibrated, vx, vy)
            row["source_validation_loss_after_train_only_bn_recalibration"] = validation_loss
            best_epoch, best_loss, stop = early_stopping_decision(
                epoch, validation_loss, best_epoch, best_loss,
                minimum_epochs=minimum_epochs, patience=patience)
            row.update(best_epoch_so_far=best_epoch, stop=stop)
            log.append(row)
            if stop:
                break
        else:
            log.append(row)
    if best_epoch < minimum_epochs or best_epoch > maximum_epochs:
        raise RuntimeError("source epoch selection failed")
    return model, normalizer, log, best_epoch, best_loss


def fit_source_fixed(train_x, train_y, seed, epochs, maximum_epochs=50):
    """Fresh final-source refit followed by one source-only BN recalibration."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= maximum_epochs:
        raise ValueError("invalid fixed source epoch count")
    seed_cpu(seed)
    x, y, normalizer = _validated_xy(train_x, train_y, "final source")
    if len(x) % BN_RECALIBRATION_BATCH_SIZE:
        raise ValueError("final source count must be a multiple of 60")
    model = EEGNet(x.shape[1], x.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=maximum_epochs)
    rng = np.random.default_rng(seed)
    log = []
    for epoch in range(1, epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        loss = train_epoch(model, x, y, optimizer, rng,
                           batch_size=SOURCE_TRAIN_BATCH_SIZE)
        scheduler.step()
        log.append({"epoch": epoch, "learning_rate": learning_rate,
                    "final_source_train_loss": loss})
    calibrated = recalibrate_batchnorm(model, x, y)
    return calibrated, normalizer, log


def select_and_refit_source(selection_x, selection_y, validation_x, validation_y,
                            final_x, final_y, seed, minimum_epochs=15,
                            maximum_epochs=50, patience=10):
    """Discard selection fit; fresh-refit exact final membership and recalibrate."""
    selection_model, selection_normalizer, selection_log, selected, best_loss = (
        fit_source_selection(
            selection_x, selection_y, validation_x, validation_y, seed,
            minimum_epochs=minimum_epochs, maximum_epochs=maximum_epochs,
            patience=patience))
    del selection_model, selection_normalizer
    final_model, final_normalizer, refit_log = fit_source_fixed(
        final_x, final_y, seed + 1, selected, maximum_epochs=maximum_epochs)
    return final_model, final_normalizer, {
        "selected_epochs": selected,
        "best_source_validation_loss": best_loss,
        "selection_seed": seed,
        "refit_seed": seed + 1,
        "selection_training_trials": len(selection_x),
        "selection_validation_trials": len(validation_x),
        "final_source_trials": len(final_x),
        "minimum_source_epochs": minimum_epochs,
        "maximum_source_epochs": maximum_epochs,
        "source_patience": patience,
        "cosine_schedule_t_max": maximum_epochs,
        "selection_log": selection_log,
        "refit_log": refit_log,
        "selection_model_discarded": True,
        "final_bn_recalibration": "exact final source once, cumulative 60-trial blocks",
        "validation_signals_used_for_bn_recalibration": False,
        "target_evaluation_used": False,
    }


def probability_summary(probability, truth):
    probability = validate_probabilities(probability, len(truth))
    truth = np.asarray(truth, dtype=int)
    if truth.shape != (len(probability),) or set(truth.tolist()) != {0, 1}:
        raise ValueError("invalid binary evaluation labels")
    predicted = probability.argmax(axis=1)
    counts = np.bincount(predicted, minlength=2)
    clipped = np.clip(probability[np.arange(len(truth)), truth], 1e-15, 1.0)
    balanced_accuracy = float(np.mean([
        np.mean(predicted[truth == label] == label) for label in (0, 1)]))
    return {
        "balanced_accuracy": balanced_accuracy,
        "log_loss": float(-np.log(clipped).mean()),
        "p_left_sd": float(probability[:, 1].std()),
        "p_left_mean": float(probability[:, 1].mean()),
        "predicted_right_n": int(counts[0]),
        "predicted_left_n": int(counts[1]),
        "single_predicted_class": bool(np.count_nonzero(counts) == 1),
    }
