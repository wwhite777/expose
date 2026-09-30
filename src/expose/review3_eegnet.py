"""Small CPU EEGNet anchor; all selection uses source people only.

Independent PyTorch implementation of EEGNet-8,2's layer specification:
Lawhern et al. (2018), original arl-eegmodels/EEGModels.py. This is a
matched-access architecture anchor, not a reproduction of published scores.
"""
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def seed_cpu(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


class EEGNet(nn.Module):
    def __init__(self, channels=8, samples=750, dropout=0.5):
        super().__init__()
        self.channels, self.samples = channels, samples
        self.temporal = nn.Conv2d(1, 8, (1, 64), bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-3, momentum=0.01)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.depth = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(16, eps=1e-3, momentum=0.01)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(16 * (samples // 4 // 8), 2)
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.samples):
            raise ValueError("expected (trials, fitted channels, fitted samples)")
        # Asymmetric padding implements exact SAME length for even kernels.
        x = self.bn1(self.temporal(F.pad(x[:, None], (31, 32, 0, 0))))
        x = self.drop(F.avg_pool2d(F.elu(self.bn2(self.spatial(x))), (1, 4)))
        x = self.point(self.depth(F.pad(x, (7, 8, 0, 0))))
        x = self.drop(F.avg_pool2d(F.elu(self.bn3(x)), (1, 8)))
        return self.classifier(x.flatten(1))

    @torch.no_grad()
    def constrain(self):
        # Per-output spatial filter norm <=1 and dense class vector norm <=.25.
        for parameter, maximum in ((self.spatial.weight, 1.0),
                                   (self.classifier.weight, 0.25)):
            norm = parameter.flatten(1).norm(dim=1).clamp_min(1e-12)
            factor = (maximum / norm).clamp_max(1.0)
            parameter.mul_(factor.reshape((-1,) + (1,) * (parameter.ndim - 1)))


def source_normalizer(source):
    x = np.asarray(source, dtype=np.float64)
    if x.ndim != 3 or not len(x) or not np.isfinite(x).all():
        raise ValueError("invalid source epochs")
    mean = x.mean(axis=(0, 2), keepdims=True)
    scale = x.std(axis=(0, 2), keepdims=True)
    if np.any(scale <= 0):
        raise ValueError("zero source channel variance")
    return mean, scale


def normalized_tensor(x, normalization):
    mean, scale = normalization
    value = (np.asarray(x) - mean) / scale
    if not np.isfinite(value).all():
        raise ValueError("nonfinite normalized epochs")
    return torch.as_tensor(value, dtype=torch.float32)


def train_epoch(model, x, y, optimizer, rng, batch_size=64, freeze_batchnorm=False):
    model.train()
    if freeze_batchnorm:
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    total = 0.0
    for indexes in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / batch_size)))):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[indexes]), y[indexes])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite neural training loss")
        loss.backward()
        optimizer.step()
        model.constrain()
        total += float(loss.detach()) * len(indexes)
    mean_loss = total / len(x)
    if not np.isfinite(mean_loss):
        raise FloatingPointError("nonfinite neural mean training loss")
    return mean_loss


def validate_probabilities(value, expected_rows):
    result = np.asarray(value)
    if (result.shape != (expected_rows, 2) or not np.isfinite(result).all()
            or np.any(result < 0) or np.any(result > 1)
            or not np.allclose(result.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise FloatingPointError("invalid neural probabilities")
    return result


@torch.no_grad()
def probabilities(model, x, batch_size=128):
    model.eval()
    value = torch.cat([model(chunk).softmax(1) for chunk in x.split(batch_size)]).cpu().numpy()
    return validate_probabilities(value, len(x))


def fit_source(train_x, train_y, seed, epochs, validation=None, patience=10):
    """Source-only early stopping when validation is supplied; otherwise fixed epochs."""
    seed_cpu(seed)
    normalizer = source_normalizer(train_x)
    x = normalized_tensor(train_x, normalizer)
    y = torch.as_tensor(train_y, dtype=torch.long)
    if set(y.tolist()) != {0, 1} or len(x) != len(y):
        raise ValueError("source labels must be valid binary labels")
    model = EEGNet(x.shape[1], x.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    log, best_loss, best_epoch = [], float("inf"), 0
    if validation is not None:
        vx = normalized_tensor(validation[0], normalizer)
        vy = torch.as_tensor(validation[1], dtype=torch.long)
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, x, y, optimizer, rng)
        row = {"epoch": epoch, "train_loss": loss}
        if validation is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(F.cross_entropy(model(vx), vy))
            if not np.isfinite(val_loss):
                raise FloatingPointError("nonfinite neural source-validation loss")
            row["source_validation_loss"] = val_loss
            if val_loss < best_loss - 1e-8:
                best_loss, best_epoch = val_loss, epoch
            if epoch - best_epoch >= patience:
                log.append(row)
                break
        log.append(row)
    # Validation fit is used only to select epoch count; final model is refitted.
    return model, normalizer, log, best_epoch if validation is not None else epochs


def adapt_source(model, normalization, history_x, history_y, seed, epochs=20):
    """Fine-tune from a source checkpoint, retaining source BN running statistics."""
    adapted = deepcopy(model)
    if len(history_x) == 0:
        return adapted, []
    seed_cpu(seed)
    x = normalized_tensor(history_x, normalization)
    y = torch.as_tensor(history_y, dtype=torch.long)
    if len(x) != len(y) or set(y.tolist()) != {0, 1}:
        raise ValueError("history must contain both classes")
    optimizer = torch.optim.Adam(adapted.parameters(), lr=1e-4)
    rng = np.random.default_rng(seed)
    losses = [train_epoch(adapted, x, y, optimizer, rng, batch_size=32,
                          freeze_batchnorm=True) for _ in range(epochs)]
    return adapted, losses


def select_and_refit_source(selection_x, selection_y, validation_x, validation_y,
                            final_x, final_y, seed, max_epochs=50, patience=10):
    """Discard the selection model and refit the declared final source membership."""
    selection_model, selection_norm, selection_log, selected = fit_source(
        selection_x, selection_y, seed, max_epochs,
        validation=(validation_x, validation_y), patience=patience)
    del selection_model, selection_norm
    if (not isinstance(selected, (int, np.integer)) or isinstance(selected, bool)
            or selected < 1 or selected > max_epochs):
        raise RuntimeError("invalid selected source epoch count")
    model, normalizer, refit_log, fitted = fit_source(
        final_x, final_y, seed + 1, selected, validation=None)
    if fitted != selected or len(refit_log) != selected:
        raise RuntimeError("source refit did not complete the selected epoch count")
    return model, normalizer, {
        "selected_epochs": selected, "selection_seed": seed, "refit_seed": seed + 1,
        "selection_training_trials": len(selection_x),
        "selection_validation_trials": len(validation_x), "final_source_trials": len(final_x),
        "selection_log": selection_log, "refit_log": refit_log,
        "selection_model_discarded": True, "target_evaluation_used": False,
    }
