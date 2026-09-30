"""Two fixed CPU baselines for engineering feasibility, not model selection."""

import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_ORDER = ("csp_lda", "ts_lr")


def validate_epochs(X, y, *, require_both_classes=True):
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 3 or min(X.shape) < 1 or X.dtype.kind not in "fiu":
        raise ValueError("epochs must be a nonempty real (trial, channel, time) array")
    if y.ndim != 1 or len(y) != len(X) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("one binary 0/1 label is required per epoch")
    if not np.all(np.isfinite(X)):
        raise ValueError("epochs contain nonfinite values")
    if require_both_classes and set(y.tolist()) != {0, 1}:
        raise ValueError("both classes are required")


def make_baseline(name):
    if name == "csp_lda":
        return make_pipeline(
            CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=False,
                cov_est="concat", component_order="mutual_info"),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
    if name == "ts_lr":
        return make_pipeline(
            Covariances(estimator="oas"),
            TangentSpace(metric="riemann", tsupdate=False),
            StandardScaler(),
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                               random_state=20260914),
        )
    raise ValueError(f"unknown fixed baseline: {name}")


def nested_history_indices(y, n_per_class, seed):
    y = np.asarray(y)
    if y.ndim != 1 or not np.all(np.isin(y, [0, 1])):
        raise ValueError("history labels must be a binary vector")
    if isinstance(n_per_class, bool) or not isinstance(n_per_class, int) or n_per_class < 0:
        raise ValueError("dose must be a nonnegative integer per class")
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        indices = rng.permutation(np.flatnonzero(y == label))
        if len(indices) < n_per_class:
            raise ValueError("insufficient history for the fixed balanced dose")
        selected.extend(indices[:n_per_class].tolist())
    return np.array(selected, dtype=int)


def assert_disjoint_trial_ids(train_ids, evaluation_ids):
    if not train_ids or not evaluation_ids:
        raise ValueError("training and evaluation must both be nonempty")
    if len(set(train_ids)) != len(train_ids) or len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("duplicate trial IDs within split")
    if set(train_ids) & set(evaluation_ids):
        raise ValueError("training and evaluation trial IDs overlap")
