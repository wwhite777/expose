"""Transfer comparators for the bounded Review-3 development extension."""

from collections import Counter
import hashlib
import json

import numpy as np
from pyriemann.classification import MDM
from pyriemann.transfer import MDWM, encode_domains
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.geodesic import geodesic
from pyriemann.utils.mean import mean_covariance
from pyriemann.utils.utils import check_metric

from .review2_controls import fit_representation, train_classifier


C_GRID = (0.1, 1.0, 10.0)
HISTORY_TOTALS = (0, 4, 10, 20, 40, 60, 100)
SOURCE_TOTALS = (100, 300, 1000, 1800)
DRAWS = 10
MAIN_MDWM_LAMBDA = 0.5
TARGET_DOMAIN = "target"
SOURCE_DOMAIN = "source"


def _covariances(value, name, allow_empty=False):
    matrices = np.asarray(value, dtype=float)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError(f"{name} must have shape (trials, channels, channels)")
    if not allow_empty and len(matrices) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(matrices).all():
        raise ValueError(f"{name} contains nonfinite values")
    if len(matrices):
        if not np.allclose(matrices, matrices.transpose(0, 2, 1), rtol=0, atol=1e-10):
            raise ValueError(f"{name} matrices must be symmetric")
        if np.any(np.linalg.eigvalsh(matrices) <= 0):
            raise ValueError(f"{name} matrices must be positive definite")
    return matrices


def _balanced_binary(labels, count, name):
    values = np.asarray(labels)
    if values.shape != (count,) or values.dtype.kind not in "biu":
        raise ValueError(f"{name} labels must be a one-dimensional integer array")
    values = values.astype(int, copy=False)
    counts = Counter(values.tolist())
    if counts != {0: count // 2, 1: count // 2} or count % 2:
        raise ValueError(f"{name} labels must contain equal nonzero counts of classes 0 and 1")
    return values


class FittedMDWM:
    """Integer-label prediction facade over pyRiemann MDM or MDWM."""

    def __init__(self, estimator, mode, lambda_target):
        self.estimator = estimator
        self.mode = mode
        self.lambda_target = lambda_target
        self.classes_ = np.array([0, 1], dtype=int)

    @property
    def covmeans_(self):
        return self.estimator.covmeans_

    @property
    def source_means_(self):
        return getattr(self.estimator, "source_means_", self.estimator.covmeans_)

    @property
    def target_means_(self):
        return getattr(self.estimator, "target_means_", None)

    def predict(self, covariances):
        predicted = self.estimator.predict(_covariances(covariances, "evaluation covariances"))
        try:
            values = np.asarray(predicted, dtype=int)
        except (TypeError, ValueError) as error:
            raise ValueError("MDWM returned labels outside the fixed 0/1 mapping") from error
        if not set(values.tolist()) <= {0, 1}:
            raise ValueError("MDWM returned labels outside the fixed 0/1 mapping")
        return values

    def predict_proba(self, covariances):
        values = self.estimator.predict_proba(_covariances(covariances, "evaluation covariances"))
        if values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("MDWM returned invalid binary probabilities")
        return values


def fit_mdwm(source_covariances, source_labels, target_covariances=None,
             target_labels=None, lambda_target=MAIN_MDWM_LAMBDA):
    """Fit official pyRiemann MDWM; use pooled source MDM when target history is empty.

    pyRiemann 0.9 defines ``domain_tradeoff`` as
    ``geodesic(source_mean, target_mean, domain_tradeoff)``. Thus zero is the
    source endpoint and one is the target endpoint.
    """
    source = _covariances(source_covariances, "source covariances")
    y_source = _balanced_binary(source_labels, len(source), "source")
    if isinstance(lambda_target, bool) or not isinstance(lambda_target, (int, float)):
        raise ValueError("lambda_target must be numeric")
    lambda_target = float(lambda_target)
    if not np.isfinite(lambda_target) or not 0 <= lambda_target <= 1:
        raise ValueError("lambda_target must lie in [0,1]")

    if target_covariances is None:
        if target_labels is not None:
            raise ValueError("target labels are forbidden when no target covariance is supplied")
        estimator = MDM(metric="riemann", n_jobs=1).fit(source, y_source)
        return FittedMDWM(estimator, "pooled_mdm_h0", 0.0)

    target = _covariances(target_covariances, "target covariances", allow_empty=True)
    if len(target) == 0:
        if target_labels is not None:
            raise ValueError("target labels are forbidden for empty h=0 history")
        estimator = MDM(metric="riemann", n_jobs=1).fit(source, y_source)
        return FittedMDWM(estimator, "pooled_mdm_h0", 0.0)
    y_target = _balanced_binary(target_labels, len(target), "target")
    covariances = np.concatenate([source, target])
    labels = np.concatenate([y_source, y_target])
    domains = np.array([SOURCE_DOMAIN] * len(source) + [TARGET_DOMAIN] * len(target))
    _, encoded = encode_domains(covariances, labels, domains)
    estimator = MDWM(domain_tradeoff=lambda_target, target_domain=TARGET_DOMAIN,
                     metric="riemann", n_jobs=1).fit(covariances, encoded)
    if estimator.classes_.tolist() != ["0", "1"]:
        raise ValueError("official MDWM class ordering differs from the fixed binary mapping")
    return FittedMDWM(estimator, "official_pyriemann_mdwm_0p9", lambda_target)


def mdwm_source_class_means(source_covariances, source_labels):
    """Compute the two unweighted source means once for MDWM memoization."""
    source = _covariances(source_covariances, "source covariances")
    labels = _balanced_binary(source_labels, len(source), "source")
    return np.stack([mean_covariance(source[labels == value], metric="riemann")
                     for value in (0, 1)])


def fit_mdwm_from_source_means(source_means, target_covariances=None,
                               target_labels=None, lambda_target=MAIN_MDWM_LAMBDA):
    """Verified MDWM reimplementation using cached official source means.

    This follows pyRiemann 0.9 MDWM exactly: unweighted Riemannian target
    class means and ``geodesic(source, target, lambda_target)``. Prediction is
    delegated to pyRiemann MDM with those fixed centroids.
    """
    means = _covariances(source_means, "source class means")
    if len(means) != 2:
        raise ValueError("source_means must contain classes 0 and 1 in order")
    if isinstance(lambda_target, bool) or not isinstance(lambda_target, (int, float)):
        raise ValueError("lambda_target must be numeric")
    lambda_target = float(lambda_target)
    if not np.isfinite(lambda_target) or not 0 <= lambda_target <= 1:
        raise ValueError("lambda_target must lie in [0,1]")
    target_means = None
    if target_covariances is not None:
        target = _covariances(target_covariances, "target covariances", allow_empty=True)
        if len(target):
            labels = _balanced_binary(target_labels, len(target), "target")
            target_means = np.stack([mean_covariance(target[labels == value], metric="riemann")
                                     for value in (0, 1)])
        elif target_labels is not None:
            raise ValueError("target labels are forbidden for empty h=0 history")
    elif target_labels is not None:
        raise ValueError("target labels are forbidden when no target covariance is supplied")
    estimator = MDM(metric="riemann", n_jobs=1)
    estimator.metric_mean, estimator.metric_dist = check_metric(estimator.metric)
    estimator.classes_ = np.array([0, 1], dtype=int)
    estimator.covmeans_ = (means.copy() if target_means is None
                           else geodesic(means, target_means, lambda_target, metric="riemann"))
    estimator.source_means_ = means.copy()
    estimator.target_means_ = None if target_means is None else target_means
    mode = "pooled_mdm_h0_cached" if target_means is None else "verified_cached_mdwm_0p9"
    return FittedMDWM(estimator, mode, 0.0 if target_means is None else lambda_target)


class RecenterReferences:
    """Frozen Riemannian recentering references; this is not RPA."""

    def __init__(self, source_whiteners, target_whitener, target_history_count):
        self.source_whiteners = source_whiteners
        self.target_whitener = target_whitener
        self.target_history_count = target_history_count


def _readonly(matrix):
    value = np.asarray(matrix, dtype=float).copy()
    value.setflags(write=False)
    return value


def fit_recenter_whitener(covariances):
    """Fit one frozen Riemannian-reference whitener from a declared signal set."""
    values = _covariances(covariances, "reference covariances")
    whitener = invsqrtm(mean_covariance(values, metric="riemann"))
    # Eigen reconstruction can differ by a few ulps across the diagonal at
    # inverse-volt scales. The inverse square root is mathematically symmetric.
    return _readonly((whitener + whitener.T) / 2)


def assemble_recenter_references(source_whiteners, target_whitener, target_history_count):
    """Assemble independently cached person references after strict validation."""
    if not isinstance(source_whiteners, dict) or not source_whiteners:
        raise ValueError("source_whiteners must be a nonempty subject mapping")
    checked = {}
    for subject, whitener in source_whiteners.items():
        if not isinstance(subject, int) or isinstance(subject, bool):
            raise ValueError("source whitener subjects must be integers")
        matrix = _covariances(np.asarray(whitener)[None], "source whitener")[0]
        checked[subject] = _readonly(matrix)
    target = _covariances(np.asarray(target_whitener)[None], "target whitener")[0]
    if (not isinstance(target_history_count, int) or isinstance(target_history_count, bool)
            or target_history_count <= 0):
        raise ValueError("target_history_count must be a positive integer")
    return RecenterReferences(checked, _readonly(target), target_history_count)


def fit_recenter_references(source_covariances, source_subjects, target_history_covariances,
                            expected_target_count=100):
    """Fit label-free source-person and target-session-1 Riemannian references."""
    source = _covariances(source_covariances, "source covariances")
    subjects = np.asarray(source_subjects)
    if subjects.shape != (len(source),) or subjects.dtype.kind not in "biu":
        raise ValueError("source_subjects must give one integer person per source covariance")
    target = _covariances(target_history_covariances, "target history covariances")
    if (isinstance(expected_target_count, bool) or not isinstance(expected_target_count, int)
            or expected_target_count <= 0 or len(target) != expected_target_count):
        raise ValueError("target recentering count differs from declared session-1 reference count")
    source_whiteners = {}
    for subject in sorted(set(subjects.astype(int).tolist())):
        selected = source[subjects == subject]
        source_whiteners[subject] = fit_recenter_whitener(selected)
    return assemble_recenter_references(
        source_whiteners, fit_recenter_whitener(target), len(target))


def _congruence(covariances, whitener):
    transformed = whitener @ covariances @ whitener.T
    transformed = (transformed + transformed.transpose(0, 2, 1)) / 2
    if not np.isfinite(transformed).all() or np.any(np.linalg.eigvalsh(transformed) <= 0):
        raise ValueError("recentring produced an invalid covariance")
    return transformed


def recenter_source(covariances, source_subjects, references):
    values = _covariances(covariances, "source covariances")
    subjects = np.asarray(source_subjects)
    if subjects.shape != (len(values),) or subjects.dtype.kind not in "biu":
        raise ValueError("source_subjects must give one integer person per covariance")
    unknown = set(subjects.astype(int).tolist()) - set(references.source_whiteners)
    if unknown:
        raise ValueError("source covariance has no fitted person reference")
    result = np.empty_like(values)
    for subject, whitener in references.source_whiteners.items():
        selected = subjects == subject
        result[selected] = _congruence(values[selected], whitener)
    return result


def recenter_target(covariances, references):
    """Apply the frozen session-1 target reference without fitting on session 2."""
    values = _covariances(covariances, "target covariances")
    return _congruence(values, references.target_whitener)


class TSLRGrid:
    """Frozen representation with the fixed three-C logistic-regression grid."""

    def __init__(self, representation, classifiers, source_count, representation_mode):
        self.representation = representation
        self.classifiers = classifiers
        self.source_count = source_count
        self.representation_mode = representation_mode
        self.C_values = tuple(sorted(classifiers))

    def transform(self, covariances):
        return self.representation.transform(_covariances(covariances, "TS-LR covariances"))

    def predict(self, covariances, C):
        if float(C) not in self.classifiers:
            raise ValueError("C is outside the fixed grid")
        return self.classifiers[float(C)].predict(self.transform(covariances))

    def predict_proba(self, covariances, C):
        if float(C) not in self.classifiers:
            raise ValueError("C is outside the fixed grid")
        return self.classifiers[float(C)].predict_proba(self.transform(covariances))


def fit_ts_lr_grid(source_covariances, source_labels, personal_covariances=None,
                   personal_labels=None, representation_mode="source_frozen",
                   C_values=C_GRID):
    """Fit the fixed C grid; selection must be performed using source-only data elsewhere."""
    if tuple(float(value) for value in C_values) != C_GRID:
        raise ValueError("TS-LR C grid must remain (0.1, 1, 10); do not outcome-search")
    source = _covariances(source_covariances, "source covariances")
    y_source = _balanced_binary(source_labels, len(source), "source")
    if personal_covariances is None:
        if personal_labels is not None:
            raise ValueError("personal labels are forbidden without personal covariances")
        personal = np.empty((0, source.shape[1], source.shape[2]))
        y_personal = np.empty(0, dtype=int)
    else:
        personal = _covariances(personal_covariances, "personal covariances", allow_empty=True)
        if len(personal) == 0:
            if personal_labels is not None:
                raise ValueError("personal labels are forbidden for h=0")
            y_personal = np.empty(0, dtype=int)
        else:
            if len(personal) not in HISTORY_TOTALS[1:]:
                raise ValueError("personal history total is outside the fixed grid")
            y_personal = _balanced_binary(personal_labels, len(personal), "personal")
    if len(source) not in SOURCE_TOTALS:
        raise ValueError("source total is outside the fixed grid")
    train_cov = np.concatenate([source, personal])
    train_y = np.concatenate([y_source, y_personal])
    representation = fit_representation(train_cov, len(source), representation_mode)
    features = representation.transform(train_cov)
    classifiers = {C: train_classifier(features, train_y, len(source), C, "pooled") for C in C_GRID}
    return TSLRGrid(representation, classifiers, len(source), representation_mode)


def _trial_ids(values, name):
    result = list(values)
    if any(not isinstance(value, str) or not value for value in result) or len(result) != len(set(result)):
        raise ValueError(f"{name} must contain unique nonempty trial IDs")
    return result


def _id_hash(values):
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def training_membership_report(source_trials, personal_label_trials, target_reference_trials,
                               evaluation_trials, representation_mode, C_values=C_GRID):
    """Return full trial identities and their exact representation/classifier roles."""
    if representation_mode not in ("source_frozen", "pooled_refit"):
        raise ValueError("invalid representation mode")
    if tuple(float(value) for value in C_values) != C_GRID:
        raise ValueError("membership report requires the fixed C grid")
    source = _trial_ids(source_trials, "source_trials")
    personal = _trial_ids(personal_label_trials, "personal_label_trials")
    reference = _trial_ids(target_reference_trials, "target_reference_trials")
    evaluation = _trial_ids(evaluation_trials, "evaluation_trials")
    if set(source) & set(personal) or set(source) & set(reference):
        raise ValueError("source trials overlap target history")
    if personal and not set(personal) <= set(reference):
        raise ValueError("personal labels must be a subset of the target session-1 reference")
    if (set(source) | set(personal) | set(reference)) & set(evaluation):
        raise ValueError("evaluation trials appear in a fitted information role")
    representation_ids = source if representation_mode == "source_frozen" else source + personal
    roles = {
        "source_trials": source,
        "personal_label_trials": personal,
        "target_reference_trials": reference,
        "evaluation_trials": evaluation,
        "representation_fit_trials": representation_ids,
        "classifier_fit_trials": source + personal,
    }
    return {
        "representation_mode": representation_mode,
        "C_values": list(C_GRID),
        "roles": {name: {"count": len(values), "sha256": _id_hash(values), "trial_ids": values}
                  for name, values in roles.items()},
        "target_reference_is_label_free": True,
        "evaluation_used_for_fit_or_selection": False,
    }
