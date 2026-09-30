"""Deterministic development-only controls adopted after review 2."""
from collections import Counter

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .baselines import nested_history_indices, validate_epochs


SOURCE_SIZES_A = (40, 70, 90, 100)
SOURCE_SIZES_B = (100, 1800)
BASE_SIZES_C = (40, 100)
C_GRID = (0.1, 1.0, 10.0)
LAMBDA_GRID = ("pooled", 0.25, 0.5, 0.75)
REPRESENTATIONS = ("source_frozen", "pooled_refit")
DONOR_KINDS = ("own", "pooled", "single_0", "single_1", "single_2")
DONOR_SELECTION_SEED = 2026092301
DONOR_BLOCK_SEED = 2026092302


def _number(value):
    return str(value).replace(".", "p")


def operations(cfg):
    _validate_cfg(cfg)
    subjects = _subjects(cfg)
    result = []

    def add(module, source_n, h_total, C, lambda_personal, representation,
            draw, target, donor_kind):
        target_text = "shared" if target is None else f"s{target:03d}"
        donor_text = "none" if donor_kind is None else donor_kind
        operation_id = (f"review2_{module}__S{source_n}_h{h_total}_C{_number(C)}_"
                        f"lam{_number(lambda_personal)}_{representation}_r{draw}_"
                        f"{target_text}_{donor_text}")
        result.append(dict(operation_id=operation_id, module=module, source_n=source_n,
                           h_total=h_total, C=C, lambda_personal=lambda_personal,
                           representation=representation, draw=draw, target=target,
                           donor_kind=donor_kind))

    for source_n in cfg["A_source_sizes"]:
        for C in cfg["C_grid"]:
            for draw in range(cfg["draws"]):
                add("A", source_n, 0, C, "pooled", "source_frozen", draw, None, None)
    for source_n in cfg["B_source_sizes"]:
        for lambda_personal in cfg["lambda_grid"]:
            for representation in cfg["representations"]:
                for C in cfg["C_grid"]:
                    for draw in range(cfg["draws"]):
                        for target in subjects:
                            add("B", source_n, 60, C, lambda_personal,
                                representation, draw, target, "own")
    for source_n in cfg["C_base_sizes"]:
        for donor_kind in cfg["donor_kinds"]:
            for draw in range(cfg["draws"]):
                for target in subjects:
                    add("C", source_n, 60 if donor_kind == "own" else 0,
                        1.0, "pooled", "pooled_refit",
                        draw, target, donor_kind)
    for h_total in cfg["D_h_total"]:
        for draw in range(cfg["draws"]):
            for target in subjects:
                add("D", cfg["D_source_n"], h_total, 1.0, "pooled",
                    cfg["D_representation"], draw,
                    target, "own" if h_total else None)
    if (len(result) != cfg["expected_operations"]
            or len({row["operation_id"] for row in result}) != cfg["expected_operations"]):
        raise ValueError("review-2 operation grid differs")
    return result


def _subjects(cfg):
    subjects = cfg.get("development_subjects")
    if not isinstance(subjects, list) or len(subjects) != 12 or len(set(subjects)) != 12:
        raise ValueError("review-2 config requires 12 unique development_subjects")
    return subjects


def _validate_cfg(cfg):
    expected = {
        "draws": 2, "A_source_sizes": list(SOURCE_SIZES_A), "C_grid": list(C_GRID),
        "B_source_sizes": list(SOURCE_SIZES_B), "lambda_grid": list(LAMBDA_GRID),
        "representations": list(REPRESENTATIONS), "C_base_sizes": list(BASE_SIZES_C),
        "donor_kinds": list(DONOR_KINDS), "donor_choice_seed": DONOR_SELECTION_SEED,
        "donor_trial_seed": DONOR_BLOCK_SEED, "D_h_total": [0, 60], "h_total": 60,
        "D_source_n": 1800, "D_unlabeled_history": 100,
        "D_representation": "source_frozen", "expected_operations": 1464,
    }
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"unexpected review-2 config value: {key}")
    source_subjects = cfg.get("source_subjects")
    if (not isinstance(source_subjects, list) or len(source_subjects) != 18
            or len(set(source_subjects)) != 18 or set(source_subjects) & set(_subjects(cfg))):
        raise ValueError("review-2 config requires 18 disjoint source_subjects")


def _validated_split(split, oldcfg):
    if not isinstance(split, dict) or len(split) != 4200:
        raise ValueError("review-2 membership requires the exact 4200-row old split")
    source_subjects = set(oldcfg["source_subjects"])
    development_subjects = set(oldcfg["development_subjects"])
    if len(source_subjects) != 18 or len(development_subjects) != 12 or source_subjects & development_subjects:
        raise ValueError("unexpected frozen participant roles")
    counts = Counter()
    normalized = {}
    for key, raw in split.items():
        if raw.get("trial_id") != key:
            raise ValueError("split key/trial_id mismatch")
        try:
            subject, session, label = int(raw["subject_id"]), int(raw["session"]), int(raw["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid split fields") from error
        role = raw.get("role")
        expected_role = "source" if subject in source_subjects else (
            "development" if subject in development_subjects else None)
        allowed_sessions = {1} if expected_role == "source" else {1, 2}
        if role != expected_role or session not in allowed_sessions or label not in (0, 1):
            raise ValueError("extra participant/session, illegal role, or invalid label in old split")
        counts[role, subject, session, label] += 1
        normalized[key] = dict(trial_id=key, role=role, subject_id=subject,
                               session=session, label=label)
    expected_cells = ({("source", subject, 1, label) for subject in source_subjects for label in (0, 1)}
                      | {("development", subject, session, label)
                         for subject in development_subjects for session in (1, 2) for label in (0, 1)})
    if set(counts) != expected_cells or any(counts[cell] != 50 for cell in expected_cells):
        raise ValueError("old split must contain exactly 50 trials per allowed class cell")
    return normalized


def _source_permutations(split, oldcfg, draw):
    if draw not in (0, 1):
        raise ValueError("draw must be 0 or 1")
    source = sorted(trial_id for trial_id, row in split.items()
                    if row["role"] == "source" and row["session"] == 1)
    rng = np.random.default_rng(oldcfg["source_draw_seeds"][draw])
    return {label: rng.permutation([trial_id for trial_id in source
                                    if split[trial_id]["label"] == label]).tolist()
            for label in (0, 1)}


def _source_prefix(permutations, total):
    if total == 1800:
        return sorted(permutations[0] + permutations[1])
    if total not in SOURCE_SIZES_A:
        raise ValueError("unsupported balanced source size")
    per_class = total // 2
    return permutations[0][:per_class] + permutations[1][:per_class]


def _own_history(split, oldcfg, draw, target, total=60):
    if total == 0:
        return []
    if total != 60:
        raise ValueError("only the frozen 60-trial own-history block is supported")
    pool = sorted(trial_id for trial_id, row in split.items()
                  if row["role"] == "development" and row["subject_id"] == target
                  and row["session"] == 1)
    labels = np.asarray([split[trial_id]["label"] for trial_id in pool])
    seed = np.random.SeedSequence([oldcfg["history_draw_seeds"][draw], target])
    return [pool[index] for index in nested_history_indices(labels, 30, seed)]


def _single_donors(split, oldcfg, cfg, draw, target, permutations):
    base100 = set(_source_prefix(permutations, 100))
    eligible = []
    for subject in sorted(oldcfg["source_subjects"]):
        enough = all(sum(1 for trial_id, row in split.items()
                         if row["role"] == "source" and row["subject_id"] == subject
                         and row["session"] == 1 and row["label"] == label
                         and trial_id not in base100) >= 30 for label in (0, 1))
        if enough:
            eligible.append(subject)
    if len(eligible) < 3:
        raise ValueError("fewer than three source donors have 30/class outside base100")
    rng = np.random.default_rng(np.random.SeedSequence([cfg["donor_choice_seed"], draw, target]))
    return rng.permutation(eligible)[:3].tolist(), base100


def _single_donor_block(split, cfg, draw, target, donor, base100):
    rng = np.random.default_rng(np.random.SeedSequence(
        [cfg["donor_trial_seed"], draw, target, donor]))
    block = []
    for label in (0, 1):
        pool = sorted(trial_id for trial_id, row in split.items()
                      if row["role"] == "source" and row["subject_id"] == donor
                      and row["session"] == 1 and row["label"] == label
                      and trial_id not in base100)
        if len(pool) < 30:
            raise ValueError("selected single donor lacks 30/class outside base100")
        block.extend(rng.permutation(pool)[:30].tolist())
    return block


def membership(op, split, oldcfg, cfg):
    _validate_cfg(cfg)
    rows = _validated_split(split, oldcfg)
    if _subjects(cfg) != oldcfg["development_subjects"]:
        raise ValueError("new config development subjects differ from frozen config")
    expected_fields = {"operation_id", "module", "source_n", "h_total", "C",
                       "lambda_personal", "representation", "draw", "target", "donor_kind"}
    if set(op) != expected_fields or op not in operations(cfg):
        raise ValueError("operation is outside the fixed review-2 grid")
    module, draw, target = op["module"], op["draw"], op["target"]
    permutations = _source_permutations(rows, oldcfg, draw)
    source = _source_prefix(permutations, op["source_n"])
    added, unlabeled, donors = [], [], []

    if module == "A":
        evaluation = sorted(trial_id for trial_id, row in rows.items()
                            if row["role"] == "development" and row["session"] == 2)
    else:
        evaluation = sorted(trial_id for trial_id, row in rows.items()
                            if row["role"] == "development" and row["subject_id"] == target
                            and row["session"] == 2)
        if module == "B":
            added = _own_history(rows, oldcfg, draw, target)
            donors = [target]
        elif module == "C":
            kind = op["donor_kind"]
            if kind == "own":
                added = _own_history(rows, oldcfg, draw, target)
                donors = [target]
            elif kind == "pooled":
                start = op["source_n"] // 2
                added = permutations[0][start:start + 30] + permutations[1][start:start + 30]
                donors = sorted({rows[trial_id]["subject_id"] for trial_id in added})
            elif kind in ("single_0", "single_1", "single_2"):
                selected, base100 = _single_donors(rows, oldcfg, cfg, draw, target, permutations)
                donor = selected[int(kind[-1])]
                added = _single_donor_block(rows, cfg, draw, target, donor, base100)
                donors = [donor]
            else:
                raise ValueError("invalid module-C donor kind")
        elif module == "D":
            unlabeled = sorted(trial_id for trial_id, row in rows.items()
                               if row["role"] == "development" and row["subject_id"] == target
                               and row["session"] == 1)
            if op["h_total"] == 60:
                added = _own_history(rows, oldcfg, draw, target)
                donors = [target]
        else:
            raise ValueError("invalid module")

    expected_added = 60 if module == "C" else op["h_total"]
    if (len(source) != op["source_n"] or len(added) != expected_added
            or len(evaluation) != (1200 if module == "A" else 100)):
        raise ValueError("membership count differs")
    if len(source) != len(set(source)) or len(added) != len(set(added)) or len(evaluation) != len(set(evaluation)):
        raise ValueError("duplicate membership trial")
    if set(source) & set(added) or set(source) & set(evaluation) or set(added) & set(evaluation):
        raise ValueError("training/evaluation memberships overlap")
    if set(unlabeled) & set(evaluation) or len(unlabeled) != len(set(unlabeled)):
        raise ValueError("unlabeled history overlaps evaluation or contains duplicates")
    if module == "D" and (len(unlabeled) != cfg["D_unlabeled_history"]
                          or not set(added) <= set(unlabeled)):
        raise ValueError("EA must use all 100 historical signals and only allowed history labels")
    if any(rows[trial_id]["role"] != "source" or rows[trial_id]["session"] != 1 for trial_id in source):
        raise ValueError("source membership role/session differs")
    if any(rows[trial_id]["role"] != "development" or rows[trial_id]["session"] != 2
           for trial_id in evaluation):
        raise ValueError("evaluation membership role/session differs")
    if added and Counter(rows[trial_id]["label"] for trial_id in added) != {0: 30, 1: 30}:
        raise ValueError("added block is not exactly 30/class")
    return source, added, evaluation, unlabeled, donors


def oas_covariances(epochs):
    epochs = np.asarray(epochs)
    validate_epochs(epochs, np.zeros(len(epochs), dtype=int), require_both_classes=False)
    covariances = Covariances(estimator="oas").fit_transform(epochs)
    if not np.isfinite(covariances).all():
        raise ValueError("OAS covariance produced nonfinite values")
    return covariances


def fit_representation(cov_train, source_count, mode):
    cov_train = np.asarray(cov_train, dtype=float)
    if (cov_train.ndim != 3 or cov_train.shape[1] != cov_train.shape[2]
            or not np.isfinite(cov_train).all() or not 0 < source_count <= len(cov_train)):
        raise ValueError("invalid covariance training array/source_count")
    if mode not in REPRESENTATIONS:
        raise ValueError("unknown representation mode")
    fit_values = cov_train[:source_count] if mode == "source_frozen" else cov_train
    representation = make_pipeline(TangentSpace(metric="riemann", tsupdate=False), StandardScaler())
    representation.fit(fit_values)
    return representation


def classifier_sample_weights(n_samples, source_count, lambda_personal):
    if (isinstance(n_samples, bool) or isinstance(source_count, bool)
            or not isinstance(n_samples, int) or not isinstance(source_count, int)
            or not 0 < source_count <= n_samples):
        raise ValueError("invalid sample/source count")
    history_count = n_samples - source_count
    if lambda_personal == "pooled":
        return np.ones(n_samples, dtype=float)
    if history_count == 0 or isinstance(lambda_personal, bool) or not isinstance(lambda_personal, (int, float)):
        raise ValueError("numeric personal weight requires personal samples")
    value = float(lambda_personal)
    if not 0 <= value <= 1:
        raise ValueError("lambda_personal must lie in [0,1]")
    total = n_samples
    weights = np.empty(n_samples, dtype=float)
    weights[:source_count] = total * (1 - value) / source_count
    weights[source_count:] = total * value / history_count
    if not np.isclose(weights.sum(), total, rtol=0, atol=1e-12):
        raise ValueError("classifier sample weights do not sum to S+h")
    return weights


def train_classifier(ztrain, y, source_count, C, lambda_personal):
    ztrain, y = np.asarray(ztrain, dtype=float), np.asarray(y)
    if (ztrain.ndim != 2 or y.shape != (len(ztrain),) or not np.isfinite(ztrain).all()
            or set(y.tolist()) != {0, 1} or not isinstance(C, (int, float))
            or isinstance(C, bool) or not math_is_positive_finite(C)):
        raise ValueError("invalid classifier training inputs")
    weights = classifier_sample_weights(len(ztrain), source_count, lambda_personal)
    classifier = LogisticRegression(penalty="l2", C=float(C), solver="lbfgs", max_iter=1000,
                                    random_state=20260914)
    classifier.fit(ztrain, y, sample_weight=weights)
    return classifier


def math_is_positive_finite(value):
    return bool(np.isfinite(value) and float(value) > 0)


def fit_ea_whitener(unlabeled_epochs):
    epochs = np.asarray(unlabeled_epochs, dtype=float)
    validate_epochs(epochs, np.zeros(len(epochs), dtype=int), require_both_classes=False)
    if len(epochs) != 100 or epochs.shape[2] < 2:
        raise ValueError("EA requires exactly 100 allowed domain trials and at least two time samples")
    mean_covariance = np.mean([np.cov(epoch, rowvar=True, ddof=1) for epoch in epochs], axis=0)
    mean_covariance = (mean_covariance + mean_covariance.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
    maximum = float(eigenvalues.max())
    if not np.isfinite(eigenvalues).all() or maximum <= 0 or eigenvalues.min() <= 1e-12 * maximum:
        raise ValueError("EA mean covariance is singular or ill-conditioned; no jitter permitted")
    whitener = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return (whitener + whitener.T) / 2


def apply_ea(epochs, whitener):
    epochs, whitener = np.asarray(epochs, dtype=float), np.asarray(whitener, dtype=float)
    validate_epochs(epochs, np.zeros(len(epochs), dtype=int), require_both_classes=False)
    if (whitener.shape != (epochs.shape[1], epochs.shape[1]) or not np.isfinite(whitener).all()
            or not np.allclose(whitener, whitener.T, rtol=0, atol=1e-12)):
        raise ValueError("invalid frozen EA whitener")
    transformed = np.einsum("ij,njt->nit", whitener, epochs)
    if not np.isfinite(transformed).all():
        raise ValueError("EA transform produced nonfinite epochs")
    return transformed


def fit_source_ea(epochs, source_subject_ids):
    epochs, source_subject_ids = np.asarray(epochs), np.asarray(source_subject_ids)
    if source_subject_ids.shape != (len(epochs),):
        raise ValueError("one source subject ID is required per epoch")
    subjects = np.unique(source_subject_ids)
    if (len(subjects) != 18
            or any(np.count_nonzero(source_subject_ids == subject) != 100 for subject in subjects)):
        raise ValueError("source EA requires all 18 source people with exactly 100 trials each")
    return {int(subject): fit_ea_whitener(epochs[source_subject_ids == subject]) for subject in subjects}


def apply_source_ea(epochs, source_subject_ids, whiteners):
    epochs, source_subject_ids = np.asarray(epochs), np.asarray(source_subject_ids)
    if source_subject_ids.shape != (len(epochs),):
        raise ValueError("one source subject ID is required per epoch")
    transformed = np.empty_like(epochs, dtype=float)
    for subject in np.unique(source_subject_ids):
        if int(subject) not in whiteners:
            raise ValueError("missing frozen source-person EA whitener")
        mask = source_subject_ids == subject
        transformed[mask] = apply_ea(epochs[mask], whiteners[int(subject)])
    return transformed
