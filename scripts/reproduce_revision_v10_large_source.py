#!/usr/bin/env python3
"""Portable replay of the saved v10 large-source donor analysis."""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

SCHEMA = "revision-v10-large-source-v1"
SEED, BOOT_DRAWS, TOL = 2026092421, 10000, 1e-10
COHORTS = {
    "openbmi_development": (12, 100, 1800, 1620),
    "bnci_loso": (9, 144, 1152, 972),
    "openbmi_confirmation": (24, 100, 1800, 1620),
}
FIT_KINDS = ("large_base", "large_own", "large_single_0", "large_single_1",
             "large_single_2", "full_base", "full_own")
FIT_CONDITION = {kind: kind.split("_", 1)[1] for kind in FIT_KINDS}
PRED_FIELDS = ["operation_id", "cohort", "target", "draw", "fit_kind", "source_n",
               "added_n", "C", "trial_id", "y_true", "y_pred", "p0", "p1",
               "base_membership", "added_membership", "evaluation_membership"]
NUMERIC_FIELDS = [
    "large_base_ba", "large_own_ba", "large_single0_ba", "large_single1_ba",
    "large_single2_ba", "large_mean_single_ba", "large_own_minus_mean_single",
    "small_base_ba", "small_own_ba", "small_mean_single_ba",
    "small_own_minus_mean_single", "large_base_minus_small_base",
    "large_own_minus_small_own", "large_mean_single_minus_small_mean_single",
    "large_minus_small_effect", "full_base_ba", "full_own_ba", "full_own_gain"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def finite_float(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(label + " is nonfinite")
    return result


def validate_plan(operations_path, memberships_path):
    operations, memberships = load_gzip_json(operations_path), load_gzip_json(memberships_path)
    if not isinstance(operations, list) or not isinstance(memberships, dict):
        raise ValueError("plan JSON types differ")
    if any(fingerprint(values) != key or len(values) != len(set(values))
           for key, values in memberships.items()):
        raise ValueError("membership fingerprint or uniqueness differs")
    by_id, contexts = {}, defaultdict(dict)
    for op in operations:
        op_id, cohort, kind = op["operation_id"], op["cohort"], op["fit_kind"]
        target, draw = int(op["target"]), int(op["draw"])
        if (op_id in by_id or cohort not in COHORTS or kind not in FIT_KINDS
                or draw not in range(10) or op.get("condition") != FIT_CONDITION[kind]
                or set(op.get("memberships", {})) != {"base", "added", "evaluation"}
                or any(ref not in memberships for ref in op["memberships"].values())):
            raise ValueError("operation identity/inventory differs")
        _, evaluation_n, full_n, large_n = COHORTS[cohort]
        source_n = large_n if kind.startswith("large_") else full_n
        added_n = 0 if kind.endswith("_base") else 60
        if (int(op["source_n"]) != source_n or int(op["added_n"]) != added_n
                or finite_float(op["C"], "C") not in (0.1, 1.0, 10.0)):
            raise ValueError("operation source/addition/C differs")
        base = memberships[op["memberships"]["base"]]
        added = memberships[op["memberships"]["added"]]
        evaluation = memberships[op["memberships"]["evaluation"]]
        if (len(base) != source_n or len(added) != added_n or len(evaluation) != evaluation_n
                or set(base) & set(added) or set(base) & set(evaluation)
                or set(added) & set(evaluation)):
            raise ValueError("operation membership sizes/disjointness differ")
        by_id[op_id] = op
        key = (cohort, target, draw)
        if kind in contexts[key]:
            raise ValueError("duplicate context fit kind")
        contexts[key][kind] = op
    expected_ops = {"openbmi_development": 840, "bnci_loso": 630,
                    "openbmi_confirmation": 1680}
    if len(by_id) != 3150 or Counter(op["cohort"] for op in operations) != expected_ops:
        raise ValueError("3150-operation cohort inventory differs")
    for cohort, (people, _, full_n, large_n) in COHORTS.items():
        cohort_keys = [key for key in contexts if key[0] == cohort]
        if len(cohort_keys) != people * 10 or len({key[1] for key in cohort_keys}) != people:
            raise ValueError("participant/draw context coverage differs")
        for target in {key[1] for key in cohort_keys}:
            refs = {contexts[key]["large_base"]["memberships"]["evaluation"]
                    for key in cohort_keys if key[1] == target}
            if len(refs) != 1: raise ValueError("evaluation membership changes across draws")
    for key, group in contexts.items():
        if set(group) != set(FIT_KINDS) or len({float(op["C"]) for op in group.values()}) != 1:
            raise ValueError("paired fit inventory/C differs")
        refs = lambda kind, which: group[kind]["memberships"][which]
        if (len({refs(kind, "evaluation") for kind in FIT_KINDS}) != 1
                or len({refs(kind, "base") for kind in FIT_KINDS[:5]}) != 1
                or len({refs(kind, "base") for kind in FIT_KINDS[5:]}) != 1):
            raise ValueError("paired base/evaluation membership differs")
        large = set(memberships[refs("large_base", "base")])
        full = set(memberships[refs("full_base", "base")])
        singles = [set(memberships[refs(f"large_single_{i}", "added")]) for i in range(3)]
        own = set(memberships[refs("large_own", "added")])
        reserved = set().union(*singles)
        if (len(reserved) != 180 or any(singles[i] & singles[j] for i in range(3) for j in range(i))
                or own & reserved or not large < full or full - large != reserved):
            raise ValueError("large/full/reserved donor membership relation differs")
        if (refs("large_own", "added") != refs("full_own", "added")
                or refs("large_base", "added") != refs("full_base", "added")):
            raise ValueError("large/full paired added membership differs")
    return operations, memberships, by_id, contexts


def load_old_scores(path):
    result = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"operation_id", "condition", "draw", "C", "balanced_accuracy", "evaluation_n"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("small-score schema differs: " + str(path))
        for row in reader:
            op_id = row["operation_id"]
            if op_id in result:
                raise ValueError("duplicate small-score operation")
            result[op_id] = row
    return result


def reconstruct_predictions(path, operations, memberships, by_id):
    state = {op_id: {"seen": set(), "correct": [0, 0], "total": [0, 0]}
             for op_id in by_id}
    evaluation_keys = {op["memberships"]["evaluation"] for op in operations}
    evaluation_sets = {key: set(memberships[key]) for key in evaluation_keys}
    truth_by_eval, rows = {}, 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != PRED_FIELDS:
            raise ValueError("prediction schema differs")
        for row in reader:
            rows += 1
            op = by_id.get(row["operation_id"])
            if op is None:
                raise ValueError("unknown prediction operation")
            current, trial = state[op["operation_id"]], row["trial_id"]
            y, pred = int(row["y_true"]), int(row["y_pred"])
            p0, p1 = finite_float(row["p0"], "p0"), finite_float(row["p1"], "p1")
            metadata_ok = (row["cohort"] == op["cohort"] and int(row["target"]) == int(op["target"])
                and int(row["draw"]) == int(op["draw"]) and row["fit_kind"] == op["fit_kind"]
                and int(row["source_n"]) == int(op["source_n"])
                and int(row["added_n"]) == int(op["added_n"])
                and abs(float(row["C"]) - float(op["C"])) <= TOL
                and row["base_membership"] == op["memberships"]["base"]
                and row["added_membership"] == op["memberships"]["added"]
                and row["evaluation_membership"] == op["memberships"]["evaluation"])
            if (not metadata_ok or trial in current["seen"] or y not in (0, 1) or pred not in (0, 1)
                    or not 0 <= p0 <= 1 or not 0 <= p1 <= 1 or abs(p0 + p1 - 1) > TOL
                    or pred != (0 if p0 >= p1 else 1)):
                raise ValueError("prediction identity/probability/duplicate differs")
            if trial not in evaluation_sets[op["memberships"]["evaluation"]]:
                raise ValueError("prediction trial outside evaluation membership")
            prior = truth_by_eval.setdefault(op["memberships"]["evaluation"], {})
            if trial in prior and prior[trial] != y:
                raise ValueError("evaluation truth differs across operations")
            prior[trial] = y
            current["seen"].add(trial); current["correct"][y] += int(pred == y); current["total"][y] += 1
    if rows != 342720:
        raise ValueError("prediction row count differs")
    scores = {}
    for op_id, current in state.items():
        op = by_id[op_id]; expected = evaluation_sets[op["memberships"]["evaluation"]]
        evaluation_n = COHORTS[op["cohort"]][1]
        if current["seen"] != expected or current["total"] != [evaluation_n // 2] * 2:
            raise ValueError("evaluation coverage/class balance differs")
        scores[op_id] = sum(current["correct"][i] / current["total"][i] for i in (0, 1)) / 2
    return rows, scores


def derive(scores, contexts, old_by_cohort):
    draws = []
    for cohort in COHORTS:
        for key in sorted(k for k in contexts if k[0] == cohort):
            _, target, draw = key; group = contexts[key]
            get = lambda kind: scores[group[kind]["operation_id"]]
            singles = [get(f"large_single_{i}") for i in range(3)]
            old = old_by_cohort[cohort]
            old_rows = {}
            for condition in ("base", "own", "single_0", "single_1", "single_2"):
                linked = group[f"large_{condition}"]["old_small_operation_id"]
                row = old.get(linked)
                old_ba = finite_float(row["balanced_accuracy"], "small BA") if row is not None else math.nan
                if (row is None or row["condition"] != condition or int(row["draw"]) != draw
                        or int(row["evaluation_n"]) != COHORTS[cohort][1]
                        or abs(finite_float(row["C"], "small C") - float(group["large_base"]["C"])) > TOL
                        or not 0 <= old_ba <= 1
                        or ("target" in row and int(row["target"]) != target)
                        or ("participant_id" in row and int(row["participant_id"]) != target)):
                    raise ValueError("linked small-score identity differs")
                old_rows[condition] = old_ba
            mean_single, small_mean = float(np.mean(singles)), float(np.mean(
                [old_rows[f"single_{i}"] for i in range(3)]))
            own, small_own = get("large_own"), old_rows["own"]
            draws.append({"cohort": cohort, "participant_id": target, "draw": draw,
                "large_base_ba": get("large_base"), "large_own_ba": own,
                **{f"large_single{i}_ba": singles[i] for i in range(3)},
                "large_mean_single_ba": mean_single, "large_own_minus_mean_single": own-mean_single,
                "small_base_ba": old_rows["base"], "small_own_ba": small_own,
                "small_mean_single_ba": small_mean, "small_own_minus_mean_single": small_own-small_mean,
                "large_base_minus_small_base": get("large_base")-old_rows["base"],
                "large_own_minus_small_own": own-small_own,
                "large_mean_single_minus_small_mean_single": mean_single-small_mean,
                "large_minus_small_effect": (own-mean_single)-(small_own-small_mean),
                "full_base_ba": get("full_base"), "full_own_ba": get("full_own"),
                "full_own_gain": get("full_own")-get("full_base")})
    grouped = defaultdict(list)
    for row in draws: grouped[(row["cohort"], row["participant_id"])].append(row)
    people = []
    for (cohort, participant), rows in sorted(grouped.items()):
        if len(rows) != 10 or {row["draw"] for row in rows} != set(range(10)):
            raise ValueError("ten-draw participant coverage differs")
        people.append({"cohort": cohort, "participant_id": participant,
                       **{name: float(np.mean([row[name] for row in rows])) for name in NUMERIC_FIELDS}})
    rng = np.random.default_rng(SEED)
    summary = {"schema": SCHEMA, "status": "completed_descriptive_posthoc",
        "units": "balanced accuracy; contrasts are BA differences",
        "bootstrap": {"seed": SEED, "draws": BOOT_DRAWS,
            "resampling_unit": "participant after averaging 10 draws", "interval": "percentile descriptive"},
        "cohorts": {}}
    for cohort in COHORTS:  # insertion order is part of the frozen RNG recipe
        rows = [row for row in people if row["cohort"] == cohort]; n = len(rows)
        indices = rng.integers(0, n, size=(BOOT_DRAWS, n))
        summary["cohorts"][cohort] = {}
        for name in NUMERIC_FIELDS:
            x = np.asarray([row[name] for row in rows]); mean = float(x.mean())
            half = float(student_t.ppf(.975, n-1) * x.std(ddof=1) / math.sqrt(n))
            boot = np.mean(x[indices], axis=1)
            summary["cohorts"][cohort][name] = {"n": n, "mean": mean,
                "median": float(np.median(x)), "standard_deviation": float(x.std(ddof=1)),
                "minimum": float(x.min()), "maximum": float(x.max()),
                "t95_low": mean-half, "t95_high": mean+half,
                "bootstrap95_low": float(np.percentile(boot, 2.5)),
                "bootstrap95_high": float(np.percentile(boot, 97.5)),
                "positive": int(np.sum(x > 1e-12)), "zero": int(np.sum(np.abs(x) <= 1e-12)),
                "negative": int(np.sum(x < -1e-12))}
    return people, summary


def compare_tree(actual, saved, path="root"):
    if isinstance(actual, dict):
        if not isinstance(saved, dict) or set(actual) != set(saved): raise ValueError(path + " keys differ")
        for key in actual: compare_tree(actual[key], saved[key], path + "." + key)
    elif isinstance(actual, float):
        if not isinstance(saved, (int, float)) or not math.isfinite(float(saved)) or abs(actual-float(saved)) > TOL:
            raise ValueError(path + " numeric value differs")
    elif actual != saved:
        raise ValueError(path + " value differs")


def validate_people(path, people):
    expected_fields = ["cohort", "participant_id", *NUMERIC_FIELDS]
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields: raise ValueError("participant-effect schema differs")
        saved = list(reader)
    if len(saved) != 45: raise ValueError("participant-effect row count differs")
    indexed = {(row["cohort"], int(row["participant_id"])): row for row in saved}
    if len(indexed) != 45: raise ValueError("duplicate participant-effect row")
    for row in people:
        found = indexed.get((row["cohort"], row["participant_id"]))
        if found is None: raise ValueError("participant-effect identity differs")
        for name in NUMERIC_FIELDS:
            if abs(row[name] - finite_float(found[name], name)) > TOL:
                raise ValueError("participant-effect value differs: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictions", "operations", "memberships", "small-openbmi", "small-bnci",
                 "small-confirmation", "summary", "participant-effects", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists(): raise FileExistsError("--out must be a new path")
    operations, memberships, by_id, contexts = validate_plan(args.operations, args.memberships)
    old = {"openbmi_development": load_old_scores(args.small_openbmi),
           "bnci_loso": load_old_scores(args.small_bnci),
           "openbmi_confirmation": load_old_scores(args.small_confirmation)}
    prediction_rows, scores = reconstruct_predictions(args.predictions, operations, memberships, by_id)
    people, calculated = derive(scores, contexts, old)
    validate_people(args.participant_effects, people)
    saved_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    compare_tree(calculated, saved_summary)
    input_names = ("predictions", "operations", "memberships", "small_openbmi", "small_bnci",
                   "small_confirmation", "summary", "participant_effects")
    receipt = {"schema": "portable-revision-v10-large-source-v1",
        "status": "complete_exact_saved_output_reproduction",
        "inputs": {name: {"sha256": sha256(getattr(args, name)),
                           "bytes": getattr(args, name).stat().st_size} for name in input_names},
        "counts": {"operations": len(scores), "prediction_rows": prediction_rows,
                   "contexts": len(contexts), "participants": len(people)},
        "operation_ba_sha256": hashlib.sha256(json.dumps(scores, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(),
        "comparison_tolerance": TOL, "summary": calculated,
        "script_sha256": sha256(__file__),
        "scope": {"reconstructs": "saved prediction BA, linked small-source contrasts, participant means, t intervals, and frozen bootstrap",
                  "does_not_reconstruct": "raw EEG preparation, training fits, source representation, C selection, or training-label balance",
                  "membership_checks": "IDs, sizes, uniqueness, evaluation coverage, base/add/evaluation disjointness, and full-minus-large donor relation",
                  "audit_status": "Saved-output reproduction; not an independent training rerun"}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "operations": len(scores)}, sort_keys=True))


if __name__ == "__main__":
    main()
