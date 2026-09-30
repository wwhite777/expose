#!/usr/bin/env python3
"""Analyze completed Review-3 donor, neural, and alignment extensions."""

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_review3 import effect, holm


SCHEMA = "review3-extension-analysis-inputs-v1"
TOLERANCE = 1e-12
DONOR_SETTINGS = ("legacy_C1_pooled_refit", "tuned_C_source_frozen")
DONOR_CONDITIONS = ("base", "own", "pooled", "single_0", "single_1", "single_2")
ALIGNMENT_PAIRS = (
    ("R2_current_prefix20_tail", "R1_historical_prefix_tail", "current_minus_historical_tail"),
    ("R2_current_all_transductive", "R1_historical_all", "current_minus_historical_all"),
)
ALIGNMENT_LABEL_GAIN_REGIMES = (
    ("R1_historical_prefix_tail", "historical_label_gain_prefix_tail"),
    ("R2_current_prefix20_tail", "current_label_gain_prefix20_tail"),
    ("R1_historical_all", "historical_label_gain_all"),
    ("R2_current_all_transductive", "current_label_gain_all_transductive"),
)
ALIGNMENT_FULL_SOURCE = {"openbmi8": "1800", "bnci": "all"}
BNCI_CORE_FAMILY = (
    "plain_label_gain_S100", "source_context_interaction",
    "mdwm_minus_plain_Sall_h60", "historical_EA_minus_plain_Sall_h0",
    "alignment_label_interaction_Sall",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_path(relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("input paths must be repository-relative strings")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError("unsafe input path")
    path = (ROOT / pure).resolve()
    if ROOT.resolve() not in path.parents:
        raise ValueError("input path escapes repository")
    return path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_spec(spec):
    required = {"schema", "core_analysis", "donors", "neural", "current_alignment"}
    if set(spec) != required or spec["schema"] != SCHEMA:
        raise ValueError("invalid extension-analysis input schema")
    core = spec["core_analysis"]
    if set(core) != {"directory", "summary_sha256", "reconstructed_scores_sha256"}:
        raise ValueError("invalid core-analysis binding")
    for name in ("summary_sha256", "reconstructed_scores_sha256"):
        if not isinstance(core[name], str) or len(core[name]) != 64:
            raise ValueError("invalid core-analysis hash")
    labels = set()
    for section in ("donors", "neural", "current_alignment"):
        if not isinstance(spec[section], list):
            raise ValueError(section + " must be a list")
        for item in spec[section]:
            if set(item) != {"label", "directory", "required"}:
                raise ValueError("invalid extension input entry")
            if (not isinstance(item["label"], str) or not item["label"]
                    or not isinstance(item["required"], bool)):
                raise ValueError("invalid extension label or requirement flag")
            key = (section, item["label"])
            if key in labels:
                raise ValueError("duplicate extension input label")
            labels.add(key)
            safe_path(item["directory"])
    safe_path(core["directory"])
    return spec


def completed_directory(item, kind):
    directory = safe_path(item["directory"])
    if not directory.exists():
        if item["required"]:
            raise FileNotFoundError(f"required {kind} output is missing: {directory}")
        return None, {"status": "missing_optional", "directory": item["directory"]}
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"{kind} output is not a regular directory")
    receipt_path, completed_path = directory / "receipt.json", directory / "COMPLETED.json"
    if not receipt_path.is_file() or not completed_path.is_file():
        raise ValueError(f"present {kind} output lacks completion records")
    receipt, completed = load_json(receipt_path), load_json(completed_path)
    completion_status_ok = (completed.get("status") == "completed"
                            or (kind == "neural" and "status" not in completed))
    if (receipt.get("status") != "completed" or not completion_status_ok
            or completed.get("receipt_sha256") != sha256(receipt_path)):
        raise ValueError(f"{kind} output is incomplete or its receipt changed")
    return directory, {"status": "completed", "directory": item["directory"],
                       "receipt_sha256": sha256(receipt_path)}


def bound_file(directory, receipt, name, completed=None, completed_key=None):
    path = directory / name
    if not path.is_file() or path.is_symlink():
        raise ValueError("missing bound output: " + name)
    digest = sha256(path)
    if receipt.get("outputs", {}).get(name) == digest:
        return path, digest
    if completed is not None and completed_key and completed.get(completed_key) == digest:
        return path, digest
    raise ValueError("output hash binding failed: " + name)


def balanced_accuracy(truth, predicted):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if truth.shape != predicted.shape or set(truth.tolist()) != {0, 1}:
        raise ValueError("balanced accuracy needs both binary classes")
    return float(np.mean([np.mean(predicted[truth == value] == value) for value in (0, 1)]))


def reconstruct_scores(path, key_fields, probability_fields, sum_tolerance=1e-8):
    grouped = defaultdict(list)
    for row in read_rows(path):
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)
    result, trial_sets = {}, {}
    for key, rows in grouped.items():
        trial_ids = [row["trial_id"] for row in rows]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("duplicate trial prediction within operation")
        truth = np.asarray([int(row["y_true"]) for row in rows])
        predicted = np.asarray([int(row["y_pred"]) for row in rows])
        probabilities = np.asarray([[float(row[name]) for name in probability_fields] for row in rows])
        if (not set(predicted.tolist()) <= {0, 1} or probabilities.shape != (len(rows), 2)
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=sum_tolerance)
                or np.any(probabilities.max(axis=1)
                          - probabilities[np.arange(len(rows)), predicted] > TOLERANCE)):
            raise ValueError("invalid prediction labels or probabilities")
        result[key] = balanced_accuracy(truth, predicted)
        trial_sets[key] = tuple(trial_ids)
    if not result:
        raise ValueError("prediction output is empty")
    return result, trial_sets


def compare_saved_scores(rows, reconstructed, key_fields, score_field="balanced_accuracy"):
    observed = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in observed or key not in reconstructed:
            raise ValueError("saved score key is duplicate or lacks trial predictions")
        score = float(row[score_field])
        if not np.isfinite(score) or abs(score - reconstructed[key]) > TOLERANCE:
            raise ValueError("saved score differs from trial-reconstructed balanced accuracy")
        observed[key] = score
    if set(observed) != set(reconstructed):
        raise ValueError("saved score inventory differs from trial predictions")
    return observed


def load_core(core_spec):
    directory = safe_path(core_spec["directory"])
    summary_path, scores_path = directory / "summary.json", directory / "reconstructed_scores.csv"
    if (not summary_path.is_file() or sha256(summary_path) != core_spec["summary_sha256"]
            or not scores_path.is_file() or sha256(scores_path) != core_spec["reconstructed_scores_sha256"]):
        raise ValueError("core analysis is missing or differs from the explicit hash binding")
    rows = list(read_rows(scores_path))
    required = {"dataset_variant", "operation_id", "study_id", "method", "target", "draw",
                "source_size", "source_n", "donor_count", "h_total", "evaluation_n",
                "balanced_accuracy"}
    if not rows or set(rows[0]) != required:
        raise ValueError("core reconstructed-score schema differs")
    by_operation = {}
    for row in rows:
        row = dict(row, target=int(row["target"]), draw=int(row["draw"]),
                   h_total=int(row["h_total"]), balanced_accuracy=float(row["balanced_accuracy"]))
        key = (row["dataset_variant"], row["operation_id"])
        if key in by_operation:
            raise ValueError("duplicate core dataset/operation ID")
        by_operation[key] = row
    return load_json(summary_path), rows, by_operation, {
        "summary_sha256": sha256(summary_path), "reconstructed_scores_sha256": sha256(scores_path)}


def analyze_donor(item, participant_rows, contrast_rows, reconstructed_rows):
    directory, verification = completed_directory(item, "donor")
    if directory is None:
        return None, verification
    receipt, completed = load_json(directory / "receipt.json"), load_json(directory / "COMPLETED.json")
    predictions, prediction_sha = bound_file(directory, receipt, "predictions.csv.gz")
    scores_path, score_sha = bound_file(directory, receipt, "operation_scores.csv")
    reconstructed, trial_sets = reconstruct_scores(predictions, ("operation_id",), ("p0", "p1"))
    score_rows = list(read_rows(scores_path))
    saved = compare_saved_scores(score_rows, reconstructed, ("operation_id",))
    by_draw = {}
    for row in score_rows:
        setting, condition = row["setting"], row["condition"]
        if setting not in DONOR_SETTINGS or condition not in DONOR_CONDITIONS:
            raise ValueError("unknown donor setting or condition")
        if int(row["evaluation_n"]) != len(trial_sets[(row["operation_id"],)]):
            raise ValueError("donor saved evaluation count differs from trial predictions")
        key = (setting, int(row["target"]), int(row["draw"]), condition)
        if key in by_draw:
            raise ValueError("duplicate donor participant/draw/condition")
        by_draw[key] = saved[(row["operation_id"],)]
        reconstructed_rows.append({"extension": "donor", "dataset_variant": item["label"],
                                   "operation_id": row["operation_id"], "variant": condition,
                                   "target": row["target"], "draw": row["draw"],
                                   "source_size": "100", "h_total": "60",
                                   "balanced_accuracy": saved[(row["operation_id"],)]})
    targets = sorted({key[1] for key in by_draw})
    draw_values = sorted({key[2] for key in by_draw})
    if draw_values != list(range(10)):
        raise ValueError("donor analysis requires exactly draws 0 through 9")
    participant = {}
    for setting in DONOR_SETTINGS:
        for target in targets:
            cell = {}
            for draw in draw_values:
                singles = [by_draw[(setting, target, draw, f"single_{index}")] for index in range(3)]
                cell.setdefault("mean_single", []).append(float(np.mean(singles)))
                for condition in ("base", "own", "pooled"):
                    cell.setdefault(condition, []).append(by_draw[(setting, target, draw, condition)])
            for condition, values in cell.items():
                participant[(setting, target, condition)] = float(np.mean(values))
                participant_rows.append({"extension": "donor", "dataset_variant": item["label"],
                                         "method_or_setting": setting, "variant": condition,
                                         "target": target, "source_size": "100", "h_total": "60",
                                         "draw_count": 10, "balanced_accuracy": participant[(setting, target, condition)]})
            pairs = {
                "own_minus_base": ("own", "base"), "pooled_minus_base": ("pooled", "base"),
                "mean_single_minus_base": ("mean_single", "base"),
                "own_minus_mean_single": ("own", "mean_single"),
                "own_minus_pooled": ("own", "pooled"),
            }
            for name, (left, right) in pairs.items():
                contrast_rows.append({"extension": "donor", "dataset_variant": item["label"],
                                      "contrast": name, "method_or_setting": setting,
                                      "target": target, "source_size": "100", "h_total": "60",
                                      "draw_count": 10,
                                      "value_pp": 100 * (participant[(setting, target, left)]
                                                         - participant[(setting, target, right)])})
    endpoint = [row["value_pp"] for row in contrast_rows
                if row["extension"] == "donor" and row["dataset_variant"] == item["label"]
                and row["method_or_setting"] == "tuned_C_source_frozen"
                and row["contrast"] == "own_minus_mean_single"]
    verification.update(predictions_sha256=prediction_sha, scores_sha256=score_sha,
                        operations=len(score_rows), reconstructed_scores=len(reconstructed))
    return effect(endpoint), verification


def analyze_neural(item, core_operations, participant_rows, contrast_rows, reconstructed_rows):
    directory, verification = completed_directory(item, "neural")
    if directory is None:
        return verification
    receipt, completed = load_json(directory / "receipt.json"), load_json(directory / "COMPLETED.json")
    predictions, prediction_sha = bound_file(directory, receipt, "predictions.csv.gz", completed,
                                              "predictions_sha256")
    scores_path, score_sha = bound_file(directory, receipt, "scores.csv", completed, "scores_sha256")
    reconstructed, _ = reconstruct_scores(predictions, ("operation_id",), ("p_right", "p_left"),
                                            sum_tolerance=1e-6)
    score_rows = list(read_rows(scores_path))
    saved = compare_saved_scores(score_rows, reconstructed, ("operation_id",))
    by_cell, paired = defaultdict(list), defaultdict(list)
    for row in score_rows:
        operation_id = row["operation_id"]
        if not operation_id.startswith("eegnet__"):
            raise ValueError("neural operation does not bind a classical grid operation")
        classical_id = operation_id[len("eegnet__"):]
        core = core_operations.get((item["label"], classical_id))
        if core is None or core["method"] != "plain_ts":
            raise ValueError("neural operation lacks its exact classical plain-TS pair")
        coordinate = (int(row["target"]), int(row["draw"]), str(row["source_size"]), int(row["h_total"]))
        if coordinate != (core["target"], core["draw"], core["source_size"], core["h_total"]):
            raise ValueError("neural/classical coordinates differ")
        if coordinate[1] not in (0, 1, 2):
            raise ValueError("neural direct comparison requires classical draws 0 through 2 only")
        score = saved[(operation_id,)]
        key = (coordinate[0], coordinate[2], coordinate[3])
        by_cell[key].append((coordinate[1], score))
        paired[key].append((coordinate[1], score - core["balanced_accuracy"]))
        reconstructed_rows.append({"extension": "neural", "dataset_variant": item["label"],
                                   "operation_id": operation_id, "variant": "eegnet",
                                   "target": coordinate[0], "draw": coordinate[1],
                                   "source_size": coordinate[2], "h_total": coordinate[3],
                                   "balanced_accuracy": score})
    for key, values in sorted(by_cell.items()):
        draws = sorted(draw for draw, _ in values)
        if draws != [0, 1, 2]:
            raise ValueError("each neural participant cell requires exactly draws 0,1,2")
        target, source, history = key
        participant_rows.append({"extension": "neural", "dataset_variant": item["label"],
                                 "method_or_setting": "eegnet", "variant": "eegnet",
                                 "target": target, "source_size": source, "h_total": history,
                                 "draw_count": 3, "balanced_accuracy": float(np.mean([value for _, value in values]))})
        differences = paired[key]
        contrast_rows.append({"extension": "neural", "dataset_variant": item["label"],
                              "contrast": "eegnet_minus_plain_ts", "method_or_setting": "eegnet",
                              "target": target, "source_size": source, "h_total": history,
                              "draw_count": 3,
                              "value_pp": 100 * float(np.mean([value for _, value in differences]))})
    verification.update(predictions_sha256=prediction_sha, scores_sha256=score_sha,
                        operations=len(score_rows), paired_draws=[0, 1, 2])
    return verification


def alignment_label_gains(dataset_variant, means, by_cell, contrast_rows, trial_sets):
    """Add post-hoc h60-minus-h0 participant pairs at the complete source size."""
    full = ALIGNMENT_FULL_SOURCE.get(dataset_variant)
    if full is None:
        raise ValueError("unknown alignment dataset for complete-source label gain")
    output = {}
    for variant, name in ALIGNMENT_LABEL_GAIN_REGIMES:
        h0 = {target for data, regime, target, source, history in means
              if data == dataset_variant and regime == variant and source == full and history == 0}
        h60 = {target for data, regime, target, source, history in means
               if data == dataset_variant and regime == variant and source == full and history == 60}
        if not h0 or h0 != h60:
            raise ValueError("alignment h0/h60 label-gain participant sets differ")
        values = []
        for target in sorted(h0):
            low_key = (dataset_variant, variant, target, full, 0)
            high_key = (dataset_variant, variant, target, full, 60)
            if (sorted(draw for draw, _, _ in by_cell[low_key]) != list(range(10))
                    or sorted(draw for draw, _, _ in by_cell[high_key]) != list(range(10))):
                raise ValueError("alignment label gain requires exact ten draws at h0 and h60")
            low_operations = {draw: opid for draw, _, opid in by_cell[low_key]}
            for draw, _, high_operation in by_cell[high_key]:
                if trial_sets[(low_operations[draw], variant)] != trial_sets[(high_operation, variant)]:
                    raise ValueError("alignment h0/h60 label gain uses different evaluation IDs")
            value = 100 * (means[high_key] - means[low_key])
            values.append(value)
            contrast_rows.append({"extension": "current_alignment", "dataset_variant": dataset_variant,
                                  "contrast": name, "method_or_setting": "ea_ts", "target": target,
                                  "source_size": full, "h_total": 60, "draw_count": 10,
                                  "value_pp": value})
        statistics = effect(values)
        statistics.update(participant_ids=sorted(h0), draw_count=10,
                          status="post_hoc_descriptive_not_in_prespecified_family")
        output[f"{name}__S{full}__h60"] = statistics
    return output


def analyze_alignment(item, core_operations, participant_rows, contrast_rows, reconstructed_rows):
    directory, verification = completed_directory(item, "current-alignment")
    if directory is None:
        return {}, verification
    receipt = load_json(directory / "receipt.json")
    predictions, prediction_sha = bound_file(directory, receipt, "trial_predictions.csv.gz")
    scores_path, score_sha = bound_file(directory, receipt, "scores.csv")
    reconstructed, trial_sets = reconstruct_scores(
        predictions, ("operation_id", "variant"), ("p0", "p1"))
    score_rows = list(read_rows(scores_path))
    saved = compare_saved_scores(score_rows, reconstructed, ("operation_id", "variant"))
    by_cell = defaultdict(list)
    for row in score_rows:
        core = core_operations.get((item["label"], row["operation_id"]))
        if core is None or core["method"] != "ea_ts":
            raise ValueError("alignment score lacks an exact classical EA-TS operation")
        if int(row["evaluation_n"]) != len(trial_sets[(row["operation_id"], row["variant"])]):
            raise ValueError("alignment saved evaluation count differs from trial predictions")
        coordinate = (int(row["target"]), int(row["draw"]), str(row["source_size"]), int(row["h_total"]))
        if coordinate != (core["target"], core["draw"], core["source_size"], core["h_total"]):
            raise ValueError("alignment/classical coordinates differ")
        dataset_variant = item["label"]
        key = (dataset_variant, row["variant"], coordinate[0], coordinate[2], coordinate[3])
        by_cell[key].append((coordinate[1], saved[(row["operation_id"], row["variant"])], row["operation_id"]))
        reconstructed_rows.append({"extension": "current_alignment", "dataset_variant": dataset_variant,
                                   "operation_id": row["operation_id"], "variant": row["variant"],
                                   "target": coordinate[0], "draw": coordinate[1],
                                   "source_size": coordinate[2], "h_total": coordinate[3],
                                   "balanced_accuracy": saved[(row["operation_id"], row["variant"])]})
    means = {}
    for key, values in sorted(by_cell.items()):
        draws = sorted(value[0] for value in values)
        if draws != list(range(10)):
            raise ValueError("each alignment participant cell requires draws 0 through 9")
        dataset_variant, variant, target, source, history = key
        means[key] = float(np.mean([value[1] for value in values]))
        participant_rows.append({"extension": "current_alignment", "dataset_variant": dataset_variant,
                                 "method_or_setting": "ea_ts", "variant": variant,
                                 "target": target, "source_size": source, "h_total": history,
                                 "draw_count": 10, "balanced_accuracy": means[key]})
    effects = {}
    datasets = sorted({key[0] for key in means})
    for dataset_variant in datasets:
        effects[dataset_variant] = {}
        coordinate_keys = sorted({key[2:] for key in means if key[0] == dataset_variant})
        for current, historical, name in ALIGNMENT_PAIRS:
            vectors = defaultdict(list)
            for target, source, history in coordinate_keys:
                left = (dataset_variant, current, target, source, history)
                right = (dataset_variant, historical, target, source, history)
                if left not in means or right not in means:
                    continue
                left_draws = by_cell[left]
                right_by_draw = {draw: (score, opid) for draw, score, opid in by_cell[right]}
                differences = []
                for draw, score, opid in left_draws:
                    historical_score, historical_opid = right_by_draw[draw]
                    if opid != historical_opid:
                        raise ValueError("alignment variants do not share the same grid operation")
                    if trial_sets[(opid, current)] != trial_sets[(opid, historical)]:
                        raise ValueError("alignment all/tail comparison uses different evaluation IDs")
                    differences.append(score - historical_score)
                value = 100 * float(np.mean(differences))
                vectors[(source, history)].append(value)
                contrast_rows.append({"extension": "current_alignment", "dataset_variant": dataset_variant,
                                      "contrast": name, "method_or_setting": "ea_ts", "target": target,
                                      "source_size": source, "h_total": history, "draw_count": 10,
                                      "value_pp": value})
            for (source, history), values in vectors.items():
                statistics = effect(values)
                statistics["harm_count"] = statistics["negative"]
                statistics["harm_tolerance_pp"] = 1e-10
                effects[dataset_variant][f"{name}__S{source}__h{history}"] = statistics
        effects[dataset_variant].update(alignment_label_gains(dataset_variant, means, by_cell, contrast_rows, trial_sets))
    verification.update(predictions_sha256=prediction_sha, scores_sha256=score_sha,
                        scores=len(score_rows), paired_evaluation_ids_verified=True)
    return effects, verification


def build_bnci_family(core_summary, donor_effects):
    endpoints = core_summary.get("endpoint_results", {}).get("bnci")
    donor = donor_effects.get("bnci")
    if endpoints is None or donor is None:
        return {"status": "unavailable", "reason": "BNCI core or donor endpoint missing"}
    contrasts = endpoints.get("contrasts", {})
    if any(name not in contrasts for name in BNCI_CORE_FAMILY):
        raise ValueError("BNCI core summary lacks a prespecified family endpoint")
    raw = {name: float(contrasts[name]["signflip_p"]) for name in BNCI_CORE_FAMILY}
    raw["donor_own_minus_mean_single_tuned"] = float(donor["signflip_p"])
    adjusted = holm(raw)
    return {"status": "complete", "method": "exact two-sided sign-flip; Holm across six",
            "excluded_descriptive_endpoint": "plain_label_gain_Sall",
            "raw_p": raw, "holm_adjusted_p": adjusted,
            "donor_effect": donor,
            "signflip_assumption": "Participant signs are exchangeable under each paired null; shared source cohorts and overlapping LOSO training folds limit independence."}


def summarize_contrasts(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["extension"], row["dataset_variant"], row["contrast"],
               row["method_or_setting"], str(row["source_size"]), int(row["h_total"]),
               int(row["draw_count"]))
        grouped[key].append((int(row["target"]), float(row["value_pp"])))
    result = {}
    for key, values in sorted(grouped.items()):
        if len(values) != len({target for target, _ in values}):
            raise ValueError("paired contrast contains duplicate participant IDs")
        extension, dataset, contrast, method, source, history, draw_count = key
        identifier = f"{extension}__{dataset}__{contrast}__{method}__S{source}__h{history}"
        statistics = effect([value for _, value in sorted(values)])
        statistics.update(participant_ids=[target for target, _ in sorted(values)],
                          draw_count=draw_count)
        if extension == "current_alignment":
            statistics.update(harm_count=statistics["negative"], harm_tolerance_pp=1e-10)
        result[identifier] = statistics
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("output directory must be new")
    spec = validate_spec(load_json(args.inputs))
    args.out_dir.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    core_summary, core_rows, core_operations, core_verification = load_core(spec["core_analysis"])
    participant_rows, contrast_rows, reconstructed_rows = [], [], []
    verification = {"core_analysis": core_verification, "donors": {}, "neural": {},
                    "current_alignment": {}}
    donor_effects, alignment_effects = {}, {}
    for item in spec["donors"]:
        endpoint, checked = analyze_donor(item, participant_rows, contrast_rows, reconstructed_rows)
        verification["donors"][item["label"]] = checked
        if endpoint is not None:
            donor_effects[item["label"]] = endpoint
    for item in spec["neural"]:
        verification["neural"][item["label"]] = analyze_neural(
            item, core_operations, participant_rows, contrast_rows, reconstructed_rows)
    for item in spec["current_alignment"]:
        effects, checked = analyze_alignment(
            item, core_operations, participant_rows, contrast_rows, reconstructed_rows)
        verification["current_alignment"][item["label"]] = checked
        for dataset_variant, values in effects.items():
            if dataset_variant in alignment_effects:
                raise ValueError("duplicate current-alignment dataset across inputs")
            alignment_effects[dataset_variant] = values
    participant_fields = ["extension", "dataset_variant", "method_or_setting", "variant", "target",
                          "source_size", "h_total", "draw_count", "balanced_accuracy"]
    contrast_fields = ["extension", "dataset_variant", "contrast", "method_or_setting", "target",
                       "source_size", "h_total", "draw_count", "value_pp"]
    reconstructed_fields = ["extension", "dataset_variant", "operation_id", "variant", "target",
                            "draw", "source_size", "h_total", "balanced_accuracy"]
    write_csv(args.out_dir / "participant_means.csv", participant_rows, participant_fields)
    write_csv(args.out_dir / "paired_contrasts.csv", contrast_rows, contrast_fields)
    write_csv(args.out_dir / "reconstructed_extension_scores.csv", reconstructed_rows,
              reconstructed_fields)
    summary = {
        "core_analysis_summary": core_summary,
        "donor_endpoints": donor_effects,
        "current_alignment_effects": alignment_effects,
        "post_hoc_descriptive_note": "Within-regime h60-minus-h0 current-alignment label gains were added after reviewing current-versus-historical gains; they are descriptive and outside the prespecified BNCI six-endpoint family.",
        "extension_contrast_effects": summarize_contrasts(contrast_rows),
        "bnci_prespecified_six_endpoint_family": build_bnci_family(core_summary, donor_effects),
        "verification": verification,
        "units": {"balanced_accuracy": "fraction", "paired_contrasts": "percentage points"},
        "aggregation": "draws averaged within participant; three single donors averaged within participant and draw before draw averaging",
        "CI_status": "conditional on observed source cohort and frozen study definitions; no independent-sample interpretation",
        "shared_statistics": {"module": "scripts/analyze_review3.py", "effect": "effect",
                              "multiplicity": "holm", "seed": 2026092304,
                              "bootstrap_draws": 10000},
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    outputs = {name: sha256(args.out_dir / name) for name in
               ("participant_means.csv", "paired_contrasts.csv",
                "reconstructed_extension_scores.csv", "summary.json")}
    receipt = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "inputs_sha256": sha256(args.inputs), "analyzer_sha256": sha256(__file__),
               "wall_seconds": time.monotonic() - started_wall,
               "cpu_seconds": time.process_time() - started_cpu,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "counts": {"participant_means": len(participant_rows),
                          "paired_contrasts": len(contrast_rows),
                          "reconstructed_scores": len(reconstructed_rows)},
               "outputs": outputs, "confirmation_data_accessed": False}
    (args.out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "counts": receipt["counts"]}, indent=2))


if __name__ == "__main__":
    main()
