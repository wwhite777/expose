#!/usr/bin/env python3
"""Summarize fixed review2 development controls from saved per-subject scores only."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {"operation_id", "module", "source_n", "h_total", "C", "lambda_personal", "representation", "draw", "target_subject", "donor_kind", "balanced_accuracy"}
BASE_FIELDS = tuple(sorted(REQUIRED - {"operation_id", "draw", "balanced_accuracy"}))
TOL, BOOTSTRAPS, SEED = 1e-12, 10_000, 20260920
SUBJECTS = tuple("2 3 8 9 16 17 33 38 45 51 53 54".split())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rows(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    if not data or set(data[0]) != fields:
        raise ValueError(f"unexpected schema: {path}")
    return data


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite value")
    return result


def canonical(value, integer=False):
    number(value)
    return str(int(number(value))) if integer else format(number(value), ".12g")


def load_grid(path):
    data = rows(path, REQUIRED)
    if len(data) != 1728:
        raise ValueError("expected exactly 1728 subject-score rows")
    operations = defaultdict(list)
    for row in data:
        for key in ("source_n", "h_total", "C", "draw", "target_subject", "balanced_accuracy"):
            number(row[key])
        if not 0 <= number(row["balanced_accuracy"]) <= 1 or row["module"] not in {"A", "B", "C", "D"}:
            raise ValueError("invalid score row")
        row["source_n"] = canonical(row["source_n"], integer=True); row["h_total"] = canonical(row["h_total"], integer=True)
        row["C"] = canonical(row["C"]); row["draw"] = canonical(row["draw"], integer=True); row["target_subject"] = canonical(row["target_subject"], integer=True)
        if row["lambda_personal"] != "pooled": row["lambda_personal"] = canonical(row["lambda_personal"])
        operations[row["operation_id"]].append(row)
    if set(row["target_subject"] for row in data) != set(SUBJECTS) or set(row["draw"] for row in data) != {"0", "1"}:
        raise ValueError("subject IDs or draw set differs from frozen development grid")
    if len(operations) != 1464:
        raise ValueError("expected exactly 1464 operation IDs")
    for operation, members in operations.items():
        if members[0]["module"] == "A":
            if len(members) != 12 or {row["target_subject"] for row in members} != set(SUBJECTS): raise ValueError("A operation must score all 12 people once")
        elif len(members) != 1:
            raise ValueError("non-A operation ID must have one target score")
    return data


def mean(values):
    return sum(values) / len(values)


def draw_mean(data):
    grouped = defaultdict(list)
    for row in data:
        key = tuple(row[k] for k in BASE_FIELDS)
        grouped[key].append((row["draw"], number(row["balanced_accuracy"])))
    result = []
    for key, values in grouped.items():
        if len(values) != 2 or {draw for draw, _ in values} != {"0", "1"}:
            raise ValueError("each logical target condition must have exactly two draws")
        result.append({**dict(zip(BASE_FIELDS, key)), "balanced_accuracy": mean([score for _, score in values])})
    return result


def filter_rows(data, **wanted):
    return [row for row in data if all(str(row[k]) == str(v) for k, v in wanted.items())]


def keyed(data, keys):
    result = {}
    for row in data:
        key = tuple(row[k] for k in keys)
        if key in result:
            raise ValueError("duplicate condition after aggregation")
        result[key] = number(row["balanced_accuracy"])
    return result


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    indices = np.random.default_rng(SEED).integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    return np.percentile(values[indices].mean(axis=1), [2.5, 97.5], method="linear").tolist()


def summarize(name, pairs, dimensions):
    result = []
    grouped = defaultdict(list)
    for row in pairs:
        grouped[tuple(row[k] for k in dimensions)].append(row)
    for key, members in sorted(grouped.items()):
        values = [row["difference"] for row in members]
        result.append({"contrast": name, **dict(zip(dimensions, key)), "n_people": len(values), "mean": mean(values),
                       "median": statistics.median(values), "minimum": min(values), "maximum": max(values),
                       "positive": sum(x > TOL for x in values), "zero": sum(abs(x) <= TOL for x in values),
                       "negative": sum(x < -TOL for x in values), "omission_means": [mean(values[:i] + values[i + 1:]) for i in range(len(values))],
                       "descriptive_ci95": bootstrap(values)})
    return result


def paired(name, left, right, dimensions):
    keys = ["target_subject", *dimensions]
    a, b = keyed(left, keys), keyed(right, keys)
    if set(a) != set(b) or len(a) != 12:
        raise ValueError(f"{name}: expected matched 12-person conditions")
    return [{"contrast": name, "target_subject": key[0], **dict(zip(dimensions, key[1:])), "left": a[key], "right": b[key], "difference": a[key] - b[key]} for key in sorted(a)]


def paired_effects(name, left, right, dimensions):
    keys = ["target_subject", *dimensions]
    a = {tuple(row[k] for k in keys): row["difference"] for row in left}
    b = {tuple(row[k] for k in keys): row["difference"] for row in right}
    if set(a) != set(b) or len(a) != 12:
        raise ValueError(f"{name}: expected matched 12-person effect conditions")
    return [{"contrast": name, "target_subject": key[0], **dict(zip(dimensions, key[1:])), "left": a[key], "right": b[key], "difference": a[key] - b[key]} for key in sorted(a)]


def singles(data):
    grouped = defaultdict(list)
    for row in data:
        if not row["donor_kind"].startswith("single_"):
            continue
        key = tuple(row[k] for k in row if k not in {"donor_kind", "balanced_accuracy"})
        grouped[key].append(number(row["balanced_accuracy"]))
    result = []
    for key, values in grouped.items():
        if len(values) != 3:
            raise ValueError("single donor control must have exactly three donors")
        row = dict(zip([k for k in next(iter(data)) if k not in {"donor_kind", "balanced_accuracy"}], key))
        row.update(donor_kind="single_average", balanced_accuracy=mean(values)); result.append(row)
    return result


def rank(values):
    rounded = [round(x, 12) for x in values]; order = sorted(range(len(values)), key=lambda i: rounded[i]); result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and rounded[order[end]] == rounded[order[start]]: end += 1
        value = (start + 1 + end) / 2
        for index in order[start:end]: result[index] = value
        start = end
    return result


def spearman(x, y):
    rx, ry = rank(x), rank(y); mx, my = mean(rx), mean(ry)
    numerator = sum((a - mx) * (b - my) for a, b in zip(rx, ry)); denominator = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if denominator == 0 else numerator / denominator


def historical_reference():
    path = ROOT / "result/day3/analysis_r002/subject_metrics.csv"
    fields = {"arm", "model", "prior_weight", "draw", "dose_per_class", "subject_id", "balanced_accuracy"}
    data = rows(path, fields)
    result = []
    for dose in (0, 30):
        values = defaultdict(list)
        for row in filter_rows(data, arm="practical", model="ts_lr", prior_weight="", dose_per_class=str(dose)):
            values[row["subject_id"]].append(number(row["balanced_accuracy"]))
        if len(values) != 12 or any(len(v) != 2 for v in values.values()): raise ValueError("historical reference grid differs")
        result.extend({"target_subject": person, "h_total": dose * 2, "balanced_accuracy": mean(scores)} for person, scores in values.items())
    return result, path


def write_csv(path, data):
    if not data: raise ValueError(f"no output rows for {path.name}")
    fields = sorted({key for row in data for key in row})
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--grid-dir", required=True, type=Path); parser.add_argument("--out-dir", required=True, type=Path); args = parser.parse_args()
    grid, out = args.grid_dir.resolve(), args.out_dir.resolve()
    scores_path, predictor_path = grid / "subject_metrics.csv", grid / "predictor_scores.csv"
    if out.exists(): raise ValueError("out-dir must be new")
    data = draw_mean(load_grid(scores_path)); out.mkdir(parents=True)
    all_means = [{k: row[k] for k in row if k != "balanced_accuracy"} | {"mean_balanced_accuracy": row["balanced_accuracy"]} for row in data]
    pairs, summaries = [], []
    # A: regularization, source-only scores.
    for source in (40, 70, 90, 100):
        for c in ("0.1", "10"):
            left, right = filter_rows(data, module="A", source_n=str(source), C=c), filter_rows(data, module="A", source_n=str(source), C="1")
            pairs += paired(f"A_C{c}_vs_1", left, right, ["source_n", "representation", "lambda_personal", "donor_kind"])
    # B: fixed weight comparison by source size and representation.
    for source in (100, 1800):
        for representation in sorted(set(r["representation"] for r in filter_rows(data, module="B", source_n=str(source)))):
            left = filter_rows(data, module="B", source_n=str(source), C="1", representation=representation, lambda_personal="0.5")
            right = filter_rows(data, module="B", source_n=str(source), C="1", representation=representation, lambda_personal="pooled")
            pairs += paired("B_lambda_0.5_vs_pooled", left, right, ["source_n", "representation"])
    for source in (100, 1800):
        b = [row for row in pairs if row["contrast"] == "B_lambda_0.5_vs_pooled" and row["source_n"] == str(source)]
        frozen = [row for row in b if row["representation"] == "source_frozen"]
        refit = [row for row in b if row["representation"] == "pooled_refit"]
        pairs += paired_effects("B_weight_by_representation", frozen, refit, ["source_n"])
    # C: three donor contrasts, after three single donors are averaged within person.
    cdata = filter_rows(data, module="C") + singles(filter_rows(data, module="C"))
    for source in (40, 100):
        base = filter_rows(cdata, source_n=str(source))
        for left_kind, right_kind, name in (("own", "pooled", "C_own_vs_pooled"), ("own", "single_average", "C_own_vs_single_average"), ("single_average", "pooled", "C_single_average_vs_pooled")):
            left, right = filter_rows(base, donor_kind=left_kind), filter_rows(base, donor_kind=right_kind)
            pairs += paired(f"C_S{source}_{name[2:]}", left, right, ["source_n"])
    # D: both aligned conditions are source-frozen. Plain h0 is the frozen old practical fit;
    # plain h60 is the current B source-frozen pooled condition, keeping representation matched.
    ddata = filter_rows(data, module="D", representation="source_frozen")
    if len(ddata) != 24:
        raise ValueError("D must contain exactly 24 two-draw-aggregated source-frozen rows")
    historical, historical_path = historical_reference()
    plain_h0 = [{"target_subject":row["target_subject"], "h_total":"0", "source_n":"1800", "C":"1", "lambda_personal":"pooled", "donor_kind":"pooled", "representation":"plain", "balanced_accuracy":row["balanced_accuracy"]} for row in historical if row["h_total"] == 0]
    plain_h60 = filter_rows(data, module="B", source_n="1800", h_total="60", C="1", lambda_personal="pooled", representation="source_frozen")
    if len(plain_h60) != 12:
        raise ValueError("B source-frozen pooled h60 reference must contain 12 people")
    for h, plain in ((0, plain_h0), (60, plain_h60)):
        pairs += paired(f"D_aligned_vs_plain_h{h}", filter_rows(ddata, h_total=str(h)), plain, [])
    pairs += paired("D_h60_vs_h0", filter_rows(ddata, h_total="60"), filter_rows(ddata, h_total="0"), [])
    plain_gain_rows = paired("D_plain_h60_vs_h0", plain_h60, plain_h0, [])
    pairs += plain_gain_rows
    aligned_gain = {r["target_subject"]: r["difference"] for r in pairs if r["contrast"] == "D_h60_vs_h0"}
    plain_gain = {r["target_subject"]: r["difference"] for r in plain_gain_rows}
    pairs += [{"contrast":"D_alignment_by_label_interaction", "target_subject":p, "representation":"aligned_minus_plain", "source_n":"1800", "C":"1", "lambda_personal":"pooled", "donor_kind":"own", "left":aligned_gain[p], "right":plain_gain[p], "difference":aligned_gain[p]-plain_gain[p]} for p in sorted(aligned_gain)]
    dimensions = ["source_n", "representation", "C", "lambda_personal", "donor_kind"]
    for name in sorted(set(row["contrast"] for row in pairs)):
        members = [r for r in pairs if r["contrast"] == name]
        summaries += summarize(name, members, [key for key in dimensions if all(key in row for row in members)])
    # Predictor diagnostics, including the frozen historical source-only reference for unaligned h=0/60.
    predictor_fields = {"subject_id", "draw", "h_total", "balanced_accuracy", "trial_ids_json"}; predictors = rows(predictor_path, predictor_fields)
    by_predictor = defaultdict(list)
    for row in predictors:
        trial_ids = json.loads(row["trial_ids_json"])
        if not isinstance(trial_ids, list) or not trial_ids or len(trial_ids) != len(set(trial_ids)):
            raise ValueError("predictor trial_ids_json must be a nonempty unique list")
        row["subject_id"] = canonical(row["subject_id"], integer=True); row["draw"] = canonical(row["draw"], integer=True); row["h_total"] = canonical(row["h_total"], integer=True)
        by_predictor[(row["subject_id"], row["h_total"])].append((row["draw"], number(row["balanced_accuracy"])))
    predictor_means = {(p, h): mean([score for _, score in v]) for (p, h), v in by_predictor.items() if len(v) == 2 and {draw for draw, _ in v} == {"0", "1"}}
    if len(predictor_means) != 36 or set(p for p, _ in predictor_means) != set(SUBJECTS) or set(h for _, h in predictor_means) != {"10", "30", "60"}: raise ValueError("predictor scores must contain 12 people x three two-draw budgets")
    wanted = {"C_S100_own_vs_single_average", "C_S100_own_vs_pooled", "D_h60_vs_h0"}
    diagnostics = []
    for contrast in sorted(wanted):
        choices = [r for r in pairs if r["contrast"] == contrast]
        if contrast.startswith("C_"): choices = [r for r in choices if r["source_n"] == "100"]
        values = {r["target_subject"]: r["difference"] for r in choices}
        if len(values) != 12: raise ValueError(f"predictor outcome missing: {contrast}")
        for budget in (10, 30, 60):
            people = sorted(values); diagnostics.append({"outcome":contrast, "history_label_budget":budget, "n_people":12, "spearman_rho":spearman([predictor_means[(p,str(budget))] for p in people], [values[p] for p in people]), "p_value": "not computed"})
    write_csv(out / "per_person_contrasts.csv", pairs); write_csv(out / "grid_means.csv", all_means); write_csv(out / "predictor_diagnostics.csv", diagnostics)
    (out / "summary.json").write_text(json.dumps({"summaries":summaries, "historical_unaligned_reference":historical, "draw_aggregation":"mean within person; three single donors averaged within person", "bootstrap":{"resamples":BOOTSTRAPS,"seed":SEED,"scope":"conditional descriptive participant bootstrap"}}, indent=2) + "\n")
    (out / "report.md").write_text("# Review2 control summaries\n\nAll summaries average two draws within person; single-donor controls average three donors within person. Intervals are conditional descriptive participant bootstrap intervals (10,000 resamples); no p-values or threshold search are reported. See CSV and JSON artifacts for all grid conditions and per-person contrasts.\n")
    inputs={str(scores_path):sha(scores_path), str(predictor_path):sha(predictor_path), str(ROOT/'research/review2_execution_20260923/PROTOCOL_v1.md'):sha(ROOT/'research/review2_execution_20260923/PROTOCOL_v1.md'), str(historical_path):sha(historical_path)}
    outputs={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    receipt={"status":"completed","created_utc":datetime.now(timezone.utc).isoformat(),"command":sys.argv,"inputs":inputs,"outputs":outputs,"script_sha256":sha(__file__),"no_fits_or_raw_reads":True,"confirmation_accessed":False}
    (out / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n"); (out / "COMPLETED.json").write_text(json.dumps({"status":"completed","outputs":{p.name:sha(p) for p in out.iterdir() if p.name != 'COMPLETED.json'}},indent=2)+"\n")


if __name__ == "__main__": main()
