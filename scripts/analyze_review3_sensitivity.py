"""Descriptive participant, montage, timing and winner diagnostics for Review 3."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr
from analyze_review3 import effect, save_csv, sha

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired_sensitivities(scores):
    keys = ("study_id", "method", "target", "draw", "source_size", "donor_count", "h_total")
    lookup = {tuple(r[k] for k in keys): r for r in scores if r["dataset_variant"] == "openbmi8"}
    vectors, summaries = [], {}
    for variant in ("openbmi20", "openbmi_offset"):
        grouped = defaultdict(list)
        for r in scores:
            if r["dataset_variant"] != variant:
                continue
            key = tuple(r[k] for k in keys)
            if key not in lookup:
                raise ValueError("unpaired sensitivity condition")
            delta = 100 * (float(r["balanced_accuracy"]) - float(lookup[key]["balanced_accuracy"]))
            grouped[(r["method"], r["source_size"], r["h_total"], r["target"])].append((int(r["draw"]), delta))
        condition = defaultdict(list)
        for (method, source, h, target), values in grouped.items():
            if sorted(draw for draw, _ in values) != list(range(10)):
                raise ValueError("sensitivity needs ten paired draws")
            value = float(np.mean([v for _, v in values]))
            vectors.append(dict(variant=variant, method=method, source_size=source,
                                h_total=h, target=target, difference_pp=value))
            condition[(method, source, h)].append((int(target), value))
        for key, people in condition.items():
            ordered = sorted(people)
            summaries["/".join((variant, *key))] = dict(
                targets=[t for t, _ in ordered], **effect([v for _, v in ordered]))
    return vectors, summaries


def diversity(people):
    selected = [r for r in people if r["dataset_variant"] == "openbmi8" and r["study_id"] == "openbmi_diversity"]
    if any(r["source_size"] != "300" for r in selected):
        raise ValueError("frozen diversity sensitivity requires S=300")
    lookup = {(r["method"], r["h_total"], r["donor_count"], int(r["target"])): float(r["balanced_accuracy"]) for r in selected}
    targets = sorted({int(r["target"]) for r in selected})
    result = {}
    for method in sorted({r["method"] for r in selected}):
        for h in ("0", "60"):
            for donors in ("3", "6"):
                delta = [100 * (lookup[method, h, donors, t] - lookup[method, h, "18", t]) for t in targets]
                result[f"{method}/h{h}/donors{donors}_minus18"] = dict(targets=targets, **effect(delta))
    return result


def winners(people):
    grouped = defaultdict(list)
    for r in people:
        if (r["dataset_variant"] in ("openbmi8", "bnci") and r["study_id"] in ("openbmi_main", "bnci_main")
                and r["donor_count"] == "all"):
            grouped[(r["dataset_variant"], r["source_size"])].append(r)
    output = []
    for (variant, source), values in grouped.items():
        targets = sorted({int(r["target"]) for r in values})
        methods = sorted({r["method"] for r in values})
        doses = sorted({int(r["h_total"]) for r in values})
        lookup = {(r["method"], int(r["h_total"]), int(r["target"])): float(r["balanced_accuracy"]) for r in values}
        for heldout in targets:
            training = [t for t in targets if t != heldout]
            means = {(m, h): 100 * np.mean([lookup[m, h, t] for t in training]) for m in methods for h in doses}
            baseline_best = max(means[m, 0] for m in methods)
            baseline = sorted(m for m in methods if means[m, 0] >= baseline_best - 1e-10)
            for h in doses:
                best = max(means[m, h] for m in methods)
                chosen = sorted(m for m in methods if means[m, h] >= best - 1e-10)
                for m in methods:
                    output.append(dict(dataset_variant=variant, source_size=source, heldout=heldout,
                                       h_total=h, method=m, training_mean_ba=means[m, h],
                                       margin_to_best_pp=means[m, h] - best,
                                       winners="|".join(chosen), h0_winners="|".join(baseline),
                                       winner_set_changed=chosen != baseline))
    return output


def decodability(spec, scores):
    lookup = {(r["dataset_variant"], r["operation_id"]): r for r in scores}
    score_cell = {(r["dataset_variant"], r["study_id"], r["method"], r["source_size"],
                   r["donor_count"], r["target"], r["draw"], r["h_total"]): float(r["balanced_accuracy"]) for r in scores}
    raw, aggregate, correlations = [], [], []
    for item in spec["historical_grids"]:
        directory = ROOT / item["directory"]
        receipt = json.loads((directory / "receipt.json").read_text())
        path = directory / "historical_decodability.csv"
        if receipt["status"] != "completed" or receipt["outputs"][path.name] != sha(path):
            raise ValueError("historical scores are incomplete or changed")
        for r in rows(path):
            score = lookup[item["label"], r["operation_id"]]
            if score["study_id"] not in ("openbmi_main", "bnci_main") or score["donor_count"] != "all":
                continue
            n, correct = int(r["history_n"]), int(r["correct_count"])
            if n != int(r["h_total"]) or n <= 0 or not 0 <= correct <= n or abs(correct/n - float(r["balanced_accuracy"])) > 1e-12:
                raise ValueError("charged balanced historical score is inconsistent")
            key = (item["label"], score["study_id"], r["method"], r["source_size"],
                   "all", r["target"], r["draw"], "0")
            gain = 100 * (float(score["balanced_accuracy"]) - score_cell[key])
            raw.append(dict(dataset_variant=item["label"], method=r["method"], source_size=r["source_size"],
                            h_total=r["h_total"], target=r["target"], draw=r["draw"],
                            historical_ba=100*float(r["balanced_accuracy"]), future_gain_pp=gain))
    grouped = defaultdict(list)
    for r in raw:
        grouped[tuple(r[k] for k in ("dataset_variant", "method", "source_size", "h_total", "target"))].append(r)
    for key, values in grouped.items():
        if sorted(int(r["draw"]) for r in values) != list(range(10)):
            raise ValueError("historical diagnostics require ten draws per participant")
        aggregate.append(dict(zip(("dataset_variant", "method", "source_size", "h_total", "target"), key),
                              historical_ba=float(np.mean([r["historical_ba"] for r in values])),
                              historical_across_draw_sd=float(np.std([r["historical_ba"] for r in values], ddof=1)),
                              future_gain_pp=float(np.mean([r["future_gain_pp"] for r in values]))))
    by_condition = defaultdict(list)
    for r in aggregate:
        by_condition[tuple(r[k] for k in ("dataset_variant", "method", "source_size", "h_total"))].append(r)
    for key, values in by_condition.items():
        def rho(v):
            x = np.round([r["historical_ba"] for r in v], 10)
            y = np.round([r["future_gain_pp"] for r in v], 10)
            return float(spearmanr(x, y).statistic) if len(set(x)) > 1 and len(set(y)) > 1 else None
        draw_rho = [rho([r for r in raw if tuple(r[k] for k in ("dataset_variant", "method", "source_size", "h_total")) == key and r["draw"] == str(draw)]) for draw in range(10)]
        correlations.append(dict(zip(("dataset_variant", "method", "source_size", "h_total"), key),
                                 n=len(values), draw_mean_spearman=rho(values), individual_draw_spearman=draw_rho))
    return raw, aggregate, correlations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(); start, cpu = time.monotonic(), time.process_time()
    spec = json.loads(args.inputs.read_text()); core = ROOT / spec["core_analysis"]["directory"]
    for name, digest in spec["core_analysis"]["files"].items():
        if sha(core / name) != digest:
            raise ValueError("core analysis changed")
    scores, people = rows(core / "reconstructed_scores.csv"), rows(core / "participant_means.csv")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    vectors, sensitivity = paired_sensitivities(scores)
    raw, aggregate, correlations = decodability(spec, scores)
    save_csv(args.out_dir / "sensitivity_participants.csv", vectors)
    save_csv(args.out_dir / "training_fold_margins.csv", winners(people))
    save_csv(args.out_dir / "decodability_draws.csv", raw)
    save_csv(args.out_dir / "decodability_participants.csv", aggregate)
    result = dict(sensitivity=sensitivity, source_diversity=diversity(people), decodability=correlations,
                  interpretation="All exploratory. Draw-averaged historical scores estimate expected h-label reliability; actually computing ten-draw averages consumes more than h distinct labels and is not a deployed h-label predictor. Individual-draw correlations are also reported. No threshold or stable-ability claim.",
                  winner_scope="Other-person target outcome means are development diagnostics; overlapping LOSO source training further limits external independence. No validated model-selection policy.",
                  cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == "__main__":
    main()
