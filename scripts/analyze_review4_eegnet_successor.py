#!/usr/bin/env python3
"""Reconstruct the frozen Review-4 EEGNet successor and paired plain-TS summary."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "result/review4_20260924/openbmi8_eegnet_successor_r001"
CLASSICAL = ROOT / "result/review3_20260923/openbmi8_grid_r003/operation_scores.csv"
OUT = ROOT / "result/review4_20260924/eegnet_successor_analysis_r003"
EXPECTED = {
    RUN / "predictions.csv.gz": "663fb45fdf0424488b2b3d0914069295a2bf9bd87cb2e02cb3ebf00f76037d37",
    RUN / "scores.csv": "2167a232d2f494670bb50d7117a4fa915c61114c4b42e5f5b06317832abebc5a",
    CLASSICAL: "6cdc704a3214bd356b3d6255cad92428149b5555246b70ab85a6132b455e2342",
}
SEED = 2026092412
RESAMPLES = 10000


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bootstrap(values, rng):
    values = np.asarray(values, dtype=float)
    samples = values[rng.integers(0, len(values), size=(RESAMPLES, len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "ci95_percentile": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "participants": len(values), "resamples": RESAMPLES, "seed": SEED}


def main():
    start_cpu, start_wall = time.process_time(), time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 65))
    if OUT.exists():
        raise FileExistsError(OUT)
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError("analysis input hash differs: " + str(path))
    OUT.mkdir(parents=True)

    groups = defaultdict(list)
    with gzip.open(RUN / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            groups[row["operation_id"]].append(row)
    if len(groups) != 144 or set(map(len, groups.values())) != {100}:
        raise ValueError("successor prediction cardinality differs")
    reconstructed = {}
    for operation, rows in groups.items():
        truth = np.asarray([int(row["y_true"]) for row in rows])
        prediction = np.asarray([int(row["y_pred"]) for row in rows])
        probability = np.asarray(
            [[float(row["p_right"]), float(row["p_left"])] for row in rows],
            dtype=np.float32)
        if Counter(truth.tolist()) != {0: 50, 1: 50}:
            raise ValueError("evaluation labels are not balanced")
        reconstructed[operation] = {
            "balanced_accuracy": float(np.mean([np.mean(prediction[truth == c] == c) for c in (0, 1)])),
            "log_loss": float(-np.log(np.clip(probability[np.arange(100), truth], 1e-15, 1)).mean()),
            "p_left_sd": float(probability[:, 1].std()),
            "predicted_right_n": int(np.count_nonzero(prediction == 0)),
            "predicted_left_n": int(np.count_nonzero(prediction == 1)),
            "single_predicted_class": bool(len(set(prediction.tolist())) == 1),
        }

    neural = {}
    with (RUN / "scores.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["target"]), int(row["draw"]), int(row["source_size"]), int(row["h_total"]))
            if key in neural:
                raise ValueError("duplicate neural score cell")
            rebuilt = reconstructed[row["operation_id"]]
            for name in ("balanced_accuracy", "log_loss", "p_left_sd"):
                if abs(float(row[name]) - rebuilt[name]) > 1e-12:
                    raise ValueError("saved neural metric differs: " + name)
            for name in ("predicted_right_n", "predicted_left_n"):
                if int(row[name]) != rebuilt[name]:
                    raise ValueError("saved neural count differs: " + name)
            if (row["single_predicted_class"].lower() == "true") != rebuilt["single_predicted_class"]:
                raise ValueError("saved collapse flag differs")
            neural[key] = {**rebuilt, "source_membership": row["source_membership"],
                           "history_membership": row["history_membership"]}
    if len(neural) != 144:
        raise ValueError("neural score cardinality differs")

    classical = {}
    with CLASSICAL.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row["study_id"] == "openbmi_main" and row["method"] == "plain_ts"
                    and row["donor_count"] == "all" and int(row["draw"]) in (0, 1, 2)
                    and int(row["source_size"]) in (300, 1800)
                    and int(row["h_total"]) in (0, 60)):
                key = (int(row["target"]), int(row["draw"]), int(row["source_size"]), int(row["h_total"]))
                classical[key] = {"balanced_accuracy": float(row["balanced_accuracy"]),
                                  "source_membership": row["source_membership"],
                                  "history_membership": row["personal_membership"]}
    if set(classical) != set(neural):
        raise ValueError("classical paired cells differ")
    for key in neural:
        if (neural[key]["source_membership"] != classical[key]["source_membership"]
                or neural[key]["history_membership"] != classical[key]["history_membership"]):
            raise ValueError("paired classical membership differs")

    participant_rows, summaries = [], []
    rng = np.random.default_rng(SEED)
    for source_size in (300, 1800):
        for history in (0, 60):
            neural_person, classical_person, difference_person = [], [], []
            for target in sorted({key[0] for key in neural}):
                keys = [(target, draw, source_size, history) for draw in (0, 1, 2)]
                nmean = float(np.mean([neural[key]["balanced_accuracy"] for key in keys]))
                cmean = float(np.mean([classical[key]["balanced_accuracy"] for key in keys]))
                participant_rows.append({"target": target, "source_size": source_size,
                                         "h_total": history, "draws": 3,
                                         "neural_mean_ba": nmean,
                                         "classical_plain_ts_mean_ba": cmean,
                                         "neural_minus_plain_ts": nmean - cmean})
                neural_person.append(nmean); classical_person.append(cmean)
                difference_person.append(nmean - cmean)
            cell_keys = [key for key in neural if key[2:] == (source_size, history)]
            neural_bootstrap = bootstrap(neural_person, rng)
            classical_bootstrap = bootstrap(classical_person, rng)
            difference_bootstrap = bootstrap(difference_person, rng)
            summaries.append({
                "source_size": source_size, "h_total": history,
                "cells": len(cell_keys), "participants": 12, "draws_per_participant": 3,
                "neural_balanced_accuracy": neural_bootstrap,
                "classical_plain_ts_balanced_accuracy": classical_bootstrap,
                "paired_neural_minus_plain_ts": difference_bootstrap,
                "paired_neural_minus_plain_ts_percentage_points": {
                    "mean": 100 * float(np.mean(difference_person)),
                    "ci95_percentile": [100 * value for value in difference_bootstrap["ci95_percentile"]]},
                "neural_single_class_cells": int(sum(neural[key]["single_predicted_class"] for key in cell_keys)),
                "neural_mean_p_left_sd_across_cells": float(np.mean([neural[key]["p_left_sd"] for key in cell_keys])),
            })

    selected = []
    for path in sorted(RUN.glob("source_*.json")):
        value = json.loads(path.read_text())
        selected.append({"draw": value["draw"], "source_size": value["source_size"],
                         "selected_epochs": value["selected_epochs"],
                         "best_source_validation_loss": value["best_source_validation_loss"],
                         "source_resubstitution_balanced_accuracy": value["source_resubstitution_metrics"]["balanced_accuracy"],
                         "source_resubstitution_p_left_sd": value["source_resubstitution_metrics"]["p_left_sd"],
                         "source_resubstitution_single_class": value["source_resubstitution_metrics"]["single_predicted_class"]})
    selected.sort(key=lambda row: (row["draw"], row["source_size"]))

    fields = list(participant_rows[0])
    with (OUT / "participant_means.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(participant_rows)
    summary = {
        "schema": "review4-eegnet-successor-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "retrospective development evaluation of one prospectively frozen source-mechanical successor; not confirmation",
        "bootstrap_unit": "participant mean across the same three draws",
        "bootstrap_seed": SEED, "bootstrap_resamples": RESAMPLES,
        "prediction_reconstruction": {"operations": 144, "trials": 14400,
                                      "all_saved_metrics_matched": True},
        "selected_source_epochs": selected,
        "by_source_size_and_history": summaries,
        "inputs": {str(path.relative_to(ROOT)): expected for path, expected in EXPECTED.items()},
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.monotonic() - start_wall,
    }
    write_json(OUT / "summary.json", summary)
    receipt = {"status": "complete", "summary_sha256": sha(OUT / "summary.json"),
               "participant_means_sha256": sha(OUT / "participant_means.csv"),
               "cpu_seconds": time.process_time() - start_cpu,
               "wall_seconds": time.monotonic() - start_wall}
    write_json(OUT / "receipt.json", receipt)


if __name__ == "__main__":
    main()
