#!/usr/bin/env python3
"""Reconstruct the focused third-dataset extension from frozen predictions."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_review3 as core
import analyze_review3_extensions as extensions

SEED = 2026092403


def summary(values):
    values = np.round(np.asarray(values, dtype=float), 12)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("Expected nine complete participant values")
    resamples = np.random.default_rng(SEED).integers(0, 9, size=(10000, 9))
    means = values[resamples].mean(axis=1)
    return {"n": 9, "mean_pp": float(values.mean()), "median_pp": float(np.median(values)),
            "ci_low_pp": float(np.quantile(means, .025)), "ci_high_pp": float(np.quantile(means, .975)),
            "positive": int(np.sum(values > 1e-10)), "negative": int(np.sum(values < -1e-10)),
            "zero": int(np.sum(np.abs(values) <= 1e-10)), "values_pp": values.tolist()}


def run(grid_dir, grid_plan, donor_dir, donor_plan, output):
    if output.exists():
        raise FileExistsError(output)
    started, cpu = time.monotonic(), time.process_time()
    donor_receipt = json.loads((donor_dir / "receipt.json").read_text())
    if donor_receipt.get("plan_receipt_sha256") != core.sha(donor_plan / "PLAN_RECEIPT.json"):
        raise ValueError("Donor run is not bound to the supplied frozen donor plan")
    plan_receipt = json.loads((donor_plan / "PLAN_RECEIPT.json").read_text())
    if plan_receipt.get("files", {}).get("donor_identity.csv") != core.sha(donor_plan / "donor_identity.csv"):
        raise ValueError("Donor identity records changed after planning")
    rows, grid_check = core.reconstruct_grid("bnci2b", grid_dir, grid_plan)
    if len(rows) != 1800:
        raise ValueError("Focused third grid must have 1800 operations")
    grouped = defaultdict(list)
    fields = ["dataset_variant", "study_id", "method", "source_size", "source_n", "donor_count", "h_total", "target"]
    for row in rows:
        grouped[tuple(row[key] for key in fields)].append(row)
    people = []
    for key, group in grouped.items():
        if len(group) != 10 or {int(row["draw"]) for row in group} != set(range(10)):
            raise ValueError("Every grid person coordinate requires ten draws")
        people.append(dict(zip(fields, key), draws=10, balanced_accuracy=float(np.mean([row["balanced_accuracy"] for row in group]))))
    by_curve = defaultdict(list)
    for row in people:
        by_curve[row["method"], str(row["source_size"]), int(row["h_total"])].append(row)
    curves = []
    for (method, source, history), group in sorted(by_curve.items()):
        group = sorted(group, key=lambda row: int(row["target"]))
        if [int(row["target"]) for row in group] != list(range(1, 10)):
            raise ValueError("Curve participants differ from all nine people")
        stats = summary([100 * row["balanced_accuracy"] for row in group])
        curves.append({"dataset_variant": "bnci2b", "method": method, "source_size": source,
                       "source_n_min": min(int(row["source_n"]) for row in group),
                       "source_n_max": max(int(row["source_n"]) for row in group),
                       "h_total": history, "n": 9, "mean_ba_percent": stats["mean_pp"],
                       "ci_low_percent": stats["ci_low_pp"], "ci_high_percent": stats["ci_high_pp"]})
    participant, contrasts, donor_scores = [], [], []
    _, donor_check = extensions.analyze_donor({"label": "bnci2b", "directory": str(donor_dir.relative_to(ROOT)), "required": True}, participant, contrasts, donor_scores)
    if len(donor_scores) != 1080:
        raise ValueError("Donor extension must have all 1080 operations")
    endpoint_groups = defaultdict(list)
    for row in contrasts:
        endpoint_groups[row["method_or_setting"], row["contrast"]].append(row)
    endpoints = {}
    for (setting, contrast), group in endpoint_groups.items():
        group.sort(key=lambda row: int(row["target"]))
        if [int(row["target"]) for row in group] != list(range(1, 10)):
            raise ValueError("Donor contrast participants differ")
        endpoints[setting + ":" + contrast] = summary([row["value_pp"] for row in group])
    lookup = {(row["method"], str(row["source_size"]), int(row["h_total"]), int(row["target"])): row["balanced_accuracy"] for row in people}
    def vec(method, source, history):
        return 100 * np.array([lookup[method, source, history, target] for target in range(1, 10)])
    grid_effects = {
        "plain_label_gain_S100": summary(vec("plain_ts", "100", 60) - vec("plain_ts", "100", 0)),
        "plain_label_gain_Sall": summary(vec("plain_ts", "all", 60) - vec("plain_ts", "all", 0)),
        "plain_S100_minus_Sall_h60": summary(vec("plain_ts", "100", 60) - vec("plain_ts", "all", 60)),
    }
    identities = list(core.read_rows(donor_plan / "donor_identity.csv"))
    singles = [row for row in identities if row["condition"].startswith("single_")]
    if len(singles) != 270 or any(int(row["added_trial_overlap_with_base"]) != 0 for row in identities):
        raise ValueError("Donor count or added trial disjointness differs")
    primary_key = "tuned_C_source_frozen:own_minus_mean_single"
    primary_rows = sorted(endpoint_groups[("tuned_C_source_frozen", "own_minus_mean_single")], key=lambda row: int(row["target"]))
    figure = {"dataset_variant": "bnci2b", "label": "BNCI2014-004", "contrast": "own_minus_mean_single",
              "method_or_setting": "tuned_C_source_frozen",
              "participant_effects": [{"target": int(row["target"]), "effect_pp": row["value_pp"]} for row in primary_rows],
              "summary": {key: endpoints[primary_key][key] for key in ("n", "mean_pp", "ci_low_pp", "ci_high_pp")}}
    output.mkdir(parents=True)
    for name, values in (("reconstructed_scores.csv", rows), ("participant_means.csv", people),
                         ("curves.csv", curves), ("donor_participant_means.csv", participant),
                         ("donor_paired_contrasts.csv", contrasts), ("reconstructed_donor_scores.csv", donor_scores)):
        core.save_csv(output / name, values)
    payload = {"status": "completed_external_extension", "dataset": "BNCI2014-004", "subjects": list(range(1, 10)),
               "primary_donor_contrast": endpoints[primary_key], "donor_endpoints": endpoints,
               "grid_contrasts": grid_effects, "curves": curves, "verification": {"grid": grid_check, "donor": donor_check},
               "donor_overlap": {"sampled_single_blocks": 270, "donors_already_in_base": sum(int(row["base_trials_from_donor"]) > 0 for row in singles), "added_trial_overlap_violations": 0},
               "bootstrap": {"seed": SEED, "resamples": 10000, "unit": "participant after averaging ten draws and donor blocks"},
               "inference": "descriptive conditional intervals; no new confirmatory significance claim or pooled-dataset test",
               "protected_OpenBMI_accessed": False}
    for name, value in (("summary.json", payload), ("third_donor_figure.json", figure)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    receipt = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "cpu_seconds": time.process_time() - cpu, "wall_seconds": time.monotonic() - started,
               "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               "code_sha256": core.sha(Path(__file__)),
               "shared_helpers": {"scripts/analyze_review3.py": core.sha(ROOT / "scripts/analyze_review3.py"), "scripts/analyze_review3_extensions.py": core.sha(ROOT / "scripts/analyze_review3_extensions.py")},
               "inputs": {str(path.relative_to(ROOT)): core.sha(path) for path in [grid_dir / "receipt.json", grid_plan / "PLAN_RECEIPT.json", donor_dir / "receipt.json", donor_plan / "PLAN_RECEIPT.json", donor_plan / "donor_identity.csv"]},
               "outputs": {path.name: core.sha(path) for path in sorted(output.iterdir())}}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"status": "completed", "primary_donor_contrast": payload["primary_donor_contrast"], "grid_contrasts": grid_effects, "cpu_seconds": receipt["cpu_seconds"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("grid-dir", "grid-plan", "donor-dir", "donor-plan", "out-dir"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.grid_dir.resolve(), args.grid_plan.resolve(), args.donor_dir.resolve(), args.donor_plan.resolve(), args.out_dir.resolve()), indent=2))
