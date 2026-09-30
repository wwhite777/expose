#!/usr/bin/env python3
"""Render review2 control summaries without recalculating any statistics."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def points(rows, contrast, where=lambda row: True):
    return [row for row in rows if row["contrast"] == contrast and where(row)]


def panel(axis, rows, title):
    people = sorted(set(row["target_subject"] for row in rows), key=int)
    labels = sorted(set(row["contrast"] for row in rows))
    colors = plt.get_cmap("tab10").colors
    for offset, label in enumerate(labels):
        values = {row["target_subject"]: float(row["difference"]) * 100 for row in rows if row["contrast"] == label}
        axis.scatter([people.index(p) + offset * .12 for p in values], list(values.values()), s=22, color=colors[offset % len(colors)], label=label.replace("_", " "))
    axis.axhline(0, color="black", linewidth=.7); axis.set_xticks(range(len(people)), people); axis.set_xlabel("Participant"); axis.set_ylabel("Difference (percentage points)"); axis.set_title(title)
    if labels: axis.legend(fontsize=7, frameon=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--analysis-dir", required=True, type=Path); parser.add_argument("--out-dir", required=True, type=Path); args = parser.parse_args()
    analysis, out = args.analysis_dir.resolve(), args.out_dir.resolve()
    if out.exists(): raise ValueError("out-dir must be new")
    pairs_path, means_path, summary_path = analysis / "per_person_contrasts.csv", analysis / "grid_means.csv", analysis / "summary.json"
    pairs, means, summary = read(pairs_path), read(means_path), json.loads(summary_path.read_text())
    out.mkdir(parents=True)
    plt.rcParams.update({"font.size":10, "axes.titlesize":11, "legend.fontsize":7})
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 7.2), constrained_layout=True)
    panel(axes[0,0], [r for r in pairs if r["contrast"].startswith("C_")], "Matched donor controls")
    panel(axes[0,1], points(pairs, "B_lambda_0.5_vs_pooled"), "Weighting by representation")
    drows = [r for r in pairs if r["contrast"].startswith("D_")]
    panel(axes[1,0], drows, "Alignment and label conditions")
    arows = [r for r in pairs if r["contrast"] == "A_C_vs_1"]
    panel(axes[1,1], arows, "Regularization versus C = 1")
    for suffix in ("png", "svg"): figure.savefig(out / f"review2_controls.{suffix}", dpi=300 if suffix == "png" else None)
    plt.close(figure)
    receipt = {"status":"completed","inputs":{str(pairs_path):sha(pairs_path),str(means_path):sha(means_path),str(summary_path):sha(summary_path)},"outputs":{name:sha(out/name) for name in ("review2_controls.png","review2_controls.svg")},"script_sha256":sha(__file__),"transformation":"balanced-accuracy differences multiplied by 100 for percentage-point axes; no statistical recalculation","panels":["donor participant effects","weight/representation","EA four-condition contrasts","regularization"]}
    (out / "receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__": main()
