#!/usr/bin/env python3
"""Render four review2 figures from completed saved-score analysis only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(rows, count, label):
    if len(rows) != count:
        raise ValueError(f"{label}: expected {count} rows, found {len(rows)}")
    people = {row["target_subject"] for row in rows if "target_subject" in row}
    if people and len(people) != 12:
        raise ValueError(f"{label}: expected 12 participants")
    return rows


def save(figure, out, name):
    for suffix in ("png", "svg"):
        figure.savefig(out / f"{name}.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--analysis-dir", required=True, type=Path); parser.add_argument("--out-dir", required=True, type=Path); args = parser.parse_args()
    analysis, out = args.analysis_dir.resolve(), args.out_dir.resolve()
    if out.exists(): raise ValueError("out-dir must be new")
    pairs = read(analysis / "per_person_contrasts.csv"); means = read(analysis / "grid_means.csv")
    summary = json.loads((analysis / "summary.json").read_text(encoding="utf-8")); out.mkdir(parents=True)
    plotted = {}
    plt.rcParams.update({"font.size":10, "axes.titlesize":11, "axes.labelsize":10})

    # 1. Donor effects: two separate source-size panels, paired participant points only.
    donor_names = ("own_vs_pooled", "own_vs_single_average")
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    donor_plot = {}
    for axis, source in zip(axes, (40, 100)):
        series = []
        for index, name in enumerate(donor_names):
            rows = require([r for r in pairs if r["contrast"] == f"C_S{source}_{name}"], 12, f"donor S{source} {name}")
            ordered = sorted(rows, key=lambda r: int(r["target_subject"]))
            ids = [r["target_subject"] for r in ordered]
            values = [100 * float(r["difference"]) for r in ordered]
            assert dict(zip(ids, values)) == {r["target_subject"]: 100 * float(r["difference"]) for r in rows}
            x = np.full(12, index) + np.linspace(-.11, .11, 12)
            axis.scatter(x, values, s=24, color=("#0072B2", "#D55E00")[index], label=("Own − pooled", "Own − mean single donor")[index])
            series.append({"label": ("own_minus_pooled", "own_minus_single_average")[index], "participant_ids": ids, "difference_percentage_points": values})
        axis.axhline(0, color="black", linewidth=.7); axis.set_xlim(-.45, 1.45); axis.set_xticks([0,1], ["Own − pooled", "Own − single"]); axis.set_title(f"Source base {source}")
        axis.set_ylabel("Balanced-accuracy difference (pp)")
        donor_plot[str(source)] = series
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8)
    figure.suptitle("Matched donor controls")
    save(figure, out, "donor_controls"); plotted["donor_controls"] = donor_plot

    # 2. Four weight-by-representation groups, mean/CI from summary plus all participant effects.
    figure, axis = plt.subplots(figsize=(6.8, 3.8)); groups=[]
    for source in (100,1800):
        for rep in ("source_frozen","pooled_refit"):
            rows = require([r for r in pairs if r["contrast"] == "B_lambda_0.5_vs_pooled" and r["source_n"] == str(source) and r["representation"] == rep], 12, f"weight S{source} {rep}")
            record = next((r for r in summary["summaries"] if r["contrast"] == "B_lambda_0.5_vs_pooled" and r.get("source_n") == str(source) and r.get("representation") == rep), None)
            if record is None: raise ValueError("missing fixed bootstrap summary")
            groups.append((f"S{source}\n{rep.replace('_',' ')}", rows, record))
    for index,(label, rows, record) in enumerate(groups):
        ordered=sorted(rows,key=lambda r:int(r["target_subject"])); ids=[r["target_subject"] for r in ordered]
        values=[100*float(r["difference"]) for r in ordered]; assert dict(zip(ids,values)) == {r["target_subject"]:100*float(r["difference"]) for r in rows}
        ci=[100*x for x in record["descriptive_ci95"]]; center=100*record["mean"]
        axis.errorbar(index, center, yerr=[[center-ci[0]],[ci[1]-center]], fmt="o", color="#009E73", capsize=4, zorder=3)
        axis.scatter(np.full(12,index)+np.linspace(-.12,.12,12),values,s=18,color="#595959",alpha=.7,zorder=2)
    axis.axhline(0,color="black",linewidth=.7); axis.set_xticks(range(4),[x[0] for x in groups]); axis.set_ylabel("λ=0.5 − pooled (pp)"); axis.set_title("Weighting contrast by source size and representation")
    save(figure,out,"weight_representation"); plotted["weight_representation"]=[{"group":g[0],"participant_ids":[r["target_subject"] for r in sorted(g[1],key=lambda r:int(r["target_subject"]))],"participant_difference_percentage_points":[100*float(r["difference"]) for r in sorted(g[1],key=lambda r:int(r["target_subject"]))],"mean_percentage_points":100*g[2]["mean"],"descriptive_ci95_percentage_points":[100*x for x in g[2]["descriptive_ci95"]]} for g in groups]

    # 3. Four absolute-BA EA conditions, thin within-person lines.
    historical = require(summary["historical_unaligned_reference"],24,"historical reference")
    plain0={r["target_subject"]:float(r["balanced_accuracy"]) for r in historical if int(r["h_total"])==0}
    plain60rows=require([r for r in means if r["module"]=="B" and r["source_n"]=="1800" and r["h_total"]=="60" and r["C"]=="1" and r["lambda_personal"]=="pooled" and r["representation"]=="source_frozen"],12,"plain h60")
    ea0rows=require([r for r in means if r["module"]=="D" and r["h_total"]=="0" and r["representation"]=="source_frozen"],12,"EA h0")
    ea60rows=require([r for r in means if r["module"]=="D" and r["h_total"]=="60" and r["representation"]=="source_frozen"],12,"EA h60")
    condition=[("Plain h0",plain0), ("Plain h60",{r["target_subject"]:float(r["mean_balanced_accuracy"]) for r in plain60rows}), ("EA h0",{r["target_subject"]:float(r["mean_balanced_accuracy"]) for r in ea0rows}), ("EA h60",{r["target_subject"]:float(r["mean_balanced_accuracy"]) for r in ea60rows})]
    if any(set(values)!=(set(plain0)) for _,values in condition): raise ValueError("EA condition participant sets differ")
    figure, axis=plt.subplots(figsize=(6.8,3.8)); people=sorted(plain0,key=int)
    for person in people: axis.plot(range(4),[100*values[person] for _,values in condition],color="#999999",linewidth=.6,alpha=.55)
    condition_means=[100*np.mean(list(values.values())) for _,values in condition]
    axis.plot(range(4),condition_means,color="#CC79A7",marker="o",linewidth=2,label="Participant mean")
    axis.set_xticks(range(4),[x[0] for x in condition]); axis.set_ylabel("Balanced accuracy (%)"); axis.set_title("Alignment and personal-label conditions"); axis.legend(frameon=False)
    save(figure,out,"ea_conditions"); plotted["ea_conditions"]={"participant_ids":people,"conditions":[{"label":label,"balanced_accuracy": [values[p] for p in people],"mean_percentage":100*np.mean(list(values.values()))} for label,values in condition]}

    # 4. Source-only regularization path: 12 participant means at every C×S condition.
    figure,axis=plt.subplots(figsize=(6.8,3.8)); reg={}
    for c,color in (("0.1","#0072B2"),("1","#000000"),("10","#D55E00")):
        xs=[]; ys=[]; series=[]
        for source in (40,70,90,100):
            rows=require([r for r in means if r["module"]=="A" and r["source_n"]==str(source) and r["C"]==c],12,f"regularization S{source} C{c}")
            ordered=sorted(rows,key=lambda r:int(r["target_subject"])); ids=[r["target_subject"] for r in ordered]; values=[float(r["mean_balanced_accuracy"]) for r in ordered]
            assert dict(zip(ids,values)) == {r["target_subject"]:float(r["mean_balanced_accuracy"]) for r in rows}
            xs.append(source); ys.append(100*np.mean(values)); series.append({"source_n":source,"participant_ids":ids,"participant_balanced_accuracy":values,"mean_percentage":100*np.mean(values)})
        axis.plot(xs,ys,marker="o",color=color,label=f"C = {c}"); reg[c]=series
    axis.set_xticks([40,70,90,100]); axis.set_xlabel("Source trials"); axis.set_ylabel("Balanced accuracy (%)"); axis.set_title("Source-only regularization path"); axis.legend(frameon=False)
    save(figure,out,"regularization_path"); plotted["regularization_path"]=reg

    (out/"plotted_values.json").write_text(json.dumps(plotted,indent=2)+"\n")
    receipt={"status":"completed","inputs":{str(analysis/"per_person_contrasts.csv"):sha(analysis/"per_person_contrasts.csv"),str(analysis/"grid_means.csv"):sha(analysis/"grid_means.csv"),str(analysis/"summary.json"):sha(analysis/"summary.json")},"outputs":{p.name:sha(p) for p in out.iterdir() if p.is_file()},"script_sha256":sha(__file__),"transformations":"balanced-accuracy differences and means multiplied by 100 only for percentage-point/percent axes","thread_limits":{"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":""}}
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__": main()
