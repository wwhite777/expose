#!/usr/bin/env python3
"""Render review-3 figures from completed analysis products, without recomputing scores.

The input directories must be complete analysis directories.  This utility only
aggregates already draw-averaged participant values for display and uses the
saved descriptive bootstrap intervals supplied by the analyzers.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CORE_PARTICIPANT = {"dataset_variant", "study_id", "method", "source_size", "source_n",
                    "donor_count", "h_total", "target", "draws", "balanced_accuracy"}
CORE_CURVES = CORE_PARTICIPANT - {"target", "draws", "balanced_accuracy"} | {
    "n", "mean_ba", "ci_low", "ci_high"}
EXT_PARTICIPANT = {"extension", "dataset_variant", "method_or_setting", "variant", "target",
                   "source_size", "h_total", "draw_count", "balanced_accuracy"}
EXT_CONTRAST = {"extension", "dataset_variant", "contrast", "method_or_setting", "target",
                "source_size", "h_total", "draw_count", "value_pp"}
TARGET_ONLY_PARTICIPANT = {"dataset_variant", "target", "h_total", "draw_count", "balanced_accuracy"}
DECODABILITY_PARTICIPANT = {"dataset_variant", "method", "source_size", "h_total", "target",
                            "historical_ba", "historical_across_draw_sd", "future_gain_pp"}
METHODS = ("plain_ts", "ea_ts", "plain_mdm", "recenter_mdm", "mdwm")
METHOD_LABELS = {"plain_ts": "TS–LR", "ea_ts": "EA–TS–LR", "plain_mdm": "MDM",
                 "recenter_mdm": "Recenter–MDM", "mdwm": "MDWM"}
COLORS = {"plain_ts": "#0072B2", "ea_ts": "#D55E00", "plain_mdm": "#56B4E9",
          "recenter_mdm": "#8C564B", "mdwm": "#009E73", "eegnet": "#CC79A7"}
METHOD_STYLES = {"plain_ts": {"marker": "o", "linestyle": "-"},
                 "ea_ts": {"marker": "s", "linestyle": "-"},
                 "plain_mdm": {"marker": "^", "linestyle": "--"},
                 "recenter_mdm": {"marker": "P", "linestyle": "--"},
                 "mdwm": {"marker": "D", "linestyle": "-."}}
FULL_SOURCE = {"openbmi8": "1800", "bnci": "all"}
DISPLAY_DATASETS = ("openbmi8", "bnci")
ENDPOINTS = (
    ("plain_label_gain_S100", "Plain label gain\nS=100"),
    ("source_context_interaction", "Source-context\ninteraction"),
    ("mdwm_minus_plain_Sall_h60", "MDWM − plain\nS=all, h=60"),
    ("historical_EA_minus_plain_Sall_h0", "EA − plain\nS=all, h=0"),
    ("alignment_label_interaction_Sall", "EA label\ninteraction"),
    ("donor_own_minus_mean_single_tuned", "Own − mean\nsingle donor"),
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path, required):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        got = list(csv.DictReader(f))
    if not got or set(got[0]) != required:
        raise ValueError(f"unexpected or empty schema: {path}")
    if any(set(r) != required for r in got):
        raise ValueError(f"inconsistent CSV row schema: {path}")
    return got


def number(row, key):
    value = float(row[key])
    if not value == value or value in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {key}")
    return value


def require_file(directory, name):
    path = Path(directory) / name
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular file missing: {path}")
    return path


def completed(directory, required):
    """Bind files guaranteed by the analyzer schemas, without inventing receipt fields."""
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"analysis directory is not a regular directory: {directory}")
    paths = {name: require_file(directory, name) for name in required}
    receipt = directory / "receipt.json"
    if receipt.is_file() and not receipt.is_symlink():
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            raise ValueError(f"present analysis receipt is not completed: {receipt}")
        paths["receipt.json"] = receipt
    return paths


def source_stats(summary, identifier):
    entry = summary["extension_contrast_effects"].get(identifier)
    if not isinstance(entry, dict) or not {"mean_pp", "ci_low_pp", "ci_high_pp"} <= set(entry):
        raise ValueError(f"missing saved contrast statistics: {identifier}")
    return (float(entry["mean_pp"]), float(entry["ci_low_pp"]), float(entry["ci_high_pp"]))


def full_source_alignment_rows(entries, dataset, contrast, h_total, expected_targets):
    """Select exactly the complete-source, ten-draw alignment participant rows."""
    source = FULL_SOURCE[dataset]
    selected = [row for row in entries if row["contrast"] == contrast and row["h_total"] == h_total
                and row["source_size"] == source]
    targets = [int(row["target"]) for row in selected]
    if set(targets) != expected_targets or len(targets) != len(set(targets)):
        raise ValueError(f"incomplete or duplicate complete-source alignment rows: {dataset}/S{source}/{contrast}/h{h_total}")
    if {row["draw_count"] for row in selected} != {"10"}:
        raise ValueError(f"alignment rows must be ten-draw participant means: {dataset}/S{source}/{contrast}/h{h_total}")
    return selected


def core_effects(core_people, core_summary, ext_contrasts, ext_summary, dataset):
    full = FULL_SOURCE[dataset]
    selected = [r for r in core_people if r["dataset_variant"] == dataset and r["study_id"] ==
                ("openbmi_main" if dataset == "openbmi8" else "bnci_main") and r["donor_count"] == "all"]
    targets = sorted({int(r["target"]) for r in selected})
    lookup = {(r["method"], r["source_size"], r["h_total"], int(r["target"])): number(r, "balanced_accuracy")
              for r in selected}
    def vector(name):
        if name == "plain_label_gain_S100":
            return [100 * (lookup["plain_ts", "100", "60", t] - lookup["plain_ts", "100", "0", t]) for t in targets]
        if name == "source_context_interaction":
            return [100 * ((lookup["plain_ts", "100", "60", t] - lookup["plain_ts", "100", "0", t]) -
                           (lookup["plain_ts", full, "60", t] - lookup["plain_ts", full, "0", t])) for t in targets]
        if name == "mdwm_minus_plain_Sall_h60":
            return [100 * (lookup["mdwm", full, "60", t] - lookup["plain_ts", full, "60", t]) for t in targets]
        if name == "historical_EA_minus_plain_Sall_h0":
            return [100 * (lookup["ea_ts", full, "0", t] - lookup["plain_ts", full, "0", t]) for t in targets]
        if name == "alignment_label_interaction_Sall":
            return [100 * ((lookup["ea_ts", full, "60", t] - lookup["ea_ts", full, "0", t]) -
                           (lookup["plain_ts", full, "60", t] - lookup["plain_ts", full, "0", t])) for t in targets]
        if name == "donor_own_minus_mean_single_tuned":
            chosen = [r for r in ext_contrasts if r["extension"] == "donor" and r["dataset_variant"] == dataset
                      and r["method_or_setting"] == "tuned_C_source_frozen"
                      and r["contrast"] == "own_minus_mean_single"]
            values = {int(r["target"]): number(r, "value_pp") for r in chosen}
            if set(values) != set(targets):
                raise ValueError(f"incomplete donor endpoint for {dataset}")
            return [values[t] for t in targets]
        raise KeyError(name)
    effects = core_summary.get("endpoint_results", {}).get(dataset, {}).get("contrasts", {})
    output = {}
    for name, _ in ENDPOINTS:
        values = vector(name)
        if name == "donor_own_minus_mean_single_tuned":
            stats = ext_summary.get("donor_endpoints", {}).get(dataset)
            if not isinstance(stats, dict):
                raise ValueError(f"missing saved donor summary for {dataset}")
        else:
            stats = effects.get(name)
        if not isinstance(stats, dict) or not {"mean_pp", "ci_low_pp", "ci_high_pp"} <= set(stats):
            raise ValueError(f"missing saved endpoint statistics {dataset}/{name}")
        output[name] = {"targets": targets, "values": values,
                        "mean": float(stats["mean_pp"]), "low": float(stats["ci_low_pp"]),
                        "high": float(stats["ci_high_pp"])}
    return output


def point_summary(ax, x, values, stats, color):
    ax.scatter([x] * len(values), values, color="#666666", alpha=.72, s=20, zorder=2)
    ax.errorbar(x, stats["mean"], yerr=[[stats["mean"] - stats["low"]],
                                         [stats["high"] - stats["mean"]]], fmt="o", color=color,
                capsize=3, lw=1.4, zorder=3)


def forest_summary(ax, y, values, stats, color):
    ax.scatter(values, [y] * len(values), color="#666666", alpha=.72, s=18, zorder=2)
    ax.errorbar(stats["mean"], y, xerr=[[stats["mean"] - stats["low"]],
                                      [stats["high"] - stats["mean"]]], fmt="o", color=color,
                capsize=3, lw=1.35, zorder=3)


def set_effect_axis(ax, title):
    ax.axhline(0, color="#333333", lw=.8, zorder=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("Paired difference (percentage points)")
    ax.grid(axis="y", color="#dddddd", lw=.6)


def output_path(out, stem, plotted):
    return {"png": f"{stem}.png", "pdf": f"{stem}.pdf", "plotted_series": plotted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-dir", type=Path, required=True)
    parser.add_argument("--extensions-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--target-only-dir", type=Path,
                        help="optional completed secondary target-only analysis directory")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("out-dir must be new and exclusive")

    core_paths = completed(args.core_dir, ("participant_means.csv", "curves.csv", "summary.json"))
    ext_paths = completed(args.extensions_dir, ("participant_means.csv", "paired_contrasts.csv", "summary.json"))
    sens_paths = completed(args.sensitivity_dir, ("summary.json", "decodability_participants.csv"))
    core_people = rows(core_paths["participant_means.csv"], CORE_PARTICIPANT)
    curves = rows(core_paths["curves.csv"], CORE_CURVES)
    ext_people = rows(ext_paths["participant_means.csv"], EXT_PARTICIPANT)
    ext_contrasts = rows(ext_paths["paired_contrasts.csv"], EXT_CONTRAST)
    decodability_people = rows(sens_paths["decodability_participants.csv"], DECODABILITY_PARTICIPANT)
    core_summary = json.loads(core_paths["summary.json"].read_text(encoding="utf-8"))
    ext_summary = json.loads(ext_paths["summary.json"].read_text(encoding="utf-8"))
    target_only_rows, target_only_summary, target_only_paths = [], {}, {}
    if args.target_only_dir is not None:
        target_only_paths = completed(args.target_only_dir, ("participant_means.csv", "summary.json"))
        target_only_rows = rows(target_only_paths["participant_means.csv"], TARGET_ONLY_PARTICIPANT)
        target_only_summary = json.loads(target_only_paths["summary.json"].read_text(encoding="utf-8"))
        if target_only_summary.get("schema") != "review3-target-only-analysis-inputs-v1":
            raise ValueError("target-only analysis summary schema differs")
        if {r["dataset_variant"] for r in target_only_rows} != set(DISPLAY_DATASETS):
            raise ValueError("target-only analysis lacks a required dataset")
    if not all(dataset in {r["dataset_variant"] for r in core_people} for dataset in DISPLAY_DATASETS):
        raise ValueError("core analysis lacks one required dataset")
    args.out_dir.mkdir(parents=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.hashsalt": "review3-figures-v1"})

    plotted, index = [], {"schema": "review3-figure-index-v1", "figures": [],
                           "input_sha256": {str(p): digest(p) for p in
                            [*core_paths.values(), *ext_paths.values(), *sens_paths.values(), *target_only_paths.values()]},
                           "notes": ["Balanced accuracy values are saved analysis values; this script does not recompute classifier outcomes.",
                                     "Intervals are analyzer-supplied descriptive participant-bootstrap 95% intervals.",
                                     "No figure contains significance stars.",
                                     "Sensitivity directory is provenance-bound; montage, offset, and donor-diversity plots are not required primary figures."]}
    def save(fig, stem, caption, series):
        fig.tight_layout(rect=getattr(fig, "_review3_layout_rect", (0, 0, 1, 1)))
        fig.savefig(args.out_dir / f"{stem}.png", dpi=220, bbox_inches="tight", metadata={"Software": "plot_review3"})
        with PdfPages(args.out_dir / f"{stem}.pdf", metadata={"Title": stem, "Creator": "plot_review3", "CreationDate": None}) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        item = output_path(args.out_dir, stem, series); item["caption"] = caption; index["figures"].append(item)

    # 1. Four core dose panels.
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.1), sharex=True, sharey=True)
    fig._review3_layout_rect = (0, .16, 1, 1)
    hvals = sorted({int(r["h_total"]) for r in curves if r["dataset_variant"] in DISPLAY_DATASETS})
    if 0 not in hvals:
        raise ValueError("core dose grid lacks h=0 reference")
    displayed_yvals = []
    for row, dataset in enumerate(DISPLAY_DATASETS):
        for col, source in enumerate(("100", FULL_SOURCE[dataset])):
            ax = axes[row, col]
            selected = [r for r in curves if r["dataset_variant"] == dataset and r["study_id"] ==
                        ("openbmi_main" if dataset == "openbmi8" else "bnci_main") and r["source_size"] == source
                        and r["donor_count"] == "all" and r["method"] in METHODS]
            for method in METHODS:
                cells = {int(r["h_total"]): r for r in selected if r["method"] == method}
                if set(cells) != set(hvals):
                    raise ValueError(f"incomplete core curve: {dataset}/S{source}/{method}")
                means = [number(cells[h], "mean_ba") for h in hvals]
                lo = [number(cells[h], "ci_low") for h in hvals]; hi = [number(cells[h], "ci_high") for h in hvals]
                displayed_yvals.extend(lo + hi)
                ax.errorbar(hvals, means, yerr=[np.subtract(means, lo), np.subtract(hi, means)],
                            color=COLORS[method], lw=1.45, capsize=2.2, label=METHOD_LABELS[method],
                            zorder=3, **METHOD_STYLES[method])
                for h, mean, low, high in zip(hvals, means, lo, hi):
                    plotted.append({"figure": "core_dose_curves", "dataset": dataset, "source_size": source,
                                    "series": method, "target": "mean", "x": h, "value": mean,
                                    "ci_low": low, "ci_high": high, "unit": "balanced_accuracy_percent"})
            plain = [r for r in core_people if r["dataset_variant"] == dataset and r["study_id"] ==
                     ("openbmi_main" if dataset == "openbmi8" else "bnci_main") and r["source_size"] == source
                     and r["donor_count"] == "all" and r["method"] == "plain_ts"]
            by_target = defaultdict(dict)
            for r in plain: by_target[int(r["target"])][int(r["h_total"])] = 100 * number(r, "balanced_accuracy")
            if any(set(v) != set(hvals) for v in by_target.values()): raise ValueError("incomplete plain participant curve")
            for target, vals in sorted(by_target.items()):
                displayed_yvals.extend(vals[h] for h in hvals)
                ax.plot(hvals, [vals[h] for h in hvals], color="#999999", alpha=.34, lw=.7, zorder=1)
                for h in hvals: plotted.append({"figure":"core_dose_curves", "dataset":dataset, "source_size":source,
                    "series":"plain_ts_participant", "target":target, "x":h, "value":vals[h], "ci_low":"", "ci_high":"", "unit":"balanced_accuracy_percent"})
            neural = [r for r in ext_people if r["extension"] == "neural" and r["dataset_variant"] == dataset
                      and r["source_size"] == source]
            if neural:
                by_h = defaultdict(list)
                for r in neural: by_h[int(r["h_total"])].append(100 * number(r, "balanced_accuracy"))
                nh = sorted(by_h)
                displayed_yvals.extend(value for h in nh for value in by_h[h])
                ax.plot(nh, [float(np.mean(by_h[h])) for h in nh], marker="s", linestyle="--", color=COLORS["eegnet"],
                        lw=1.6, label="EEGNet (3 draws)", zorder=4)
                for h in nh:
                    for value in by_h[h]: plotted.append({"figure":"core_dose_curves", "dataset":dataset, "source_size":source,
                        "series":"eegnet_3draw", "target":"participant", "x":h, "value":value, "ci_low":"", "ci_high":"", "unit":"balanced_accuracy_percent"})
            if target_only_rows:
                by_h = defaultdict(list)
                for r in target_only_rows:
                    if r["dataset_variant"] == dataset:
                        by_h[int(r["h_total"])].append(100 * number(r, "balanced_accuracy"))
                target_h = sorted(by_h)
                expected_ids = sorted({int(r["target"]) for r in target_only_rows if r["dataset_variant"] == dataset})
                if target_h != [h for h in hvals if h != 0] or any(len(by_h[h]) != len(expected_ids) for h in target_h):
                    raise ValueError("target-only plotted participant inventory differs from core dose grid")
                curve = target_only_summary["target_only_curves"]
                stats = [curve.get(f"{dataset}__h{h}") for h in target_h]
                if any(not isinstance(v, dict) or not {"mean_pp", "ci_low_pp", "ci_high_pp"} <= set(v) for v in stats):
                    raise ValueError("target-only saved curve interval is missing")
                means = [float(v["mean_pp"]) for v in stats]; lo = [float(v["ci_low_pp"]) for v in stats]; hi = [float(v["ci_high_pp"]) for v in stats]
                displayed_yvals.extend(lo + hi)
                ax.errorbar(target_h, means, yerr=[np.subtract(means, lo), np.subtract(hi, means)], marker="D", linestyle=":", color="#CC79A7", lw=1.5, capsize=2.5, zorder=5, label="Personal-only C=1")
                for h, mean, low, high in zip(target_h, means, lo, hi):
                    plotted.append({"figure":"core_dose_curves", "dataset":dataset, "source_size":source, "series":"target_only_C1", "target":"mean", "x":h, "value":mean, "ci_low":low, "ci_high":high, "unit":"balanced_accuracy_percent"})
            ax.set_title(f"{'OpenBMI-8' if dataset == 'openbmi8' else 'BNCI'}; S={source}", fontweight="bold", fontsize=9)
            ax.set_xticks(hvals); ax.grid(axis="y", color="#dddddd", lw=.6)
            ax.tick_params(labelsize=8)
            if row == 1: ax.set_xlabel("Historical labels (h)", fontsize=8.5)
            if col == 0: ax.set_ylabel("Balanced accuracy (%)", fontsize=8.5)
    if not displayed_yvals:
        raise ValueError("core figure has no displayed values")
    ymin, ymax = max(0, min(displayed_yvals) - 2), min(100, max(displayed_yvals) + 2)
    if ymin >= ymax:
        raise ValueError("invalid displayed core curve bounds")
    for ax in axes.flat:
        ax.set_ylim(ymin, ymax)
    handles, labels = [], []
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle); labels.append(label)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005), ncol=4,
               fontsize=8, frameon=False, columnspacing=.9, handlelength=1.8, handletextpad=.4)
    save(fig, "core_dose_curves", "Core dose curves. Thin gray lines are individual plain TS-LR participants; colored lines show saved participant means and descriptive 95% intervals. The five classical arms are shown in every panel." + (" The optional target-only overlay uses fixed C=1 and is not source-tuned." if target_only_rows else "") + (" EEGNet uses three draws; classical curves use ten draws." if any(r["extension"] == "neural" for r in ext_people) else ""), ["core curves"])

    # 2. Six external/development paired endpoints. `x` in plotted_values.csv
    # remains the endpoint category index even though this is a horizontal plot.
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.5), sharey=True)
    for ax, dataset in zip(axes, DISPLAY_DATASETS):
        effects = core_effects(core_people, core_summary, ext_contrasts, ext_summary, dataset)
        ax.axvline(0, color="#333333", lw=.8, zorder=0)
        for y, (name, label) in enumerate(ENDPOINTS):
            stat = effects[name]; forest_summary(ax, y, stat["values"], stat, "#0072B2")
            for target, value in zip(stat["targets"], stat["values"]): plotted.append({"figure":"paired_endpoints", "dataset":dataset, "source_size":"endpoint", "series":name, "target":target, "x":y, "value":value, "ci_low":stat["low"], "ci_high":stat["high"], "unit":"percentage_points"})
        ax.set_title("External BNCI replication" if dataset == "bnci" else "OpenBMI development", fontsize=10, fontweight="bold")
        ax.set_xlabel("Paired difference (percentage points)", fontsize=8.5)
        ax.set_yticks(range(len(ENDPOINTS)), [label for _, label in ENDPOINTS], fontsize=7.5)
        ax.invert_yaxis(); ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#dddddd", lw=.6)
    axes[1].tick_params(axis="y", labelleft=False)
    save(fig, "paired_endpoints", "Six paired endpoint contrasts. Dots are participants; blue markers and bars are saved participant means and descriptive 95% intervals.", [x[0] for x in ENDPOINTS])

    # 3. Donor controls.
    donor_names = ("own_minus_pooled", "own_minus_mean_single")
    donor_labels = ("Own − pooled other", "Own − mean single other")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.5), sharey=True)
    for ax, dataset in zip(axes, DISPLAY_DATASETS):
        for offset, setting in [(-.16, "legacy_C1_pooled_refit"), (.16, "tuned_C_source_frozen")]:
            for x, name in enumerate(donor_names):
                chosen = [r for r in ext_contrasts if r["extension"] == "donor" and r["dataset_variant"] == dataset
                          and r["method_or_setting"] == setting and r["contrast"] == name]
                if not chosen: raise ValueError(f"missing donor condition {dataset}/{setting}/{name}")
                ident = f"donor__{dataset}__{name}__{setting}__S100__h60"
                stat = source_stats(ext_summary, ident); values = [number(r, "value_pp") for r in chosen]
                color = "#666666" if setting.startswith("legacy") else "#D55E00"
                ax.scatter([x + offset] * len(values), values, color=color, alpha=.6, s=18, zorder=2)
                ax.errorbar(x + offset, stat[0], yerr=[[stat[0]-stat[1]], [stat[2]-stat[0]]], fmt="o", color=color, capsize=3, zorder=3,
                            label=("Legacy C=1 pooled-refit" if setting.startswith("legacy") else "Source-tuned frozen") if x == 0 else None)
                for r, value in zip(chosen, values): plotted.append({"figure":"donor_controls", "dataset":dataset, "source_size":"100", "series":setting+"/"+name, "target":r["target"], "x":x, "value":value, "ci_low":stat[1], "ci_high":stat[2], "unit":"percentage_points"})
        set_effect_axis(ax, "OpenBMI development" if dataset == "openbmi8" else "BNCI2014-001")
        ax.set_xticks(range(2), donor_labels, fontsize=8); ax.tick_params(axis="y", labelsize=8); ax.legend(fontsize=7.5, loc="upper left")
    save(fig, "donor_controls", "Matched-donor controls at base 100 plus 60 historical labels. Dots are draw-averaged participants; bars use saved descriptive intervals.", list(donor_names))

    # 4. Current-session alignment: prefix/tail and complete-batch variants on matched rows.
    align_names = ("current_minus_historical_tail", "current_minus_historical_all")
    align_labels = ("Prefix-20\ncurrent whitening\n(tail score)", "Full-batch\ncurrent whitening\n(transductive)")
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.5), sharey=True)
    alignment_expected = {}
    for ax, dataset in zip(axes, DISPLAY_DATASETS):
        source = FULL_SOURCE[dataset]
        expected_targets = {int(r["target"]) for r in core_people if r["dataset_variant"] == dataset
                            and r["study_id"] == ("openbmi_main" if dataset == "openbmi8" else "bnci_main")
                            and r["donor_count"] == "all" and r["source_size"] == source}
        expected_n = 12 if dataset == "openbmi8" else 9
        if len(expected_targets) != expected_n:
            raise ValueError(f"unexpected complete-source participant inventory: {dataset}/S{source}")
        alignment_expected[dataset] = expected_targets
        entries = [r for r in ext_contrasts if r["extension"] == "current_alignment" and r["dataset_variant"] == dataset
                   and r["contrast"] in align_names and r["h_total"] in ("0", "60")]
        if not entries: raise ValueError(f"missing current-alignment rows for {dataset}")
        for h, offset, color in (("0", -.17, "#0072B2"), ("60", .17, "#D55E00")):
            for x, name in enumerate(align_names):
                selected = full_source_alignment_rows(entries, dataset, name, h, expected_targets)
                ident = f"current_alignment__{dataset}__{name}__ea_ts__S{source}__h{h}"
                stat = source_stats(ext_summary, ident); values = [number(r, "value_pp") for r in selected]
                ax.scatter([x+offset]*len(values), values, color=color, alpha=.6, s=18, zorder=2)
                ax.errorbar(x+offset, stat[0], yerr=[[stat[0]-stat[1]], [stat[2]-stat[0]]], fmt="o", color=color, capsize=3, zorder=3,
                            label=f"h={h}" if x == 0 else None)
                for r, value in zip(selected, values): plotted.append({"figure":"current_alignment", "dataset":dataset, "source_size":source, "series":name+f"/h{h}", "target":r["target"], "x":x, "value":value, "ci_low":stat[1], "ci_high":stat[2], "unit":"percentage_points"})
        set_effect_axis(ax, f"{'OpenBMI development' if dataset == 'openbmi8' else 'BNCI2014-001'}; S={source}")
        ax.set_xticks(range(2), align_labels, fontsize=7.2); ax.tick_params(axis="y", labelsize=8); ax.legend(fontsize=7.5, loc="upper left")
    # Four h-by-transform series must contain exactly the complete-source people.
    for dataset, expected_n in (("openbmi8", 12), ("bnci", 9)):
        records = [row for row in plotted if row["figure"] == "current_alignment" and row["dataset"] == dataset]
        expected_series = {name + f"/h{h}" for name in align_names for h in ("0", "60")}
        if {row["series"] for row in records} != expected_series or len(records) != 4 * expected_n:
            raise ValueError(f"wrong current-alignment plotted inventory: {dataset}")
        for series in expected_series:
            values = [row for row in records if row["series"] == series]
            if len(values) != expected_n or {int(row["target"]) for row in values} != alignment_expected[dataset]:
                raise ValueError(f"wrong current-alignment plotted participants: {dataset}/{series}")
            if {row["source_size"] for row in values} != {FULL_SOURCE[dataset]}:
                raise ValueError(f"wrong current-alignment plotted source: {dataset}/{series}")
    save(fig, "current_alignment", "Current-session alignment effects compare each current transform with its historical control on identical evaluation rows. Full-batch current alignment is explicitly transductive.", list(align_names))

    # 5. Descriptive historical-score sensitivity. Values are already averaged
    # within participant across ten draws by the sensitivity analyzer.
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.5), sharex=False, sharey=False)
    for row, source_kind in enumerate(("100", "complete")):
        for col, dataset in enumerate(DISPLAY_DATASETS):
            source = FULL_SOURCE[dataset] if source_kind == "complete" else "100"
            ax = axes[row, col]
            selected = [r for r in decodability_people if r["dataset_variant"] == dataset
                        and r["source_size"] == source and r["h_total"] == "60"
                        and r["method"] in ("plain_ts", "ea_ts")]
            by_method = {}
            for method in ("plain_ts", "ea_ts"):
                method_rows = [r for r in selected if r["method"] == method]
                by_method[method] = {int(r["target"]): r for r in method_rows}
                if len(by_method[method]) != len(method_rows):
                    raise ValueError("duplicate descriptive historical-score participant")
            target_sets = [set(values) for values in by_method.values()]
            expected_targets = {int(r["target"]) for r in core_people
                                if r["dataset_variant"] == dataset}
            if not expected_targets or any(ids != expected_targets for ids in target_sets):
                raise ValueError(f"incomplete descriptive historical-score sensitivity: {dataset}/S{source}")
            for method, color, marker, label in (("plain_ts", COLORS["plain_ts"], "o", "Plain TS–LR"),
                                                 ("ea_ts", COLORS["ea_ts"], "s", "Historical EA–TS–LR")):
                values = by_method[method]
                ordered = [values[target] for target in sorted(values)]
                x = [number(item, "historical_ba") for item in ordered]
                y = [number(item, "future_gain_pp") for item in ordered]
                ax.scatter(x, y, color=color, marker=marker, s=26, alpha=.82, label=label, zorder=2)
                for target, x_value, y_value in zip(sorted(values), x, y):
                    plotted.append({"figure": "history_score_sensitivity", "dataset": dataset, "source_size": source,
                                    "series": method, "target": target, "x": x_value, "value": y_value,
                                    "ci_low": "", "ci_high": "", "unit": "historical_BA_percent_vs_future_gain_pp"})
            ax.axhline(0, color="#333333", lw=.8, zorder=0)
            ax.grid(color="#e0e0e0", lw=.55)
            ax.set_title(f"{'OpenBMI-8' if dataset == 'openbmi8' else 'BNCI'}; S={source}", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=8)
            if row == 1: ax.set_xlabel("Historical session-1 score (%)", fontsize=8)
            if col == 0: ax.set_ylabel("Future h=60 − h=0 gain (pp)", fontsize=8)
            if row == 0 and col == 1: ax.legend(fontsize=7.2, frameon=False, loc="best")
    save(fig, "history_score_sensitivity", "Descriptive historical-score sensitivity at h=60. Each point is a participant’s ten-draw average historical session-1 score and paired future h=60-minus-h=0 gain. These draw averages use more than one h-label block and are not a deployable h-label predictor; participants, not draws, are the displayed units.", ["plain_ts", "ea_ts"])

    fields = ["figure", "dataset", "source_size", "series", "target", "x", "value", "ci_low", "ci_high", "unit"]
    with (args.out_dir / "plotted_values.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(plotted)
    index["plotted_values_csv"] = "plotted_values.csv"
    index["output_sha256"] = {p.name: digest(p) for p in sorted(args.out_dir.iterdir()) if p.is_file()}
    (args.out_dir / "FIGURE_INDEX.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
