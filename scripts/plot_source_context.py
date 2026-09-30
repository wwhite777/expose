#!/usr/bin/env python3
"""Redraw verified source-context values; no fitting or statistical reanalysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def render(data_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = json.loads(data_path.read_text())
    cohorts = data["cohort_order"]
    people, stats = data["participants"], data["summary_statistics"]
    assert len(people) == 45 and len(cohorts) == 3
    labels = ["OpenBMI\nDevelopment\n(n=12)", "BNCI 2a\nDevelopment\n(n=9)",
              "OpenBMI\nSecondary\n(n=24)"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3, "legend.fontsize": 8, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25))
    blue, orange, gray = "#0072B2", "#C77900", "#68727D"

    def point(ax, x, item, color, marker="o", filled=True, z=4):
        y = 100 * item["mean"]
        err = [[100 * (item["mean"] - item["t95_low"])],
               [100 * (item["t95_high"] - item["mean"])]]
        ax.errorbar(x, y, yerr=err, fmt=marker, color=color, markersize=5,
                    markerfacecolor=color if filled else "white", markeredgewidth=1,
                    capsize=2.4, elinewidth=1, zorder=z)

    for i, cohort in enumerate(cohorts):
        rows = [p for p in people if p["cohort"] == cohort]
        for p in rows:
            axes[0].plot([i - .15, i + .15],
                         [100 * p["small_effect"], 100 * p["large_effect"]],
                         color="#B2B9BF", lw=.6, alpha=.65, zorder=1)
            axes[0].scatter([i - .15, i + .15],
                            [100 * p["small_effect"], 100 * p["large_effect"]],
                            c=[gray, blue], s=9, alpha=.6, zorder=2)
        point(axes[0], i - .15, stats[cohort]["small_effect"], gray)
        point(axes[0], i + .15, stats[cohort]["large_effect"], blue)
        for offset, series in zip([-.27, -.09, .09, .27],
                ["small_own", "small_donor", "large_own", "large_donor"]):
            point(axes[1], i + offset, stats[cohort][series],
                  blue if series.endswith("own") else orange,
                  "o" if series.endswith("own") else "s", series.startswith("large"))
    axes[0].axhline(0, color="#444444", ls="--", lw=.8, zorder=0)
    axes[0].set_ylabel("Own − donor (percentage points)")
    axes[0].set_title("A  Paired archive advantage", loc="left", weight="bold")
    axes[0].set_ylim(-10, 19)
    axes[0].legend(handles=[
        Line2D([], [], color=gray, marker="o", ls="", label="100 source trials"),
        Line2D([], [], color=blue, marker="o", ls="", label="Large source base")],
        loc="lower left", frameon=False, handletextpad=.35)
    # Dots, rather than truncated bars, make the nonzero axis explicit and honest.
    axes[1].set_ylim(40, 85)
    axes[1].set_yticks([40, 50, 60, 70, 80])
    axes[1].axhline(50, color="#969DA4", ls=":", lw=.8, zorder=0)
    axes[1].set_ylabel("Balanced accuracy (%)")
    axes[1].set_title("B  Absolute accuracy", loc="left", weight="bold")
    axes[1].legend(handles=[
        Line2D([], [], color=blue, marker="o", ls="", label="Own 60"),
        Line2D([], [], color=orange, marker="s", ls="", label="Mean donor 60"),
        Line2D([], [], color=gray, marker="o", mfc="white", ls="", label="100 source"),
        Line2D([], [], color=gray, marker="o", ls="", label="Large source")],
        loc="lower right", frameon=False, ncol=2, columnspacing=.65,
        handlelength=.9, handletextpad=.25)
    for ax in axes:
        ax.set_xticks(range(3), labels)
        ax.set_xlim(-.5, 2.5)
        ax.grid(axis="y", color="#E7EAED", lw=.5, zorder=0)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=.085, right=.985, top=.895, bottom=.28, wspace=.34)
    out.mkdir(parents=True, exist_ok=False)
    for ext in ("pdf", "svg", "png"):
        kw = {"dpi": 320} if ext == "png" else {}
        if ext == "pdf":
            kw["metadata"] = {"Title": "Source-context comparison", "CreationDate": None,
                               "ModDate": None, "Creator": "EXPOSE figure script"}
        fig.savefig(out / ("figure2_source_context." + ext), facecolor="white", **kw)
    plt.close(fig)
    receipt = {"source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "scope": "Presentation only: all 45 participant values and saved means/t intervals unchanged",
        "panel_b": "Dot-and-whisker plot; axis 40–85 percent; no truncated bars",
        "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(out.iterdir()) if p.is_file()}}
    (out / "figure2_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    render(args.data, args.out)
