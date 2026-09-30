#!/usr/bin/env python3
"""Render the saved-score review-2 participant and fold diagnostics."""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
INPUTS = [
    "result/review2_20260923/saved_score_diagnostics_v1/participant_ts_effects.csv",
    "result/review2_20260923/saved_score_diagnostics_v1/fold_margins.csv",
]
EXPECTED_MEAN_PP = 5.708333333333333
TOL = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_plot_values(root):
    effect_rows = read_csv(root / INPUTS[0])
    margin_rows = read_csv(root / INPUTS[1])
    subjects = sorted(int(row["subject_id"]) for row in effect_rows)
    if len(subjects) != 12 or len(set(subjects)) != 12:
        raise ValueError("expected 12 unique participant effect rows")
    effects = {int(row["subject_id"]): 100 * float(row["net_fixed_total_replacement"])
               for row in effect_rows}
    if not all(math.isfinite(value) for value in effects.values()):
        raise ValueError("nonfinite participant effect")
    mean = sum(effects.values()) / len(effects)
    if abs(mean - EXPECTED_MEAN_PP) > TOL:
        raise ValueError(f"cohort mean {mean} differs from expected {EXPECTED_MEAN_PP}")

    selected = [row for row in margin_rows
                if row["analysis_arm"] == "fixed_total" and int(row["dose_per_class"]) == 5
                and row["alternative_model"] == "csp_lda"]
    margins = {}
    for subject in subjects:
        rows = [row for row in selected if int(row["heldout_subject_id"]) == subject]
        averaged = [row for row in rows if row["aggregation"] == "draw_averaged_person_first"]
        single = [row for row in rows if row["aggregation"] == "single_source_history_draw"]
        if len(averaged) != 1 or len(single) != 2 or {int(row["draw"]) for row in single} != {0, 1}:
            raise ValueError(f"fixed-total margin coverage differs for participant {subject}")
        draw = {int(row["draw"]): 100 * float(row["ts_lr_minus_alternative_training_mean_ba"])
                for row in single}
        average = 100 * float(averaged[0]["ts_lr_minus_alternative_training_mean_ba"])
        if abs(average - (draw[0] + draw[1]) / 2) > TOL:
            raise ValueError(f"draw average differs for participant {subject}")
        margins[subject] = {"draw0": draw[0], "draw1": draw[1], "person_first_draw_average": average}

    plotted = []
    for subject in subjects:
        plotted.append({"panel": "A", "heldout_or_participant_id": subject,
                        "series": "net_fixed_total_replacement", "value_percentage_points": effects[subject],
                        "definition": "TS-LR BA(40 source + 60 personal) minus BA(100 source)"})
    for subject in subjects:
        for series in ("draw0", "draw1", "person_first_draw_average"):
            plotted.append({"panel": "B", "heldout_or_participant_id": subject, "series": series,
                            "value_percentage_points": margins[subject][series],
                            "definition": "TS-LR minus CSP-LDA other-11 training-fold mean at 5 personal trials/class"})
    return subjects, effects, margins, plotted


def render(out, subjects, effects, margins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
                         "legend.fontsize": 8, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.9))
    fig.subplots_adjust(left=.075, right=.985, bottom=.29, top=.80, wspace=.16)
    x = np.arange(len(subjects))

    effect_values = np.asarray([effects[subject] for subject in subjects])
    effect_colors = np.where(effect_values >= 0, "#0072B2", "#D55E00")
    axes[0].vlines(x, 0, effect_values, color=effect_colors, linewidth=1.5, alpha=.8)
    axes[0].scatter(x, effect_values, color=effect_colors, s=36, zorder=3,
                    edgecolor="white", linewidth=.5)
    axes[0].axhline(0, color="#333333", linewidth=.8)
    axes[0].axhline(EXPECTED_MEAN_PP, color="#CC79A7", linestyle="--", linewidth=1.6,
                    label="Cohort mean = 5.7083 pp")
    axes[0].set_title("A  Matched-count replacement effect")
    axes[0].set_ylabel("Net TS–LR balanced-accuracy difference (pp)")
    axes[0].set_xlabel("Development participant ID")
    axes[0].set_xticks(x, [str(subject) for subject in subjects])
    axes[0].grid(axis="y", color="#d4d4d4", linewidth=.6, alpha=.8)
    axes[0].legend(frameon=False, loc="upper left")

    series = [
        ("draw0", "Draw 0", "#0072B2", "o", "-"),
        ("draw1", "Draw 1", "#D55E00", "s", "-"),
        ("person_first_draw_average", "Person-first draw average", "#000000", "D", "--"),
    ]
    for key, label, color, marker, linestyle in series:
        values = [margins[subject][key] for subject in subjects]
        axes[1].plot(x, values, color=color, marker=marker, linestyle=linestyle,
                     linewidth=1.25, markersize=4.5, label=label)
    axes[1].axhline(0, color="#333333", linewidth=.8)
    axes[1].set_title("B  Fixed-total fold margin at 10 personal labels")
    axes[1].set_ylabel("TS–LR minus CSP–LDA training-fold mean (pp)")
    axes[1].set_xlabel("Held-out development participant ID")
    axes[1].set_xticks(x, [str(subject) for subject in subjects])
    axes[1].grid(axis="y", color="#d4d4d4", linewidth=.6, alpha=.8)
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.20),
                   ncol=3, columnspacing=1.4, handletextpad=.5)

    fig.suptitle("Saved-score participant and source/history-draw diagnostics", fontsize=12)
    fig.text(.5, .035,
             "Descriptive exposed-development results. LOPO training folds share 10 of 11 people; "
             "lines aid reading only. No independent-sample error bars.",
             ha="center", va="bottom", fontsize=8, color="#444444")
    for extension in ("svg", "png"):
        fig.savefig(out / f"review2_diagnostics.{extension}", dpi=240, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def apply_limits():
    gib = 16 * 1024 ** 3
    _, hard_as = resource.getrlimit(resource.RLIMIT_AS)
    hard_as = gib if hard_as == resource.RLIM_INFINITY else min(hard_as, gib)
    resource.setrlimit(resource.RLIMIT_AS, (hard_as, hard_as))
    _, hard_cpu = resource.getrlimit(resource.RLIMIT_CPU)
    hard_cpu = 125 if hard_cpu == resource.RLIM_INFINITY else min(hard_cpu, 125)
    resource.setrlimit(resource.RLIMIT_CPU, (min(120, hard_cpu), hard_cpu))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("180-second wall limit")))
    signal.alarm(180)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--replace-owned-render", action="store_true",
                        help="replace only this renderer's four existing output files")
    args = parser.parse_args(argv)
    out = args.out_dir.resolve()
    if out.exists():
        expected = {"receipt.json", "plotted_values.csv", "review2_diagnostics.svg", "review2_diagnostics.png"}
        if not args.replace_owned_render or {path.name for path in out.iterdir()} != expected:
            raise FileExistsError(f"output directory exists or differs from owned render set: {out}")
    else:
        out.mkdir(parents=True)
    started_wall, started_cpu = time.monotonic(), time.process_time()
    command = [sys.executable, str(Path(__file__).resolve()), "--out-dir", str(out)]
    if args.replace_owned_render:
        command.append("--replace-owned-render")
    receipt = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
               "command": command,
               "pid": os.getpid(), "input_scope": "two completed saved-score diagnostic CSVs only",
               "thread_environment": {key: os.environ[key] for key in
                                      ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
               "limits": {"cpu_seconds": 120, "wall_seconds": 180, "address_space_gib": 16}}
    atomic_json(out / "receipt.json", receipt)
    exit_code = 1
    try:
        apply_limits()
        before = {path: sha256(ROOT / path) for path in INPUTS}
        code_before = sha256(Path(__file__).resolve())
        subjects, effects, margins, plotted = load_plot_values(ROOT)
        atomic_csv(out / "plotted_values.csv", plotted)
        render(out, subjects, effects, margins)
        after = {path: sha256(ROOT / path) for path in INPUTS}
        code_after = sha256(Path(__file__).resolve())
        if before != after or code_before != code_after:
            raise ValueError("render input or code changed during execution")
        receipt.update(status="completed", exit_code=0, input_hashes_before=before,
                       input_hashes_after=after, code_hash_before=code_before, code_hash_after=code_after,
                       participant_ids=subjects, panel_a_cohort_mean_pp=EXPECTED_MEAN_PP,
                       overlap_note="LOPO training folds overlap; no independent-sample error bars",
                       output_hashes={path.name: sha256(path) for path in sorted(out.iterdir())
                                      if path.is_file() and path.name != "receipt.json"})
        exit_code = 0
    except BaseException as error:
        receipt.update(status="failed", exit_code=1, error=repr(error))
        traceback.print_exc()
    finally:
        signal.alarm(0)
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - started_wall,
                       cpu_seconds=time.process_time() - started_cpu,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
