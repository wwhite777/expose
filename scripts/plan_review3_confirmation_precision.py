#!/usr/bin/env python3
"""Create the proposed n=24 confirmation precision/design-sensitivity table.

This is an author-review planning calculation, not a protocol freeze or an
analysis of participant outcomes.  It assumes independent Normal person-level
differences and integrates over the sampling distribution of the sample SD.
"""

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

from scipy.integrate import quad
from scipy import stats
import scipy


N = 24
DF = N - 1
CONFIDENCE_LEVEL = 0.95
PRACTICAL_CUTOFF_PP = 2.0
ASSUMED_SDS_PP = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0)
TRUE_MEANS_PP = (0.0, 2.0, 4.0, 6.0)
STATUS = "author-review planning; not freeze"
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/review3_20260923/CONFIRMATION_PROTOCOL_FOR_AUTHOR_REVIEW_v1.md"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_probability(true_mean_pp, assumed_sd_pp, t_critical,
                      practical_cutoff_pp):
    """Integrate P(mean clears thresholds | sample-S chi-square draw)."""
    standard_error = assumed_sd_pp / math.sqrt(N)

    def integrand(u):
        if not math.isfinite(u):
            return 0.0
        sample_sd = assumed_sd_pp * math.sqrt(u / DF)
        threshold = t_critical * sample_sd / math.sqrt(N)
        if practical_cutoff_pp is not None:
            threshold = max(threshold, practical_cutoff_pp)
        conditional = stats.norm.sf((threshold - true_mean_pp) / standard_error)
        return float(conditional * stats.chi2.pdf(u, DF))

    if practical_cutoff_pp is None:
        value, error = quad(integrand, 0.0, math.inf, epsabs=1e-12,
                            epsrel=1e-10, limit=250)
    else:
        switch = DF * (
            practical_cutoff_pp * math.sqrt(N) /
            (t_critical * assumed_sd_pp)
        ) ** 2
        low, low_error = quad(integrand, 0.0, switch, epsabs=1e-12,
                              epsrel=1e-10, limit=250)
        high, high_error = quad(integrand, switch, math.inf, epsabs=1e-12,
                                epsrel=1e-10, limit=250)
        value, error = low + high, low_error + high_error
    if not math.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12:
        raise RuntimeError("quadrature produced an invalid probability")
    return min(1.0, max(0.0, value)), error


def calculate():
    t_critical = float(stats.t.ppf((1.0 + CONFIDENCE_LEVEL) / 2.0, DF))
    rows = []
    maximum_nct_error = 0.0
    maximum_quad_error = 0.0

    for assumed_sd_pp in ASSUMED_SDS_PP:
        previous = -math.inf
        for true_mean_pp in TRUE_MEANS_PP:
            joint, joint_error = event_probability(
                true_mean_pp, assumed_sd_pp, t_critical, PRACTICAL_CUTOFF_PP)
            ci_only, ci_error = event_probability(
                true_mean_pp, assumed_sd_pp, t_critical, None)
            noncentrality = true_mean_pp * math.sqrt(N) / assumed_sd_pp
            analytic = float(stats.nct.sf(t_critical, DF, noncentrality))
            comparison_error = abs(ci_only - analytic)
            maximum_nct_error = max(maximum_nct_error, comparison_error)
            maximum_quad_error = max(maximum_quad_error, joint_error, ci_error)

            if comparison_error > 5e-9:
                raise RuntimeError("CI-only quadrature disagrees with noncentral t")
            if (true_mean_pp == 0.0
                    and (joint > 0.025 + 5e-10 or ci_only > 0.025 + 5e-10)):
                raise RuntimeError("null event probability exceeds 0.025")
            if joint + 1e-12 < previous:
                raise RuntimeError("joint-event probability is not monotone in mean")
            if joint > ci_only + 1e-12:
                raise RuntimeError("joint criterion is less restrictive than CI-only")
            previous = joint

            rows.append({
                "n_people": N,
                "df": DF,
                "true_mean_pp": true_mean_pp,
                "assumed_person_sd_pp": assumed_sd_pp,
                "nominal_ci_halfwidth_pp": (
                    t_critical * assumed_sd_pp / math.sqrt(N)),
                "joint_support_probability": joint,
                "ci_positive_probability": ci_only,
            })

    return rows, {
        "t_critical": t_critical,
        "maximum_ci_only_vs_nct_absolute_error": maximum_nct_error,
        "maximum_quadrature_reported_absolute_error": maximum_quad_error,
        "checks": {
            "ci_only_matches_noncentral_t": True,
            "null_joint_and_ci_only_probabilities_at_most_0.025": True,
            "joint_probability_monotone_in_true_mean_within_each_sd": True,
            "joint_probability_not_above_ci_only_probability": True,
        },
    }


def write_csv(path, rows):
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: (f"{value:.12g}" if isinstance(value, float) else value)
                for key, value in row.items()
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True,
                        help="exclusive new directory for planning outputs")
    args = parser.parse_args()
    started_wall = time.monotonic()
    started_cpu = time.process_time()

    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    rows, validation = calculate()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    table_path = out_dir / "confirmation_precision.csv"
    summary_path = out_dir / "summary.json"
    receipt_path = out_dir / "receipt.json"
    write_csv(table_path, rows)

    summary = {
        "status": STATUS,
        "design": {
            "n_people": N,
            "df": DF,
            "confidence_level": CONFIDENCE_LEVEL,
            "practical_cutoff_pp": PRACTICAL_CUTOFF_PP,
            "assumed_person_sd_pp": list(ASSUMED_SDS_PP),
            "true_mean_pp": list(TRUE_MEANS_PP),
        },
        "event": (
            "two-sided 95% one-sample t interval lower endpoint > 0 and "
            "observed sample mean >= 2 percentage points"
        ),
        "calculation": (
            "deterministic scipy quadrature over (n-1)S^2/sigma^2 ~ "
            "chi-square(n-1), using independence of sample mean and S under "
            "independent Normal person differences"
        ),
        "validation": validation,
        "rows": rows,
        "limitations": [
            "This calculation uses no protected or development outcomes.",
            "It does not validate Normality or independence of person-level differences.",
            "It is not a power guarantee, protocol adoption, protocol freeze, or final-study result.",
            "The nominal half-width substitutes each assumed population SD for the sample SD.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    receipt = {
        "status": STATUS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "inputs": {
            str(PROTOCOL.relative_to(ROOT)): sha256(PROTOCOL),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__),
        },
        "outputs": {
            table_path.name: {"sha256": sha256(table_path), "bytes": table_path.stat().st_size},
            summary_path.name: {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "python": sys.version.split()[0],
            "scipy": scipy.__version__,
            "threads": 1,
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": STATUS, "out_dir": str(out_dir),
                      "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
