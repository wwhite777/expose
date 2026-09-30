#!/usr/bin/env python3
"""Analyze a future fixed Review-4 confirmation score inventory."""

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["operation_id", "participant_id", "draw", "condition", "setting", "source_n",
          "added_n", "h_total", "C", "balanced_accuracy", "evaluation_n"]
PRIMARY = ("own", "single_0", "single_1", "single_2")
OPTIONAL = ("base", "pooled")
SETTING = "tuned_C_source_frozen"
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 2026092408
SIGNFLIP_DRAWS = 100000
SIGNFLIP_SEED = 2026092409
TOL = 1e-12


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "bytes": path.stat().st_size}


def percentile(values, probability):
    return float(np.quantile(np.asarray(values, dtype=float), probability, method="linear"))


def interval_summary(values, bootstrap_draws=BOOTSTRAP_DRAWS):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("effect vector must be finite and nonempty")
    mean = float(np.mean(x))
    if len(x) >= 2:
        sd = float(np.std(x, ddof=1))
        half = float(student_t.ppf(0.975, len(x) - 1) * sd / math.sqrt(len(x)))
        t_low, t_high = mean - half, mean + half
    else:
        sd = t_low = t_high = None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(x), size=(bootstrap_draws, len(x)))
    boot = np.mean(x[indices], axis=1)
    positive = int(np.sum(x > TOL)); negative = int(np.sum(x < -TOL))
    return {"n": len(x), "mean_pp": mean, "median_pp": float(np.median(x)),
            "sd_pp": sd, "min_pp": float(np.min(x)), "max_pp": float(np.max(x)),
            "t_ci_low_pp": t_low, "t_ci_high_pp": t_high,
            "bootstrap_ci_low_pp": percentile(boot, 0.025),
            "bootstrap_ci_high_pp": percentile(boot, 0.975),
            "positive": positive, "zero": len(x) - positive - negative, "negative": negative}


def signflip_p(values, draws=SIGNFLIP_DRAWS):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("sign-flip vector must be finite and nonempty")
    observed = abs(float(np.mean(x)))
    rng = np.random.default_rng(SIGNFLIP_SEED)
    extreme = 0
    remaining = draws
    while remaining:
        count = min(10000, remaining)
        signs = rng.integers(0, 2, size=(count, len(x)), dtype=np.int8) * 2 - 1
        permuted = np.abs(np.mean(signs * x, axis=1))
        extreme += int(np.sum(permuted >= observed - TOL))
        remaining -= count
    return (extreme + 1) / (draws + 1)


def classify(summary, complete_people):
    if complete_people < 20 or summary["t_ci_low_pp"] is None:
        return "operationally_incomplete"
    if summary["t_ci_low_pp"] > 0 and summary["mean_pp"] >= 2.0:
        return "supportive"
    if summary["t_ci_low_pp"] > 0 and summary["mean_pp"] < 2.0:
        return "positive_below_threshold"
    if summary["t_ci_high_pp"] < 0:
        return "adverse"
    return "uncertain"


def analyze_rows(rows, operations, protected_ids, bootstrap_draws=BOOTSTRAP_DRAWS,
                 signflip_draws=SIGNFLIP_DRAWS):
    expected = {op["operation_id"]: op for op in operations}
    primary_ids = {key for key, op in expected.items() if op["condition"] in PRIMARY}
    optional_ids = {key for key, op in expected.items() if op["condition"] in OPTIONAL}
    if len(primary_ids) != 960 or len(optional_ids) != 480 or len(protected_ids) != 24:
        raise ValueError("packet operation inventory differs")
    scores = {}
    optional_seen = set()
    for row in rows:
        if set(row) != set(FIELDS):
            raise ValueError("score CSV schema differs")
        operation_id = row["operation_id"]
        if operation_id not in expected:
            raise ValueError("extra or unknown operation ID")
        if operation_id in scores:
            raise ValueError("duplicate operation score")
        op = expected[operation_id]
        participant, draw = int(row["participant_id"]), int(row["draw"])
        source_n, added_n, h_total = int(row["source_n"]), int(row["added_n"]), int(row["h_total"])
        C = float(row["C"]); ba = float(row["balanced_accuracy"]); evaluation_n = int(row["evaluation_n"])
        if (participant != op["participant_id"] or draw != op["draw"]
                or row["condition"] != op["condition"] or row["setting"] != SETTING
                or source_n != 100 or source_n != op["source_n"] or added_n != op["added_n"]
                or h_total != op["h_total"] or C != float(op["C"])
                or evaluation_n != 100):
            raise ValueError("score coordinate, dose, setting, C, or evaluation count differs")
        if not math.isfinite(ba) or not 0 <= ba <= 1:
            raise ValueError("balanced accuracy lies outside [0,1]")
        scores[operation_id] = ba
        if operation_id in optional_ids:
            optional_seen.add(operation_id)
    effect_rows, draw_rows, status_rows = [], [], []
    by_coordinate = {(op["participant_id"], op["draw"], op["condition"]): op["operation_id"]
                     for op in operations}
    for participant in protected_ids:
        required = [by_coordinate[(participant, draw, condition)]
                    for draw in range(10) for condition in PRIMARY]
        missing = [operation_id for operation_id in required if operation_id not in scores]
        if missing:
            status_rows.append({"participant_id": participant, "status": "incomplete",
                                "required_scores": 40, "observed_scores": 40 - len(missing),
                                "missing_scores": len(missing),
                                "missing_operation_ids": "|".join(missing),
                                "reason": "missing required primary score row; no partial-draw endpoint"})
            continue
        own_values, donor_values, effects = [], [], []
        for draw in range(10):
            own = scores[by_coordinate[(participant, draw, "own")]]
            singles = [scores[by_coordinate[(participant, draw, f"single_{index}")]]
                       for index in range(3)]
            donor = statistics.mean(singles); effect = (own - donor) * 100
            own_values.append(own); donor_values.append(donor); effects.append(effect)
            draw_rows.append({"participant_id": participant, "draw": draw,
                              "own_ba": own, "mean_single_ba": donor, "effect_pp": effect})
        row = {"participant_id": participant, "draws": 10,
               "own_mean_ba": statistics.mean(own_values),
               "mean_single_ba": statistics.mean(donor_values),
               "effect_pp": statistics.mean(effects)}
        effect_rows.append(row)
        status_rows.append({"participant_id": participant, "status": "complete",
                            "required_scores": 40, "observed_scores": 40,
                            "missing_scores": 0, "missing_operation_ids": "", "reason": ""})

    values = [row["effect_pp"] for row in effect_rows]
    primary_summary = interval_summary(values, bootstrap_draws) if values else {
        "n": 0, "mean_pp": None, "median_pp": None, "sd_pp": None, "min_pp": None,
        "max_pp": None, "t_ci_low_pp": None, "t_ci_high_pp": None,
        "bootstrap_ci_low_pp": None, "bootstrap_ci_high_pp": None,
        "positive": 0, "zero": 0, "negative": 0}
    complete = len(effect_rows); missing = 24 - complete
    complete_ids = {row["participant_id"] for row in effect_rows}
    for participant in protected_ids:
        expected_optional = {by_coordinate[(participant, draw, condition)]
                             for draw in range(10) for condition in OPTIONAL}
        observed_optional = expected_optional & optional_seen
        if observed_optional and observed_optional != expected_optional:
            raise ValueError("optional base/pooled inventory is partial within a participant")
        if optional_seen and ((participant in complete_ids) != bool(observed_optional)):
            raise ValueError("optional inventory must match complete primary participants")
    primary_summary["classification"] = classify(primary_summary, complete)
    primary_summary["supportive_rule"] = "two-sided 95% t CI lower >0 and mean >=2 pp"
    primary_summary["signflip_two_sided_mc_p"] = (signflip_p(values, signflip_draws)
                                                   if values else None)
    primary_summary["signflip_draws"] = signflip_draws
    observed_sum = sum(values)
    primary_summary["full24_missing_bound_low_pp"] = (observed_sum - 100 * missing) / 24
    primary_summary["full24_missing_bound_high_pp"] = (observed_sum + 100 * missing) / 24

    optional_summary = None
    if optional_seen:
        base_person, pooled_person = {}, {}
        for participant in sorted(complete_ids):
            base_person[participant] = statistics.mean(
                scores[by_coordinate[(participant, draw, "base")]] for draw in range(10)) * 100
            pooled_person[participant] = statistics.mean(
                scores[by_coordinate[(participant, draw, "pooled")]] for draw in range(10)) * 100
        optional_summary = {
            "base_absolute_ba_percent": interval_summary(list(base_person.values()), bootstrap_draws),
            "pooled_absolute_ba_percent": interval_summary(list(pooled_person.values()), bootstrap_draws),
            "pooled_minus_base_pp": interval_summary(
                [pooled_person[p] - base_person[p] for p in sorted(complete_ids)], bootstrap_draws),
            "own_minus_base_pp_complete_primary_people": interval_summary(
                [row["own_mean_ba"] * 100 - base_person[row["participant_id"]] for row in effect_rows],
                bootstrap_draws) if effect_rows else None,
            "status": "descriptive_secondary_not_primary_decision",
        }
    return {"primary": primary_summary, "optional": optional_summary,
            "complete_people": complete, "incomplete_people": missing,
            "primary_rows_observed": len(scores.keys() & primary_ids),
            "optional_rows_observed": len(optional_seen)}, effect_rows, draw_rows, status_rows


def load_packet(packet_dir):
    receipt_path = packet_dir / "PACKET_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "completed_source_only_packet" or receipt.get("protected_accessed") is not False:
        raise ValueError("source packet receipt differs")
    for name, item in receipt["outputs"].items():
        if sha256(packet_dir / name) != item["sha256"]:
            raise ValueError("source packet output hash differs")
    machine = json.loads((packet_dir / "MACHINE_CONFIG.json").read_text(encoding="utf-8"))
    with gzip.open(packet_dir / "operations.json.gz", "rt", encoding="utf-8") as handle:
        operations = json.load(handle)
    return receipt_path, machine, operations


def validate_runner_binding(receipt_path, completed_path, scores_path, packet_receipt,
                            authorization_path, code_freeze_path):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    code_freeze = json.loads(code_freeze_path.read_text(encoding="utf-8"))
    complete_people = receipt.get("complete_people")
    output_dir = receipt_path.parent
    expected_outputs = {"scores.csv", "trial_predictions.csv.gz", "person_status.csv",
                        "protected_covariance_guards.json", "protected_memberships.json",
                        "access_log.jsonl", "STAGED_INTEGRITY.json"}
    if (receipt.get("status") != "completed_score_inventory_no_analysis"
            or receipt.get("aggregate_analysis_called") is not False
            or receipt.get("raw_MAT_persisted") is not False
            or not isinstance(complete_people, int) or not 0 <= complete_people <= 24
            or receipt.get("failed_people") != 24 - complete_people
            or receipt.get("operations") != complete_people * 60
            or receipt.get("predictions") != complete_people * 6000
            or receipt.get("packet_receipt_sha256") != sha256(packet_receipt)
            or receipt.get("authorization_sha256") != sha256(authorization_path)
            or receipt.get("code_freeze_sha256") != sha256(code_freeze_path)
            or receipt.get("runner_sha256") != code_freeze.get("code", {}).get("runner", {}).get("sha256")
            or scores_path != output_dir / "scores.csv"
            or completed_path != output_dir / "COMPLETED.json"
            or set(receipt.get("outputs", {})) != expected_outputs
            or receipt.get("outputs", {}).get("scores.csv") != sha256(scores_path)
            or completed != {"status": receipt["status"],
                             "receipt_sha256": sha256(receipt_path)}):
        raise ValueError("runner receipt/COMPLETED binding differs")
    for name, digest in receipt["outputs"].items():
        path = output_dir / name
        if not path.is_file() or sha256(path) != digest:
            raise ValueError("runner output changed after completion: " + name)
    staged = json.loads((output_dir / "STAGED_INTEGRITY.json").read_text(encoding="utf-8"))
    if (staged.get("status") != "staged_inventory_locked_before_analysis"
            or staged.get("complete_people") != complete_people
            or staged.get("failed_people") != 24 - complete_people
            or staged.get("scores_written") != complete_people * 60
            or staged.get("predictions_written") != complete_people * 6000
            or staged.get("scores_partial_sha256") != sha256(scores_path)):
        raise ValueError("staged inventory binding differs")
    return receipt


def validate_code_freeze(path, packet_receipt, authorization):
    value = json.loads(path.read_text(encoding="utf-8"))
    analyzer = value.get("code", {}).get("analyzer")
    if (value.get("schema") != "review4-confirmation-code-freeze-v1"
            or value.get("status") != "complete_code_freeze_requires_author_adoption"
            or value.get("protected_access_authorized") is not False
            or value.get("packet_receipt", {}).get("sha256") != sha256(packet_receipt)
            or analyzer != file_identity(__file__)
            or authorization.get("code_freeze_sha256") != sha256(path)):
        raise PermissionError("analyzer does not match the authorized confirmation code freeze")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path,
                        help="Future author-adoption/protected-access authorization binding")
    parser.add_argument("--code-freeze", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--runner-receipt", required=True, type=Path)
    parser.add_argument("--runner-completed", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.packet_dir, args.authorization = args.packet_dir.resolve(), args.authorization.resolve()
    args.code_freeze = args.code_freeze.resolve()
    args.scores, args.out_dir = args.scores.resolve(), args.out_dir.resolve()
    args.runner_receipt = args.runner_receipt.resolve()
    args.runner_completed = args.runner_completed.resolve()
    start_wall, start_cpu = time.monotonic(), time.process_time()
    if args.out_dir.exists():
        raise FileExistsError("output directory must be new")
    receipt_path, machine, operations = load_packet(args.packet_dir)
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    adoption_name = authorization.get("adoption_record")
    adoption_path = ((ROOT / adoption_name).resolve()
                     if isinstance(adoption_name, str) else None)
    if (set(authorization) != {"schema", "author_adopted", "protected_access_authorized",
                               "code_freeze_sha256", "packet_receipt_sha256", "adoption_record",
                               "adoption_record_sha256", "decision"}
            or authorization.get("schema") != "review4-confirmation-authorization-v1"
            or authorization.get("packet_receipt_sha256") != sha256(receipt_path)
            or authorization.get("author_adopted") is not True
            or authorization.get("protected_access_authorized") is not True
            or authorization.get("decision")
            != "adopt_and_authorize_protected_confirmation_execution"
            or adoption_path is None or ROOT not in adoption_path.parents
            or not adoption_path.is_file()
            or authorization.get("adoption_record_sha256") != sha256(adoption_path)):
        raise PermissionError("packet lacks explicit bound author adoption and protected-access authorization")
    validate_code_freeze(args.code_freeze, receipt_path, authorization)
    validate_runner_binding(args.runner_receipt, args.runner_completed, args.scores,
                            receipt_path, args.authorization, args.code_freeze)
    with args.scores.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS:
            raise ValueError("score CSV columns or order differ")
        rows = list(reader)
    result, effects, draws, statuses = analyze_rows(
        rows, operations, machine["protected_participant_ids"])
    result.update({"status": "completed_confirmation_analysis",
                   "primary_endpoint": "own minus mean of three single-other donors",
                   "setting": SETTING, "draw_aggregation": "ten draws averaged within participant",
                   "primary_interval": "two-sided Student-t 95% interval",
                   "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
                   "signflip": {"draws": SIGNFLIP_DRAWS, "seed": SIGNFLIP_SEED,
                                "method": "fixed Monte Carlo two-sided mean sign flip with plus-one correction"}})
    args.out_dir.mkdir(parents=True)
    for name, values in (("person_effects.csv", effects), ("draw_effects.csv", draws),
                         ("person_status.csv", statuses)):
        fields = list(values[0]) if values else (["participant_id"] if name != "draw_effects.csv" else ["participant_id", "draw"])
        with (args.out_dir / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(values)
    (args.out_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    inputs = {"packet_receipt": receipt_path, "authorization": args.authorization,
              "code_freeze": args.code_freeze,
              "runner_receipt": args.runner_receipt,
              "runner_completed": args.runner_completed, "scores": args.scores}
    outputs = ("person_effects.csv", "draw_effects.csv", "person_status.csv", "summary.json")
    receipt = {"status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "command": [sys.executable, *sys.argv], "threads": 1,
               "inputs": {name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                          for name, path in inputs.items()},
               "code": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(__file__)},
               "outputs": {name: {"sha256": sha256(args.out_dir / name),
                                  "bytes": (args.out_dir / name).stat().st_size} for name in outputs},
               "runtime": {"cpu_seconds": time.process_time() - start_cpu,
                           "wall_seconds": time.monotonic() - start_wall,
                           "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}}
    (args.out_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "classification": result["primary"]["classification"],
                      "complete_people": result["complete_people"]}, sort_keys=True))


if __name__ == "__main__":
    main()
