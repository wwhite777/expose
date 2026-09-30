#!/usr/bin/env python3
"""Portable saved-output reconstruction of the final confirmation endpoint."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import argparse, csv, gzip, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

SCORE_FIELDS = ["operation_id", "participant_id", "draw", "condition", "setting",
                "source_n", "added_n", "h_total", "C", "balanced_accuracy", "evaluation_n"]
PRED_FIELDS = ["operation_id", "participant_id", "draw", "condition", "trial_id",
               "y_true", "y_pred", "p0", "p1"]
EFFECT_FIELDS = ["participant_id", "draws", "own_mean_ba", "mean_single_ba", "effect_pp"]
CONDITIONS = ("base", "own", "pooled", "single_0", "single_1", "single_2")
PRIMARY = ("own", "single_0", "single_1", "single_2")
PEOPLE = (1,4,7,11,13,15,19,22,23,24,27,28,30,34,35,37,40,41,43,44,46,48,49,50)
TOL = 1e-12

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def close(a, b, tol=1e-10):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol

def load_scores(path):
    scores, coords = {}, {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != SCORE_FIELDS: raise ValueError("score schema differs")
        for row in reader:
            op, p, d, cond = row["operation_id"], int(row["participant_id"]), int(row["draw"]), row["condition"]
            coord = (p, d, cond); base = cond == "base"
            numeric = [float(row["C"]), float(row["balanced_accuracy"])]
            if (op in scores or coord in coords or p not in PEOPLE or d not in range(10)
                    or cond not in CONDITIONS or row["setting"] != "tuned_C_source_frozen"
                    or int(row["source_n"]) != 100 or int(row["added_n"]) != (0 if base else 60)
                    or int(row["h_total"]) != (0 if base else 60) or int(row["evaluation_n"]) != 100
                    or not all(math.isfinite(x) for x in numeric) or numeric[0] <= 0 or not 0 <= numeric[1] <= 1):
                raise ValueError("invalid, duplicate, or unexpected score row")
            scores[op] = {"coord": coord, "ba": numeric[1], "C": numeric[0]}; coords[coord] = op
    expected = {(p,d,c) for p in PEOPLE for d in range(10) for c in CONDITIONS}
    if len(scores) != 1440 or set(coords) != expected: raise ValueError("score inventory differs")
    for p in PEOPLE:
        for d in range(10):
            if len({scores[coords[p,d,c]]["C"] for c in CONDITIONS}) != 1:
                raise ValueError("C differs within a paired coordinate")
    return scores, coords

def reconstruct_predictions(path, scores):
    state = {op: {"trials": {}, "correct": [0,0]} for op in scores}; rows = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != PRED_FIELDS: raise ValueError("prediction schema differs")
        for row in reader:
            rows += 1; op = row["operation_id"]
            if op not in state: raise ValueError("unknown prediction operation")
            p,d,c = int(row["participant_id"]), int(row["draw"]), row["condition"]
            y, pred = int(row["y_true"]), int(row["y_pred"]); p0,p1 = float(row["p0"]),float(row["p1"])
            trial = row["trial_id"]; current = state[op]
            if ((p,d,c) != scores[op]["coord"] or trial in current["trials"] or y not in (0,1)
                    or pred not in (0,1) or not all(math.isfinite(x) and 0 <= x <= 1 for x in (p0,p1))
                    or abs(p0+p1-1) > 1e-10 or pred != int(p1 > p0)):
                raise ValueError("invalid or duplicate prediction row")
            current["trials"][trial] = y; current["correct"][y] += int(pred == y)
    if rows != 144000: raise ValueError("prediction row count differs")
    person_trials, reconstructed, max_diff = {}, {}, 0.0
    for op,item in state.items():
        truths = item["trials"]
        if len(truths) != 100 or list(truths.values()).count(0) != 50 or list(truths.values()).count(1) != 50:
            raise ValueError("operation trial/count balance differs")
        p = scores[op]["coord"][0]
        if p in person_trials and truths != person_trials[p]: raise ValueError("evaluation trials differ within person")
        person_trials.setdefault(p, truths)
        ba = sum(item["correct"]) / 100.0; reconstructed[op] = ba
        max_diff = max(max_diff, abs(ba - scores[op]["ba"]))
    if max_diff > TOL: raise ValueError("reconstructed balanced accuracy differs from score CSV")
    digest = hashlib.sha256(json.dumps(reconstructed, sort_keys=True, separators=(",",":")).encode()).hexdigest()
    return rows, max_diff, digest

def derive_effects(scores, coords):
    result = {}
    for p in PEOPLE:
        own, donor = [], []
        for d in range(10):
            own.append(scores[coords[p,d,"own"]]["ba"])
            donor.append(statistics.mean(scores[coords[p,d,f"single_{i}"]]["ba"] for i in range(3)))
        result[p] = {"own_mean_ba":statistics.mean(own), "mean_single_ba":statistics.mean(donor),
                     "effect_pp":statistics.mean((a-b)*100 for a,b in zip(own,donor))}
    return result

def validate_effect_file(path, effects):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EFFECT_FIELDS: raise ValueError("participant-effect schema differs")
        rows = list(reader)
    if len(rows) != 24 or {int(r["participant_id"]) for r in rows} != set(PEOPLE):
        raise ValueError("participant-effect inventory differs")
    for row in rows:
        p=int(row["participant_id"])
        if int(row["draws"]) != 10 or any(not close(row[k], effects[p][k]) for k in EFFECT_FIELDS[2:]):
            raise ValueError("saved participant effect differs from reconstruction")

def summarize(effects):
    x=np.asarray([effects[p]["effect_pp"] for p in PEOPLE]); mean=float(x.mean()); sd=float(x.std(ddof=1))
    half=float(student_t.ppf(.975,23)*sd/math.sqrt(24)); rng=np.random.default_rng(2026092408)
    boot=np.mean(x[rng.integers(0,24,size=(10000,24))],axis=1)
    rng=np.random.default_rng(2026092409); extreme=0; observed=abs(mean)
    for _ in range(10):
        signs=rng.integers(0,2,size=(10000,24),dtype=np.int8)*2-1
        extreme += int(np.sum(np.abs(np.mean(signs*x,axis=1)) >= observed-TOL))
    low,high=mean-half,mean+half
    classification = "supportive" if low>0 and mean>=2 else ("positive_below_threshold" if low>0 else ("adverse" if high<0 else "uncertain"))
    return {"n":24,"mean_pp":mean,"median_pp":float(np.median(x)),"sd_pp":sd,
            "t_ci_low_pp":low,"t_ci_high_pp":high,
            "bootstrap_ci_low_pp":float(np.quantile(boot,.025,method="linear")),
            "bootstrap_ci_high_pp":float(np.quantile(boot,.975,method="linear")),
            "signflip_two_sided_mc_p":(extreme+1)/100001,
            "positive":int(np.sum(x>TOL)),"zero":int(np.sum(np.abs(x)<=TOL)),"negative":int(np.sum(x<-TOL)),
            "classification":classification}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("predictions","scores","participant-effects","summary"): ap.add_argument("--"+name,required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path); args=ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    scores,coords=load_scores(args.scores); rows,max_diff,op_digest=reconstruct_predictions(args.predictions,scores)
    effects=derive_effects(scores,coords); validate_effect_file(args.participant_effects,effects); calculated=summarize(effects)
    saved=json.loads(args.summary.read_text(encoding="utf-8")); primary=saved.get("primary",{})
    if (saved.get("status") != "completed_confirmation_analysis" or saved.get("complete_people") != 24
            or saved.get("primary_endpoint") != "own minus mean of three single-other donors"):
        raise ValueError("saved confirmation summary identity differs")
    keys=("n","mean_pp","median_pp","sd_pp","t_ci_low_pp","t_ci_high_pp","bootstrap_ci_low_pp",
          "bootstrap_ci_high_pp","signflip_two_sided_mc_p","positive","zero","negative","classification")
    for key in keys:
        if (calculated[key] != primary.get(key) if isinstance(calculated[key],(str,int))
                else not close(calculated[key],primary.get(key))): raise ValueError("saved primary differs: "+key)
    inputs={k:getattr(args,k.replace("-","_")) for k in ("predictions","scores","participant-effects","summary")}
    output={"status":"complete_exact_saved_output_reproduction","schema":"portable-final-confirmation-v1",
            "inputs":{k:{"sha256":sha(p),"bytes":p.stat().st_size} for k,p in inputs.items()},
            "counts":{"prediction_rows":rows,"operations":len(scores),"participants":24},
            "max_abs_operation_ba_difference":max_diff,"operation_ba_sha256":op_digest,
            "participant_effects":[{"participant_id":p,**effects[p]} for p in PEOPLE],"primary":calculated,
            "recipe":{"bootstrap":{"draws":10000,"seed":2026092408,"quantile":"linear"},
                      "signflip":{"draws":100000,"seed":2026092409,"chunk":10000,"plus_one":True},
                      "classification":"t95 lower>0 and mean>=2pp; lower>0 below2; upper<0 adverse; else uncertain"},
            "script_sha256":sha(__file__)}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":output["status"],"classification":calculated["classification"]},sort_keys=True))

if __name__ == "__main__": main()
