"""Run the frozen compact CPU EEGNet grid with separately refitted checkpoints."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from run_review3_grid import (DatasetStore, atomic_json, fingerprint, read_gzip_json,
                             seeded_rng, sha256)


def balanced_ids(pool, store, wanted, rng):
    y = store.labels(pool)
    out = []
    for label in (0, 1):
        available = [p for p, z in zip(pool, y) if z == label]
        if len(available) < wanted // 2:
            raise ValueError("not enough source-selection trials")
        out += rng.permutation(available).tolist()[:wanted // 2]
    return out


def load_epochs(store):
    """Read only index-declared people/sessions, validating original identities."""
    import numpy as np
    from expose.review2_controls import oas_covariances
    records = {}
    if store.dataset == "openbmi":
        original = json.loads((ROOT / "result/day3/preparation_r001/schema.json").read_text())["records"]
        records = {(r["subject"], r["session"]): r for r in original}
    result = {}
    for record in store.records:
        subject, session = record["subject"], record["session"]
        if store.dataset == "openbmi":
            old = records[(subject, session)]
            cache = old["cache"]
            path = ROOT / cache["array_path"]
            if sha256(path) != cache["array_sha256"] or old["role"] != record["role"]:
                raise ValueError("neural source cache/role mismatch")
            with np.load(path, allow_pickle=False) as data:
                x, y = data["X"], data["y"]
        else:
            from expose.bnci2014 import load_bnci2014_epochs
            raw = ROOT / "data/raw/bnci2014_001_review3" / f"A{subject:02d}{'T' if session == 1 else 'E'}.mat"
            rows = list(csv.DictReader((ROOT / "research/review3_20260923/BNCI_DOWNLOAD_RECEIPT.tsv").open(), delimiter="\t"))
            matches = [r for r in rows if Path(r.get("path", r.get("file", r.get("filename", "")))).name == raw.name]
            if len(matches) != 1 or sha256(raw) != matches[0]["sha256"]:
                raise ValueError("BNCI neural raw identity mismatch")
            epochs = load_bnci2014_epochs(raw, subject=subject, session="T" if session == 1 else "E")
            x, y = epochs.X, epochs.y
            if list(epochs.trial_ids) != record["trial_ids"]:
                raise ValueError("BNCI neural trial order differs from covariance path")
        if len(x) != len(record["trial_ids"]) or not np.array_equal(y, record["y"]):
            raise ValueError("neural trial/label identity mismatch")
        # Check preprocessing against the independently persisted covariance path.
        cov = store.load_arrays(record["record_id"])["cov"]
        np.testing.assert_allclose(oas_covariances(x[:3]), cov[:3], rtol=1e-10, atol=1e-20)
        for i, trial in enumerate(record["trial_ids"]):
            result[trial] = x[i].astype(np.float32)
    return result


def run(config_path, out):
    import numpy as np
    import torch
    from expose.review3_eegnet import (adapt_source, normalized_tensor, probabilities,
                                      select_and_refit_source, seed_cpu)
    config = json.loads(config_path.read_text())
    required = {"frozen_files", "seed", "cpu_seconds", "wall_seconds", "index_path", "index_sha256",
                "operations_path", "memberships_path", "study_id", "dataset", "montage", "mode",
                "draws", "source_sizes", "history_totals", "expected_scores",
                "max_source_epochs", "source_patience", "fine_epochs"}
    if set(config) != required or (config["max_source_epochs"], config["source_patience"], config["fine_epochs"]) != (50, 10, 20):
        raise ValueError("neural config or fixed training settings differ")
    if config["draws"] not in ([0, 1, 2], [0], [1], [2]) or config["history_totals"] != [0, 4, 10, 20, 40, 60, 100]:
        raise ValueError("neural draw/history grid differs")
    for relative, expected in config["frozen_files"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError("frozen neural input/code changed: " + relative)
    if out.exists():
        raise FileExistsError("neural run output already exists")
    out.mkdir(parents=True)
    start, cpu = time.monotonic(), time.process_time()
    receipt = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                   config_sha256=sha256(config_path), source_fits=0, adaptations=0,
                   numeric_threads=1, interop_threads=1, confirmation_data_accessed=False,
                   cuda_used=False, predictions=0, scores=0, checkpoints=[])
    atomic_json(out / "receipt.json", receipt)
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (config["cpu_seconds"], config["cpu_seconds"] + 10))
    def expired(signum, _frame):
        raise TimeoutError("neural resource limit signal " + str(signum))
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGXCPU, expired)
    signal.alarm(config["wall_seconds"])
    try:
        seed_cpu(config["seed"])
        store = DatasetStore(ROOT / config["index_path"], config["index_sha256"])
        plan = read_gzip_json(ROOT / config["operations_path"])
        sets = read_gzip_json(ROOT / config["memberships_path"])
        operations = [p for p in plan if p["method"] == "plain_ts"
                      and p["study_id"] == config["study_id"]
                      and p["dataset"] == config["dataset"] and p["montage"] == config["montage"]
                      and p["mode"] == config["mode"] and p["donor_count"] == "all"
                      and p["draw"] in config["draws"]
                      and p["source_size"] in config["source_sizes"]
                      and p["h_total"] in config["history_totals"]]
        if len(operations) != config["expected_scores"]:
            raise ValueError("neural planned score count differs")
        epochs = load_epochs(store)
        take = lambda ids: np.stack([epochs[t] for t in ids])
        checkpoints = {}
        score_rows = []
        prediction_path = out / "predictions.csv.gz"
        with gzip.open(prediction_path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["operation_id", "target", "draw", "source_size", "h_total", "trial_id", "y_true", "y_pred", "p_right", "p_left"])
            writer.writeheader()
            for op in operations:
                refs = op["memberships"]
                source_ids, history_ids, eval_ids = (sets[refs[k]] for k in ("source", "personal", "evaluation"))
                source_key = fingerprint([refs["source"], op["draw"]])
                if source_key not in checkpoints:
                    source_people = sorted({int(store.subjects([t])[0]) for t in source_ids})
                    source_pool = [t for r in store.records if r["session"] == 1 and r["subject"] in source_people for t in r["trial_ids"]]
                    rng = seeded_rng(config["seed"], "neural_source_validation", source_key)
                    validation_people = sorted(rng.permutation(source_people).tolist()[:3])
                    validation_ids = [t for t in source_pool if int(store.subjects([t])[0]) in validation_people]
                    selection_pool = [t for t in source_pool if int(store.subjects([t])[0]) not in validation_people]
                    selected_ids = balanced_ids(selection_pool, store, min(len(source_ids), len(selection_pool)), rng)
                    if set(selected_ids) & set(validation_ids) or (set(source_pool) & set(eval_ids)):
                        raise ValueError("source validation overlap")
                    fit_seed = int(rng.integers(1, 2**30))
                    model, normalizer, fitting = select_and_refit_source(
                        take(selected_ids), store.labels(selected_ids), take(validation_ids), store.labels(validation_ids),
                        take(source_ids), store.labels(source_ids), fit_seed,
                        max_epochs=config["max_source_epochs"], patience=config["source_patience"])
                    checkpoint = out / f"source_{source_key}.pt"
                    torch.save({"state_dict": model.state_dict(), "normalizer_mean": normalizer[0],
                                "normalizer_scale": normalizer[1], "source_membership": refs["source"]}, checkpoint)
                    fitting.update(source_ids=source_ids, selection_ids=selected_ids, validation_ids=validation_ids,
                                   validation_people=validation_people, checkpoint_sha256=sha256(checkpoint))
                    atomic_json(out / f"source_{source_key}.json", fitting)
                    checkpoints[source_key] = model, normalizer
                    receipt["source_fits"] += 2
                    receipt["checkpoints"].append(checkpoint.name)
                model, normalizer = checkpoints[source_key]
                fine_seed = int(seeded_rng(config["seed"], "fine", source_key, op["target"]).integers(1, 2**30))
                if history_ids:
                    adapted, losses = adapt_source(model, normalizer, take(history_ids), store.labels(history_ids),
                                                  fine_seed, epochs=config["fine_epochs"])
                    receipt["adaptations"] += 1
                else:
                    adapted, losses = model, []
                p = probabilities(adapted, normalized_tensor(take(eval_ids), normalizer))
                if (p.shape != (len(eval_ids), 2) or not np.isfinite(p).all()
                        or np.any(p < 0) or np.any(p > 1)
                        or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
                    raise FloatingPointError("invalid neural probabilities before output")
                predicted, truth = p.argmax(1), store.labels(eval_ids)
                ba = float(np.mean([np.mean(predicted[truth == c] == c) for c in (0, 1)]))
                operation_id = "eegnet__" + op["operation_id"]
                score_rows.append({"operation_id": operation_id, "method": "eegnet", "target": op["target"],
                                   "draw": op["draw"], "source_size": op["source_size"], "h_total": op["h_total"],
                                   "balanced_accuracy": ba, "source_checkpoint": source_key,
                                   "fine_seed": fine_seed, "fine_losses": json.dumps(losses),
                                   "source_membership": refs["source"], "history_membership": refs["personal"]})
                for i, trial in enumerate(eval_ids):
                    writer.writerow(dict(operation_id=operation_id, target=op["target"], draw=op["draw"],
                                         source_size=op["source_size"], h_total=op["h_total"], trial_id=trial,
                                         y_true=int(truth[i]), y_pred=int(predicted[i]),
                                         p_right=float(p[i, 0]), p_left=float(p[i, 1])))
                receipt["scores"] += 1
                receipt["predictions"] += len(eval_ids)
                if receipt["scores"] % 21 == 0:
                    receipt.update(cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start)
                    atomic_json(out / "receipt.json", receipt)
                    print(json.dumps({k: receipt[k] for k in ("scores", "source_fits", "adaptations", "cpu_seconds")}), flush=True)
        with (out / "scores.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
            writer.writeheader(); writer.writerows(score_rows)
        outputs = {"predictions.csv.gz": sha256(out / "predictions.csv.gz"),
                   "scores.csv": sha256(out / "scores.csv")}
        checkpoint_names = receipt["checkpoints"]
        if len(checkpoint_names) != len(set(checkpoint_names)):
            raise ValueError("duplicate source checkpoint names")
        expected_sidecars = {Path(name).with_suffix(".json").name for name in checkpoint_names}
        if ({path.name for path in out.glob("source_*.pt")} != set(checkpoint_names)
                or {path.name for path in out.glob("source_*.json")} != expected_sidecars):
            raise ValueError("source checkpoint/sidecar inventory differs")
        for checkpoint_name in checkpoint_names:
            checkpoint = out / checkpoint_name
            metadata = out / (checkpoint.stem + ".json")
            fitting = json.loads(metadata.read_text())
            if fitting.get("checkpoint_sha256") != sha256(checkpoint):
                raise ValueError("source checkpoint differs from its fit sidecar")
            outputs[checkpoint.name] = sha256(checkpoint)
            outputs[metadata.name] = sha256(metadata)
        receipt["outputs"] = outputs
        receipt["output_binding_schema"] = "review3-neural-output-bindings-v2"
        receipt["status"] = "completed"
    except BaseException as error:
        receipt.update(status="failed", error=repr(error))
        traceback.print_exc()
    finally:
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                       cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-start,
                       peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(out / "receipt.json", receipt)
    if receipt["status"] != "completed":
        raise SystemExit(1)
    atomic_json(out / "COMPLETED.json", {"receipt_sha256": sha256(out / "receipt.json"),
                                         "predictions_sha256": sha256(out / "predictions.csv.gz"),
                                         "scores_sha256": sha256(out / "scores.csv"),
                                         "outputs": receipt["outputs"],
                                         "output_binding_schema": "review3-neural-output-bindings-v2"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config.resolve(), arguments.run_dir.resolve())
