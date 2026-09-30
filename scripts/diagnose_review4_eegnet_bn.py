#!/usr/bin/env python3
"""Source-only BatchNorm diagnostics for the six frozen OpenBMI8 EEGNet fits.

This script never reads development or confirmation epochs, target-session
predictions, or evaluation outcomes.  Its losses are source-training
resubstitution summaries and must not be presented as held-out performance.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from copy import deepcopy
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

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_RELATIVE = "result/day3/preparation_r001/schema.json"
CHECKPOINT_RELATIVE = "result/review3_20260923/openbmi8_neural_r001"
CORE_RELATIVE = "src/expose/review3_eegnet.py"
SCHEMA_SHA256 = "4cce8f8f8baf3107df60ee0df23ac650f85cc2639d156d1949f35633050401c3"
CORE_SHA256 = "dd779d9b462725f09ac2567fce5df2e7be7d51b37f4d0831b13dc6ffc1fb4bb5"
EXPECTED_FILES = {
    "source_28b3759b6f9c868ae54ea41cf97977f924e96b1d437dde013a26df66cd8addd4": {
        "json": "a7afd2cc542eb789f8178c7af63de6cb9b898670c62d2a9db45fbc652861157b",
        "pt": "309b029d631264112be1653ceb21ffbc18b610402aa34261a4853d62d036042e",
    },
    "source_3f30781c15296e5774a64740011083a9c043d622cb2b2f356b5529dd5c833630": {
        "json": "6eeddf084e04e1ce83c862dc5c29666e4bf662c0378e3c69d57042be96ecbf7b",
        "pt": "b8ded2dd14438d8e260082995ed96e2f2cd9c1a5d328ad1aaf46ed95a692d97c",
    },
    "source_4d13420865129d742acf57f42c1da68dded1a65a1fec8597d303ebe85046675b": {
        "json": "a730251d475303cebd420cecd7c54fc43bdb131d86df3bfb1eeacc19983d5fa2",
        "pt": "811e92d8d3d2037b2e99e1019ae337cc999021c6ff52b27a3353dde54cdc97bb",
    },
    "source_56b98dd4434877c26b72be5432922d7bf1d1e15a5b99133b07f4e0009ca5fffb": {
        "json": "5c07c55a83e7140e0b3c17d7f6502075ff356f8bf5e8f982bc722bf05eb6382a",
        "pt": "7a9e8ce50ba6a6979214b9810108a621c296b6672e33676cc25ea2e2cce14ece",
    },
    "source_8f36e33bdc80adb5dde913085b4312fc12508783278f51bcfc81614c5919a93b": {
        "json": "a862002e3eb4b951b10e3c89a8a3de4e230b6c4b05f457de57214b9eca04f827",
        "pt": "8594d577b4b1b28f01c3318e7f65f146e26e143516b0d32376cfd3d5122c1779",
    },
    "source_dca8d006cc16d465fd80022d79d25f6f1eb9498f430eb7a697f350b1cf4140ca": {
        "json": "2a9f3781179697c94d061ce112c8adbb57e86ff47230fbf3a231cfea66ea401c",
        "pt": "8b7abe474ff3a3230f2919c67a696d79a20b6c25afefcedd6d892b42d424a603",
    },
}
TRAIN_BATCH_SIZE = 64
DIAGNOSTIC_BATCH_SIZE = 60  # Divides both frozen source sizes, 300 and 1800.


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = Path(str(path) + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields):
    temporary = Path(str(path) + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def overlap_count(left, right):
    return len(set(left) & set(right))


def validate_sidecar(sidecar, available_ids):
    required = {
        "checkpoint_sha256", "final_source_trials", "refit_log", "refit_seed",
        "selected_epochs", "selection_ids", "selection_log",
        "selection_model_discarded", "selection_seed", "selection_training_trials",
        "selection_validation_trials", "source_ids", "target_evaluation_used",
        "validation_ids", "validation_people",
    }
    if set(sidecar) != required:
        raise ValueError("source sidecar fields differ")
    source = sidecar["source_ids"]
    selection = sidecar["selection_ids"]
    validation = sidecar["validation_ids"]
    for name, values in (("source", source), ("selection", selection),
                         ("validation", validation)):
        if (not isinstance(values, list) or not values
                or len(values) != len(set(values))
                or not set(values) <= available_ids):
            raise ValueError(name + " IDs are not unique SOURCE-session1 trials")
    if len(source) != sidecar["final_source_trials"]:
        raise ValueError("final source count differs")
    if len(selection) != sidecar["selection_training_trials"]:
        raise ValueError("selection source count differs")
    if len(validation) != sidecar["selection_validation_trials"]:
        raise ValueError("selection validation count differs")
    if set(selection) & set(validation):
        raise ValueError("selection training and validation IDs overlap")
    if sidecar["target_evaluation_used"] is not False:
        raise ValueError("source fit records target evaluation access")
    if sidecar["selection_model_discarded"] is not True:
        raise ValueError("selection model was not recorded as discarded")
    epochs = sidecar["selected_epochs"]
    if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
            or len(sidecar["refit_log"]) != epochs):
        raise ValueError("selected/refit epoch contract differs")
    return {
        "source_selection_overlap_n": overlap_count(source, selection),
        "source_validation_overlap_n": overlap_count(source, validation),
        "selection_validation_overlap_n": 0,
        "final_checkpoint_validation_status": (
            "not_held_out_from_final_refit"
            if overlap_count(source, validation) else "disjoint_from_final_refit"
        ),
    }


def load_source_pool(root, schema):
    records = [record for record in schema.get("records", [])
               if record.get("role") == "source" and record.get("session") == 1]
    if len(records) != 18 or len({record["subject"] for record in records}) != 18:
        raise ValueError("expected exactly 18 SOURCE people in session 1")
    pool = {}
    cache_inputs = {}
    for record in sorted(records, key=lambda row: row["subject"]):
        cache = record.get("cache", {})
        relative = cache.get("array_path")
        expected = cache.get("array_sha256")
        path = root / relative if isinstance(relative, str) else None
        if path is None or not isinstance(expected, str) or sha256(path) != expected:
            raise ValueError("SOURCE epoch cache identity differs")
        with np.load(path, allow_pickle=False) as values:
            if set(values.files) < {"X", "y"}:
                raise ValueError("SOURCE cache lacks X or y")
            x = np.asarray(values["X"])
            y = np.asarray(values["y"])
        if (x.shape != (100, 8, 750) or y.shape != (100,)
                or not np.isfinite(x).all() or set(y.astype(int).tolist()) != {0, 1}
                or not np.array_equal(y, y.astype(np.int64))):
            raise ValueError("SOURCE epoch cache contents differ")
        for index in range(100):
            trial = (f"lee2019:s{record['subject']:03d}:session1:"
                     f"offline:t{index:03d}")
            if trial in pool:
                raise ValueError("duplicate SOURCE trial identity")
            # The original neural loader explicitly converted every epoch to float32.
            pool[trial] = (x[index].astype(np.float32), int(y[index]))
        cache_inputs[relative] = expected
    if len(pool) != 1800:
        raise ValueError("SOURCE session-1 pool must contain 1800 trials")
    return pool, cache_inputs


def take(pool, trial_ids):
    missing = [trial for trial in trial_ids if trial not in pool]
    if missing:
        raise ValueError("requested non-SOURCE trial: " + missing[0])
    x = np.stack([pool[trial][0] for trial in trial_ids])
    y = np.asarray([pool[trial][1] for trial in trial_ids], dtype=np.int64)
    if x.dtype != np.float32 or x.shape[1:] != (8, 750) or set(y.tolist()) != {0, 1}:
        raise ValueError("invalid selected SOURCE epochs")
    return x, y


def batchnorm_layers(model):
    result = [(name, layer) for name, layer in model.named_modules()
              if isinstance(layer, nn.BatchNorm2d)]
    if [name for name, _ in result] != ["bn1", "bn2", "bn3"]:
        raise ValueError("EEGNet BatchNorm layer contract differs")
    return result


def layer_summary(layer, initial_momentum=0.01):
    tracked = int(layer.num_batches_tracked.detach().cpu())
    mean = layer.running_mean.detach().cpu().numpy().astype(np.float64)
    variance = layer.running_var.detach().cpu().numpy().astype(np.float64)
    if (tracked < 0 or not np.isfinite(mean).all() or not np.isfinite(variance).all()
            or np.any(variance < 0)):
        raise FloatingPointError("invalid BatchNorm running statistics")
    return {
        "num_batches_tracked": tracked,
        "initial_stat_fraction": float((1.0 - initial_momentum) ** tracked),
        "running_mean_mean": float(mean.mean()),
        "running_mean_sd": float(mean.std()),
        "running_mean_abs_max": float(np.abs(mean).max()),
        "running_var_mean": float(variance.mean()),
        "running_var_sd": float(variance.std()),
        "running_var_min": float(variance.min()),
        "running_var_max": float(variance.max()),
    }


@torch.no_grad()
def predict(model, x, batch_size, batch_statistics=False):
    diagnostic = deepcopy(model) if batch_statistics else model
    diagnostic.eval()
    if batch_statistics:
        # Only BatchNorm enters train mode. Dropout and all other modules remain in
        # evaluation mode. Running-stat updates happen only on this discarded copy.
        for _, layer in batchnorm_layers(diagnostic):
            layer.train()
    outputs = []
    for indexes in range(0, len(x), batch_size):
        outputs.append(diagnostic(x[indexes:indexes + batch_size]).softmax(1))
    value = torch.cat(outputs).cpu().numpy()
    if (value.shape != (len(x), 2) or not np.isfinite(value).all()
            or not np.allclose(value.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
        raise FloatingPointError("invalid diagnostic probabilities")
    return value


@torch.no_grad()
def recalibrate_batchnorm(model, x, batch_size=DIAGNOSTIC_BATCH_SIZE):
    calibrated = deepcopy(model)
    calibrated.eval()
    layers = batchnorm_layers(calibrated)
    for _, layer in layers:
        layer.reset_running_stats()
        layer.momentum = None  # PyTorch cumulative moving average.
        layer.train()
    for indexes in range(0, len(x), batch_size):
        calibrated(x[indexes:indexes + batch_size])
    expected = math.ceil(len(x) / batch_size)
    if any(int(layer.num_batches_tracked) != expected for _, layer in layers):
        raise RuntimeError("source-only cumulative BatchNorm pass count differs")
    calibrated.eval()
    return calibrated


def probability_metrics(probability, y):
    predicted = probability.argmax(axis=1)
    clipped = np.clip(probability[np.arange(len(y)), y], 1e-15, 1.0)
    counts = np.bincount(predicted, minlength=2)
    return {
        "source_resubstitution_log_loss": float(-np.log(clipped).mean()),
        "p_left_mean": float(probability[:, 1].mean()),
        "p_left_sd": float(probability[:, 1].std()),
        "predicted_right_n": int(counts[0]),
        "predicted_left_n": int(counts[1]),
        "single_predicted_class": bool(np.count_nonzero(counts) == 1),
        "source_resubstitution_accuracy": float(np.mean(predicted == y)),
    }


def comparison_metrics(reference, candidate):
    return {
        "changed_predictions_n": int(np.count_nonzero(
            reference.argmax(axis=1) != candidate.argmax(axis=1))),
        "mean_abs_probability_change": float(np.abs(reference - candidate).mean()),
        "max_abs_probability_change": float(np.abs(reference - candidate).max()),
    }


def load_checkpoint(path, expected_hash, expected_membership, eegnet_class):
    if sha256(path) != expected_hash:
        raise ValueError("checkpoint hash differs: " + path.name)
    # This is a trusted, project-owned checkpoint whose exact hash is checked
    # before deserialization. It contains NumPy normalizer arrays, so the legacy
    # weights_only=False loader is required.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if set(payload) != {"state_dict", "normalizer_mean", "normalizer_scale",
                       "source_membership"}:
        raise ValueError("checkpoint payload fields differ")
    if payload["source_membership"] != expected_membership:
        raise ValueError("checkpoint source membership differs")
    mean = np.asarray(payload["normalizer_mean"], dtype=np.float64)
    scale = np.asarray(payload["normalizer_scale"], dtype=np.float64)
    if (mean.shape != (1, 8, 1) or scale.shape != (1, 8, 1)
            or not np.isfinite(mean).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError("checkpoint normalizer differs")
    model = eegnet_class(8, 750)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, (mean, scale)


def run(root, checkpoint_dir, schema_path, output, cpu_seconds=120, wall_seconds=300):
    start_wall = time.monotonic()
    start_cpu = time.process_time()
    if output.exists():
        raise FileExistsError("diagnostic output already exists")
    output.mkdir(parents=True)
    receipt = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "six frozen OpenBMI8 SOURCE-session1 EEGNet checkpoints only",
        "development_epochs_accessed": False,
        "confirmation_epochs_accessed": False,
        "target_predictions_or_outcomes_accessed": False,
        "diagnostic_batch_size": DIAGNOSTIC_BATCH_SIZE,
        "source_loss_interpretation": "training resubstitution; never held-out validation",
    }
    atomic_json(output / "receipt.json", receipt)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 10))
        def expired(signum, _frame):
            raise TimeoutError("diagnostic resource signal " + str(signum))
        signal.signal(signal.SIGALRM, expired)
        signal.signal(signal.SIGXCPU, expired)
        signal.alarm(wall_seconds)
        if sha256(schema_path) != SCHEMA_SHA256 or sha256(root / CORE_RELATIVE) != CORE_SHA256:
            raise ValueError("frozen schema or EEGNet source changed")
        actual = {path.stem for path in checkpoint_dir.glob("source_*.json")}
        actual_pt = {path.stem for path in checkpoint_dir.glob("source_*.pt")}
        if actual != set(EXPECTED_FILES) or actual_pt != set(EXPECTED_FILES):
            raise ValueError("expected exactly six frozen checkpoint pairs")

        sys.path.insert(0, str(root / "src"))
        from expose.review3_eegnet import EEGNet, source_normalizer

        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        pool, cache_inputs = load_source_pool(root, schema)
        layer_rows, mode_rows, comparisons, checkpoint_rows = [], [], [], []
        checkpoint_inputs = {}
        for stem, hashes in sorted(EXPECTED_FILES.items()):
            sidecar_path = checkpoint_dir / (stem + ".json")
            checkpoint_path = checkpoint_dir / (stem + ".pt")
            if sha256(sidecar_path) != hashes["json"]:
                raise ValueError("source sidecar hash differs: " + stem)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar["checkpoint_sha256"] != hashes["pt"]:
                raise ValueError("sidecar checkpoint hash differs: " + stem)
            overlap = validate_sidecar(sidecar, set(pool))
            source_x, source_y = take(pool, sidecar["source_ids"])
            # The membership identifier is stored only in the checkpoint. Read it
            # after the fixed file hash is verified, then bind it during full load.
            trusted = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            membership = trusted.get("source_membership")
            del trusted
            if not isinstance(membership, str) or not membership:
                raise ValueError("invalid checkpoint source membership")
            model, normalization = load_checkpoint(
                checkpoint_path, hashes["pt"], membership, EEGNet)
            expected_norm = source_normalizer(source_x)
            if (not np.array_equal(normalization[0], expected_norm[0])
                    or not np.array_equal(normalization[1], expected_norm[1])):
                raise ValueError("checkpoint normalizer is not from exact source IDs")
            normalized = torch.as_tensor(
                (source_x.astype(np.float64) - normalization[0]) / normalization[1],
                dtype=torch.float32)

            expected_batches = (sidecar["selected_epochs"]
                                * math.ceil(len(source_x) / TRAIN_BATCH_SIZE))
            original_layers = {}
            for name, layer in batchnorm_layers(model):
                summary = layer_summary(layer)
                original_layers[name] = summary
                layer_rows.append({
                    "checkpoint": stem, "layer": name, "source_n": len(source_x),
                    "selected_epochs": sidecar["selected_epochs"],
                    "expected_training_batches": expected_batches,
                    "batch_count_matches_recipe": (
                        summary["num_batches_tracked"] == expected_batches),
                    **summary,
                })

            eval_probability = predict(model, normalized, batch_size=128)
            batch_probability = predict(
                model, normalized, batch_size=DIAGNOSTIC_BATCH_SIZE,
                batch_statistics=True)
            recalibrated = recalibrate_batchnorm(
                model, normalized, batch_size=DIAGNOSTIC_BATCH_SIZE)
            recal_probability = predict(recalibrated, normalized, batch_size=128)
            probabilities = {
                "checkpoint_eval_running_stats": eval_probability,
                "batch_stats_copy_dropout_disabled": batch_probability,
                "source_cumulative_bn_recalibration_copy": recal_probability,
            }
            for mode, probability in probabilities.items():
                mode_rows.append({
                    "checkpoint": stem, "mode": mode, "source_n": len(source_y),
                    "source_right_n": int(np.count_nonzero(source_y == 0)),
                    "source_left_n": int(np.count_nonzero(source_y == 1)),
                    **probability_metrics(probability, source_y),
                })
            for mode in ("batch_stats_copy_dropout_disabled",
                         "source_cumulative_bn_recalibration_copy"):
                comparisons.append({
                    "checkpoint": stem, "candidate_mode": mode,
                    **comparison_metrics(eval_probability, probabilities[mode]),
                })

            recalibrated_layers = {
                name: layer_summary(layer) for name, layer in batchnorm_layers(recalibrated)
            }
            checkpoint_rows.append({
                "checkpoint": stem,
                "checkpoint_sha256": hashes["pt"],
                "source_membership": membership,
                "source_n": len(source_y),
                "selected_epochs": sidecar["selected_epochs"],
                "refit_seed": sidecar["refit_seed"],
                **overlap,
                "original_batchnorm": original_layers,
                "recalibrated_batchnorm": recalibrated_layers,
            })
            checkpoint_inputs[str(sidecar_path.relative_to(root))] = hashes["json"]
            checkpoint_inputs[str(checkpoint_path.relative_to(root))] = hashes["pt"]

        layer_fields = [
            "checkpoint", "layer", "source_n", "selected_epochs",
            "expected_training_batches", "batch_count_matches_recipe",
            "num_batches_tracked", "initial_stat_fraction", "running_mean_mean",
            "running_mean_sd", "running_mean_abs_max", "running_var_mean",
            "running_var_sd", "running_var_min", "running_var_max",
        ]
        mode_fields = [
            "checkpoint", "mode", "source_n", "source_right_n", "source_left_n",
            "source_resubstitution_log_loss", "source_resubstitution_accuracy",
            "p_left_mean", "p_left_sd", "predicted_right_n", "predicted_left_n",
            "single_predicted_class",
        ]
        comparison_fields = [
            "checkpoint", "candidate_mode", "changed_predictions_n",
            "mean_abs_probability_change", "max_abs_probability_change",
        ]
        write_csv(output / "batchnorm_layers.csv", layer_rows, layer_fields)
        write_csv(output / "source_prediction_modes.csv", mode_rows, mode_fields)
        write_csv(output / "source_prediction_comparisons.csv", comparisons,
                  comparison_fields)
        atomic_json(output / "checkpoint_details.json", checkpoint_rows)
        inputs = {
            SCHEMA_RELATIVE: SCHEMA_SHA256,
            CORE_RELATIVE: CORE_SHA256,
            **cache_inputs,
            **checkpoint_inputs,
        }
        atomic_json(output / "INPUTS.json", inputs)
        receipt.update({
            "status": "complete",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoints": len(checkpoint_rows),
            "source_cache_records": 18,
            "batchnorm_layer_rows": len(layer_rows),
            "prediction_mode_rows": len(mode_rows),
            "comparison_rows": len(comparisons),
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.monotonic() - start_wall,
            "outputs": {},
        })
        for name in ("batchnorm_layers.csv", "source_prediction_modes.csv",
                     "source_prediction_comparisons.csv", "checkpoint_details.json",
                     "INPUTS.json"):
            receipt["outputs"][name] = sha256(output / name)
        atomic_json(output / "receipt.json", receipt)
    except BaseException as error:
        receipt.update({
            "status": "failed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "cpu_seconds": time.process_time() - start_cpu,
            "wall_seconds": time.monotonic() - start_wall,
        })
        atomic_json(output / "receipt.json", receipt)
        raise
    finally:
        signal.alarm(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-seconds", type=int, default=120)
    parser.add_argument("--wall-seconds", type=int, default=300)
    args = parser.parse_args()
    root = args.root.resolve()
    checkpoint_dir = (args.checkpoint_dir or root / CHECKPOINT_RELATIVE).resolve()
    schema = (args.schema or root / SCHEMA_RELATIVE).resolve()
    output = args.output.resolve()
    if args.cpu_seconds < 1 or args.cpu_seconds > 120 or args.wall_seconds < 1:
        raise ValueError("resource limits differ from bounded diagnostic")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    run(root, checkpoint_dir, schema, output,
        cpu_seconds=args.cpu_seconds, wall_seconds=args.wall_seconds)


if __name__ == "__main__":
    main()
