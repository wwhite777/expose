import csv
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_merger(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "merge_review3_neural", REPO / "scripts/merge_review3_neural.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fields, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def build_fixture(tmp_path, merger):
    marker = tmp_path / "frozen.txt"; marker.write_text("frozen\n")
    parent = {
        "frozen_files": {"frozen.txt": merger.sha256(marker)}, "seed": 2026092305,
        "cpu_seconds": 10800, "wall_seconds": 14000,
        "index_path": "index.json", "index_sha256": "a" * 64,
        "operations_path": "operations.json.gz", "memberships_path": "memberships.json.gz",
        "study_id": "bnci_main", "dataset": "bnci2014", "montage": "22ch", "mode": "loso",
        "draws": [0, 1, 2], "source_sizes": ["all"],
        "history_totals": list(merger.HISTORY_TOTALS), "expected_scores": 189,
        "max_source_epochs": 50, "source_patience": 10, "fine_epochs": 20,
    }
    parent_path = tmp_path / "parent.json"; write_json(parent_path, parent)
    manifest = {"schema": merger.SCHEMA,
                "parent_config": {"path": "parent.json", "sha256": merger.sha256(parent_path)},
                "parts": []}
    for draw in range(3):
        config = dict(parent, draws=[draw], expected_scores=63)
        config_path = tmp_path / f"config{draw}.json"; write_json(config_path, config)
        directory = tmp_path / f"part{draw}"; directory.mkdir()
        checkpoints = []
        for target in range(1, 10):
            checkpoint = directory / f"source_key{draw}_{target}.pt"
            checkpoint.write_bytes(b"weights" + bytes([draw, target]))
            metadata = directory / f"source_key{draw}_{target}.json"
            write_json(metadata, {"checkpoint_sha256": merger.sha256(checkpoint),
                                  "selected_epochs": 3})
            checkpoints.append(checkpoint.name)
        scores, predictions = [], []
        for target in range(1, 10):
            for history in merger.HISTORY_TOTALS:
                operation = f"eegnet__bnci-d{draw}-t{target}-h{history}"
                scores.append({"operation_id": operation, "method": "eegnet", "target": target,
                               "draw": draw, "source_size": "all", "h_total": history,
                               "balanced_accuracy": 1.0,
                               "source_checkpoint": f"key{draw}_{target}",
                               "fine_seed": 123, "fine_losses": "[]",
                               "source_membership": f"membership-{draw}-{target}",
                               "history_membership": f"history-{draw}-{target}-{history}"})
                for trial in range(144):
                    truth = trial % 2
                    predictions.append({"operation_id": operation, "target": target, "draw": draw,
                                        "source_size": "all", "h_total": history,
                                        "trial_id": f"target{target}-trial{trial}",
                                        "y_true": truth, "y_pred": truth,
                                        "p_right": 1.0 if truth == 0 else 0.0,
                                        "p_left": 1.0 if truth == 1 else 0.0})
        score_path, prediction_path = directory / "scores.csv", directory / "predictions.csv.gz"
        write_csv(score_path, scores, merger.SCORE_FIELDS)
        write_csv(prediction_path, predictions, merger.PREDICTION_FIELDS, compressed=True)
        outputs = {"predictions.csv.gz": merger.sha256(prediction_path),
                   "scores.csv": merger.sha256(score_path)}
        for name in checkpoints:
            checkpoint = directory / name
            metadata = directory / (checkpoint.stem + ".json")
            outputs[checkpoint.name] = merger.sha256(checkpoint)
            outputs[metadata.name] = merger.sha256(metadata)
        receipt = {"status": "completed", "config_sha256": merger.sha256(config_path),
                   "source_fits": 18, "adaptations": 54, "predictions": 9072, "scores": 63,
                   "checkpoints": checkpoints, "cpu_seconds": 10 + draw,
                   "wall_seconds": 20 + draw, "numeric_threads": 1, "interop_threads": 1,
                   "confirmation_data_accessed": False, "cuda_used": False,
                   "outputs": outputs,
                   "output_binding_schema": "review3-neural-output-bindings-v2"}
        receipt_path = directory / "receipt.json"; write_json(receipt_path, receipt)
        completed = {"receipt_sha256": merger.sha256(receipt_path),
                     "predictions_sha256": merger.sha256(prediction_path),
                     "scores_sha256": merger.sha256(score_path), "outputs": outputs,
                     "output_binding_schema": "review3-neural-output-bindings-v2"}
        write_json(directory / "COMPLETED.json", completed)
        manifest["parts"].append({"draw": draw, "config_path": config_path.name,
                                  "config_sha256": merger.sha256(config_path),
                                  "directory": directory.name})
    manifest_path = tmp_path / "parts.json"; write_json(manifest_path, manifest)
    return manifest_path


def test_merge_three_hash_bound_partitions_without_copying_weights(tmp_path):
    merger = load_merger(tmp_path)
    manifest = build_fixture(tmp_path, merger)
    output = tmp_path / "aggregate"
    receipt = merger.merge(manifest, output)
    assert receipt["status"] == "completed" and receipt["type"] == "partition_merge"
    assert receipt["counts"] == {"scores": 189, "predictions": 27216,
                                 "targets": 9, "checkpoint_references": 27}
    assert receipt["summed_partition_cpu_seconds"] == 33
    assert not list(output.glob("*.pt"))
    provenance = json.loads((output / "PARTITION_PROVENANCE.json").read_text())
    assert provenance["weights_copied"] is False and len(provenance["parts"]) == 3
    for part in provenance["parts"]:
        assert len(part["checkpoint_references"]) == 9
        assert {item["source_membership"] for item in part["checkpoint_references"]} == {
            f"membership-{part['draw']}-{target}" for target in range(1, 10)}
    completed = json.loads((output / "COMPLETED.json").read_text())
    assert completed["receipt_sha256"] == merger.sha256(output / "receipt.json")


def test_merge_rejects_missing_draw_partition(tmp_path):
    merger = load_merger(tmp_path)
    manifest_path = build_fixture(tmp_path, merger)
    manifest = json.loads(manifest_path.read_text()); manifest["parts"].pop()
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="exactly three|complement"):
        merger.merge(manifest_path, tmp_path / "aggregate")


def test_merge_rejects_duplicate_operation_across_hash_valid_parts(tmp_path):
    merger = load_merger(tmp_path)
    manifest_path = build_fixture(tmp_path, merger)
    first = next(csv.DictReader((tmp_path / "part0/scores.csv").open()))["operation_id"]
    score_path = tmp_path / "part1/scores.csv"
    with score_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["operation_id"] = first
    write_csv(score_path, rows, merger.SCORE_FIELDS)
    receipt_path = tmp_path / "part1/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"]["scores.csv"] = merger.sha256(score_path)
    write_json(receipt_path, receipt)
    completed_path = tmp_path / "part1/COMPLETED.json"
    completed = json.loads(completed_path.read_text())
    completed["receipt_sha256"] = merger.sha256(receipt_path)
    completed["scores_sha256"] = merger.sha256(score_path)
    completed["outputs"] = receipt["outputs"]
    write_json(completed_path, completed)
    with pytest.raises(ValueError, match="duplicate operation ID"):
        merger.merge(manifest_path, tmp_path / "aggregate")


def test_merge_rejects_joint_checkpoint_and_sidecar_tamper(tmp_path):
    merger = load_merger(tmp_path)
    manifest_path = build_fixture(tmp_path, merger)
    checkpoint = tmp_path / "part1/source_key1_4.pt"
    sidecar = tmp_path / "part1/source_key1_4.json"
    checkpoint.write_bytes(b"replacement weights")
    metadata = json.loads(sidecar.read_text())
    metadata["checkpoint_sha256"] = merger.sha256(checkpoint)
    write_json(sidecar, metadata)
    with pytest.raises(ValueError, match="completed output bindings"):
        merger.merge(manifest_path, tmp_path / "aggregate")
