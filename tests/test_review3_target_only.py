import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("review3_target_only", ROOT / "scripts/run_review3_target_only.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_only_fit_never_receives_source_covariances(monkeypatch):
    runner = load_runner()
    calls = []

    class Representation:
        def transform(self, values):
            return np.asarray(values).reshape(len(values), -1)

    class Model:
        def predict(self, values):
            return np.zeros(len(values), dtype=int)
        def predict_proba(self, values):
            return np.tile([.75, .25], (len(values), 1))

    def representation(values, source_count, mode):
        calls.append((len(values), source_count, mode))
        return Representation()

    monkeypatch.setattr(runner, "fit_representation", representation)
    monkeypatch.setattr(runner, "train_classifier", lambda z, y, source_count, C, weights: Model())
    personal = np.stack([np.eye(2), np.eye(2) * 2])
    predicted, probabilities = runner.fit_target_only(personal, np.array([0, 1]), np.stack([np.eye(2)] * 3))
    assert calls == [(2, 2, "source_frozen")]
    assert predicted.tolist() == [0, 0, 0]
    assert probabilities.shape == (3, 2)


def test_target_only_accepts_a_probability_tie_when_model_label_is_within_core_tolerance(monkeypatch):
    runner = load_runner()
    class Representation:
        def transform(self, values): return np.asarray(values).reshape(len(values), -1)
    class Model:
        def predict(self, values): return np.ones(len(values), dtype=int)
        def predict_proba(self, values): return np.tile([.5, .5], (len(values), 1))
    monkeypatch.setattr(runner, "fit_representation", lambda *unused: Representation())
    monkeypatch.setattr(runner, "train_classifier", lambda *unused: Model())
    X = np.stack([np.eye(2), np.eye(2) * 2])
    predicted, _ = runner.fit_target_only(X, np.array([0, 1]), np.stack([np.eye(2)]))
    assert predicted.tolist() == [1]


def test_openbmi_fixture_writes_one_score_and_all_evaluation_predictions_per_anchor(tmp_path, monkeypatch):
    runner = load_runner()
    config_path, index_path, plan_dir, out = (tmp_path / "CONFIG.json", tmp_path / "INDEX.json", tmp_path / "plan", tmp_path / "out")
    config_path.write_text("{}", encoding="utf-8"); index_path.write_text("{}", encoding="utf-8"); plan_dir.mkdir()
    (plan_dir / "PLAN_RECEIPT.json").write_text("{}", encoding="utf-8")
    sets, operations = {}, []
    for draw in range(10):
        for target in range(101, 113):
            for h in runner.HISTORY_TOTALS:
                source, personal, evaluation = (f"source-{draw}", f"personal-{draw}-{target}-{h}", f"eval-{target}")
                sets[source] = ["source-only"]
                sets[personal] = [f"p{target}-{h}-{i}" for i in range(h)]
                sets[evaluation] = [f"e{target}-0", f"e{target}-1", f"e{target}-2", f"e{target}-3"]
                operations.append({"operation_id": f"op-{draw}-{target}-{h}", "study_id": "openbmi_main", "dataset": "openbmi",
                    "montage": "8ch", "mode": "frozen_source", "method": "plain_ts", "draw": draw, "target": target,
                    "donor_count": "all", "source_size": 1800, "source_n": 1800, "h_total": h, "reference_trials_per_person": 4,
                    "memberships": {"source": source, "personal": personal, "evaluation": evaluation, "source_reference": None, "target_reference": None}, "tuning_id": "source-tuned"})
    config = {"draws": 10, "index_sha256": runner.sha256(index_path),
              "studies": [{"study_id": "openbmi_main", "mode": "frozen_source", "source_sizes": [100, 1800]}]}

    class Store:
        source_matrix_requests = 0
        def __init__(self, *unused): pass
        def subjects(self, ids):
            return np.array([int(value.split("-")[0][1:]) for value in ids])
        def labels(self, ids):
            return np.array([index % 2 for index, _ in enumerate(ids)], dtype=int)
        def matrices(self, ids, kind):
            if ids == ["source-only"]:
                self.source_matrix_requests += 1
                raise AssertionError("source covariance must not be loaded")
            return np.stack([np.eye(2) * (i + 1) for i in range(len(ids))])

    monkeypatch.setattr(runner.grid, "validate_config", lambda *unused: config)
    monkeypatch.setattr(runner.grid, "validate_plan", lambda *unused: ({"files": {}}, operations, sets, []))
    monkeypatch.setattr(runner.grid, "DatasetStore", Store)
    monkeypatch.setattr(runner, "fit_target_only", lambda X, y, E: (np.zeros(len(E), dtype=int), np.tile([.6, .4], (len(E), 1))))
    receipt = runner.run(config_path, index_path, plan_dir, out)
    assert receipt["counts"] == {"operations": 720, "predictions": 2880}
    with gzip.open(out / "predictions.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with (out / "operation_scores.csv").open(newline="", encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    assert len(predictions) == 2880 and len(scores) == 720
    assert {row["C"] for row in scores} == {"1.0"}
    assert json.loads((out / "PLAN_BINDING.json").read_text())["source_covariances_used_for_fitting"] is False
