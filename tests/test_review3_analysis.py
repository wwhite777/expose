import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analyze_review3 import reconstruct_grid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gzip_json(path, value):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(value, handle)


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def fixture_grid(tmp_path):
    plan, grid = tmp_path / 'plan', tmp_path / 'grid'
    plan.mkdir(); grid.mkdir()
    ids = ['t0', 't1', 't2', 't3']
    membership = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    op = {'operation_id': 'op', 'study_id': 'openbmi_main', 'method': 'plain_ts', 'draw': 0, 'target': 2,
          'donor_count': 'all', 'source_size': 100, 'source_n': 100, 'h_total': 0, 'C': 1.0,
          'tuning_id': 'tune', 'memberships': {'evaluation': membership, 'source': 'source', 'personal': 'personal'}}
    gzip_json(plan / 'operations.json.gz', [op]); gzip_json(plan / 'membership_sets.json.gz', {membership: ids})
    plan_receipt = {'status': 'planned_no_fits', 'files': {name: sha(plan / name) for name in ('operations.json.gz', 'membership_sets.json.gz')}}
    (plan / 'PLAN_RECEIPT.json').write_text(json.dumps(plan_receipt))
    predictions = [dict(operation_id='op', trial_id=trial, y_true=truth, y_pred=pred, p0=p0, p1=p1)
                   for trial, truth, pred, p0, p1 in [('t0', 0, 0, .9, .1), ('t1', 0, 1, .4, .6),
                                                       ('t2', 1, 1, .2, .8), ('t3', 1, 1, .1, .9)]]
    with gzip.open(grid / 'predictions.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    score = {key: op[key] for key in ('operation_id', 'study_id', 'method', 'draw', 'target', 'donor_count', 'source_size', 'source_n', 'h_total', 'C')}
    score.update(balanced_accuracy=.75, evaluation_n=4, source_membership='source', personal_membership='personal')
    write_csv(grid / 'operation_scores.csv', [score])
    write_csv(grid / 'selected_c.csv', [{'tuning_id': 'tune', 'selected_C': 1.0}])
    receipt = {'status': 'completed', 'plan_receipt_sha256': sha(plan / 'PLAN_RECEIPT.json'),
               'outputs': {'predictions.csv.gz': sha(grid / 'predictions.csv.gz'), 'operation_scores.csv': sha(grid / 'operation_scores.csv'),
                           'selected_c.csv': sha(grid / 'selected_c.csv')}}
    (grid / 'receipt.json').write_text(json.dumps(receipt))
    (grid / 'COMPLETED.json').write_text(json.dumps({'receipt_sha256': sha(grid / 'receipt.json')}))
    return grid, plan


def refresh_receipt(grid):
    receipt = json.loads((grid / 'receipt.json').read_text())
    for name in ('predictions.csv.gz', 'operation_scores.csv', 'selected_c.csv'):
        receipt['outputs'][name] = sha(grid / name)
    (grid / 'receipt.json').write_text(json.dumps(receipt))
    (grid / 'COMPLETED.json').write_text(json.dumps({'receipt_sha256': sha(grid / 'receipt.json')}))


def test_reconstruction_binds_plan_ids_and_saved_score(tmp_path):
    grid, plan = fixture_grid(tmp_path)
    rows, verified = reconstruct_grid('fixture', grid, plan)
    assert rows[0]['balanced_accuracy'] == .75 and verified['operations'] == 1


def test_reconstruction_rejects_hash_valid_corrupted_score(tmp_path):
    grid, plan = fixture_grid(tmp_path)
    rows = list(csv.DictReader((grid / 'operation_scores.csv').open()))
    rows[0]['balanced_accuracy'] = '.5'; write_csv(grid / 'operation_scores.csv', rows); refresh_receipt(grid)
    with pytest.raises(ValueError, match='saved score differs'):
        reconstruct_grid('fixture', grid, plan)


def test_reconstruction_rejects_missing_expected_prediction_row(tmp_path):
    grid, plan = fixture_grid(tmp_path)
    with gzip.open(grid / 'predictions.csv.gz', 'rt', newline='') as handle:
        rows = list(csv.DictReader(handle))[:-1]
    with gzip.open(grid / 'predictions.csv.gz', 'wt', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    refresh_receipt(grid)
    with pytest.raises(ValueError, match='evaluation IDs'):
        reconstruct_grid('fixture', grid, plan)
