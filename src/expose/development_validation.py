"""Fail-closed structural checks for the frozen development continuation."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .development import logical_cells, membership, operations
from .provenance import file_sha256 as sha
from .runtime import _validate_completion
from .scout import trial_id


def completed_stage(root, directory, required):
    root, directory = Path(root), Path(directory)
    supervisor = json.loads((directory / 'supervisor_receipt.json').read_text())
    if supervisor['status'] != 'completed' or supervisor['exit_code'] != 0:
        raise ValueError('producer supervisor did not complete')
    identities = _validate_completion(directory, supervisor['child_pid'])
    if identities['run_id'] != directory.name or any(supervisor.get(k) != v for k, v in identities.items()):
        raise ValueError('producer supervisor identity differs')
    marker = json.loads((directory / 'COMPLETED.json').read_text())
    if not set(required) <= set(marker['outputs']):
        raise ValueError('producer omitted required output')
    manifest = json.loads((directory / 'INPUT_MANIFEST.json').read_text())
    if not manifest or any(sha(root / name) != digest for name, digest in manifest.items()):
        raise ValueError('producer dependency differs')
    return marker


def validate_split(rows, cfg):
    expected = {(s, 1) for s in cfg['source_subjects']} | {(s, t) for s in cfg['development_subjects'] for t in (1, 2)}
    if len(rows) != 4200:
        raise ValueError('split trial count differs')
    result, cells = {}, defaultdict(list)
    for row in rows:
        s, t, i, y = (int(row[k]) for k in ('subject_id', 'session', 'trial_index_zero_based', 'label'))
        tid = row['trial_id']
        role = 'source' if s in cfg['source_subjects'] else 'development'
        if ((s, t) not in expected or row['role'] != role or tid != trial_id(s, t, i)
                or tid in result or not 0 <= i < 100 or y not in (0, 1)):
            raise ValueError('split identity, role or label differs')
        if row['class_name'] != ['right_hand', 'left_hand'][y] or row['mat_variable'] != 'EEG_MI_train':
            raise ValueError('split class/segment differs')
        result[tid] = row
        cells[s, t].append(y)
    if set(cells) != expected or any(Counter(y) != {0: 50, 1: 50} for y in cells.values()):
        raise ValueError('split group count/balance differs')
    for key in ('raw_support_sha256', 'epoch_sha256'):
        if len({r[key] for r in rows}) != 4200:
            raise ValueError('duplicate EEG content in split')
    return result


def membership_rows(op, source, history, evaluation):
    source_usage = 'source_representation' if op['arm'] == 'calibration' and op['dose'] else 'source_training'
    history_usage = 'target_update' if op['arm'] == 'calibration' else 'history_training'
    return [dict(operation_id=op['operation_id'], usage=usage, trial_id=tid)
            for usage, ids in [(source_usage, source), (history_usage, history), ('evaluation', evaluation)]
            for tid in ids]


def validate_operation(op, predictions, members, split, cfg):
    source, history, evaluation = membership(op, split, cfg)
    wanted_members = {(r['usage'], r['trial_id']) for r in membership_rows(op, source, history, evaluation)}
    observed_members = {(r['usage'], r['trial_id']) for r in members}
    if len(members) != len(observed_members) or observed_members != wanted_members:
        raise ValueError('operation membership differs: ' + op['operation_id'])
    expected = {(draw, '' if prior is None else str(prior), tid)
                for draw, prior in logical_cells(op, cfg) for tid in evaluation}
    observed, broadcast = set(), {}
    for r in predictions:
        if r['operation_id'] != op['operation_id'] or r['arm'] != op['arm'] or r['model'] != op['model'] or int(r['dose_per_class']) != op['dose']:
            raise ValueError('prediction operation identity differs')
        tid = r['trial_id']
        key = (int(r['draw']), str(r['prior_weight']), tid)
        if key not in expected or key in observed:
            raise ValueError('duplicate/unexpected prediction cell')
        observed.add(key)
        truth, prediction = int(r['y_true']), int(r['y_pred'])
        probability = np.asarray([float(r['p_class0_right']), float(r['p_class1_left'])])
        if (int(r['subject_id']) != int(split[tid]['subject_id']) or truth != int(split[tid]['label'])
                or prediction not in (0, 1) or not np.isfinite(probability).all()
                or (probability < 0).any() or (probability > 1).any()
                or abs(probability.sum() - 1) > 1e-10 or prediction != int(probability.argmax())):
            raise ValueError('prediction label/probability contract differs')
        value = (truth, prediction, *probability.tolist())
        if op['draw'] is None and broadcast.setdefault(tid, value) != value:
            raise ValueError('shared source prediction changed across logical cells')
    if observed != expected or any(r['operation_id'] != op['operation_id'] for r in members):
        raise ValueError('missing predictions or wrong membership operation')


def validate_grid(predictions, members, timings, split_rows, cfg):
    split = validate_split(split_rows, cfg)
    expected = {o['operation_id']: o for o in operations(cfg)}
    by_prediction, by_member = defaultdict(list), defaultdict(list)
    for row in predictions:
        by_prediction[row['operation_id']].append(row)
    for row in members:
        by_member[row['operation_id']].append(row)
    costs = {r['operation_id']: r for r in timings}
    if (set(by_prediction) != set(expected) or set(by_member) != set(expected)
            or set(costs) != set(expected) or len(timings) != len(expected)
            or len(predictions) != cfg['expected_prediction_rows']):
        raise ValueError('grid operation/row set differs')
    for oid, op in expected.items():
        validate_operation(op, by_prediction[oid], by_member[oid], split, cfg)
        cost = costs[oid]
        optional = lambda v: '' if v is None else str(v)
        if (cost['arm'] != op['arm'] or cost['model'] != op['model'] or int(cost['dose_per_class']) != op['dose']
                or optional(cost['draw']) != optional(op['draw'])
                or optional(cost['target_subject']) != optional(op['target'])
                or optional(cost['prior_weight']) != optional(op['prior'])
                or cost['operation_kind'] != ('target_update' if op['arm']=='calibration' and op['dose'] else 'fit')):
            raise ValueError('timing operation identity differs')
        source, history, evaluation = membership(op, split, cfg)
        if (int(cost['source_trials']) != len(source) or int(cost['history_trials']) != len(history)
                or int(cost['evaluation_trials']) != len(evaluation)):
            raise ValueError('timing membership counts differ')
        for field in ('fit_cpu_seconds', 'predict_cpu_seconds', 'fit_wall_seconds', 'predict_wall_seconds'):
            value = float(cost[field])
            if not np.isfinite(value) or value < 0:
                raise ValueError('invalid operation timing')
    return dict(operations=len(expected), prediction_rows=len(predictions), membership_rows=len(members),
                prepared_trials=len(split), confirmation_accessed=False,
                class_order=[0, 1], shared_zero_history_predictions_checked=True)
