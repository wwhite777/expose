import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from expose.development import load_config, logical_cells, membership, operations
from expose.development_validation import completed_stage, membership_rows, validate_operation
from expose.provenance import file_sha256
from expose.runtime import atomic_write_json
from expose.scout import trial_id


def fixture_operation():
    cfg = load_config(Path(__file__).resolve().parents[1])
    split = {}
    for role, subjects in [('source', cfg['source_subjects']), ('development', cfg['development_subjects'])]:
        for subject in subjects:
            for session in ([1] if role=='source' else [1, 2]):
                for i in range(100):
                    tid = trial_id(subject, session, i)
                    split[tid] = dict(trial_id=tid, role=role, subject_id=subject, session=session, label=i % 2)
    op = next(o for o in operations(cfg) if o['arm']=='calibration' and o['dose']==0)
    source, history, query = membership(op, split, cfg)
    members = membership_rows(op, source, history, query)
    rows = [dict(operation_id=op['operation_id'], arm=op['arm'], model=op['model'], draw=d,
                 dose_per_class=0, prior_weight=str(w), subject_id=split[t]['subject_id'], trial_id=t,
                 y_true=split[t]['label'], y_pred=0, p_class0_right=.6, p_class1_left=.4)
            for d, w in logical_cells(op, cfg) for t in query]
    return cfg, split, op, rows, members


def test_valid_shared_prediction_and_membership():
    cfg, split, op, rows, members = fixture_operation()
    validate_operation(op, rows, members, split, cfg)


@pytest.mark.parametrize('defect', ['missing_cell', 'duplicate_cell', 'wrong_label', 'swapped_probability',
                                   'nan', 'shared_drift', 'training_overlap', 'wrong_member_operation',
                                   'wrong_subject', 'unexpected_prior'])
def test_rejects_prediction_and_membership_corruption(defect):
    cfg, split, op, rows, members = fixture_operation()
    if defect=='missing_cell':
        rows.pop()
    elif defect=='duplicate_cell':
        rows[-1] = copy.deepcopy(rows[0])
    elif defect=='wrong_label':
        rows[0]['y_true'] = 1-rows[0]['y_true']
    elif defect=='swapped_probability':
        rows[0]['p_class0_right'], rows[0]['p_class1_left'] = .4, .6
    elif defect=='nan':
        rows[0]['p_class0_right'] = float('nan')
    elif defect=='shared_drift':
        rows[0]['p_class0_right'], rows[0]['p_class1_left'] = .7, .3
    elif defect=='training_overlap':
        members[0]['trial_id'] = rows[0]['trial_id']
    elif defect=='wrong_member_operation':
        members[0]['operation_id'] = 'wrong'
    elif defect=='wrong_subject':
        rows[0]['subject_id'] = 54
    elif defect=='unexpected_prior':
        rows[0]['prior_weight'] = '500'
    with pytest.raises(ValueError):
        validate_operation(op, rows, members, split, cfg)


def completed_fixture(tmp_path):
    (tmp_path/'input.txt').write_text('input')
    out = tmp_path/'example_r001'
    out.mkdir()
    atomic_write_json(out/'INPUT_MANIFEST.json', {'input.txt': file_sha256(tmp_path/'input.txt')})
    atomic_write_json(out/'receipt.json', dict(run_id=out.name, pid=os.getpid(), status='completed', exit_code=0))
    atomic_write_json(out/'COMPLETED.json', dict(run_id=out.name, receipt_sha256=file_sha256(out/'receipt.json'),
                      outputs={name:file_sha256(out/name) for name in ['receipt.json', 'INPUT_MANIFEST.json']}))
    atomic_write_json(out/'supervisor_receipt.json', dict(status='completed', exit_code=0, child_pid=os.getpid(),
                      run_id=out.name, receipt_sha256=file_sha256(out/'receipt.json'), completion_sha256=file_sha256(out/'COMPLETED.json')))
    return out


def test_consumer_verifies_dependencies_and_required_outputs(tmp_path):
    out = completed_fixture(tmp_path)
    completed_stage(tmp_path, out, ['INPUT_MANIFEST.json'])
    with pytest.raises(ValueError, match='omitted'):
        completed_stage(tmp_path, out, ['missing.csv'])
    (tmp_path/'input.txt').write_text('changed')
    with pytest.raises(ValueError, match='dependency'):
        completed_stage(tmp_path, out, ['INPUT_MANIFEST.json'])


def test_grid_wrapper_preserves_failed_receipt_and_refuses_overwrite(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('day3_runner_fixture', root/'scripts/run_day3.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # Keep the test process's own resource/signal state; production calls are exercised elsewhere.
    monkeypatch.setattr(runner.resource, 'setrlimit', lambda *args: None)
    monkeypatch.setattr(runner.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(runner.signal, 'alarm', lambda *args: None)
    def invalid_producer(*args):
        raise ValueError('fixture incomplete producer')
    monkeypatch.setattr(runner, 'completed_stage', invalid_producer)
    monkeypatch.setattr(runner.sys, 'argv', ['run_day3.py', '--run-dir', str(tmp_path/'grid_fixture')])
    assert runner.main() == 1
    receipt = tmp_path/'grid_fixture/receipt.json'
    before = receipt.read_bytes()
    assert json.loads(before)['status']=='failed'
    assert not (receipt.parent/'COMPLETED.json').exists()
    with pytest.raises(FileExistsError):
        runner.main()
    assert receipt.read_bytes()==before
