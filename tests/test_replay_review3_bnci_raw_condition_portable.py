import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "replay_review3_bnci_raw_condition_portable",
        ROOT / "scripts/replay_review3_bnci_raw_condition_portable.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regular_file_resolves_and_records_exact_identity(tmp_path):
    module = load_module()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"portable-fixture")
    digest = module.sha256(payload)

    resolved = module.regular_file(payload, "fixture", digest)
    assert resolved == payload.resolve()
    assert module.input_identity(payload) == {
        "resolved_path": str(payload.resolve()),
        "bytes": len(b"portable-fixture"),
        "sha256": digest,
    }


def test_regular_file_rejects_missing_wrong_hash_and_symlink(tmp_path):
    module = load_module()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"portable-fixture")

    with pytest.raises(ValueError, match="absent, not regular, or a symlink"):
        module.regular_file(tmp_path / "missing.bin", "fixture")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        module.regular_file(payload, "fixture", "0" * 64)

    link = tmp_path / "payload-link.bin"
    link.symlink_to(payload)
    with pytest.raises(ValueError, match="symlink"):
        module.regular_file(link, "fixture")


def test_source_validation_rejects_tampered_package_initializer(tmp_path):
    module = load_module()
    package = tmp_path / "src/expose"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("tampered\n", encoding="utf-8")
    (package / "bnci2014.py").write_bytes((ROOT / "src/expose/bnci2014.py").read_bytes())

    with pytest.raises(ValueError, match="package initializer SHA-256 differs"):
        module.validate_source_files(tmp_path / "src")


def test_resource_setup_failure_finalizes_failed_receipt(tmp_path, monkeypatch):
    module = load_module()
    output = tmp_path / "failed-output"

    def reject_limit(*args, **kwargs):
        raise OSError("fixture lower host limit")

    monkeypatch.setattr(module.resource, "setrlimit", reject_limit)
    with pytest.raises(OSError, match="lower host limit"):
        module.run(
            tmp_path / "unused-plan",
            tmp_path / "unused-grid",
            tmp_path / "unused-raw",
            tmp_path / "unused-manifest.tsv",
            tmp_path / "unused-src",
            output,
        )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert receipt["error"] == {
        "type": "OSError",
        "message": "fixture lower host limit",
    }
    assert receipt["finished_utc"]
    assert receipt["cpu_seconds"] >= 0
    assert receipt["wall_seconds"] >= 0
    assert not (output / "COMPLETED.json").exists()


def test_portable_script_has_no_repository_data_defaults():
    module = load_module()
    source = (ROOT / "scripts/replay_review3_bnci_raw_condition_portable.py").read_text(
        encoding="utf-8"
    )
    assert module.OPERATION_ID == "bnci_main__plain_ts__d0__t1__kall__Sall__h60"
    assert module.PROBABILITY_TOLERANCE == 1e-10
    assert "ROOT =" not in source
    assert "PLAN_DIR =" not in source
    assert "GRID_DIR =" not in source
    assert "DOWNLOAD_RECEIPT =" not in source
    for flag in (
        '"--plan-dir"',
        '"--grid-dir"',
        '"--raw-dir"',
        '"--download-manifest"',
        '"--source-root"',
        '"--out-dir"',
    ):
        assert flag in source
