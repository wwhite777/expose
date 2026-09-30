import importlib.util
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def load_plotter():
    spec = importlib.util.spec_from_file_location("plot_review3", REPO / "scripts/plot_review3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(target, source="1800", draws="10"):
    return {"target": str(target), "source_size": source, "draw_count": draws,
            "contrast": "current_minus_historical_tail", "h_total": "60"}


def test_current_alignment_uses_only_complete_source_participants():
    module = load_plotter()
    full = [row(target) for target in (3, 5, 7)]
    lower_source = [row(target, source="100") for target in (3, 5, 7)]
    selected = module.full_source_alignment_rows(
        full + lower_source, "openbmi8", "current_minus_historical_tail", "60", {3, 5, 7})
    assert len(selected) == 3
    assert {item["source_size"] for item in selected} == {"1800"}


def test_current_alignment_rejects_duplicate_or_non_ten_draw_complete_source_rows():
    module = load_plotter()
    with pytest.raises(ValueError, match="incomplete or duplicate"):
        module.full_source_alignment_rows([row(1), row(1)], "openbmi8", "current_minus_historical_tail", "60", {1})
    with pytest.raises(ValueError, match="ten-draw"):
        module.full_source_alignment_rows([row(1, draws="3")], "openbmi8", "current_minus_historical_tail", "60", {1})
