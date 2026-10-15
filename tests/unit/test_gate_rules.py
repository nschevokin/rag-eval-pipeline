import pytest

from rag.evaluate.gate import (
    GateConfig,
    floors_from_baseline,
    judge,
    load_gate_config,
    render_gate_config,
)

FLOORS = {"recall_at_5": 0.6, "recall_at_10": 0.7, "mrr": 0.4, "ndcg_at_10": 0.45}
CONFIG = GateConfig(floors=FLOORS, regression_tolerance=0.02)
BASE = {"recall_at_5": 0.70, "recall_at_10": 0.80, "mrr": 0.50, "ndcg_at_10": 0.55}


def test_passes_above_floors_without_baseline():
    assert judge(BASE, CONFIG, None) == (True, [])


def test_fails_below_an_absolute_floor():
    passed, reasons = judge({**BASE, "mrr": 0.39}, CONFIG, None)
    assert not passed
    assert reasons == ["mrr 0.3900 is below the floor 0.4000"]


def test_regression_within_tolerance_passes_and_beyond_fails():
    assert judge({**BASE, "recall_at_5": 0.68}, CONFIG, BASE)[0]
    passed, reasons = judge({**BASE, "recall_at_5": 0.67}, CONFIG, BASE)
    assert not passed
    assert "regresses" in reasons[0]


def test_floors_come_from_baseline_minus_margin():
    assert floors_from_baseline(BASE, 0.03) == {
        "recall_at_5": 0.67,
        "recall_at_10": 0.77,
        "mrr": 0.47,
        "ndcg_at_10": 0.52,
    }


def test_gate_file_round_trips(tmp_path):
    path = tmp_path / "gate.toml"
    path.write_text(render_gate_config(CONFIG, baseline_eval_id=3))
    assert load_gate_config(path) == CONFIG


def test_committed_gate_file_starts_without_floors():
    from conftest import ROOT

    config = load_gate_config(ROOT / "eval" / "gate.toml")
    assert config.regression_tolerance == pytest.approx(0.02)
