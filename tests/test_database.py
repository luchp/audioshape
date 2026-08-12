"""Optional integration tests against the separately obtained local DB."""

from pathlib import Path

import pytest

from audioshape.database import parse_database
from audioshape.ranking import evaluate, rank
from audioshape.scenario import Scenario

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "VituixCAD_driver_db.txt"
pytestmark = pytest.mark.skipif(
    not DB_PATH.is_file(),
    reason="local VituixCAD database is not distributed with the repository",
)


@pytest.fixture(scope="module")
def db():
    return parse_database(DB_PATH)


def test_parse_database(db):
    assert len(db.drivers) > 500
    td15h = [d for d in db.drivers if d.model == "TD15H"]
    assert len(td15h) == 1
    d = td15h[0]
    assert d.manufacturer == "AE Speakers"
    assert d.fs == 21.0
    assert d.sd == pytest.approx(855e-4)
    assert d.xmax == pytest.approx(14e-3)
    assert d.vas == pytest.approx(0.467)
    assert d.mms == pytest.approx(0.129)
    # rows with missing required data are reported, not silently dropped
    assert all(":" in reason for reason in db.skipped)


def test_rank_produces_sorted_feasible_first(db):
    sc = Scenario()
    evals = rank(db.drivers, sc, min_size_in=12)
    assert evals, "no 12in+ drivers evaluated"
    feas_flags = [e.feasible for e in evals]
    assert feas_flags == sorted(feas_flags, reverse=True)
    feasible = [e for e in evals if e.feasible]
    keys = [e.sort_key() for e in feasible]
    assert keys == sorted(keys)
    assert all(e.pareto_rank >= 0 for e in feasible)


def test_more_units_reduce_separate_risks(db):
    sc = Scenario()
    d = next(dr for dr in db.drivers if dr.model == "TD15H")
    e1 = evaluate(d, sc, n_units=1)
    e2 = evaluate(d, sc, n_units=2)
    assert e2.xi_x == pytest.approx(e1.xi_x / 2)
    assert e2.xi_x_transient < e1.xi_x_transient
    assert e2.doppler_im == pytest.approx(e1.doppler_im / 2)
    assert not hasattr(e2, "total_distortion")


def test_high_qts_driver_is_not_rejected_for_alignment_alone(db):
    sc = Scenario(alignment_qtc=0.55)
    high_qts = next(d for d in db.drivers if d.qts >= 0.55)
    ev = evaluate(high_qts, sc)
    assert not any("Qtc" in reason for reason in ev.reasons)
    assert any("preferred alignment not reached" in note for note in ev.notes)
