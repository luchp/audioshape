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
    dist = [e.total_distortion for e in feasible]
    assert dist == sorted(dist)


def test_more_units_reduce_distortion(db):
    sc = Scenario()
    d = next(dr for dr in db.drivers if dr.model == "TD15H")
    e1 = evaluate(d, sc, n_units=1)
    e2 = evaluate(d, sc, n_units=2)
    assert e2.xi_x == pytest.approx(e1.xi_x / 2)
    assert e2.total_distortion < e1.total_distortion


def test_infeasible_high_qts_driver_flagged(db):
    sc = Scenario(qtc=0.55)
    high_qts = next(d for d in db.drivers if d.qts >= 0.55)
    ev = evaluate(high_qts, sc)
    assert not ev.feasible
    # Unreachable alignment is no longer a hard rejection by itself (see
    # test_qtc_ceiling_rescues_overshooting_driver / MAX_VB_VAS_RATIO
    # fallback): this driver is genuinely infeasible on excursion/thermal
    # clipping even in the large-box + EQ fallback.
    assert any("clip" in r for r in ev.reasons)
    assert any("Fc can't reach" in n for n in ev.notes)
