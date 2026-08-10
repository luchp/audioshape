"""Tests for role-restricted evaluation, pair ranking, and recipe loading."""

from pathlib import Path

import pytest

from audioshape.driver import Driver
from audioshape.ranking import PairEvaluation, evaluate, pair_rank
from audioshape.recipe import load_recipe
from audioshape.scenario import Scenario

RECIPE_PATH = Path(__file__).resolve().parents[1] / "examples" / "example_recipe.toml"


@pytest.fixture
def driver_s() -> Driver:
    """Sub-suited driver (paper's S): huge Vd, low Fs, low f_L."""
    return Driver(
        manufacturer="Example", model="S18", size_in=18,
        fs=20.0, qes=0.543, qms=4.0, re=3.5, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0, bl=18.0, le=4e-3)


@pytest.fixture
def driver_m() -> Driver:
    """Attack-suited driver (paper's M): small Vd, high Fs, high f_L."""
    return Driver(
        manufacturer="Example", model="M12", size_in=12,
        fs=48.0, qes=0.353, qms=6.0, re=5.2, mms=0.065,
        sd=0.052, xmax=0.008, vas=0.0648, p_max=350.0, bl=17.0, le=0.7e-3)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(v_room=60.0, l_max=6.0, r_listen=1.0,
                    sub_target_spl=110.0, attack_target_spl=110.0, distortion_budget=0.03,
                    qtc=0.55, f_low=15.0, f_split=80.0, f_high=250.0)


def test_sub_role_uses_own_band(driver_s, scenario):
    ev_sub = evaluate(driver_s, scenario, n_units=2,
                      band_low=scenario.f_low, band_high=scenario.f_split,
                      doppler_ref=scenario.f_split)
    ev_full = evaluate(driver_s, scenario, n_units=2)  # default: full range
    # same band -> identical excursion demand and Doppler reference
    assert ev_sub.xi_x == pytest.approx(ev_full.xi_x)
    assert ev_sub.doppler_im == pytest.approx(ev_full.doppler_im)


def test_attack_role_ignores_subs_own_excursion(driver_m, scenario):
    """Attack driver's Doppler must come from its OWN excursion at its own
    band top (f_high), not the sub's -- physically separate sources."""
    ev = evaluate(driver_m, scenario, n_units=1,
                 band_low=scenario.f_split, band_high=scenario.f_high,
                 doppler_ref=scenario.f_high, role="attack")
    # its own excursion utilization, not the sub's huge one
    assert ev.xi_x < 3.0  # sane, driven by its own small Vd vs band-top demand
    # Doppler uses f_high (250 Hz) as reference, matching band_high
    from audioshape import physics
    x1 = min(ev.xi_x, 1.0) * driver_m.xmax
    expected = physics.doppler_im(scenario.f_high, x1)
    assert ev.doppler_im == pytest.approx(expected)


def test_corner_rate_gate_only_binds_low_band(driver_m, scenario):
    """A driver whose corner rate would fail the room-closure gate is not
    flagged for it when scored only in the attack band (band_low > f_pz)."""
    ev_attack = evaluate(driver_m, scenario, n_units=1,
                         band_low=scenario.f_split, band_high=scenario.f_high,
                         doppler_ref=scenario.f_high, role="attack")
    assert not any("f_pz" in r for r in ev_attack.reasons)


def test_qtc_ceiling_rescues_overshooting_driver(scenario):
    """A driver whose corner rate would overshoot f_pz at the fixed ceiling
    Qtc is not simply rejected: a lower Qtc (bigger box) that still clears
    Qts is used instead, same as the paper's own undershoot-is-free
    argument after eq:Fsrule."""
    # Fs/Qts = 100 Hz > f_pz/Qtc(0.55) = 28.6/0.55 = 52 Hz at the ceiling,
    # so the old fixed-ceiling gate rejected this driver outright even
    # though Fs=20 Hz itself is well below f_pz -- a bigger box rescues it.
    high_corner_rate = Driver(
        manufacturer="Example", model="HighFsQts", size_in=18,
        fs=20.0, qes=0.2083, qms=5.0, re=4.0, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0)
    assert high_corner_rate.corner_rate == pytest.approx(100.0, rel=0.02)

    ev = evaluate(high_corner_rate, scenario, n_units=2, role="sub")
    assert ev.feasible
    # actual alignment undershoots the ceiling to keep Fc at/below f_pz
    assert ev.boxed.qtc < scenario.qtc
    assert ev.boxed.fc <= scenario.f_pz + 1e-6

    # a still-higher corner rate (Fs alone already past f_pz) no longer
    # hard-rejects: it falls back to a large practical box (capped at
    # MAX_VB_VAS_RATIO * Vas) and leans on EQ to fill the remaining gap,
    # with feasibility now decided by the real excursion/thermal budgets.
    unreachable = Driver(
        manufacturer="Example", model="FsAboveTarget", size_in=18,
        fs=35.0, qes=0.4083, qms=5.0, re=4.0, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0)
    ev_bad = evaluate(unreachable, scenario, n_units=2, role="sub")
    assert any("Fc can't reach" in n for n in ev_bad.notes)
    assert ev_bad.boxed.fc > scenario.f_pz  # EQ has to fill the remaining gap


def test_large_room_unreachable_corner_scales_to_feasible_with_more_units():
    """Real-world case: a large room (f_pz well below any commercial driver's
    Fs/Qts corner rate) makes ideal Fc-at-f_pz alignment unreachable by any
    finite box for a driver like the Dayton UMII18-22 (Fs=22, Qts=0.53).
    This must not hard-reject the driver -- MAX_VB_VAS_RATIO's large-box +
    EQ fallback is used, and feasibility is decided by real excursion/
    thermal headroom, which more units can supply."""
    umii18 = Driver(
        manufacturer="Dayton Audio", model="UMII18-22", size_in=18,
        fs=22.0, qes=0.67, qms=23.06, re=0.124, mms=0.2482,
        sd=0.192, xmax=0.028, vas=1.184, p_max=1200.0)
    sc = Scenario(v_room=150.0, l_max=10.0)
    assert umii18.qts >= sc.f_pz / umii18.corner_rate  # corner unreachable

    ev_1 = evaluate(umii18, sc, n_units=1, role="sub")
    assert any("Fc can't reach" in n for n in ev_1.notes)
    assert not ev_1.feasible  # too little headroom with just one unit
    assert any("clip" in r for r in ev_1.reasons)  # real excursion/thermal reason

    ev_8 = evaluate(umii18, sc, n_units=8, role="sub")
    assert any("Fc can't reach" in n for n in ev_8.notes)
    assert ev_8.feasible  # enough array headroom makes it buildable after all


def test_pair_rank_combines_independent_rankings(driver_s, driver_m, scenario):
    pairs = pair_rank([driver_s, driver_m], scenario,
                      sub_units=2, attack_units=1, top_k_each=5)
    assert pairs, "expected at least one pair"
    best = pairs[0]
    assert isinstance(best, PairEvaluation)
    # S (huge Vd, low Fs) should win the sub role; M (small, high fs) the attack role
    assert best.sub.driver.model == "S18"
    assert best.attack.driver.model == "M12"
    assert best.total_distortion == pytest.approx(
        best.sub.total_distortion + best.attack.total_distortion)
    # sorted by (feasible-first, combined distortion)
    keys = [p.sort_key() for p in pairs]
    assert keys == sorted(keys)


def test_load_recipe():
    recipe = load_recipe(RECIPE_PATH)
    assert recipe.db.name == "VituixCAD_driver_db.txt"
    assert recipe.db.is_file()
    assert recipe.scenario.sub_target_spl == 115.0
    assert recipe.scenario.attack_target_spl == 105.0
    assert recipe.scenario.qtc == 0.55
    assert recipe.sub_units == 2
    assert recipe.attack_units == 1
    assert recipe.sub_size_min == 15
    assert recipe.attack_size_max == 10


def test_load_recipe_rejects_unknown_scenario_key(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('db = "x.txt"\n[scenario]\nbogus_key = 1\n')
    with pytest.raises(ValueError, match="unknown"):
        load_recipe(bad)
