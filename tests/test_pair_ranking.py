"""Tests for role evaluation, Pareto ranking, and recipe loading."""

from pathlib import Path

import pytest

from audioshape.driver import Driver
from audioshape.ranking import (
    PairEvaluation,
    RiskVector,
    evaluate,
    pair_rank,
)
from audioshape.recipe import load_recipe
from audioshape.scenario import Scenario

RECIPE_PATH = Path(__file__).resolve().parents[1] / "examples" / "example_recipe.toml"


@pytest.fixture
def driver_s() -> Driver:
    return Driver(
        manufacturer="Example", model="S18", size_in=18,
        fs=20.0, qes=0.543, qms=4.0, re=3.5, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0,
        bl=18.0, le=4e-3,
    )


@pytest.fixture
def driver_m() -> Driver:
    return Driver(
        manufacturer="Example", model="M12", size_in=12,
        fs=48.0, qes=0.353, qms=6.0, re=5.2, mms=0.065,
        sd=0.052, xmax=0.008, vas=0.0648, p_max=350.0,
        bl=17.0, le=0.7e-3,
    )


@pytest.fixture
def relaxed_scenario() -> Scenario:
    return Scenario(
        r_listen=1.0,
        sub_target_spl=100.0,
        attack_target_spl=100.0,
        room_model="ideal_pressure_zone",
        leakage_corner_hz=0.0,
        amplifier_voltage_rms=1000.0,
        amplifier_current_rms=1000.0,
        amplifier_power_continuous=1e6,
        amplifier_power_burst=1e6,
        max_role_box_volume_m3=3.0,
    )


def test_xmax_is_hard_limit_and_80_percent_is_only_margin(driver_s):
    scenario = Scenario(
        amplifier_voltage_rms=1000.0,
        amplifier_current_rms=1000.0,
        amplifier_power_continuous=1e6,
        amplifier_power_burst=1e6,
        max_role_box_volume_m3=3.0,
    )
    evaluation = evaluate(
        driver_s, scenario, n_units=3,
        band_low=scenario.f_low, band_high=scenario.f_split,
        doppler_ref=scenario.f_split, role="sub",
    )
    assert evaluation.feasible
    assert evaluation.xi_x < 1.0
    assert scenario.preferred_excursion < evaluation.xi_x_transient < 1.0
    assert not evaluation.is_preferred_excursion
    assert evaluation.n_units_preferred == 4


def test_complete_band_extrema_and_amplifier_gate(driver_s, relaxed_scenario):
    evaluation = evaluate(
        driver_s, relaxed_scenario, n_units=2,
        band_low=relaxed_scenario.f_low,
        band_high=relaxed_scenario.f_split,
        doppler_ref=relaxed_scenario.f_split,
        role="sub",
    )
    assert relaxed_scenario.f_low <= evaluation.electrical.power_frequency <= (
        relaxed_scenario.f_split
    )
    assert relaxed_scenario.f_low <= evaluation.transient.shape_frequency <= (
        relaxed_scenario.f_split
    )

    voltage_limited = Scenario(
        r_listen=relaxed_scenario.r_listen,
        sub_target_spl=relaxed_scenario.sub_target_spl,
        attack_target_spl=relaxed_scenario.attack_target_spl,
        room_model="ideal_pressure_zone",
        leakage_corner_hz=0.0,
        amplifier_voltage_rms=5.0,
        amplifier_current_rms=1000.0,
        amplifier_power_continuous=1e6,
        amplifier_power_burst=1e6,
    )
    limited = evaluate(
        driver_s, voltage_limited, n_units=2,
        band_low=voltage_limited.f_low,
        band_high=voltage_limited.f_split,
        role="sub",
    )
    assert not limited.feasible
    assert any("voltage" in reason for reason in limited.reasons)


def test_attack_channel_gets_no_stereo_summing_credit(
    driver_m, relaxed_scenario
):
    direct = evaluate(
        driver_m, relaxed_scenario, n_units=1,
        band_low=relaxed_scenario.f_split,
        band_high=relaxed_scenario.f_high,
        doppler_ref=relaxed_scenario.f_high,
        role="attack",
    )
    pair = pair_rank(
        [driver_m], relaxed_scenario,
        sub_units=1, attack_units=1, top_k_each=1,
    )[0]
    assert pair.attack.xi_x == pytest.approx(direct.xi_x)
    assert pair.attack.scenario.attack_target_spl == 100.0


def test_default_doppler_reference_is_role_band_high(
    driver_m, relaxed_scenario
):
    default = evaluate(
        driver_m,
        relaxed_scenario,
        n_units=1,
        band_low=relaxed_scenario.f_split,
        band_high=relaxed_scenario.f_high,
        role="attack",
    )
    explicit = evaluate(
        driver_m,
        relaxed_scenario,
        n_units=1,
        band_low=relaxed_scenario.f_split,
        band_high=relaxed_scenario.f_high,
        doppler_ref=relaxed_scenario.f_high,
        role="attack",
    )
    assert default.doppler_im == pytest.approx(explicit.doppler_im)


def test_qtc_ceiling_and_large_box_fallback(driver_s, relaxed_scenario):
    high_corner_rate = Driver(
        manufacturer="Example", model="HighFsQts", size_in=18,
        fs=20.0, qes=0.2083, qms=5.0, re=4.0, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0,
    )
    evaluation = evaluate(
        high_corner_rate, relaxed_scenario, n_units=2, role="sub"
    )
    assert evaluation.boxed.qtc < relaxed_scenario.qtc
    assert evaluation.boxed.fc <= relaxed_scenario.f_pz + 1e-6

    unreachable = Driver(
        manufacturer="Example", model="FsAboveTarget", size_in=18,
        fs=35.0, qes=0.4083, qms=5.0, re=4.0, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0,
    )
    fallback = evaluate(
        unreachable, relaxed_scenario, n_units=2, role="sub"
    )
    assert any("alignment limited" in note for note in fallback.notes)
    assert fallback.boxed.fc > relaxed_scenario.f_pz


def test_large_room_respects_total_box_volume_cap():
    driver = Driver(
        manufacturer="Dayton Audio", model="UMII18-22", size_in=18,
        fs=22.0, qes=0.67, qms=23.06, re=0.124, mms=0.2482,
        sd=0.192, xmax=0.028, vas=1.184, p_max=1200.0,
    )
    scenario = Scenario(
        v_room=150.0,
        l_max=10.0,
        amplifier_voltage_rms=1e4,
        amplifier_current_rms=1e4,
        amplifier_power_continuous=1e9,
        amplifier_power_burst=1e9,
    )
    one = evaluate(driver, scenario, n_units=1, role="sub")
    four = evaluate(driver, scenario, n_units=4, role="sub")
    assert any("alignment limited" in note for note in one.notes)
    assert not one.feasible
    assert not four.feasible
    assert four.xi_x == pytest.approx(one.xi_x / 4.0)
    assert four.risk.box_volume_m3 == pytest.approx(1.0)
    assert any("Qtc" in reason for reason in four.reasons)


def test_risk_vector_pareto_dominance():
    better = RiskVector(
        0.5, 0.6, 0.01, 0.001, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2
    )
    worse = RiskVector(
        0.6, 0.7, 0.02, 0.002, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.3
    )
    tradeoff = RiskVector(
        0.4, 0.55, 0.01, 0.001, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3
    )
    assert better.dominates(worse, "sub")
    assert not worse.dominates(better, "sub")
    assert not better.dominates(tradeoff, "sub")


def test_pair_rank_keeps_role_keys_separate(
    driver_s, driver_m, relaxed_scenario
):
    pairs = pair_rank(
        [driver_s, driver_m],
        relaxed_scenario,
        sub_units=2,
        attack_units=1,
        top_k_each=2,
    )
    assert pairs
    best = pairs[0]
    assert isinstance(best, PairEvaluation)
    assert best.sub.driver.model == "S18"
    assert best.attack.driver.model == "M12"
    assert not hasattr(best, "total_distortion")
    assert [pair.sort_key() for pair in pairs] == sorted(
        pair.sort_key() for pair in pairs
    )


def test_load_recipe():
    recipe = load_recipe(RECIPE_PATH)
    assert recipe.db.name == "VituixCAD_driver_db.txt"
    assert recipe.db.is_file()
    assert recipe.scenario.sub_target_spl == 110.0
    assert recipe.scenario.attack_target_spl == 105.0
    assert recipe.scenario.room_model == "leaky_pressure_zone"
    assert recipe.scenario.leakage_corner_hz == 10.0
    assert recipe.scenario.preferred_excursion == 0.8
    assert recipe.scenario.amplifier_voltage_rms == 90.0
    assert recipe.scenario.max_box_vas_ratio == 10.0
    assert recipe.scenario.max_role_box_volume_m3 == 1.0
    assert recipe.sub_units == 4
    assert recipe.attack_units == 1
    assert recipe.require_even_sub_units
    assert recipe.sub_size_min == 15
    assert recipe.sub_size_max == 18
    assert recipe.attack_size_min == 12
    assert recipe.attack_size_max == 12


def test_load_recipe_rejects_unknown_scenario_key(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text('db = "x.txt"\n[scenario]\nbogus_key = 1\n')
    with pytest.raises(ValueError, match="unknown"):
        load_recipe(bad)


def test_load_recipe_rejects_odd_force_cancelling_manifold(tmp_path):
    bad = tmp_path / "odd.toml"
    bad.write_text(
        'db = "x.txt"\n[pair]\nsub_units = 3\n'
        'require_even_sub_units = true\n'
    )
    with pytest.raises(ValueError, match="even unit count"):
        load_recipe(bad)
