"""Tests pinned to the worked example of sealed_driver_criteria.tex (v2).

Driver S (18"): Fs 20 Hz, Mms 400 g, Re 3.5, Bl 18, Qms 4, Sd 0.115 m^2,
Xmax 20 mm, Le 4 mH, Pmax 600 W  ->  Qes 0.543, Qts 0.478, Vas 297 L,
eta0 0.43 %, Vd 2.30 L, SPL_T 116.3 dB, f_x 32.5 Hz, Fs/Qts 42 Hz.
"""

import math

import pytest

from audioshape import physics
from audioshape.driver import BoxedDriver, Driver
from audioshape.scenario import Scenario


@pytest.fixture
def driver_s() -> Driver:
    return Driver(
        manufacturer="Example", model="S18", size_in=18,
        fs=20.0, qes=0.543, qms=4.0, re=3.5, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0, bl=18.0, le=4e-3)


@pytest.fixture
def scenario() -> Scenario:
    return Scenario(v_room=60.0, l_max=6.0, r_listen=1.0,
                    target_spl=110.0, distortion_budget=0.03,
                    qtc=0.55, f_low=15.0, f_split=80.0)


def test_derived_parameters(driver_s):
    assert driver_s.qts == pytest.approx(0.478, abs=0.002)
    assert driver_s.vd == pytest.approx(2.30e-3, rel=1e-3)
    assert driver_s.eta0 == pytest.approx(0.00428, rel=0.01)
    assert driver_s.corner_rate == pytest.approx(41.8, rel=0.01)
    assert driver_s.f_le == pytest.approx(139.0, rel=0.01)


def test_box_alignment(driver_s):
    b055 = BoxedDriver(driver_s, qtc=0.55)
    assert b055.vb == pytest.approx(0.918, rel=0.01)
    assert b055.fc == pytest.approx(23.0, rel=0.01)
    b071 = BoxedDriver(driver_s, qtc=0.707)
    assert b071.vb == pytest.approx(0.250, rel=0.03)
    assert b071.fc == pytest.approx(29.6, rel=0.01)


def test_qtc_below_qts_rejected(driver_s):
    with pytest.raises(ValueError):
        BoxedDriver(driver_s, qtc=0.40)


def test_spl_excursion_ceiling_matches_108p5_constant():
    # SPL_L(f) = 108.5 + 20 log10(f^2 Vd) at 1 m, 2pi (eq:SPLL)
    vd = 2.30e-3
    f = 30.0
    expected = 108.5 + 20 * math.log10(f * f * vd)
    assert physics.spl_excursion_ceiling(f, vd, r=1.0) == pytest.approx(
        expected, abs=0.05)


def test_thermal_ceiling(driver_s):
    spl = physics.spl_thermal_ceiling(driver_s.eta0, driver_s.p_max, r=1.0)
    assert spl == pytest.approx(116.3, abs=0.1)


def test_regime_boundary(driver_s):
    fx = physics.regime_boundary_fx(driver_s.fs, driver_s.p_max,
                                    driver_s.qes, driver_s.mms, driver_s.xmax)
    assert fx == pytest.approx(32.5, rel=0.01)
    assert physics.burst_boundary_fx(fx) == pytest.approx(2 * fx, rel=0.01)


def test_pressure_zone(scenario):
    assert scenario.f_pz == pytest.approx(28.6, abs=0.1)
    # PZ branch of the demand: sqrt(2) p_t V_room / (rho0 c^2) = 3.79 L
    p_t = physics.pressure_from_spl(110.0)
    v_pz = math.sqrt(2) * p_t * 60.0 / (physics.RHO0 * physics.C_AIR**2)
    assert v_pz == pytest.approx(3.80e-3, rel=0.01)
    # demand is flat below f_pz and falls ~1/f^2 above
    assert scenario.demand_volume(10.0) == scenario.demand_volume(20.0)
    ratio = scenario.demand_volume(40.0) / scenario.demand_volume(80.0)
    assert ratio == pytest.approx(4.0, rel=1e-6)


def test_eq_tax_passband_power(scenario, driver_s):
    # 110 dB @ 1 m half-space needs ~0.61 W acoustic -> ~143 W at eta0
    w_ac = physics.acoustic_power_halfspace(scenario.target_pressure, 1.0)
    assert w_ac == pytest.approx(0.61, abs=0.01)
    assert w_ac / driver_s.eta0 == pytest.approx(142.0, rel=0.02)


def test_distortion_laws():
    # anchored: D(1) = 0.10 at Xmax (IEC 62458)
    assert physics.harmonic_distortion(1.0) == pytest.approx(0.10)
    xi = physics.utilization_for_distortion(0.03)
    assert physics.harmonic_distortion(xi) == pytest.approx(0.03, rel=1e-9)
    assert 0.3 < xi < 0.55  # paper's bracket for D* = 3 %
    # Doppler: X1 = 18.9 mm carrying 80 Hz -> 1.4 % (worked example)
    assert physics.doppler_im(80.0, 18.9e-3) == pytest.approx(0.0138, abs=0.001)


def test_box_hd2_worked_example():
    # two S in 473 L each at Qtc 0.61, V_dem 2.18 L/unit -> ~0.11 %
    hd2 = physics.box_hd2(2.18e-3, 0.473, 0.478, 0.61)
    assert hd2 == pytest.approx(0.00107, rel=0.15)


def test_corner_rate_rule(scenario, driver_s):
    # S: Fs/Qts = 42 Hz <= f_pz/Qtc = 52 Hz -> compliant (eq:Fsrule)
    assert scenario.max_corner_rate == pytest.approx(52.0, abs=0.5)
    assert driver_s.corner_rate < scenario.max_corner_rate


def test_motor_bound_worked_example(driver_s):
    # Sub class: driver S's own EBP with an illustrative overhang u=3,
    # B=1.0 T (Sec. "materials", "Numerical content") -> 331 Hz / 125 g.
    ebp_u2 = physics.motor_bound_ebp_u2(driver_s.ebp, u=3.0)
    assert ebp_u2 == pytest.approx(331.0, abs=1.0)
    beta = physics.implied_coil_mass_fraction(driver_s.ebp, u=3.0, b_field=1.0)
    assert beta * driver_s.mms * 1e3 == pytest.approx(125.0, abs=1.0)  # g
