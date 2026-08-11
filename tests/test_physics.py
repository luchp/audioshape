"""Unit tests for the sealed-box and room-demand models."""

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
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0,
        bl=18.0, le=4e-3,
    )


def test_derived_parameters(driver_s):
    assert driver_s.qts == pytest.approx(0.478, abs=0.002)
    assert driver_s.vd == pytest.approx(2.30e-3, rel=1e-3)
    assert driver_s.eta0 == pytest.approx(0.00428, rel=0.01)
    assert driver_s.corner_rate == pytest.approx(41.8, rel=0.01)
    assert driver_s.f_le == pytest.approx(139.0, rel=0.01)


def test_force_factor_can_be_derived_from_ts_data(driver_s):
    derived = physics.force_factor_from_qes(
        driver_s.fs, driver_s.qes, driver_s.mms, driver_s.re
    )
    assert derived == pytest.approx(driver_s.bl, rel=0.02)

    without_bl = Driver(
        manufacturer="Example", model="NoBl", size_in=18,
        fs=driver_s.fs, qes=driver_s.qes, qms=driver_s.qms,
        re=driver_s.re, mms=driver_s.mms, sd=driver_s.sd,
        xmax=driver_s.xmax, vas=driver_s.vas, p_max=driver_s.p_max,
    )
    assert without_bl.effective_bl == pytest.approx(derived)


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
    vd = 2.30e-3
    frequency = 30.0
    expected = 108.5 + 20 * math.log10(frequency * frequency * vd)
    assert physics.spl_excursion_ceiling(
        frequency, vd, r=1.0
    ) == pytest.approx(expected, abs=0.05)


def test_thermal_ceiling(driver_s):
    spl = physics.spl_thermal_ceiling(driver_s.eta0, driver_s.p_max, r=1.0)
    assert spl == pytest.approx(116.3, abs=0.1)


def test_regime_boundary(driver_s):
    fx = physics.regime_boundary_fx(
        driver_s.fs, driver_s.p_max, driver_s.qes,
        driver_s.mms, driver_s.xmax,
    )
    assert fx == pytest.approx(32.5, rel=0.01)


def test_ideal_and_leaky_pressure_zone_models():
    ideal = Scenario(room_model="ideal_pressure_zone", leakage_corner_hz=0.0)
    leaky = Scenario(room_model="leaky_pressure_zone", leakage_corner_hz=10.0)

    assert ideal.f_pz == pytest.approx(28.6, abs=0.1)
    assert ideal.demand_volume(10.0) == ideal.demand_volume(20.0)
    assert leaky.demand_volume(15.0) == pytest.approx(4.57e-3, rel=0.01)
    assert leaky.demand_volume(10.0) > leaky.demand_volume(15.0)
    assert leaky.demand_volume(40.0) == ideal.demand_volume(40.0)

    ratio = ideal.demand_volume(40.0) / ideal.demand_volume(80.0)
    assert ratio == pytest.approx(4.0, rel=1e-6)
    assert physics.room_leakage_displacement_factor(
        10.0, 10.0
    ) == pytest.approx(math.sqrt(2.0))


def test_power_current_and_voltage_are_consistent(driver_s):
    boxed = BoxedDriver(driver_s, qtc=0.61)
    displacement = 10e-3
    frequency = 40.0
    power = physics.power_at_excursion_limit(
        frequency, driver_s.mms, driver_s.qes, driver_s.fs,
        displacement, boxed.wc, driver_s.sigma_m,
    )
    current = physics.current_at_excursion_limit(
        frequency, driver_s.mms, driver_s.qes, driver_s.fs, driver_s.re,
        displacement, boxed.wc, driver_s.sigma_m,
    )
    voltage = physics.voltage_at_excursion_limit(
        frequency, driver_s.mms, driver_s.bl, driver_s.re,
        displacement, boxed.wc, boxed.wc / boxed.qtc,
    )
    assert current * current * driver_s.re == pytest.approx(power)
    assert voltage > 0


def test_burst_solver_reaches_free_mass_limit_and_scales(driver_s):
    base = physics.sealed_burst_requirements(
        f=10_000.0,
        x_sine_peak=1e-3,
        fc=20.0,
        qtc=0.55,
        mms=driver_s.mms,
        re=driver_s.re,
        bl=driver_s.bl,
        phase_samples=4,
        steps_per_cycle=192,
    )
    assert base.shape_factor == pytest.approx(2.0 * math.pi, rel=0.03)

    doubled = physics.sealed_burst_requirements(
        f=10_000.0,
        x_sine_peak=2e-3,
        fc=20.0,
        qtc=0.55,
        mms=driver_s.mms,
        re=driver_s.re,
        bl=driver_s.bl,
        phase_samples=4,
        steps_per_cycle=192,
    )
    assert doubled.shape_factor == pytest.approx(base.shape_factor)
    assert doubled.displacement_peak == pytest.approx(
        2.0 * base.displacement_peak
    )
    assert doubled.voltage_rms == pytest.approx(2.0 * base.voltage_rms)
    assert doubled.current_rms == pytest.approx(2.0 * base.current_rms)
    assert doubled.coil_power_w == pytest.approx(4.0 * base.coil_power_w)


def test_burst_factor_depends_on_alignment_and_window(driver_s):
    common = dict(
        f=80.0,
        x_sine_peak=1e-3,
        fc=80.0,
        qtc=0.55,
        mms=driver_s.mms,
        re=driver_s.re,
        bl=driver_s.bl,
        phase_samples=8,
    )
    rectangular = physics.sealed_burst_requirements(
        window="rectangular", **common
    )
    hann = physics.sealed_burst_requirements(window="hann", **common)
    assert rectangular.shape_factor < 2.0
    assert hann.shape_factor != pytest.approx(rectangular.shape_factor)
    assert hann.coil_power_w < rectangular.coil_power_w


def test_box_spring_indicator_is_separate():
    indicator = physics.box_spring_nonlinearity(
        2.18e-3, 0.473, 0.478, 0.61
    )
    assert indicator == pytest.approx(0.00107, rel=0.15)


def test_corner_rate_rule(driver_s):
    scenario = Scenario()
    assert scenario.max_corner_rate == pytest.approx(52.0, abs=0.5)
    assert driver_s.corner_rate < scenario.max_corner_rate
