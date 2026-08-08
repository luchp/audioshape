"""Tests for the vented-box comparison model (`vented.py`).

Pins the numerical checks performed by hand/ad-hoc script while deriving
Sec. "Why sealed, not vented" of `sealed_driver_criteria.tex` (v3): the
port-shut limit must reduce to the already-validated sealed-box formula,
the ideal-port low-frequency pressure slope must be 24 dB/octave (exponent
4 in frequency), cone excursion must flatten to the driver's free-air
limit, and the sealed-vs-vented DC excursion ratio must equal (Fc/Fs)^2 =
1+alpha, reusing the same driver S fixture as `tests/test_physics.py`.
"""

import math

import pytest

from audioshape import physics, vented
from audioshape.driver import BoxedDriver, Driver


@pytest.fixture
def driver_s() -> Driver:
    return Driver(
        manufacturer="Example", model="S18", size_in=18,
        fs=20.0, qes=0.543, qms=4.0, re=3.5, mms=0.400,
        sd=0.115, xmax=0.020, vas=0.297, p_max=600.0, bl=18.0, le=4e-3)


def test_sealed_limit_consistency(driver_s):
    """Delta(s)/Dp(s) with a near-shut port (huge Map) must match
    `sealed_limit_delta`, and `sealed_limit_delta(0)` must match the
    sealed-box k_t = Mms*wc^2 implied by `BoxedDriver` (same Qtc=0.61 box
    as the paper's worked example, 473 L)."""
    boxed = BoxedDriver(driver_s, qtc=0.61)
    va = vented.VentedAlignment(driver=driver_s, vb=boxed.vb,
                                l_eff=1.0e6, s_port=0.008, r_ap=0.0)

    s = 1j * 2 * math.pi * 25.0
    ratio = (va.delta(s) / va.dp(s)) / va.sealed_limit_delta(s)
    assert ratio.real == pytest.approx(1.0, rel=1e-4)
    assert ratio.imag == pytest.approx(0.0, abs=1e-4)

    wc = 2 * math.pi * boxed.fc
    kt = driver_s.mms * wc * wc
    assert va.sealed_limit_delta(0).real == pytest.approx(kt, rel=0.01)


def test_implied_qtc_round_trips_box_volume_for_qtc(driver_s):
    """implied_qtc is the exact inverse of physics.box_volume_for_qtc."""
    vb = physics.box_volume_for_qtc(driver_s.vas, driver_s.qts, qtc=0.61)
    assert vented.implied_qtc(driver_s.vas, driver_s.qts, vb) == pytest.approx(
        0.61, rel=1e-9)


def test_low_frequency_pressure_slope_is_24db_per_octave(driver_s):
    """Ideal port (Rap=0): pressure proxy f*|U_tot(f)| must grow with
    exponent ~4 in frequency well below Fb (i.e. 24 dB/octave in SPL)."""
    boxed = BoxedDriver(driver_s, qtc=0.61)
    va = vented.VentedAlignment.tuned(driver_s, vb=boxed.vb, fb=23.0,
                                      s_port=0.008, r_ap=0.0)
    assert va.fb == pytest.approx(23.0, rel=1e-6)

    def pressure_proxy(f: float) -> float:
        s = 1j * 2 * math.pi * f
        return f * abs(va.u_total(s))

    for f1, f2 in ((1.0, 2.0), (2.0, 4.0)):
        slope = math.log(pressure_proxy(f2) / pressure_proxy(f1)) / math.log(f2 / f1)
        assert slope == pytest.approx(4.0, abs=0.1)


def test_low_frequency_excursion_matches_free_air_limit(driver_s):
    """Cone excursion at very low frequency (well below Fb) must match the
    driver's own free-air (no box at all) DC excursion limit -- the vented
    box supplies no restoring force there."""
    boxed = BoxedDriver(driver_s, qtc=0.61)
    va = vented.VentedAlignment.tuned(driver_s, vb=boxed.vb, fb=23.0,
                                      s_port=0.008, r_ap=0.0)
    cms = vented.compliance_from_vas(driver_s.vas, driver_s.sd)
    free_air = vented.free_air_dc_excursion(driver_s.bl, cms, driver_s.re)

    x_tiny = abs(va.x(1j * 2 * math.pi * 0.02))
    assert x_tiny == pytest.approx(free_air, rel=0.01)


def test_relative_excursion_factor_is_one_plus_alpha(driver_s):
    """Same driver, same box volume: vented DC excursion must exceed sealed
    DC excursion by exactly (1+alpha) = (Fc/Fs)^2 (Prop. box invariance)."""
    boxed = BoxedDriver(driver_s, qtc=0.61)
    va = vented.VentedAlignment.tuned(driver_s, vb=boxed.vb, fb=23.0,
                                      s_port=0.008, r_ap=0.0)

    x_tiny = abs(va.x(1j * 2 * math.pi * 0.02))
    wc = 2 * math.pi * boxed.fc
    kt = driver_s.mms * wc * wc
    sealed_dc = (driver_s.bl / driver_s.re) / kt
    one_plus_alpha = (boxed.fc / driver_s.fs) ** 2

    assert x_tiny / sealed_dc == pytest.approx(one_plus_alpha, rel=1e-3)


def test_required_port_area_and_velocity_roundtrip():
    """required_port_area/port_velocity are exact inverses; a "typical"
    10 cm port at the same target is far beyond the turbulence limit --
    the concrete numbers quoted in the paper's port-velocity example
    (r_listen=3 m, self-consistent with L_max=6 m via r ~ L_max/2,
    Sec. "Room closure")."""
    p_t = physics.pressure_from_spl(110.0)
    r, fb = 3.0, 23.0

    area = vented.required_port_area(p_t, r, fb)
    assert vented.port_velocity(p_t, r, fb, area) == pytest.approx(
        vented.TURBULENCE_VELOCITY_MAX, rel=1e-9)
    assert vented.port_diameter(area) == pytest.approx(0.227, abs=0.001)

    typical_area = math.pi * (0.10 / 2) ** 2  # 10 cm diameter port
    v_typical = vented.port_velocity(p_t, r, fb, typical_area)
    assert v_typical == pytest.approx(87.5, abs=0.5)
    assert v_typical > vented.TURBULENCE_VELOCITY_MAX
