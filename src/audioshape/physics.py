"""Pure physics of the sealed-box driver criteria.

All functions are pure and use SI units (metres, kilograms, seconds, watts,
pascals) unless a name says otherwise.  No I/O, no plotting, no globals other
than physical constants -- this module must stay importable from a web backend.

The equations implement `papers/26325/sealed_driver_criteria.tex`.
Section/equation references in the docstrings point at that document.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RHO0 = 1.2  # air density [kg/m^3]
C_AIR = 343.0  # speed of sound [m/s]
P0 = 2e-5  # reference pressure [Pa]
GAMMA = 1.4  # adiabatic index of air


def spl_from_pressure(p_rms: float) -> float:
    """SPL [dB] from rms pressure [Pa]."""
    return 20.0 * math.log10(p_rms / P0)


def pressure_from_spl(spl_db: float) -> float:
    """rms pressure [Pa] from SPL [dB]."""
    return P0 * 10.0 ** (spl_db / 20.0)


# ----------------------------------------------------------------------
# Radiation and room pressure zone (half-space / soffit mounting)
# ----------------------------------------------------------------------

def radiation_pressure_rms(f: float, v_disp_peak: float, r: float) -> float:
    """rms pressure [Pa] at distance r [m] of a half-space piston displacing
    peak volume v_disp_peak [m^3] sinusoidally at frequency f [Hz].

    p_peak = rho0 * omega^2 * V / (2 pi r);   rms = peak / sqrt(2).
    Equals eq. (SPL_L) with its 108.5 dB constant at r = 1 m.
    """
    w = 2.0 * math.pi * f
    return RHO0 * w * w * v_disp_peak / (2.0 * math.pi * r) / math.sqrt(2.0)


def pressure_zone_frequency(l_max: float) -> float:
    """f_pz = c / (2 L_max): below this the room acts as a pressure vessel."""
    return C_AIR / (2.0 * l_max)


def pressure_zone_pressure_rms(v_disp_peak: float, v_room: float) -> float:
    """rms pressure [Pa] in the pressure zone: p = rho0 c^2 V_disp / V_room
    (adiabatic compression of the room air), peak -> rms with sqrt(2)."""
    return RHO0 * C_AIR * C_AIR * v_disp_peak / v_room / math.sqrt(2.0)


def room_leakage_displacement_factor(f: float,
                                     leakage_corner_hz: float) -> float:
    """Extra displacement required by a pressure-zone leakage corner.

    A room compliance shunted by an acoustic leakage resistance gives a
    first-order high-pass relation from source displacement to pressure.
    The ideal pressure-zone demand is therefore multiplied by
    ``sqrt(1 + (f_leak/f)^2)``.  This is a low-frequency limiting model, not
    a claim that the complete room is one low-order system.
    """
    if f <= 0:
        raise ValueError("frequency must be positive")
    if leakage_corner_hz < 0:
        raise ValueError("leakage corner must be non-negative")
    return math.sqrt(1.0 + (leakage_corner_hz / f) ** 2)


def demand_volume(f: float, target_spl: float, r: float,
                  v_room: float, l_max: float,
                  room_model: str = "leaky_pressure_zone",
                  leakage_corner_hz: float = 10.0) -> float:
    """Peak displaced volume V_dem(f) [m^3] required to reach `target_spl`
    [dB rms] at the listening position.

    Above f_pz: radiation branch, V_dem ~ 1/f^2 (rises 12 dB/oct downward).
    Below f_pz: the larger of the radiation demand frozen at f_pz and the
    pressure-zone compliance demand.  ``ideal_pressure_zone`` leaves that
    branch flat.  ``leaky_pressure_zone`` applies the declared first-order
    leakage factor, so demand rises 6 dB/oct below the leakage corner.
    """
    if f <= 0:
        raise ValueError("frequency must be positive")
    if room_model not in ("ideal_pressure_zone", "leaky_pressure_zone"):
        raise ValueError(f"unknown room model {room_model!r}")

    p_t = pressure_from_spl(target_spl)
    f_pz = pressure_zone_frequency(l_max)

    def radiation_branch(freq: float) -> float:
        w = 2.0 * math.pi * freq
        return math.sqrt(2.0) * p_t * 2.0 * math.pi * r / (RHO0 * w * w)

    if f >= f_pz:
        return radiation_branch(f)
    v_pz = math.sqrt(2.0) * p_t * v_room / (RHO0 * C_AIR * C_AIR)
    if room_model == "leaky_pressure_zone":
        v_pz *= room_leakage_displacement_factor(f, leakage_corner_hz)
    return max(radiation_branch(f_pz), v_pz)


# ----------------------------------------------------------------------
# Sealed-box alignment (Prop. box-invariance)
# ----------------------------------------------------------------------

def box_volume_for_qtc(vas: float, qts: float, qtc: float) -> float:
    """Vb = Vas / ((Qtc/Qts)^2 - 1) [same unit as Vas]. Requires qtc > qts."""
    ratio = (qtc / qts) ** 2 - 1.0
    if ratio <= 0.0:
        raise ValueError(f"Qtc={qtc} must exceed Qts={qts} for a sealed box")
    return vas / ratio


def qtc_for_box_volume(vas: float, qts: float, vb: float) -> float:
    """Inverse of box_volume_for_qtc: the Qtc a box of volume vb actually
    gives this driver, Qtc = Qts * sqrt(1 + Vas/Vb)."""
    return qts * math.sqrt(1.0 + vas / vb)


def fc_for_qtc(fs: float, qts: float, qtc: float) -> float:
    """Fc = Qtc * Fs / Qts (box invariance: Fc/Qtc = Fs/Qts = sigma/2pi)."""
    return qtc * fs / qts


# ----------------------------------------------------------------------
# SPL ceilings (Sec. frequency domain / time domain)
# ----------------------------------------------------------------------

def spl_excursion_ceiling(f: float, vd: float, r: float) -> float:
    """Steady-sine displacement-limited SPL [dB rms] at distance ``r``.

    Finite bursts are evaluated with :func:`sealed_burst_requirements`
    rather than a universal shape factor.
    """
    p = radiation_pressure_rms(f, vd, r)
    return spl_from_pressure(p)


def spl_thermal_ceiling(eta0: float, p_max: float, r: float) -> float:
    """Thermal (dissipation-limited) passband SPL [dB] at distance r,
    half-space: SPL_T = 112.2 + 10 log10(eta0 * Pmax) - 20 log10(r)."""
    return 112.2 + 10.0 * math.log10(eta0 * p_max) - 20.0 * math.log10(r)


def eta0(fs: float, vas_m3: float, qes: float) -> float:
    """Reference efficiency eta0 = 9.78e-7 * Fs^3 * Vas / Qes (Vas in m^3)."""
    return 9.78e-7 * fs ** 3 * vas_m3 / qes


def force_factor_from_qes(fs: float, qes: float, mms: float, re: float) -> float:
    """Force factor [T m] implied by Fs, Qes, Mms, and Re.

    From ``Qes = omega_s Mms Re / Bl^2``.  This is used only when a public
    record omits Bl; the evaluation reports that the value was derived.
    """
    return math.sqrt(2.0 * math.pi * fs * mms * re / qes)


def regime_boundary_fx(fs: float, p_max: float, qes: float,
                       mms: float, xmax: float) -> float:
    """f_x [Hz]: excursion-limited below, dissipation-limited above (eq:fx).

    f_x = (1/2pi) * (4 pi Fs Pmax / (Qes Mms Xmax^2))^(1/4).
    """
    val = 4.0 * math.pi * fs * p_max / (qes * mms * xmax * xmax)
    return val ** 0.25 / (2.0 * math.pi)


def power_at_excursion_limit(f: float, mms: float, qes: float, fs: float,
                             xmax: float, wc: float, sigma_m: float) -> float:
    """Electrical power [W] needed to hold x = Xmax at frequency f (eq:PL).

    P_L(w) = (Mms Qes Xmax^2 / (4 pi Fs)) * ((wc^2 - w^2)^2 + sigma_m^2 w^2).
    Uses only the mechanical damping sigma_m (the electrical part is supplied
    back by the source).
    """
    w = 2.0 * math.pi * f
    return (mms * qes * xmax * xmax / (4.0 * math.pi * fs)) * (
        (wc * wc - w * w) ** 2 + sigma_m * sigma_m * w * w
    )


def voltage_at_excursion_limit(f: float, mms: float, bl: float, re: float,
                               xmax: float, wc: float, sigma: float) -> float:
    """RMS terminal voltage [V] needed to hold x = xmax at frequency f
    (eq:xofe, inverted for e).

    e_hat = xmax * (Re*Mms/Bl) * sqrt((wc^2-w^2)^2 + sigma^2 w^2), V_rms =
    e_hat/sqrt(2). Unlike power_at_excursion_limit (current drive, eq:iofx,
    mechanical damping sigma_m only), a *voltage* source also has to
    overcome the electrical damping sigma_e = Bl^2/(Re Mms), so this uses
    the *total* box-invariant damping rate sigma = wc/Qtc = sigma_m+sigma_e
    (Prop. box invariance), not sigma_m alone.
    """
    w = 2.0 * math.pi * f
    e_hat = xmax * (re * mms / bl) * math.sqrt(
        (wc * wc - w * w) ** 2 + sigma * sigma * w * w)
    return e_hat / math.sqrt(2.0)


def current_at_excursion_limit(f: float, mms: float, qes: float, fs: float,
                               re: float, xmax: float, wc: float,
                               sigma_m: float) -> float:
    """RMS coil current [A] needed for a peak displacement ``xmax``."""
    power = power_at_excursion_limit(
        f, mms, qes, fs, xmax, wc, sigma_m
    )
    return math.sqrt(power / re)


def eq_tax_power(f: float, p_passband: float, wc: float, sigma_m: float) -> float:
    """Power [W] required at f after EQ to flat response (eq:EQtax):
    P_req = P_pb * ((wc^2 - w^2)^2 + sigma_m^2 w^2) / w^4."""
    w = 2.0 * math.pi * f
    return p_passband * ((wc * wc - w * w) ** 2 + sigma_m * sigma_m * w * w) / w ** 4


def acoustic_power_halfspace(p_rms: float, r: float) -> float:
    """Radiated acoustic power [W] for rms pressure p at distance r into 2pi:
    W = 2 pi r^2 p^2 / (rho0 c)."""
    return 2.0 * math.pi * r * r * p_rms * p_rms / (RHO0 * C_AIR)


# ----------------------------------------------------------------------
# Transient response and separate nonlinear-risk indicators
# ----------------------------------------------------------------------

def excursion_utilization(v_dem: float, vd: float) -> float:
    """xi_x = V_dem / Vd (per driver unit)."""
    return v_dem / vd


@dataclass(frozen=True)
class BurstRequirements:
    """Worst phase of one declared voltage burst at a target output.

    ``x_sine_peak`` is the steady-sine displacement that would produce the
    target pressure at the same frequency.  The numerical sealed-box ODE is
    scaled so the burst reaches the same peak acceleration during its active
    window.  RMS voltage, current, and coil power are evaluated over that
    active window; peak displacement includes the subsequent ring-down.
    """

    shape_factor: float
    displacement_peak: float
    voltage_rms: float
    current_rms: float
    coil_power_w: float


def sealed_burst_requirements(
    f: float,
    x_sine_peak: float,
    fc: float,
    qtc: float,
    mms: float,
    re: float,
    bl: float,
    *,
    cycles: float = 1.0,
    window: str = "rectangular",
    phase_samples: int = 8,
    steps_per_cycle: int = 96,
    decay_time_constants: float = 8.0,
) -> BurstRequirements:
    """Numerically solve the sealed ODE for a finite voltage burst.

    Start phases are sampled uniformly over one cycle.  A multiple of four
    includes both sine and cosine starts.  This replaces a universal
    free-mass burst constant with a frequency- and alignment-dependent
    result.
    """
    positive = (f, x_sine_peak, fc, qtc, mms, re, bl, cycles)
    if any(value <= 0 for value in positive):
        raise ValueError("burst frequencies, parameters, and target must be positive")
    if window not in ("rectangular", "hann"):
        raise ValueError(f"unknown burst window {window!r}")
    if phase_samples < 4 or phase_samples % 4:
        raise ValueError("phase samples must be a multiple of four")
    if steps_per_cycle < 32:
        raise ValueError("steps per cycle must be at least 32")

    omega = 2.0 * math.pi * f
    desired_acceleration = omega * omega * x_sine_peak

    shape_factor = 0.0
    displacement_peak = 0.0
    voltage_rms = 0.0
    current_rms = 0.0
    coil_power_w = 0.0

    for phase_index in range(phase_samples):
        phase = 2.0 * math.pi * phase_index / phase_samples
        response = _sealed_burst_unit_voltage(
            f=f,
            fc=fc,
            qtc=qtc,
            mms=mms,
            re=re,
            bl=bl,
            cycles=cycles,
            window=window,
            phase=phase,
            steps_per_cycle=steps_per_cycle,
            decay_time_constants=decay_time_constants,
        )
        scale = desired_acceleration / response.peak_active_acceleration
        phase_displacement = response.peak_displacement * scale
        phase_shape = phase_displacement / x_sine_peak
        phase_voltage = response.voltage_rms * scale
        phase_current = response.current_rms * scale
        phase_power = response.coil_power_w * scale * scale

        shape_factor = max(shape_factor, phase_shape)
        displacement_peak = max(displacement_peak, phase_displacement)
        voltage_rms = max(voltage_rms, phase_voltage)
        current_rms = max(current_rms, phase_current)
        coil_power_w = max(coil_power_w, phase_power)

    return BurstRequirements(
        shape_factor=shape_factor,
        displacement_peak=displacement_peak,
        voltage_rms=voltage_rms,
        current_rms=current_rms,
        coil_power_w=coil_power_w,
    )


@dataclass(frozen=True)
class _UnitBurstResponse:
    peak_active_acceleration: float
    peak_displacement: float
    voltage_rms: float
    current_rms: float
    coil_power_w: float


def _sealed_burst_unit_voltage(
    *,
    f: float,
    fc: float,
    qtc: float,
    mms: float,
    re: float,
    bl: float,
    cycles: float,
    window: str,
    phase: float,
    steps_per_cycle: int,
    decay_time_constants: float,
) -> _UnitBurstResponse:
    omega = 2.0 * math.pi * f
    wc = 2.0 * math.pi * fc
    sigma = wc / qtc
    forcing_gain = bl / (re * mms)
    duration = cycles / f

    def voltage(t: float) -> float:
        if not 0.0 <= t < duration:
            return 0.0
        envelope = 1.0
        if window == "hann":
            envelope = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / duration)
        return envelope * math.sin(omega * t + phase)

    def derivatives(t: float, x: float, velocity: float) -> tuple[float, float]:
        acceleration = (
            forcing_gain * voltage(t) - sigma * velocity - wc * wc * x
        )
        return velocity, acceleration

    def rk4_step(t: float, x: float, velocity: float,
                 dt: float) -> tuple[float, float]:
        k1_x, k1_v = derivatives(t, x, velocity)
        k2_x, k2_v = derivatives(
            t + dt / 2.0,
            x + dt * k1_x / 2.0,
            velocity + dt * k1_v / 2.0,
        )
        k3_x, k3_v = derivatives(
            t + dt / 2.0,
            x + dt * k2_x / 2.0,
            velocity + dt * k2_v / 2.0,
        )
        k4_x, k4_v = derivatives(
            t + dt,
            x + dt * k3_x,
            velocity + dt * k3_v,
        )
        return (
            x + dt * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x) / 6.0,
            velocity + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0,
        )

    x = 0.0
    velocity = 0.0
    t = 0.0
    peak_acceleration = 0.0
    peak_displacement = 0.0
    voltage_squared = 0.0
    current_squared = 0.0

    active_steps = max(
        steps_per_cycle,
        math.ceil(duration * steps_per_cycle * max(f, fc)),
    )
    active_dt = duration / active_steps
    for _ in range(active_steps):
        e = voltage(t)
        acceleration = forcing_gain * e - sigma * velocity - wc * wc * x
        current = (e - bl * velocity) / re
        peak_acceleration = max(peak_acceleration, abs(acceleration))
        peak_displacement = max(peak_displacement, abs(x))
        voltage_squared += e * e
        current_squared += current * current
        x, velocity = rk4_step(t, x, velocity, active_dt)
        t += active_dt

    # Avoid a floating-point value infinitesimally below the burst boundary:
    # the first, much larger ring-down step must never sample active voltage.
    t = duration
    peak_displacement = max(peak_displacement, abs(x))
    decay_duration = decay_time_constants * 2.0 / sigma
    ring_steps = max(1, math.ceil(decay_duration * steps_per_cycle * fc))
    ring_dt = decay_duration / ring_steps
    for _ in range(ring_steps):
        peak_displacement = max(peak_displacement, abs(x))
        x, velocity = rk4_step(t, x, velocity, ring_dt)
        t += ring_dt
    peak_displacement = max(peak_displacement, abs(x))

    if peak_acceleration <= 0:
        raise RuntimeError("burst simulation produced no active acceleration")
    voltage_rms = math.sqrt(voltage_squared / active_steps)
    current_rms = math.sqrt(current_squared / active_steps)
    return _UnitBurstResponse(
        peak_active_acceleration=peak_acceleration,
        peak_displacement=peak_displacement,
        voltage_rms=voltage_rms,
        current_rms=current_rms,
        coil_power_w=current_rms * current_rms * re,
    )


def doppler_im(f_high: float, x1_peak: float) -> float:
    """Small-index first-order Doppler sideband amplitude ratio.

    The phase-modulation index is ``m = 2*pi*f_high*x1_peak/c``.  For
    ``m << 1``, either first-order sideband has relative amplitude
    ``J1(m)/J0(m) ~= m/2 = pi*f_high*x1_peak/c``.  This kinematic indicator
    is not a perceptual distortion estimate.
    """
    return math.pi * f_high * x1_peak / C_AIR


def box_spring_nonlinearity(v_dem: float, vb: float,
                            qts: float, qtc: float) -> float:
    """Leading dimensionless box air-spring nonlinearity indicator.

    It is reported separately and is not interpreted or summed as acoustic
    harmonic distortion.
    """
    return (GAMMA + 1.0) / 4.0 * (v_dem / vb) * (1.0 - (qts / qtc) ** 2)


# ----------------------------------------------------------------------
# Corner-rate rule (eq:Fsrule)
# ----------------------------------------------------------------------

def corner_rate(fs: float, qts: float) -> float:
    """sigma / 2pi = Fs / Qts [Hz]: box-invariant corner placement rate."""
    return fs / qts


def max_corner_rate(f_pz: float, qtc_target: float) -> float:
    """Largest admissible Fs/Qts so the box corner can sit at f_pz with the
    target Qtc: Fs/Qts <= f_pz / Qtc* (eq:Fsrule)."""
    return f_pz / qtc_target


def qtc_for_target_corner(corner_rate: float, f_target: float,
                          qtc_ceiling: float) -> float:
    """Qtc to actually use: the smaller of a configured ceiling and the Qtc
    that would land Fc exactly at f_target (box invariance Fc/Qtc =
    corner_rate, eq:Fsrule). A corner below the target needs no correction
    boost but costs enclosure volume; a corner above it costs boost and
    headroom. A driver that would overshoot at the ceiling is therefore
    assigned a lower Qtc (larger box) when the box cap permits."""
    return min(qtc_ceiling, f_target / corner_rate)
