"""Pure physics of the sealed-box driver criteria.

All functions are pure and use SI units (metres, kilograms, seconds, watts,
pascals) unless a name says otherwise.  No I/O, no plotting, no globals other
than physical constants -- this module must stay importable from a web backend.

The equations implement `sealed_driver_criteria.tex` (v2).  Section/equation
references in the docstrings point at that document.
"""

from __future__ import annotations

import math

RHO0 = 1.2  # air density [kg/m^3]
C_AIR = 343.0  # speed of sound [m/s]
P0 = 2e-5  # reference pressure [Pa]
GAMMA = 1.4  # adiabatic index of air

# IEC 62458 anchoring of the harmonic-distortion law D(xi) = d2*xi + d3*xi^2
# with D(1) = d2 + d3 = 0.10 at x = Xmax (Sec. "Non-correctable distortion").
D2_DEFAULT = 0.05
D3_DEFAULT = 0.05

# Motor materials bound (Theorem "Motor materials bound", eq:thmbound):
# K_mat = B^2 sigma_c/rho_c, conductivity/density of the coil conductor.
K_MAT_CU_PER_B2 = 1059.0  # Hz/T^2 (copper, sigma_c/rho_c = 6.65e3 S m^2/kg)
K_MAT_AL_PER_B2 = 2063.0  # Hz/T^2 (aluminium, sigma_c/rho_c = 1.30e4 S m^2/kg)


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


def demand_volume(f: float, target_spl: float, r: float,
                  v_room: float, l_max: float) -> float:
    """Peak displaced volume V_dem(f) [m^3] required to reach `target_spl`
    [dB rms] at the listening position (eq:demand).

    Above f_pz: radiation branch, V_dem ~ 1/f^2 (rises 12 dB/oct downward).
    Below f_pz: flat -- the larger of the radiation demand frozen at f_pz and
    the pressure-zone demand sqrt(2) p_t V_room / (rho0 c^2).
    """
    p_t = pressure_from_spl(target_spl)
    f_pz = pressure_zone_frequency(l_max)

    def radiation_branch(freq: float) -> float:
        w = 2.0 * math.pi * freq
        return math.sqrt(2.0) * p_t * 2.0 * math.pi * r / (RHO0 * w * w)

    if f >= f_pz:
        return radiation_branch(f)
    v_pz = math.sqrt(2.0) * p_t * v_room / (RHO0 * C_AIR * C_AIR)
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

def spl_excursion_ceiling(f: float, vd: float, r: float,
                          shape_factor: float = 1.0) -> float:
    """Displacement-limited SPL [dB rms-equivalent] at distance r.

    Sine: shape_factor C = 1.  Burst/pulse: C = 2 (cosine burst, Lemma shape),
    which costs 20 log10 C dB of ceiling.  Independent of Qtc/Bl/Mms
    (Cor. onlyVd).
    """
    p = radiation_pressure_rms(f, vd / shape_factor, r)
    return spl_from_pressure(p)


def spl_thermal_ceiling(eta0: float, p_max: float, r: float) -> float:
    """Thermal (dissipation-limited) passband SPL [dB] at distance r,
    half-space: SPL_T = 112.2 + 10 log10(eta0 * Pmax) - 20 log10(r)."""
    return 112.2 + 10.0 * math.log10(eta0 * p_max) - 20.0 * math.log10(r)


def eta0(fs: float, vas_m3: float, qes: float) -> float:
    """Reference efficiency eta0 = 9.78e-7 * Fs^3 * Vas / Qes (Vas in m^3)."""
    return 9.78e-7 * fs ** 3 * vas_m3 / qes


def regime_boundary_fx(fs: float, p_max: float, qes: float,
                       mms: float, xmax: float) -> float:
    """f_x [Hz]: excursion-limited below, dissipation-limited above (eq:fx).

    f_x = (1/2pi) * (4 pi Fs Pmax / (Qes Mms Xmax^2))^(1/4).
    """
    val = 4.0 * math.pi * fs * p_max / (qes * mms * xmax * xmax)
    return val ** 0.25 / (2.0 * math.pi)


def burst_boundary_fx(fx: float, kappa: float = 4.0, c_shape: float = 2.0) -> float:
    """Transient boundary f^_x = f_x * (kappa * C^2)^(1/4) (eq:fxburst)."""
    return fx * (kappa * c_shape * c_shape) ** 0.25


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
# Non-correctable distortion (Sec. distortion)
# ----------------------------------------------------------------------

def excursion_utilization(v_dem: float, vd: float) -> float:
    """xi_x = V_dem / Vd (per driver unit)."""
    return v_dem / vd


def harmonic_distortion(xi_x: float,
                        d2: float = D2_DEFAULT, d3: float = D3_DEFAULT) -> float:
    """Motor/suspension HD estimate D(xi) = d2 xi + d3 xi^2 (eq:HDscale),
    anchored so D(1) = 0.10 at the IEC 62458 Xmax."""
    return d2 * xi_x + d3 * xi_x * xi_x


def utilization_for_distortion(d_target: float,
                               d2: float = D2_DEFAULT,
                               d3: float = D3_DEFAULT) -> float:
    """Invert D(xi) = d_target for xi (positive root of d3 xi^2 + d2 xi - D)."""
    if d3 == 0.0:
        return d_target / d2
    return (-d2 + math.sqrt(d2 * d2 + 4.0 * d3 * d_target)) / (2.0 * d3)


def doppler_im(f_high: float, x1_peak: float) -> float:
    """Doppler FM sideband index m/2 = pi f2 X1 / c (eq:doppler): the
    fraction of f_high's amplitude thrown into sidebands by a bass excursion
    X1 [m peak] on the same cone."""
    return math.pi * f_high * x1_peak / C_AIR


def box_hd2(v_dem: float, vb: float, qts: float, qtc: float) -> float:
    """Second harmonic of the box air spring (eq:boxHD):
    HD2 = ((gamma+1)/4) (V_dem/Vb) (1 - (Qts/Qtc)^2)."""
    return (GAMMA + 1.0) / 4.0 * (v_dem / vb) * (1.0 - (qts / qtc) ** 2)


def thermal_compression_db(xi_p: float, delta_t_rated: float = 100.0) -> float:
    """Thermal AM level shift [dB]: -0.034 * xi_P * dT_rated (Sec. distortion).
    xi_p is the power utilization P_req / Pmax."""
    return -0.034 * xi_p * delta_t_rated


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
    corner_rate, eq:Fsrule). Undershooting a target corner is free (EQ cut);
    overshooting it is taxed (EQ boost, costs excursion) -- Sec. "Sizing
    rule and Fs criterion" -- so a driver whose corner rate would overshoot
    f_target at the ceiling is better served by a lower Qtc (bigger box)
    than by rejecting it or forcing the fixed ceiling regardless."""
    return min(qtc_ceiling, f_target / corner_rate)


# ----------------------------------------------------------------------
# Motor materials bound (Sec. "EBP as coil-mass fraction", eq:thmbound)
# ----------------------------------------------------------------------

def motor_bound_ebp_u2(ebp: float, u: float) -> float:
    """EBP * u^2 [Hz]: the LHS of the motor materials bound (eq:thmbound),
    given the driver's own EBP = Fs/Qes and an assumed overhang factor u
    (u is not a datasheet field -- it is the coil/gap geometry ratio
    h_c/h_g' of Sec. "materials", chosen illustratively per motor class)."""
    return ebp * u * u


def implied_coil_mass_fraction(ebp: float, u: float, b_field: float,
                               k_mat_per_b2: float = K_MAT_CU_PER_B2) -> float:
    """beta = m_c/Mms implied by matching the motor materials bound
    (eq:thmbound) at equality, given an assumed flux density b_field [T]
    and overhang factor u.  B and u are illustrative motor-topology
    assumptions, not datasheet fields (Sec. "materials", "Numerical
    content")."""
    k_mat = k_mat_per_b2 * b_field * b_field
    return motor_bound_ebp_u2(ebp, u) / k_mat
