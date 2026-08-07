"""Vented (ported) box comparison model.

NOT part of the ranking/CLI pipeline -- the tool stays sealed-only (see
`docs/architecture.md`).  This module exists solely to back
`sealed_driver_criteria.tex`, Sec. "Why sealed, not vented": a first
principles 4th-order transfer-function model used to derive

  (i)   the 24 dB/octave low-frequency pressure asymptote, vs. sealed's
        12 dB/octave (already in `physics.py`/the sealed model),
  (ii)  that cone excursion at low frequency is *not* limited by the box at
        all -- it relaxes to the driver's own free-air value, independent
        of box/port sizing, and
  (iii) the port air-velocity relation behind the turbulence/chuffing
        argument.

Steady-sine phasors, s = j*omega (the same substitution already implicit in
`eq:xofe` of the paper).  Pure functions + one small dataclass; no I/O, no
plotting, no numpy -- mirrors `physics.py`'s style and unit conventions (SI
throughout).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from audioshape import physics
from audioshape.driver import Driver

# Empirical chuffing/turbulence-onset air velocity (Dickason, "The
# Loudspeaker Design Cookbook", 7th ed.) -- an anchor, not a first-principles
# result, exactly like the IEC-62458 Xmax anchor (A7 in the paper).
TURBULENCE_VELOCITY_MAX = 17.0  # m/s


def compliance_from_vas(vas: float, sd: float) -> float:
    """Cms = Vas / (rho0 c^2 Sd^2), from Vas === rho0 c^2 Cms Sd^2
    (`eq:newton`).  Lets this module recover Cms without adding it to
    `Driver` (kept out of the core/ranking pipeline by design)."""
    return vas / (physics.RHO0 * physics.C_AIR ** 2 * sd * sd)


def free_air_dc_excursion(bl: float, cms: float, re: float, e: float = 1.0) -> float:
    """X(0) = Bl Cms E / Re: a driver's own free-air (no enclosure at all)
    low-frequency excursion limit for drive amplitude e [V, default 1 V].

    Sec. "Why sealed, not vented", low-frequency asymptotics: this is the
    value vented cone excursion relaxes to below tuning -- the box supplies
    *no* restoring force there, unlike sealed.
    """
    return bl * cms * e / re


def port_mass(l_eff: float, s_port: float) -> float:
    """Map = rho0 * l_eff / s_port [kg/m^4]: port acoustic mass, l_eff the
    end-corrected port length [m], s_port its cross-sectional area [m^2]."""
    return physics.RHO0 * l_eff / s_port


def box_compliance(vb: float) -> float:
    """Cab = Vb / (rho0 c^2) [m^3/Pa]: acoustic compliance of the box air,
    repackaging the stiffness term already used in `eq:newton`
    (Sd^2/Cab == rho0 c^2 Sd^2 / Vb)."""
    return vb / (physics.RHO0 * physics.C_AIR ** 2)


def implied_qtc(vas: float, qts: float, vb: float) -> float:
    """The sealed Qtc a box of volume vb *would* give this driver (inverse
    of `physics.box_volume_for_qtc`): alpha = Vas/Vb, Qtc = Qts sqrt(1+alpha).

    Lets a sealed-vs-vented comparison figure put the same driver in the
    same box volume both ways (Sec. "Why sealed, not vented")."""
    alpha = vas / vb
    return qts * math.sqrt(1.0 + alpha)


@dataclass(frozen=True)
class VentedAlignment:
    """A driver in a vented (ported) box: comparison-only construct for the
    paper's Sec. "Why sealed, not vented" (never used by `ranking`/`cli`).

    Model (steady-sine, s = j*omega):
        L(s)  = Mms (s^2 + sigma s + ws^2)          -- free-air cone (no box)
        Dp(s) = Cab Map s^2 + Cab Rap s + 1           -- port+box Helmholtz poly
        Delta(s) = L(s) Dp(s) + Sd^2 s (Map s + Rap)  -- 4th-order denominator
        X(s)      = (Bl/Re) E Dp(s) / Delta(s)
        U_port(s) = -Sd s X(s) / Dp(s)
        U_tot(s)  = Sd s X(s) (Dp(s) - 1) / Dp(s)
    """

    driver: Driver
    vb: float                # box volume [m^3]
    l_eff: float               # port effective (end-corrected) length [m]
    s_port: float               # port cross-sectional area [m^2]
    r_ap: float = 0.0           # port acoustic resistance [Pa s/m^3] (0 = ideal)

    @classmethod
    def tuned(cls, driver: Driver, vb: float, fb: float, s_port: float,
             r_ap: float = 0.0) -> "VentedAlignment":
        """Build an alignment whose port length is chosen to hit tuning
        `fb` [Hz] for box volume `vb` [m^3] and port area `s_port` [m^2]."""
        cab = box_compliance(vb)
        map_needed = 1.0 / ((2.0 * math.pi * fb) ** 2 * cab)
        l_eff = map_needed * s_port / physics.RHO0
        return cls(driver=driver, vb=vb, l_eff=l_eff, s_port=s_port, r_ap=r_ap)

    @property
    def cab(self) -> float:
        return box_compliance(self.vb)

    @property
    def map(self) -> float:
        return port_mass(self.l_eff, self.s_port)

    @property
    def fb(self) -> float:
        """Port tuning frequency Fb = 1/(2 pi sqrt(Map Cab))."""
        return 1.0 / (2.0 * math.pi * math.sqrt(self.map * self.cab))

    @property
    def cms(self) -> float:
        return compliance_from_vas(self.driver.vas, self.driver.sd)

    @property
    def sigma(self) -> float:
        """sigma = 2 pi Fs/Qts (box-invariant, Prop. box invariance of the
        damping rate): reused unchanged, since the vented cone equation
        carries no box-stiffness term (box dynamics enter via Pb only)."""
        return 2.0 * math.pi * physics.corner_rate(self.driver.fs, self.driver.qts)

    @property
    def ws(self) -> float:
        return 2.0 * math.pi * self.driver.fs

    def dp(self, s: complex) -> complex:
        """Dp(s) = Cab Map s^2 + Cab Rap s + 1 (port+box Helmholtz
        polynomial; Dp=0 at s = j 2 pi Fb when Rap=0)."""
        return self.cab * self.map * s * s + self.cab * self.r_ap * s + 1.0

    def cone_operator(self, s: complex) -> complex:
        """L(s) = Mms (s^2 + sigma s + ws^2): the driver's own free-air
        second-order operator (box stiffness is not in here for vented --
        it enters through Pb/Dp instead)."""
        d = self.driver
        return d.mms * (s * s + self.sigma * s + self.ws * self.ws)

    def delta(self, s: complex) -> complex:
        """Delta(s) = L(s) Dp(s) + Sd^2 s (Map s + Rap): the 4th-order
        system denominator."""
        d = self.driver
        return (self.cone_operator(s) * self.dp(s)
                + d.sd * d.sd * s * (self.map * s + self.r_ap))

    def x(self, s: complex, e: complex = 1.0) -> complex:
        """Cone displacement phasor X(s) = (Bl/Re) E Dp(s) / Delta(s)."""
        d = self.driver
        return (d.bl / d.re) * e * self.dp(s) / self.delta(s)

    def u_port(self, s: complex, e: complex = 1.0) -> complex:
        """Port-only volume-velocity phasor U_p(s) = -Sd s X(s) / Dp(s)."""
        d = self.driver
        return -d.sd * s * self.x(s, e) / self.dp(s)

    def u_total(self, s: complex, e: complex = 1.0) -> complex:
        """Total (cone + port) volume-velocity phasor:
        U_tot(s) = Sd s X(s) (Dp(s) - 1) / Dp(s)."""
        d = self.driver
        dp_s = self.dp(s)
        return d.sd * s * self.x(s, e) * (dp_s - 1.0) / dp_s

    def sealed_limit_delta(self, s: complex) -> complex:
        """Delta(s)/Dp(s) as Map, Rap -> infinity (port sealed shut): the
        consistency corollary says this must equal the sealed-box
        denominator Mms(s^2 + sigma s + ws^2) + Sd^2/Cab (`eq:newton`'s
        k_t, reused here as Sd^2/Cab). Used as the ground truth in
        `tests/test_vented.py`, approached numerically by a near-closed
        port (very large Map)."""
        d = self.driver
        return self.cone_operator(s) + d.sd * d.sd / self.cab


# ----------------------------------------------------------------------
# Port air velocity / turbulence (Sec. "Why sealed, not vented", Sec. 1.4)
# ----------------------------------------------------------------------

def required_port_area(p_t: float, r: float, fb: float,
                       v_max: float = TURBULENCE_VELOCITY_MAX) -> float:
    """Minimum port area [m^2] keeping peak port air velocity <= v_max
    [m/s] while reaching rms pressure p_t [Pa] at distance r [m], tuning
    fb [Hz].

    Near Fb the port carries essentially all the output (U_port ~ U_total;
    cone-null result, Dp(j 2 pi Fb) -> 0 for a low-loss port), so combined
    with the monopole relation (A3/`eq:pofx`):
        S_port >= U_total(Fb) / v_max = 2 pi r p_t / (rho0 wb v_max).
    """
    wb = 2.0 * math.pi * fb
    u_total = 2.0 * math.pi * r * p_t / (physics.RHO0 * wb)
    return u_total / v_max


def port_velocity(p_t: float, r: float, fb: float, s_port: float) -> float:
    """Peak port air velocity [m/s] at tuning for a port of area s_port
    [m^2] (inverse of `required_port_area`)."""
    wb = 2.0 * math.pi * fb
    u_total = 2.0 * math.pi * r * p_t / (physics.RHO0 * wb)
    return u_total / s_port


def port_diameter(area: float) -> float:
    """Equivalent round-port diameter [m] for cross-sectional area [m^2]."""
    return math.sqrt(4.0 * area / math.pi)
