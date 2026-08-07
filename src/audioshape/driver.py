"""Driver and box dataclasses with derived Thiele-Small quantities.

`Driver` stores datasheet values in SI units.  `BoxedDriver` binds a driver to
a sealed alignment (target Qtc).  Both are plain data + derived properties;
all heavy math lives in `physics`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from audioshape import physics


@dataclass(frozen=True)
class Driver:
    """Sealed-box candidate driver, datasheet values in SI units."""

    manufacturer: str
    model: str
    size_in: float          # nominal diameter [inch]
    fs: float               # free-air resonance [Hz]
    qes: float
    qms: float
    re: float               # DC resistance [ohm]
    mms: float              # moving mass [kg]
    sd: float               # piston area [m^2]
    xmax: float             # linear excursion, one-way peak [m]
    vas: float              # equivalent compliance volume [m^3]
    p_max: float            # driver rated power [W]
    bl: float = float("nan")    # force factor [T m]
    le: float = float("nan")    # voice-coil inductance [H]
    type_code: str = ""         # VituixCAD Type column (S, W, ...)

    @property
    def qts(self) -> float:
        return self.qes * self.qms / (self.qes + self.qms)

    @property
    def vd(self) -> float:
        """Displacement volume Vd = Sd * Xmax [m^3] (invariant 1)."""
        return self.sd * self.xmax

    @property
    def eta0(self) -> float:
        """Reference efficiency (invariant 2 with Pmax)."""
        return physics.eta0(self.fs, self.vas, self.qes)

    @property
    def corner_rate(self) -> float:
        """sigma/2pi = Fs/Qts [Hz] (invariant 3)."""
        return physics.corner_rate(self.fs, self.qts)

    @property
    def ebp(self) -> float:
        return self.fs / self.qes

    @property
    def f_le(self) -> float:
        """Inductance corner f_L = Re/(2 pi Le) [Hz] (invariant 4);
        inf when Le is unknown or zero."""
        if not (self.le > 0.0):
            return float("inf")
        return self.re / (2.0 * math.pi * self.le)

    @property
    def sigma_m(self) -> float:
        """Mechanical damping rate sigma_m = ws/Qms [1/s]."""
        return 2.0 * math.pi * self.fs / self.qms

    def label(self) -> str:
        return f"{self.manufacturer} {self.model}"


@dataclass(frozen=True)
class BoxedDriver:
    """A driver in a sealed box aligned to a target Qtc."""

    driver: Driver
    qtc: float
    n_units: int = 1  # identical drivers in identical boxes (A9 scaling)

    vb: float = field(init=False)   # box volume per driver [m^3]
    fc: float = field(init=False)   # closed-box resonance [Hz]

    def __post_init__(self) -> None:
        d = self.driver
        object.__setattr__(self, "vb",
                           physics.box_volume_for_qtc(d.vas, d.qts, self.qtc))
        object.__setattr__(self, "fc",
                           physics.fc_for_qtc(d.fs, d.qts, self.qtc))

    @property
    def wc(self) -> float:
        return 2.0 * math.pi * self.fc

    @property
    def vd_total(self) -> float:
        return self.n_units * self.driver.vd

    def spl_gain_db(self) -> float:
        """Coherent SPL gain of n identical units over one: 20 log10 n for
        displacement-limited output (each unit contributes Vd at same drive)."""
        return 20.0 * math.log10(self.n_units)
