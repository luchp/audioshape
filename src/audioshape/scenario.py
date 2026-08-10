"""User scenario: room, listening position, targets.

A `Scenario` is a plain serializable config object (AGENTS.md: separate the
serializable config from the actionable objects).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from audioshape import physics


@dataclass(frozen=True)
class Scenario:
    """Everything the user chooses; SI units, SPL in dB."""

    v_room: float = 60.0        # room volume [m^3]
    l_max: float = 6.0          # largest room dimension [m] -> f_pz
    r_listen: float = 3.0       # listening distance [m], to the single
                                 # fixed listening position (the couch)
    target_spl: float = 110.0   # rms-equivalent SPL at the listening
                                 # position [dB], required from *this one
                                 # source* (the mono sub manifold, or one
                                 # tower channel on its own for worst-case
                                 # panned content) -- no stereo summing
                                 # correction is applied anywhere in this
                                 # tool.  If a design instead used two
                                 # separate stereo subs, their sum at the
                                 # couch would exceed a single sub's
                                 # target_spl by roughly +3 dB (decorrelated)
                                 # to +6 dB (correlated/mono bass) -- not
                                 # modeled here, not this tool's default
                                 # architecture.
    distortion_budget: float = 0.03   # allowed HD (fraction), D*
    doppler_budget: float = 0.02      # allowed Doppler FM index, D*_IM
                                        # (eq:doppler); attack-role feasibility
                                        # gate only (sec_procedure.tex step 4)
    qtc: float = 0.55           # target closed-box alignment
    f_low: float = 15.0         # lowest analysis frequency [Hz]
    f_split: float = 80.0       # sub / attack split [Hz]
    f_high: float = 250.0       # top of the attack band [Hz]
    burst_shape: float = 2.0    # C (cosine burst, Lemma shape)
    burst_headroom: float = 4.0 # kappa (short-burst power headroom)

    def __post_init__(self) -> None:
        if self.l_max <= 0 or self.v_room <= 0 or self.r_listen <= 0:
            raise ValueError("room dimensions and distance must be positive")
        if not 0 < self.distortion_budget < 1:
            raise ValueError("distortion budget must be a fraction in (0, 1)")
        if self.doppler_budget <= 0:
            raise ValueError("doppler budget must be positive")

    @property
    def f_pz(self) -> float:
        """Room pressure-zone corner c/(2 L_max) [Hz]."""
        return physics.pressure_zone_frequency(self.l_max)

    @property
    def max_corner_rate(self) -> float:
        """Admissible Fs/Qts (eq:Fsrule)."""
        return physics.max_corner_rate(self.f_pz, self.qtc)

    def demand_volume(self, f: float) -> float:
        """Peak displaced volume [m^3] needed for the target SPL at f."""
        return physics.demand_volume(f, self.target_spl, self.r_listen,
                                     self.v_room, self.l_max)

    @property
    def v_dem_max(self) -> float:
        """Worst-case (flat, below f_pz) demand volume [m^3]."""
        return self.demand_volume(min(self.f_low, self.f_pz))

    @property
    def utilization_budget(self) -> float:
        """xi* such that D(xi*) = D* (inverts eq:HDscale)."""
        return physics.utilization_for_distortion(self.distortion_budget)

    @property
    def target_pressure(self) -> float:
        return physics.pressure_from_spl(self.target_spl)

    def required_vd(self, band_low: float | None = None) -> float:
        """Total Vd [m^3] to hold the target SPL at the distortion budget
        down to `band_low` (default: f_low; boxed sizing rule eq:Vdreq)."""
        f = self.f_low if band_low is None else band_low
        return self.demand_volume(f) / self.utilization_budget

    def units_required(self, vd_per_unit: float,
                       band_low: float | None = None) -> int:
        """Driver count for the sizing rule, rounded up."""
        return max(1, math.ceil(self.required_vd(band_low) / vd_per_unit))
