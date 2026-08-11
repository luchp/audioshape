"""User scenario: room, listening position, targets.

A `Scenario` is a plain serializable config object (AGENTS.md: separate the
serializable config from the actionable objects).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from audioshape import physics

Role = Literal["sub", "attack", "full"]
RoomModel = Literal["ideal_pressure_zone", "leaky_pressure_zone"]
TransientWindow = Literal["rectangular", "hann"]


@dataclass(frozen=True)
class Scenario:
    """Serializable system requirements; SI units and rms SPL in dB.

    The mono sub target applies to the complete manifold.  The upper-bass
    target applies independently to each stereo channel; no stereo summing
    credit is used.  ``Xmax`` is the clipping boundary.  The lower
    ``preferred_excursion`` value is a design-margin marker, not a gate and
    not an inferred distortion threshold.
    """

    v_room: float = 60.0
    l_max: float = 6.0
    r_listen: float = 3.0
    sub_target_spl: float = 110.0
    attack_target_spl: float = 105.0

    room_model: RoomModel = "leaky_pressure_zone"
    leakage_corner_hz: float = 10.0

    qtc: float = 0.55
    max_box_vas_ratio: float = 10.0
    max_role_box_volume_m3: float = 1.0
    f_low: float = 15.0
    f_split: float = 80.0
    f_high: float = 250.0

    preferred_excursion: float = 0.80
    doppler_budget: float = 0.02

    amplifier_voltage_rms: float = 90.0
    amplifier_current_rms: float = 15.0
    amplifier_power_continuous: float = 500.0
    amplifier_power_burst: float = 1000.0

    transient_cycles: float = 1.0
    transient_window: TransientWindow = "rectangular"
    transient_phase_samples: int = 8

    def __post_init__(self) -> None:
        if self.l_max <= 0 or self.v_room <= 0 or self.r_listen <= 0:
            raise ValueError("room dimensions and distance must be positive")
        if self.room_model not in ("ideal_pressure_zone", "leaky_pressure_zone"):
            raise ValueError(f"unknown room model {self.room_model!r}")
        if self.leakage_corner_hz < 0:
            raise ValueError("leakage corner must be non-negative")
        if not 0 < self.preferred_excursion <= 1:
            raise ValueError("preferred excursion must be a fraction in (0, 1]")
        if self.doppler_budget <= 0:
            raise ValueError("doppler budget must be positive")
        if self.qtc <= 0:
            raise ValueError("Qtc ceiling must be positive")
        if self.max_box_vas_ratio <= 0 or self.max_role_box_volume_m3 <= 0:
            raise ValueError("box-volume limits must be positive")
        if not 0 < self.f_low < self.f_split < self.f_high:
            raise ValueError("frequencies must satisfy 0 < f_low < f_split < f_high")
        amplifier_limits = (
            self.amplifier_voltage_rms,
            self.amplifier_current_rms,
            self.amplifier_power_continuous,
            self.amplifier_power_burst,
        )
        if any(limit <= 0 for limit in amplifier_limits):
            raise ValueError("amplifier limits must be positive")
        if self.transient_cycles <= 0:
            raise ValueError("transient cycles must be positive")
        if self.transient_window not in ("rectangular", "hann"):
            raise ValueError(f"unknown transient window {self.transient_window!r}")
        if self.transient_phase_samples < 4 or self.transient_phase_samples % 4:
            raise ValueError("transient phase samples must be a multiple of four")

    @property
    def f_pz(self) -> float:
        """Room pressure-zone corner c/(2 L_max) [Hz]."""
        return physics.pressure_zone_frequency(self.l_max)

    @property
    def max_corner_rate(self) -> float:
        """Admissible Fs/Qts (eq:Fsrule)."""
        return physics.max_corner_rate(self.f_pz, self.qtc)

    def target_spl_for(self, role: Role) -> float:
        """Couch-area target for the mono manifold or one stereo channel."""
        if role in ("sub", "full"):
            return self.sub_target_spl
        if role == "attack":
            return self.attack_target_spl
        raise ValueError(f"unknown role {role!r}")

    def target_corner_hz(self, role: Role) -> float:
        """This role's own box-corner target (eq:Fsrule): the room's
        pressure zone for sub (and 'full', a single driver whose band also
        reaches down into it), the sub/attack split for attack
        (sec_procedure.tex step 5's "Fc<~f_sp")."""
        if role == "attack":
            return self.f_split
        if role in ("sub", "full"):
            return self.f_pz
        raise ValueError(f"unknown role {role!r}")

    def max_box_volume_per_driver(self, vas: float, n_units: int) -> float:
        """Per-driver box cap: min(ratio*Vas, role-volume/N)."""
        if vas <= 0 or n_units < 1:
            raise ValueError("Vas and unit count must be positive")
        return min(
            self.max_box_vas_ratio * vas,
            self.max_role_box_volume_m3 / n_units,
        )

    def demand_volume(self, f: float, role: Role = "sub") -> float:
        """Peak displaced volume [m^3] needed for the target SPL at f."""
        return physics.demand_volume(f, self.target_spl_for(role), self.r_listen,
                                     self.v_room, self.l_max,
                                     room_model=self.room_model,
                                     leakage_corner_hz=self.leakage_corner_hz)

    def v_dem_max(self, role: Role = "sub") -> float:
        """Worst demand at the lower edge of the role's operating band."""
        band_low = self.f_split if role == "attack" else self.f_low
        return self.demand_volume(band_low, role)

    def target_pressure(self, role: Role = "sub") -> float:
        return physics.pressure_from_spl(self.target_spl_for(role))

    def required_vd(self, band_low: float | None = None,
                    role: Role = "sub",
                    utilization: float | None = None) -> float:
        """Total Vd needed at a declared excursion utilization.

        The default is the preferred 80% design margin.  Pass ``1.0`` for
        the physical clipping requirement.
        """
        if utilization is None:
            utilization = self.preferred_excursion
        if not 0 < utilization <= 1:
            raise ValueError("utilization must be a fraction in (0, 1]")
        f = (self.f_split if role == "attack" else self.f_low
             ) if band_low is None else band_low
        return self.demand_volume(f, role) / utilization

    def units_required(self, vd_per_unit: float,
                       band_low: float | None = None,
                       role: Role = "sub",
                       utilization: float | None = None) -> int:
        """Driver count for the declared utilization, rounded up."""
        return max(
            1,
            math.ceil(
                self.required_vd(band_low, role, utilization) / vd_per_unit
            ),
        )
