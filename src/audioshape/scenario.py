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

    The mono sub target applies to the complete manifold. The upper-bass
    target applies independently to each stereo channel. A full-band stereo
    channel receives the configured mono-bass summing credit only below the
    split. ``Xmax`` is the clipping boundary. The lower
    ``preferred_excursion`` value is a design-margin marker, not a gate and
    not an inferred distortion threshold.
    """

    v_room: float = 60.0
    l_max: float = 6.0
    r_listen: float = 3.0
    sub_target_spl: float = 110.0
    attack_target_spl: float = 105.0
    stereo_low_bass_summing_db: float = 6.0

    room_model: RoomModel = "leaky_pressure_zone"
    leakage_corner_hz: float = 10.0

    alignment_qtc: float = 0.55
    max_box_vas_ratio: float = 4.0
    max_box_volume_per_driver_m3: float | None = None
    f_low: float = 15.0
    f_split: float = 80.0
    manifold_crossover_ceiling_hz: float = 80.0
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
        if not 0 <= self.stereo_low_bass_summing_db <= 6.1:
            raise ValueError("stereo low-bass summing must lie between 0 and 6.1 dB")
        if not 0 < self.preferred_excursion <= 1:
            raise ValueError("preferred excursion must be a fraction in (0, 1]")
        if self.doppler_budget <= 0:
            raise ValueError("doppler budget must be positive")
        if self.alignment_qtc <= 0:
            raise ValueError("alignment Qtc target must be positive")
        if self.max_box_vas_ratio <= 0:
            raise ValueError("maximum box/Vas ratio must be positive")
        if (
            self.max_box_volume_per_driver_m3 is not None
            and self.max_box_volume_per_driver_m3 <= 0
        ):
            raise ValueError("per-driver box-volume limit must be positive")
        if not 0 < self.f_low < self.f_split < self.f_high:
            raise ValueError("frequencies must satisfy 0 < f_low < f_split < f_high")
        if not self.f_low < self.manifold_crossover_ceiling_hz < self.f_high:
            raise ValueError(
                "manifold crossover ceiling must lie between f_low and f_high"
            )
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
        """Fs/Qts that places Fc at f_pz at the preferred alignment Qtc."""
        return physics.max_corner_rate(self.f_pz, self.alignment_qtc)

    @property
    def is_manifold_crossover_valid(self) -> bool:
        """Whether the nominal role split respects the manifold ceiling.

        The ceiling is a declared acoustic-integration constraint, not a
        prediction of the T/S driver model. Role-only evaluations may inspect
        counterfactual splits above it; pair/recipe APIs call
        :meth:`require_valid_manifold_crossover`.
        """
        return self.f_split <= self.manifold_crossover_ceiling_hz

    def require_valid_manifold_crossover(self) -> None:
        """Reject a pair architecture whose nominal split exceeds the
        declared manifold ceiling.

        A real crossover must also provide enough attenuation above the
        manifold's validated operating band; finite-slope filter design is
        outside this scenario model.
        """
        if not self.is_manifold_crossover_valid:
            raise ValueError(
                f"f_split={self.f_split:g} Hz exceeds the nominal manifold "
                f"crossover ceiling of "
                f"{self.manifold_crossover_ceiling_hz:g} Hz"
            )

    def target_spl_for(self, role: Role) -> float:
        """Representative target for role summaries.

        Full-band stereo calculations use :meth:`target_spl_at`, because the
        two channels share the mono low-bass demand below ``f_split`` but
        independently meet the upper-bass target above it.
        """
        if role == "sub":
            return self.sub_target_spl
        if role == "attack":
            return self.attack_target_spl
        if role == "full":
            return max(
                self.sub_target_spl - self.stereo_low_bass_summing_db,
                self.attack_target_spl,
            )
        raise ValueError(f"unknown role {role!r}")

    def target_spl_at(self, f: float, role: Role) -> float:
        """Target SPL carried by one evaluated source/channel at frequency."""
        if f <= 0:
            raise ValueError("frequency must be positive")
        if role == "full":
            if f < self.f_split:
                return (
                    self.sub_target_spl
                    - self.stereo_low_bass_summing_db
                )
            return self.attack_target_spl
        return self.target_spl_for(role)

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

    def box_volume_cap_per_driver(self, vas: float, n_units: int) -> float:
        """Configured per-driver box cap.

        The default is ``max_box_vas_ratio * Vas``. An optional absolute
        per-driver limit can further constrain it. Unit count is validated
        here but does not silently shrink each enclosure as more drivers are
        selected; total volume is an explicit ranking output.
        """
        if vas <= 0 or n_units < 1:
            raise ValueError("Vas and unit count must be positive")
        cap = self.max_box_vas_ratio * vas
        if self.max_box_volume_per_driver_m3 is not None:
            cap = min(cap, self.max_box_volume_per_driver_m3)
        return cap

    def demand_volume(self, f: float, role: Role = "sub") -> float:
        """Peak displaced volume [m^3] needed for the target SPL at f."""
        return physics.demand_volume(f, self.target_spl_at(f, role), self.r_listen,
                                     self.v_room, self.l_max,
                                     room_model=self.room_model,
                                     leakage_corner_hz=self.leakage_corner_hz)

    def v_dem_max(self, role: Role = "sub") -> float:
        """Worst demand at the lower edge of the role's operating band."""
        band_low = self.f_split if role == "attack" else self.f_low
        return self.demand_volume(band_low, role)

    def target_pressure(self, role: Role = "sub", f: float | None = None) -> float:
        target = (
            self.target_spl_for(role)
            if f is None
            else self.target_spl_at(f, role)
        )
        return physics.pressure_from_spl(target)

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
