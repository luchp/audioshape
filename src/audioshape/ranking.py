"""Evaluate and rank sealed-box candidates for the declared system.

The core policy is deliberately non-compensatory:

1. reject candidates that cross a physical or electrical limit;
2. assign Pareto fronts using separate, named risk indicators;
3. apply a declared role-specific lexicographic order.

Unlike mechanisms are never added into a synthetic distortion percentage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable, Literal

from audioshape import physics
from audioshape.driver import BoxedDriver, Driver
from audioshape.scenario import Scenario

Role = Literal["sub", "attack", "full"]

@dataclass(frozen=True)
class RiskVector:
    """Separate minimization objectives; no arithmetic total is defined."""

    steady_excursion: float
    transient_excursion: float
    doppler_modulation: float
    box_spring: float
    driver_power: float
    amplifier_voltage: float
    amplifier_current: float
    amplifier_continuous_power: float
    amplifier_burst_power: float
    inductive: float
    box_volume_m3: float

    def pareto_values(self, role: Role) -> tuple[float, ...]:
        """Core non-compensatory objectives used for Pareto fronts.

        The complete vector remains available for reporting. Keeping the
        front to the main role tradeoffs avoids the degenerate result where
        nearly every candidate is non-dominated in eleven dimensions.
        """
        excursion = max(self.steady_excursion, self.transient_excursion)
        amplifier = max(
            self.amplifier_voltage,
            self.amplifier_current,
            self.amplifier_continuous_power,
            self.amplifier_burst_power,
        )
        if role == "sub":
            return excursion, amplifier, self.box_volume_m3
        return (
            excursion,
            self.doppler_modulation,
            amplifier,
            self.box_volume_m3,
        )

    def dominates(
        self,
        other: "RiskVector",
        role: Role,
        tolerance: float = 1e-12,
    ) -> bool:
        """Whether every core role risk is no worse and one is better."""
        pairs = tuple(zip(
            self.pareto_values(role),
            other.pareto_values(role),
        ))
        no_worse = all(left <= right + tolerance for left, right in pairs)
        strictly_better = any(left < right - tolerance for left, right in pairs)
        return no_worse and strictly_better


@dataclass(frozen=True)
class ElectricalExtrema:
    """Worst steady-sine per-driver demand across the complete role band."""

    voltage_rms: float
    voltage_frequency: float
    current_rms: float
    current_frequency: float
    coil_power_w: float
    power_frequency: float


@dataclass(frozen=True)
class TransientExtrema:
    """Worst declared finite-burst results across the sampled role band."""

    shape_factor: float
    shape_frequency: float
    displacement_peak: float
    displacement_frequency: float
    voltage_rms: float
    voltage_frequency: float
    current_rms: float
    current_frequency: float
    coil_power_w: float
    power_frequency: float


@dataclass(frozen=True)
class Evaluation:
    boxed: BoxedDriver
    scenario: Scenario
    role: Role
    band_low: float
    band_high: float

    feasible: bool
    reasons: tuple[str, ...]
    notes: tuple[str, ...]

    risk: RiskVector
    electrical: ElectricalExtrema
    transient: TransientExtrema

    spl_sine_floor: float
    spl_transient_floor: float
    steady_excursion_frequency: float
    f_x: float
    n_units_preferred: int
    pareto_rank: int = -1

    @property
    def driver(self) -> Driver:
        return self.boxed.driver

    @property
    def eta0_pmax(self) -> float:
        return self.driver.eta0 * self.driver.p_max

    @property
    def xi_x(self) -> float:
        return self.risk.steady_excursion

    @property
    def xi_x_transient(self) -> float:
        return self.risk.transient_excursion

    @property
    def doppler_im(self) -> float:
        return self.risk.doppler_modulation

    @property
    def box_nonlinearity(self) -> float:
        return self.risk.box_spring

    @property
    def xi_p(self) -> float:
        return self.risk.driver_power

    @property
    def amplifier_utilization(self) -> float:
        return max(
            self.risk.amplifier_voltage,
            self.risk.amplifier_current,
            self.risk.amplifier_continuous_power,
            self.risk.amplifier_burst_power,
        )

    @property
    def is_preferred_excursion(self) -> bool:
        return max(self.xi_x, self.xi_x_transient) <= (
            self.scenario.preferred_excursion
        )

    def policy_key(self) -> tuple:
        """Role-specific order used only within a Pareto front."""
        excursion = max(self.xi_x, self.xi_x_transient)
        electrical = max(self.xi_p, self.amplifier_utilization)
        if self.role == "attack":
            inductance_unknown = not self.driver.has_inductance
            f_le_order = -self.driver.f_le if self.driver.has_inductance else 0.0
            return (
                not self.is_preferred_excursion,
                -self.eta0_pmax,
                inductance_unknown,
                f_le_order,
                excursion,
                electrical,
                self.risk.box_volume_m3,
                self.driver.label(),
            )
        return (
            excursion,
            self.xi_x,
            self.doppler_im,
            electrical,
            self.box_nonlinearity,
            self.risk.box_volume_m3,
            self.n_units_preferred,
            self.driver.label(),
        )

    def sort_key(self) -> tuple:
        pareto = self.pareto_rank if self.pareto_rank >= 0 else math.inf
        return (not self.feasible, pareto, self.policy_key())


def evaluate(
    driver: Driver,
    scenario: Scenario,
    n_units: int = 1,
    band_low: float | None = None,
    band_high: float | None = None,
    doppler_ref: float | None = None,
    role: Role = "sub",
) -> Evaluation:
    """Evaluate one mono manifold or one independent stereo channel.

    ``n_units`` identical drivers share that role/channel's acoustic target.
    Each physical driver is checked against the full per-driver amplifier
    envelope. The attack role receives no opposite-channel credit. The full
    role uses the scenario's configured coherent mono-bass credit below the
    split and the independent-channel target above it.
    """
    if n_units < 1:
        raise ValueError("n_units must be at least one")

    sc = scenario
    band_low = sc.f_low if band_low is None else band_low
    band_high = sc.f_high if band_high is None else band_high
    doppler_ref = band_high if doppler_ref is None else doppler_ref
    if not 0 < band_low < band_high:
        raise ValueError("evaluation band must satisfy 0 < low < high")

    reasons: list[str] = []
    notes: list[str] = []

    f_target = sc.target_corner_hz(role)
    desired_qtc = physics.qtc_for_target_corner(
        driver.corner_rate, f_target, sc.alignment_qtc
    )
    box_cap = sc.box_volume_cap_per_driver(driver.vas, n_units)
    desired_box = math.inf
    if desired_qtc > driver.qts:
        desired_box = physics.box_volume_for_qtc(
            driver.vas, driver.qts, desired_qtc
        )
    if desired_box > box_cap:
        qtc = physics.qtc_for_box_volume(driver.vas, driver.qts, box_cap)
        cap_description = f"{sc.max_box_vas_ratio:g}x Vas"
        if sc.max_box_volume_per_driver_m3 is not None:
            cap_description += (
                f", capped at {sc.max_box_volume_per_driver_m3*1e3:.0f} L"
            )
        notes.append(
            f"preferred alignment not reached within {cap_description}; "
            f"using {box_cap*1e3:.0f} L per driver gives "
            f"Qtc={qtc:.2f}, Fc={physics.fc_for_qtc(driver.fs, driver.qts, qtc):.1f} Hz"
        )
    else:
        qtc = desired_qtc

    boxed = BoxedDriver(driver, qtc=qtc, n_units=n_units)
    if not driver.has_force_factor:
        notes.append("Bl missing; derived from Fs, Qes, Mms, and Re")
    if not driver.has_inductance:
        notes.append("Le missing; inductive screening remains uncertain")
    elif driver.f_le < band_high:
        reasons.append(
            f"inductive screen f_L={driver.f_le:.0f} Hz < {band_high:.0f} Hz"
        )

    steady_grid = _frequency_grid(
        band_low,
        band_high,
        samples=121,
        extras=(boxed.fc, sc.f_pz, sc.leakage_corner_hz),
    )
    steady_excursion = 0.0
    maximum_displacement = (0.0, band_low)
    maximum_volume_per_driver = 0.0
    voltage = (0.0, band_low)
    current = (0.0, band_low)
    power = (0.0, band_low)
    sigma_total = boxed.wc / boxed.qtc

    for frequency in steady_grid:
        volume_total = sc.demand_volume(frequency, role)
        displacement = volume_total / (n_units * driver.sd)
        excursion = displacement / driver.xmax
        steady_excursion = max(steady_excursion, excursion)
        maximum_displacement = _larger_extreme(
            maximum_displacement, displacement, frequency
        )
        maximum_volume_per_driver = max(
            maximum_volume_per_driver, volume_total / n_units
        )

        power_at_frequency = physics.power_at_excursion_limit(
            frequency,
            driver.mms,
            driver.qes,
            driver.fs,
            displacement,
            boxed.wc,
            driver.sigma_m,
        )
        current_at_frequency = physics.current_at_excursion_limit(
            frequency,
            driver.mms,
            driver.qes,
            driver.fs,
            driver.re,
            displacement,
            boxed.wc,
            driver.sigma_m,
        )
        voltage_at_frequency = physics.voltage_at_excursion_limit(
            frequency,
            driver.mms,
            driver.effective_bl,
            driver.re,
            displacement,
            boxed.wc,
            sigma_total,
        )
        voltage = _larger_extreme(voltage, voltage_at_frequency, frequency)
        current = _larger_extreme(current, current_at_frequency, frequency)
        power = _larger_extreme(power, power_at_frequency, frequency)

    electrical = ElectricalExtrema(
        voltage_rms=voltage[0],
        voltage_frequency=voltage[1],
        current_rms=current[0],
        current_frequency=current[1],
        coil_power_w=power[0],
        power_frequency=power[1],
    )

    transient_grid = _frequency_grid(
        band_low,
        band_high,
        samples=5,
        extras=(boxed.fc, sc.f_pz, sc.leakage_corner_hz),
    )
    transient_shape = (0.0, band_low)
    transient_displacement = (0.0, band_low)
    transient_voltage = (0.0, band_low)
    transient_current = (0.0, band_low)
    transient_power = (0.0, band_low)
    for frequency in transient_grid:
        x_sine_peak = (
            sc.demand_volume(frequency, role) / (n_units * driver.sd)
        )
        burst = physics.sealed_burst_requirements(
            frequency,
            x_sine_peak,
            boxed.fc,
            boxed.qtc,
            driver.mms,
            driver.re,
            driver.effective_bl,
            cycles=sc.transient_cycles,
            window=sc.transient_window,
            phase_samples=sc.transient_phase_samples,
        )
        transient_shape = _larger_extreme(
            transient_shape, burst.shape_factor, frequency
        )
        transient_displacement = _larger_extreme(
            transient_displacement, burst.displacement_peak, frequency
        )
        transient_voltage = _larger_extreme(
            transient_voltage, burst.voltage_rms, frequency
        )
        transient_current = _larger_extreme(
            transient_current, burst.current_rms, frequency
        )
        transient_power = _larger_extreme(
            transient_power, burst.coil_power_w, frequency
        )

    transient = TransientExtrema(
        shape_factor=transient_shape[0],
        shape_frequency=transient_shape[1],
        displacement_peak=transient_displacement[0],
        displacement_frequency=transient_displacement[1],
        voltage_rms=transient_voltage[0],
        voltage_frequency=transient_voltage[1],
        current_rms=transient_current[0],
        current_frequency=transient_current[1],
        coil_power_w=transient_power[0],
        power_frequency=transient_power[1],
    )

    transient_excursion = transient.displacement_peak / driver.xmax
    doppler = physics.doppler_im(doppler_ref, maximum_displacement[0])
    box_spring = physics.box_spring_nonlinearity(
        maximum_volume_per_driver, boxed.vb, driver.qts, boxed.qtc
    )
    inductive = (
        band_high / driver.f_le if driver.has_inductance else 1.0
    )
    risk = RiskVector(
        steady_excursion=steady_excursion,
        transient_excursion=transient_excursion,
        doppler_modulation=doppler,
        box_spring=box_spring,
        driver_power=electrical.coil_power_w / driver.p_max,
        amplifier_voltage=max(
            electrical.voltage_rms, transient.voltage_rms
        ) / sc.amplifier_voltage_rms,
        amplifier_current=max(
            electrical.current_rms, transient.current_rms
        ) / sc.amplifier_current_rms,
        amplifier_continuous_power=(
            electrical.coil_power_w / sc.amplifier_power_continuous
        ),
        amplifier_burst_power=(
            transient.coil_power_w / sc.amplifier_power_burst
        ),
        inductive=inductive,
        box_volume_m3=n_units * boxed.vb,
    )

    if risk.steady_excursion > 1.0:
        reasons.append(
            f"steady excursion clip xi_x={risk.steady_excursion:.2f} "
            f"at {maximum_displacement[1]:.1f} Hz / "
            f"{sc.target_spl_at(maximum_displacement[1], role):.0f} dB"
        )
    if risk.transient_excursion > 1.0:
        reasons.append(
            f"transient excursion clip xi_x={risk.transient_excursion:.2f} "
            f"at {transient.displacement_frequency:.1f} Hz / "
            f"{sc.target_spl_at(transient.displacement_frequency, role):.0f} dB"
        )
    if role in ("attack", "full") and doppler > sc.doppler_budget:
        reasons.append(
            f"Doppler sideband ratio={doppler:.3f} > "
            f"{sc.doppler_budget:.3f}"
        )
    if risk.driver_power > 1.0:
        reasons.append(
            f"driver power={electrical.coil_power_w:.0f} W > "
            f"{driver.p_max:.0f} W"
        )
    if electrical.voltage_rms > sc.amplifier_voltage_rms:
        reasons.append(
            f"steady voltage={electrical.voltage_rms:.1f} V rms > "
            f"{sc.amplifier_voltage_rms:.1f} V rms"
        )
    if electrical.current_rms > sc.amplifier_current_rms:
        reasons.append(
            f"steady current={electrical.current_rms:.1f} A rms > "
            f"{sc.amplifier_current_rms:.1f} A rms"
        )
    if electrical.coil_power_w > sc.amplifier_power_continuous:
        reasons.append(
            f"steady power={electrical.coil_power_w:.0f} W > "
            f"{sc.amplifier_power_continuous:.0f} W continuous"
        )
    if transient.voltage_rms > sc.amplifier_voltage_rms:
        reasons.append(
            f"burst voltage={transient.voltage_rms:.1f} V rms > "
            f"{sc.amplifier_voltage_rms:.1f} V rms"
        )
    if transient.current_rms > sc.amplifier_current_rms:
        reasons.append(
            f"burst current={transient.current_rms:.1f} A rms > "
            f"{sc.amplifier_current_rms:.1f} A rms"
        )
    if transient.coil_power_w > sc.amplifier_power_burst:
        reasons.append(
            f"burst power={transient.coil_power_w:.0f} W > "
            f"{sc.amplifier_power_burst:.0f} W"
        )

    sine_floor = sc.target_spl_at(
        maximum_displacement[1], role
    ) + 20.0 * math.log10(
        1.0 / steady_excursion
    )
    transient_floor = sc.target_spl_at(
        transient.displacement_frequency, role
    ) + 20.0 * math.log10(
        1.0 / transient_excursion
    )
    preferred_units = max(
        1,
        math.ceil(
            n_units
            * max(steady_excursion, transient_excursion)
            / sc.preferred_excursion
        ),
    )

    return Evaluation(
        boxed=boxed,
        scenario=sc,
        role=role,
        band_low=band_low,
        band_high=band_high,
        feasible=not reasons,
        reasons=tuple(reasons),
        notes=tuple(notes),
        risk=risk,
        electrical=electrical,
        transient=transient,
        spl_sine_floor=sine_floor,
        spl_transient_floor=transient_floor,
        steady_excursion_frequency=maximum_displacement[1],
        f_x=physics.regime_boundary_fx(
            driver.fs,
            driver.p_max,
            driver.qes,
            driver.mms,
            driver.xmax,
        ),
        n_units_preferred=preferred_units,
    )


def rank(
    drivers: list[Driver],
    scenario: Scenario,
    n_units: int = 1,
    min_size_in: float = 0.0,
    max_size_in: float = float("inf"),
    band_low: float | None = None,
    band_high: float | None = None,
    doppler_ref: float | None = None,
    role: Role = "sub",
) -> list[Evaluation]:
    evaluations = [
        evaluate(
            driver,
            scenario,
            n_units,
            band_low,
            band_high,
            doppler_ref,
            role,
        )
        for driver in drivers
        if min_size_in <= driver.size_in <= max_size_in
    ]
    evaluations = _assign_pareto_ranks(evaluations)
    return sorted(evaluations, key=Evaluation.sort_key)


@dataclass(frozen=True)
class PairEvaluation:
    """A mono low-bass manifold plus one candidate per stereo channel."""

    sub: Evaluation
    attack: Evaluation

    @property
    def feasible(self) -> bool:
        return self.sub.feasible and self.attack.feasible

    @property
    def physical_driver_count(self) -> int:
        return self.sub.boxed.n_units + 2 * self.attack.boxed.n_units

    @property
    def total_box_volume_m3(self) -> float:
        return (
            self.sub.risk.box_volume_m3
            + 2.0 * self.attack.risk.box_volume_m3
        )

    def sort_key(self) -> tuple:
        sub_pareto = self.sub.pareto_rank if self.sub.pareto_rank >= 0 else math.inf
        attack_pareto = (
            self.attack.pareto_rank
            if self.attack.pareto_rank >= 0
            else math.inf
        )
        return (
            not self.feasible,
            sub_pareto,
            attack_pareto,
            self.sub.policy_key(),
            self.attack.policy_key(),
            self.physical_driver_count,
            round(self.total_box_volume_m3, 6),
        )


def pair_rank(
    drivers: list[Driver],
    scenario: Scenario,
    sub_units: int = 1,
    attack_units: int = 1,
    sub_size_min: float = 0.0,
    sub_size_max: float = float("inf"),
    attack_size_min: float = 0.0,
    attack_size_max: float = float("inf"),
    top_k_each: int = 15,
    require_even_sub_units: bool = False,
) -> list[PairEvaluation]:
    """Combine independently ranked role candidates without summing risks."""
    scenario.require_valid_manifold_crossover()
    if require_even_sub_units and sub_units % 2:
        raise ValueError(
            "force-cancelling sub manifolds require an even unit count"
        )
    sc = scenario
    sub_evaluations = rank(
        drivers,
        sc,
        n_units=sub_units,
        min_size_in=sub_size_min,
        max_size_in=sub_size_max,
        band_low=sc.f_low,
        band_high=sc.f_split,
        doppler_ref=sc.f_split,
        role="sub",
    )[:top_k_each]
    attack_evaluations = rank(
        drivers,
        sc,
        n_units=attack_units,
        min_size_in=attack_size_min,
        max_size_in=attack_size_max,
        band_low=sc.f_split,
        band_high=sc.f_high,
        doppler_ref=sc.f_high,
        role="attack",
    )[:top_k_each]
    pairs = [
        PairEvaluation(sub, attack)
        for sub in sub_evaluations
        for attack in attack_evaluations
    ]
    return sorted(pairs, key=PairEvaluation.sort_key)


def pareto_front(evaluations: Iterable[Evaluation]) -> list[Evaluation]:
    """Return the feasible non-dominated evaluations."""
    candidates = [evaluation for evaluation in evaluations if evaluation.feasible]
    return [
        candidate
        for candidate in candidates
        if not any(
            other.risk.dominates(candidate.risk, candidate.role)
            for other in candidates
            if other is not candidate
        )
    ]


def _assign_pareto_ranks(
    evaluations: list[Evaluation],
) -> list[Evaluation]:
    feasible_indices = [
        index for index, evaluation in enumerate(evaluations)
        if evaluation.feasible
    ]
    dominates: dict[int, list[int]] = {index: [] for index in feasible_indices}
    domination_count = {index: 0 for index in feasible_indices}

    for position, left_index in enumerate(feasible_indices):
        left = evaluations[left_index]
        for right_index in feasible_indices[position + 1:]:
            right = evaluations[right_index]
            if left.risk.dominates(right.risk, left.role):
                dominates[left_index].append(right_index)
                domination_count[right_index] += 1
            elif right.risk.dominates(left.risk, right.role):
                dominates[right_index].append(left_index)
                domination_count[left_index] += 1

    current_front = [
        index for index in feasible_indices if domination_count[index] == 0
    ]
    ranks: dict[int, int] = {}
    rank_index = 1
    while current_front:
        next_front: list[int] = []
        for index in current_front:
            ranks[index] = rank_index
            for dominated_index in dominates[index]:
                domination_count[dominated_index] -= 1
                if domination_count[dominated_index] == 0:
                    next_front.append(dominated_index)
        current_front = next_front
        rank_index += 1

    return [
        replace(evaluation, pareto_rank=ranks.get(index, -1))
        for index, evaluation in enumerate(evaluations)
    ]


def _frequency_grid(
    low: float,
    high: float,
    *,
    samples: int,
    extras: Iterable[float] = (),
) -> tuple[float, ...]:
    if samples < 2:
        raise ValueError("frequency grid needs at least two samples")
    ratio = high / low
    values = {
        low * ratio ** (index / (samples - 1))
        for index in range(samples)
    }
    values.update(value for value in extras if low <= value <= high)
    return tuple(sorted(values))


def _larger_extreme(
    current: tuple[float, float],
    candidate_value: float,
    candidate_frequency: float,
) -> tuple[float, float]:
    if candidate_value > current[0]:
        return candidate_value, candidate_frequency
    return current
