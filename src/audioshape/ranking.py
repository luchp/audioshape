"""Evaluate drivers against a scenario and rank them.

Pure computation on `Driver` + `Scenario`; no I/O.  The ranking implements the
selection procedure of the paper: feasibility gates first, then sort by
predicted non-correctable distortion at the target (Prop. equivalence:
distortion sorting == headroom maximization).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from audioshape import physics
from audioshape.driver import BoxedDriver, Driver
from audioshape.scenario import Scenario


@dataclass(frozen=True)
class Evaluation:
    """All criteria for one driver (n_units identical units) in one scenario."""

    boxed: BoxedDriver
    scenario: Scenario

    # feasibility
    feasible: bool
    reasons: tuple[str, ...]        # why infeasible (empty when feasible)

    # excursion / distortion at the target SPL
    xi_x: float                     # worst-case excursion utilization
    hd: float                       # motor/suspension HD (eq:HDscale)
    doppler_im: float               # in-band Doppler index (eq:doppler)
    box_hd2: float                  # box air-spring HD2 (eq:boxHD)
    total_distortion: float         # sum of the three above

    # power at the target SPL
    xi_p: float                     # worst-case power utilization P_req/Pmax
    thermal_compression_db: float   # -0.034 xi_P dT (Sec. distortion)

    # ceilings and boundaries (at the listening position)
    spl_sine_floor: float           # sine ceiling at f_low
    spl_burst_floor: float          # burst ceiling at f_low
    f_x: float                      # sine regime boundary (eq:fx)
    f_x_burst: float                # transient boundary (eq:fxburst)

    n_units_required: int           # units to meet the distortion budget

    @property
    def driver(self) -> Driver:
        return self.boxed.driver

    def sort_key(self) -> tuple:
        """Feasible first, then lowest total distortion, then fewest units."""
        return (not self.feasible, self.total_distortion, self.n_units_required)


def evaluate(driver: Driver, scenario: Scenario, n_units: int = 1,
            band_low: float | None = None, band_high: float | None = None,
            doppler_ref: float | None = None) -> Evaluation:
    """Evaluate one driver (n identical units) against the scenario, over the
    band [band_low, band_high] (default: [f_low, f_high], the whole range --
    i.e. this driver alone covers everything).

    Pass a role-restricted band to score a driver for only part of the range,
    e.g. band=(f_low, f_split) for a sub or (f_split, f_high) for an attack
    driver in a two-driver system (see pair_rank).  Excursion demand is worst
    at band_low (the demand curve falls with frequency and is flat below
    f_pz, eq:demand).  Doppler is self-Doppler within the driver's own band:
    its own excursion modulating `doppler_ref` (default: band_high, the top
    of its own passband) -- no other driver's excursion is involved, since a
    physically separate sub and tower do not couple acoustically.
    """
    sc = scenario
    band_low = sc.f_low if band_low is None else band_low
    band_high = sc.f_high if band_high is None else band_high
    doppler_ref = sc.f_split if doppler_ref is None else doppler_ref
    reasons: list[str] = []

    if driver.qts >= sc.qtc:
        return _infeasible(driver, sc, n_units, band_low,
                           f"Qts={driver.qts:.2f} >= target Qtc={sc.qtc:.2f}")

    boxed = BoxedDriver(driver, qtc=sc.qtc, n_units=n_units)

    # --- corner-rate gate (eq:Fsrule): only binds the driver that must ----
    # reach into the room's pressure zone; an attack driver starting above
    # f_pz does not need its box corner placed there.
    if band_low <= sc.f_pz and driver.corner_rate > sc.max_corner_rate:
        reasons.append(
            f"Fs/Qts={driver.corner_rate:.0f} Hz > "
            f"f_pz/Qtc={sc.max_corner_rate:.0f} Hz (corner cannot reach f_pz)")

    # --- inductance corner must clear this driver's own band -----------
    if driver.f_le < band_high:
        reasons.append(
            f"f_L={driver.f_le:.0f} Hz inside band (< {band_high:.0f} Hz)")

    # --- excursion and distortion at the target, within this band ------
    v_dem = sc.demand_volume(band_low)
    xi_x = physics.excursion_utilization(v_dem, boxed.vd_total)
    hd = physics.harmonic_distortion(xi_x)

    x1 = min(xi_x, 1.0) * driver.xmax  # own excursion, capped
    doppler = physics.doppler_im(doppler_ref, x1)

    box_hd = physics.box_hd2(min(v_dem / n_units, driver.vd),
                             boxed.vb, driver.qts, sc.qtc)
    total = hd + doppler + box_hd

    if xi_x > 1.0:
        reasons.append(f"excursion clip: xi_x={xi_x:.2f} > 1 at "
                       f"{sc.target_spl:.0f} dB, {band_low:.0f} Hz")

    # --- power at the target (EQ tax, worst at the bottom of the band) --
    p_t = sc.target_pressure
    w_ac = physics.acoustic_power_halfspace(p_t, sc.r_listen)
    p_passband = w_ac / (driver.eta0 * n_units * n_units)
    p_req = physics.eq_tax_power(max(band_low, sc.f_pz), p_passband,
                                 boxed.wc, driver.sigma_m)
    xi_p = p_req / driver.p_max
    if xi_p > 1.0:
        reasons.append(f"thermal clip: xi_P={xi_p:.2f} > 1")

    # --- ceilings at the listening position, referenced to band_low -----
    spl_sine = _room_spl_ceiling(boxed.vd_total, sc, band_low, shape_factor=1.0)
    spl_burst = _room_spl_ceiling(boxed.vd_total, sc, band_low,
                                  shape_factor=sc.burst_shape)

    f_x = physics.regime_boundary_fx(driver.fs, driver.p_max, driver.qes,
                                     driver.mms, driver.xmax)
    f_x_burst = physics.burst_boundary_fx(f_x, sc.burst_headroom, sc.burst_shape)

    return Evaluation(
        boxed=boxed, scenario=sc,
        feasible=not reasons, reasons=tuple(reasons),
        xi_x=xi_x, hd=hd, doppler_im=doppler, box_hd2=box_hd,
        total_distortion=total,
        xi_p=xi_p,
        thermal_compression_db=physics.thermal_compression_db(min(xi_p, 1.0)),
        spl_sine_floor=spl_sine, spl_burst_floor=spl_burst,
        f_x=f_x, f_x_burst=f_x_burst,
        n_units_required=sc.units_required(driver.vd, band_low),
    )


def rank(drivers: list[Driver], scenario: Scenario, n_units: int = 1,
         min_size_in: float = 0.0, max_size_in: float = float("inf"),
         band_low: float | None = None, band_high: float | None = None,
         doppler_ref: float | None = None) -> list[Evaluation]:
    """Evaluate a size category and sort: feasible first, lowest total
    non-correctable distortion first."""
    evals = [evaluate(d, scenario, n_units, band_low, band_high, doppler_ref)
             for d in drivers if min_size_in <= d.size_in <= max_size_in]
    return sorted(evals, key=Evaluation.sort_key)


@dataclass(frozen=True)
class PairEvaluation:
    """A sub driver (own band [f_low, f_split]) paired with an attack driver
    (own band [f_split, f_high]); no cross-driver coupling term, since the
    two are physically separate sources (soffit-wall sub manifold vs.
    mid/high tower)."""

    sub: Evaluation
    attack: Evaluation

    @property
    def feasible(self) -> bool:
        return self.sub.feasible and self.attack.feasible

    @property
    def total_distortion(self) -> float:
        return self.sub.total_distortion + self.attack.total_distortion

    @property
    def n_units_required(self) -> tuple[int, int]:
        return (self.sub.n_units_required, self.attack.n_units_required)

    def sort_key(self) -> tuple:
        return (not self.feasible, self.total_distortion,
                sum(self.n_units_required))


def pair_rank(drivers: list[Driver], scenario: Scenario,
             sub_units: int = 1, attack_units: int = 1,
             sub_size_min: float = 0.0, sub_size_max: float = float("inf"),
             attack_size_min: float = 0.0, attack_size_max: float = float("inf"),
             top_k_each: int = 15) -> list[PairEvaluation]:
    """Rank sub and attack candidates independently within their own bands
    and size windows, then combine the top-K of each into pairs sorted by
    combined distortion (feasible pairs first)."""
    sc = scenario
    sub_evals = rank(drivers, sc, n_units=sub_units,
                     min_size_in=sub_size_min, max_size_in=sub_size_max,
                     band_low=sc.f_low, band_high=sc.f_split,
                     doppler_ref=sc.f_split)[:top_k_each]
    attack_evals = rank(drivers, sc, n_units=attack_units,
                        min_size_in=attack_size_min, max_size_in=attack_size_max,
                        band_low=sc.f_split, band_high=sc.f_high,
                        doppler_ref=sc.f_high)[:top_k_each]
    pairs = [PairEvaluation(s, a) for s in sub_evals for a in attack_evals]
    return sorted(pairs, key=PairEvaluation.sort_key)


def _room_spl_ceiling(vd_total: float, sc: Scenario, band_low: float,
                      shape_factor: float) -> float:
    """Ceiling SPL at the listening position at band_low, including the room
    pressure zone: the SPL at which V_dem(band_low) equals Vd/C."""
    v_dem_unit = sc.demand_volume(band_low) / sc.target_pressure  # per Pa
    p_max_rms = (vd_total / shape_factor) / v_dem_unit
    return physics.spl_from_pressure(p_max_rms)


def _infeasible(driver: Driver, sc: Scenario, n_units: int, band_low: float,
                reason: str) -> Evaluation:
    """Driver cannot form the requested box at all."""
    inf = float("inf")
    boxed = BoxedDriver(driver, qtc=max(sc.qtc, driver.qts * 1.01) + 1e-9,
                        n_units=n_units)
    return Evaluation(
        boxed=boxed, scenario=sc, feasible=False, reasons=(reason,),
        xi_x=inf, hd=inf, doppler_im=inf, box_hd2=inf, total_distortion=inf,
        xi_p=inf, thermal_compression_db=-inf,
        spl_sine_floor=-inf, spl_burst_floor=-inf,
        f_x=math.nan, f_x_burst=math.nan,
        n_units_required=sc.units_required(driver.vd, band_low),
    )
