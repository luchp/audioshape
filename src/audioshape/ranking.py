"""Evaluate drivers against a scenario and rank them.

Pure computation on `Driver` + `Scenario`; no I/O.  The ranking implements the
selection procedure of the paper: feasibility gates first, then sort by
predicted non-correctable distortion at the target (Prop. equivalence:
distortion sorting == headroom maximization).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from audioshape import physics
from audioshape.driver import BoxedDriver, Driver
from audioshape.scenario import Scenario

Role = Literal["sub", "attack"]


@dataclass(frozen=True)
class Evaluation:
    """All criteria for one driver (n_units identical units, n_channels
    coherent-signal channels) in one scenario."""

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
    n_channels: int = 1             # coherent-signal channels (see evaluate())
    role: Role = "sub"              # which selection rule sort_key() applies

    @property
    def driver(self) -> Driver:
        return self.boxed.driver

    @property
    def eta0_pmax(self) -> float:
        """Dissipation invariant eta0 * Pmax (sec_procedure.tex step 2):
        the attack role's primary ranking criterion."""
        return self.driver.eta0 * self.driver.p_max

    def sort_key(self) -> tuple:
        """Paper's selection procedure (sec_procedure.tex / worked-example
        ranking): feasible first, then role-specific ordering.

        Sub role: maximize Vd (displacement) -- equivalent to minimizing
        total_distortion, since D(xi_x) is monotonic in 1/Vd; tie-break on
        Sd (Doppler) and eta0*Pmax (both captured by total_distortion's
        Doppler term and the thermal gate) -- unchanged from before.

        Attack role: maximize eta0*Pmax (dissipation invariant), subject to
        its own excursion utilization and Doppler index staying within
        budget (enforced as feasibility gates in evaluate(), not scored
        here); tie-break on f_L (inductance corner, higher is better) and
        fewest units required."""
        if self.role == "attack":
            return (not self.feasible, -self.eta0_pmax, -self.driver.f_le,
                    self.n_units_required)
        return (not self.feasible, self.total_distortion, self.n_units_required)


def evaluate(driver: Driver, scenario: Scenario, n_units: int = 1,
            band_low: float | None = None, band_high: float | None = None,
            doppler_ref: float | None = None, n_channels: int = 1,
            role: Role = "sub") -> Evaluation:
    """Evaluate one driver (n identical units, on n_channels coherent-signal
    channels) against the scenario, over the band [band_low, band_high]
    (default: [f_low, f_high], the whole range -- i.e. this driver alone
    covers everything).

    `role` selects the ranking rule applied by `Evaluation.sort_key()`
    (sec_procedure.tex): "sub" maximizes Vd (displacement); "attack"
    maximizes eta0*Pmax (dissipation) subject to its own excursion and
    Doppler index staying within budget -- for "attack", exceeding either
    budget here makes the driver infeasible (reasons), it is not merely
    scored worse.

    Pass a role-restricted band to score a driver for only part of the range,
    e.g. band=(f_low, f_split) for a sub or (f_split, f_high) for an attack
    driver in a two-driver system (see pair_rank).  Excursion demand is worst
    at band_low (the demand curve falls with frequency and is flat below
    f_pz, eq:demand).  Doppler is self-Doppler within the driver's own band:
    its own excursion modulating `doppler_ref` (default: band_high, the top
    of its own passband) -- no other driver's excursion is involved, since a
    physically separate sub and tower do not couple acoustically.

    `n_units` is the number of identical drivers sharing one electrical
    channel (a manifold): they add coherently both acoustically (Vd -> N*Vd)
    and electrically (A9's N^2 power-sharing, since they share one signal
    and one amplifier channel). `n_channels` is the number of separate,
    independently-driven channels (e.g. stereo L/R) carrying a common-signal
    (e.g. mono-panned) program: for genuinely correlated content these also
    sum acoustically at the listening position, but each channel's own
    driver(s) still only ever dissipate their own channel's power -- no
    electrical/thermal credit crosses channels. So `n_units` scales both the
    acoustic and thermal budget (A9 array), while `n_channels` scales only
    the acoustic budget.
    """
    sc = scenario
    band_low = sc.f_low if band_low is None else band_low
    band_high = sc.f_high if band_high is None else band_high
    doppler_ref = sc.f_split if doppler_ref is None else doppler_ref
    reasons: list[str] = []

    # --- Qtc ceiling vs. this role's own corner target (eq:Fsrule) -------
    # Use the smaller of the configured ceiling and whatever Qtc lands Fc
    # exactly at this role's own corner (f_pz for sub/full, f_split for
    # attack): undershooting a target corner is free (EQ cut), overshooting
    # it is taxed (EQ boost, costs excursion), so never pin every driver to
    # the same fixed ceiling when a lower Qtc (bigger box) both avoids the
    # overshoot and is still available under it. This one check subsumes
    # the old "Qts >= ceiling" gate: Fs >= f_target forces qtc <= driver.qts
    # regardless of the ceiling, so it is caught here too; a driver already
    # compliant at the fixed ceiling gets qtc == sc.qtc, unchanged.
    f_target = sc.target_corner_hz(role)
    qtc = physics.qtc_for_target_corner(driver.corner_rate, f_target, sc.qtc)
    if driver.qts >= qtc:
        corner_name = "f_sp" if role == "attack" else "f_pz"
        return _infeasible(
            driver, sc, n_units, band_low,
            f"Qts={driver.qts:.2f} >= usable Qtc={qtc:.2f} "
            f"(ceiling {sc.qtc:.2f}, {corner_name}={f_target:.0f} Hz)",
            n_channels=n_channels, role=role)

    boxed = BoxedDriver(driver, qtc=qtc, n_units=n_units)
    n_acoustic = n_units * n_channels  # coherent-sum units across channels

    # --- inductance corner must clear this driver's own band -----------
    if driver.f_le < band_high:
        reasons.append(
            f"f_L={driver.f_le:.0f} Hz inside band (< {band_high:.0f} Hz)")

    # --- excursion and distortion at the target, within this band ------
    # (acoustic totals: coherent sum across both units-per-channel and
    # channels -- see docstring above)
    v_dem = sc.demand_volume(band_low, role)
    vd_acoustic_total = n_channels * boxed.vd_total
    xi_x = physics.excursion_utilization(v_dem, vd_acoustic_total)
    hd = physics.harmonic_distortion(xi_x)

    x1 = min(xi_x, 1.0) * driver.xmax  # own excursion, capped
    doppler = physics.doppler_im(doppler_ref, x1)

    box_hd = physics.box_hd2(min(v_dem / n_acoustic, driver.vd),
                             boxed.vb, driver.qts, qtc)
    total = hd + doppler + box_hd

    if xi_x > 1.0:
        reasons.append(f"excursion clip: xi_x={xi_x:.2f} > 1 at "
                       f"{sc.target_spl_for(role):.0f} dB, {band_low:.0f} Hz")

    # --- attack-role budget gates (sec_procedure.tex step 4/2): the paper
    # ranks attack drivers by maximizing eta0*Pmax "subject to" their own
    # excursion utilization and Doppler index staying within budget -- i.e.
    # these are feasibility gates for this role, not scored terms.
    if role == "attack":
        if xi_x > sc.utilization_budget:
            reasons.append(
                f"xi_x={xi_x:.3f} > distortion budget xi*="
                f"{sc.utilization_budget:.3f}")
        if doppler > sc.doppler_budget:
            reasons.append(
                f"Doppler index={doppler:.3f} > budget D*_IM="
                f"{sc.doppler_budget:.3f}")

    # --- power at the target (EQ tax, worst at the bottom of the band) --
    # Thermal/power budget is per-physical-driver only: n_units (same
    # electrical channel/manifold) gets A9's N^2 power-sharing since those
    # drivers share one amplifier signal, but n_channels never does --
    # each channel's own driver(s) dissipate only their own channel's power,
    # heat never crosses channels (this session's explicit design decision).
    p_t = sc.target_pressure(role)
    w_ac = physics.acoustic_power_halfspace(p_t, sc.r_listen)
    p_passband = w_ac / (driver.eta0 * n_units * n_units)
    p_req = physics.eq_tax_power(max(band_low, sc.f_pz), p_passband,
                                 boxed.wc, driver.sigma_m)
    xi_p = p_req / driver.p_max
    if xi_p > 1.0:
        reasons.append(f"thermal clip: xi_P={xi_p:.2f} > 1")

    # --- ceilings at the listening position, referenced to band_low -----
    spl_sine = _room_spl_ceiling(vd_acoustic_total, sc, band_low, role,
                                 shape_factor=1.0)
    spl_burst = _room_spl_ceiling(vd_acoustic_total, sc, band_low, role,
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
        n_units_required=sc.units_required(driver.vd, band_low, role),
        n_channels=n_channels,
        role=role,
    )


def rank(drivers: list[Driver], scenario: Scenario, n_units: int = 1,
         min_size_in: float = 0.0, max_size_in: float = float("inf"),
         band_low: float | None = None, band_high: float | None = None,
         doppler_ref: float | None = None, n_channels: int = 1,
         role: Role = "sub") -> list[Evaluation]:
    """Evaluate a size category and sort per the role's ranking rule
    (sec_procedure.tex): sub maximizes Vd; attack maximizes eta0*Pmax
    subject to its own excursion/Doppler budgets (see Evaluation.sort_key)."""
    evals = [evaluate(d, scenario, n_units, band_low, band_high, doppler_ref,
                      n_channels=n_channels, role=role)
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
             top_k_each: int = 15, sub_channels: int = 1,
             attack_channels: int = 2) -> list[PairEvaluation]:
    """Rank sub and attack candidates independently within their own bands
    and size windows, then combine the top-K of each into pairs sorted by
    combined distortion (feasible pairs first).

    `sub_channels` (default 1, mono manifold) and `attack_channels` (default
    2, stereo L/R) scale the *acoustic* coherent-sum only -- see
    `evaluate()`'s docstring for why thermal/power budget never gets that
    credit."""
    sc = scenario
    sub_evals = rank(drivers, sc, n_units=sub_units,
                     min_size_in=sub_size_min, max_size_in=sub_size_max,
                     band_low=sc.f_low, band_high=sc.f_split,
                     doppler_ref=sc.f_split, n_channels=sub_channels,
                     role="sub")[:top_k_each]
    attack_evals = rank(drivers, sc, n_units=attack_units,
                        min_size_in=attack_size_min, max_size_in=attack_size_max,
                        band_low=sc.f_split, band_high=sc.f_high,
                        doppler_ref=sc.f_high, n_channels=attack_channels,
                        role="attack")[:top_k_each]
    pairs = [PairEvaluation(s, a) for s in sub_evals for a in attack_evals]
    return sorted(pairs, key=PairEvaluation.sort_key)


def _room_spl_ceiling(vd_total: float, sc: Scenario, band_low: float,
                      role: str, shape_factor: float) -> float:
    """Ceiling SPL at the listening position at band_low, including the room
    pressure zone: the SPL at which V_dem(band_low) equals Vd/C."""
    v_dem_unit = sc.demand_volume(band_low, role) / sc.target_pressure(role)  # per Pa
    p_max_rms = (vd_total / shape_factor) / v_dem_unit
    return physics.spl_from_pressure(p_max_rms)


def _infeasible(driver: Driver, sc: Scenario, n_units: int, band_low: float,
                reason: str, n_channels: int = 1, role: Role = "sub") -> Evaluation:
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
        n_units_required=sc.units_required(driver.vd, band_low, role),
        n_channels=n_channels,
        role=role,
    )
