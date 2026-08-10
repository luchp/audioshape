"""Matplotlib reporting layer.

Only this module (and `cli`) may import matplotlib; the core stays plot-free
so it can back a web service later.  Each function draws onto a provided or
fresh figure and returns it -- callers decide about show()/savefig().
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from audioshape import physics, vented
from audioshape.driver import BoxedDriver, Driver
from audioshape.ranking import Evaluation
from audioshape.scenario import Scenario


def _freq_axis(ev: Evaluation, n: int = 400, f_min: float | None = None) -> np.ndarray:
    sc = ev.scenario
    lo = sc.f_low * 0.7 if f_min is None else f_min
    return np.geomspace(lo, sc.f_high, n)


def _style_log_xaxis(ax) -> None:
    """Force labeled minor ticks (2, 3, 4, 6 x each decade) on a log
    frequency axis. Matplotlib's default LogFormatter only labels minor
    ticks below ~1 decade of span, so otherwise-similar figures whose axis
    happens to cross that threshold (e.g. the driver S vs. M worked-example
    plots) render with inconsistent tick density -- override it so every
    figure reads the same regardless of exactly how many decades it spans.
    """
    ax.xaxis.set_minor_locator(LogLocator(subs=(2, 3, 4, 6)))
    ax.xaxis.set_minor_formatter(
        LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf)))


def _room_gain_factor(f: float, ev: Evaluation) -> float:
    """Linear pressure gain of the room relative to free half-space radiation
    at the listening distance (>= 1 below f_pz, 1 above)."""
    sc = ev.scenario
    w = 2.0 * np.pi * f
    v_radiation = (np.sqrt(2.0) * sc.target_pressure(ev.role) * 2.0 * np.pi
                   * sc.r_listen / (physics.RHO0 * w * w))
    return v_radiation / sc.demand_volume(f, ev.role)


def _shade_crossover(ax, f: np.ndarray, crossover: float | None,
                     crossover_shade: str | None) -> None:
    """Mark the sub/attack driver crossover f_split with a dotted line, and
    lightly shade the side of the plot that is the *other* driver's band
    (crossover_shade='below': this driver's own band starts at crossover,
    e.g. the attack driver; 'above': its own band ends at crossover, e.g.
    the sub driver) -- distinct from f_pz, the room's pressure-zone corner,
    which is a property of the room, not of the two-driver handoff."""
    if crossover is None or not (f[0] <= crossover <= f[-1]):
        return
    if crossover_shade == "below":
        ax.axvspan(f[0], crossover, color="grey", alpha=0.12, lw=0,
                  label="other driver's band")
    elif crossover_shade == "above":
        ax.axvspan(crossover, f[-1], color="grey", alpha=0.12, lw=0,
                  label="other driver's band")
    ax.axvline(crossover, color="grey", lw=0.8, ls=":")
    # A taller offset than the f_pz/Fc/f_x row (xytext=(2, 6)) keeps the
    # label legible even when f_split lands close to Fc, as for the attack
    # driver, whose box is tuned near the crossover by design.
    ax.annotate("$f_{sp}$", (crossover, ax.get_ylim()[0]),
                xytext=(2, 20), textcoords="offset points", fontsize=9)


def spl_figure(ev: Evaluation, fig: Figure | None = None,
              show_power_axis: bool = False, f_min: float | None = None,
              crossover: float | None = None,
              crossover_shade: str | None = None,
              legend_loc: str = "lower right") -> Figure:
    """Achievable SPL at the listening position vs frequency.

    Curves: sine excursion ceiling, burst (pulse) excursion ceiling, thermal
    ceiling (with EQ tax below Fc), all including room pressure-zone gain;
    plus the target line and the markers f_pz, Fc, f_x.

    show_power_axis adds a secondary (right-hand, log) axis with the
    per-unit electrical power actually needed to sit at the target line --
    using the real two-branch demand curve, not a flat-EQ straw man --
    which is what answers whether A6's unconstrained amplifier ever implies
    an absurd power draw (eq:EQtax): it stays bounded, because the demand
    curve itself saturates below Fc/f_pz rather than growing without limit.
    Only meaningful when `ev.scenario.r_listen` is the *room-consistent*
    listening distance (f_pz computed from the same room, e.g. SC_SUB):
    at the r=1 m driver-pricing basis (e.g. SC_ATTACK), demand_volume's
    below-f_pz branch mixes an unrelated length scale and the resulting
    curve is not physically meaningful there, so leave this off for that
    case.

    f_min overrides the default `f_low*0.7` plot-axis floor. Use this either
    to crop out the below-f_pz region for a scenario whose r_listen is not
    the room-consistent distance (e.g. SC_ATTACK: below f_pz, demand_volume's
    two branches are evaluated at mismatched length scales there and the
    resulting curve is not physically meaningful, see above), or -- for a
    room-consistent scenario, e.g. SC_SUB -- to trim the uninformative flat
    run-in below f_pz (the pressure-zone branch is frequency-independent by
    construction, so nothing new is shown there beyond a single value).

    crossover draws the sub/attack driver handoff frequency f_split as a
    dotted line, with crossover_shade in {"below", "above"} lightly shading
    the side of the plot that belongs to the *other* driver's band ("below"
    for an attack driver whose own band starts at f_split, "above" for a
    sub driver whose own band ends there) -- distinct from f_pz, which is a
    property of the room, not of the two-driver handoff.

    legend_loc overrides the default "lower right" legend placement; e.g.
    for the attack driver, whose box is tuned close to f_split, the f_pz/
    Fc/f_x/f_sp marker cluster sits right where that corner would go, so
    "upper left" (clear of curves there) reads better.
    """
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    sc, d, boxed = ev.scenario, ev.driver, ev.boxed
    f = _freq_axis(ev, f_min=f_min)

    # Excursion ceilings: SPL at which V_dem(f) = Vd_total / C.
    spl_sine = np.array([
        sc.target_spl_for(ev.role) + 20.0 * np.log10(boxed.vd_total / sc.demand_volume(x, ev.role))
        for x in f])
    spl_burst = spl_sine - 20.0 * np.log10(sc.burst_shape)

    # Thermal ceiling: passband value minus the EQ tax, plus room gain.
    spl_pb = physics.spl_thermal_ceiling(
        d.eta0, d.p_max, sc.r_listen) + boxed.spl_gain_db()
    tax = np.array([physics.eq_tax_power(x, 1.0, boxed.wc, d.sigma_m)
                    for x in f])
    room = np.array([20.0 * np.log10(_room_gain_factor(x, ev)) for x in f])
    spl_thermal = spl_pb - 10.0 * np.log10(tax) + room

    ax.plot(f, spl_sine, label="excursion ceiling, sine", lw=2)
    ax.plot(f, spl_burst, label=f"excursion ceiling, burst (C={sc.burst_shape:g})",
            lw=2, ls="--")
    ax.plot(f, spl_thermal, label="thermal ceiling (driver $P_{max}$, EQ tax)",
            lw=2, ls="-.")
    ax.axhline(sc.target_spl_for(ev.role), color="k", lw=1,
               label=f"target {sc.target_spl_for(ev.role):g} dB")
    # Fix the y-limits before placing marker labels: annotate() below anchors
    # to ax.get_ylim()[0], which must reflect the *final* view (not a
    # provisional pre-set_ylim autoscale value) or the label can land outside
    # the eventually-visible range and silently disappear.
    ax.set_ylim(sc.target_spl_for(ev.role) - 25, None)

    for x, name in ((sc.f_pz, "$f_{pz}$"), (boxed.fc, "$F_c$"), (ev.f_x, "$f_x$")):
        if np.isfinite(x) and f[0] <= x <= f[-1]:
            ax.axvline(x, color="grey", lw=0.8, ls=":")
            ax.annotate(name, (x, ax.get_ylim()[0]),
                        xytext=(2, 6), textcoords="offset points", fontsize=9)
    _shade_crossover(ax, f, crossover, crossover_shade)

    ax.set_xscale("log")
    _style_log_xaxis(ax)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(f"SPL at {sc.r_listen:g} m [dB]")
    ax.set_title(f"{d.label()}  |  {ev.boxed.n_units}x in {boxed.vb*1e3:.0f} L "
                 f"(Qtc={sc.qtc:g}, Fc={boxed.fc:.1f} Hz)")
    ax.grid(True, which="both", alpha=0.3)

    if not show_power_axis:
        ax.legend(loc=legend_loc, fontsize=9)
        return fig

    # Secondary axis: per-unit electrical power actually needed to sit at
    # the target line (real demand curve, room gain included), not at Xmax.
    x_dem = np.array([sc.demand_volume(x, ev.role) for x in f]) / (boxed.n_units * d.sd)
    p_req = physics.power_at_excursion_limit(f, d.mms, d.qes, d.fs, x_dem,
                                             boxed.wc, d.sigma_m)
    ax2 = ax.twinx()
    ax2.plot(f, p_req, color="tab:purple", lw=1.2, ls=(0, (1, 1)),
             label="power at target, per unit (right axis)")
    ax2.set_yscale("log")
    ax2.set_ylabel("electrical power at target, per unit [W]", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_loc, fontsize=9)
    return fig


def distortion_figure(ev: Evaluation, fig: Figure | None = None,
                      f_min: float | None = None, crossover: float | None = None,
                      crossover_shade: str | None = None) -> Figure:
    """Predicted non-correctable distortion vs frequency at the target SPL.

    Curves: motor/suspension HD (eq:HDscale), Doppler IM onto the top of the
    band (eq:doppler), box air-spring HD2 (eq:boxHD), and their sum, against
    the distortion budget D*.

    f_min overrides the default `f_low*0.7` plot-axis floor; crossover and
    crossover_shade mark/shade the sub/attack driver handoff f_split -- see
    `spl_figure`'s docstring for both.
    """
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    sc, d, boxed = ev.scenario, ev.driver, ev.boxed
    f = _freq_axis(ev, f_min=f_min)

    v_dem = np.array([sc.demand_volume(x, ev.role) for x in f])
    xi = v_dem / boxed.vd_total
    hd = np.array([physics.harmonic_distortion(x) for x in xi])
    x1 = np.minimum(xi, 1.0) * d.xmax
    doppler = np.array([physics.doppler_im(sc.f_split, x) for x in x1])
    box = np.array([physics.box_hd2(min(v, boxed.n_units * d.vd) / boxed.n_units,
                                    boxed.vb, d.qts, sc.qtc) for v in v_dem])

    ax.plot(f, 100 * hd, label="motor/suspension HD", lw=2)
    ax.plot(f, 100 * doppler,
            label=f"Doppler IM onto {sc.f_split:g} Hz", lw=2, ls="--")
    ax.plot(f, 100 * box, label="box air-spring HD2", lw=2, ls="-.")
    ax.plot(f, 100 * (hd + doppler + box), label="total", lw=2.5, color="k")
    ax.axhline(100 * sc.distortion_budget, color="r", lw=1,
               label=f"budget $D^*$ = {100*sc.distortion_budget:g} %")
    # Switch to log y before placing marker labels: annotate() below anchors
    # to ax.get_ylim()[0], which must reflect the final (log-scale) view, not
    # the linear-mode autoscale value, or the label silently lands off-plot.
    ax.set_yscale("log")

    ax.axvline(sc.f_pz, color="grey", lw=0.8, ls=":")
    ax.annotate("$f_{pz}$", (sc.f_pz, ax.get_ylim()[0]),
                xytext=(2, 6), textcoords="offset points", fontsize=9)
    _shade_crossover(ax, f, crossover, crossover_shade)

    ax.set_xscale("log")
    _style_log_xaxis(ax)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(f"distortion at {sc.target_spl_for(ev.role):g} dB target, r={sc.r_listen:g} m [%]")
    ax.set_title(f"{d.label()}  |  non-correctable distortion, "
                 f"{ev.boxed.n_units} unit(s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def demand_figure(sc: Scenario, fig: Figure | None = None,
                  role: str = "sub") -> Figure:
    """Required peak displaced volume V_dem(f) at the listening position
    (eq:demand): rises 12 dB/oct above f_pz (free-field/modal radiation),
    flat below it (adiabatic room pressure-zone) -- Sec. "Room closure"."""
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    f = np.geomspace(sc.f_pz * 0.25, sc.f_high, 400)
    v_dem_l = np.array([sc.demand_volume(x, role) * 1e3 for x in f])

    ax.plot(f, v_dem_l, lw=2, color="tab:blue")
    # Switch to log/log before placing the marker label: annotate() below
    # anchors to ax.get_ylim()[0], which must reflect the final (log-scale)
    # view, not the linear-mode autoscale value, or the label silently lands
    # off-plot (see the identical fix in spl_figure/distortion_figure).
    ax.set_xscale("log")
    _style_log_xaxis(ax)
    ax.set_yscale("log")
    ax.axvline(sc.f_pz, color="grey", lw=0.8, ls=":")
    ax.annotate("$f_{pz}$", (sc.f_pz, ax.get_ylim()[0]),
                xytext=(2, 6), textcoords="offset points", fontsize=9)

    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$V_{\mathrm{dem}}$ [L]")
    ax.set_title(f"Required displaced volume at {sc.target_spl_for(role):g} dB, "
                 f"{sc.r_listen:g} m  "
                 f"($V_{{\\mathrm{{room}}}}$={sc.v_room:g} m$^3$, "
                 f"$L_{{\\max}}$={sc.l_max:g} m)")
    ax.grid(True, which="both", alpha=0.3)
    return fig


def vented_comparison_figure(driver: Driver, sc: Scenario, vb: float, fb: float,
                             s_port: float, fig: Figure | None = None) -> Figure:
    """Sealed vs. vented, same driver and same box volume: SPL(f) and cone
    excursion X(f).

    Both share the driver's mass-controlled asymptote at high frequency
    (box/port irrelevant there -- both curves are scaled to the same
    passband level at `sc.f_split`); they diverge below their respective
    corners: sealed rolls off at 12 dB/oct with excursion saturating
    (box-limited); vented rolls off at 24 dB/oct while excursion is *not*
    limited by the box at all (Sec. "Why sealed, not vented").
    """
    if fig is None:
        fig = Figure(figsize=(9, 8), constrained_layout=True)
    ax_spl, ax_x = fig.subplots(2, 1, sharex=True)

    qtc = vented.implied_qtc(driver.vas, driver.qts, vb)
    boxed = BoxedDriver(driver, qtc=qtc, n_units=1)
    va = vented.VentedAlignment.tuned(driver, vb=vb, fb=fb, s_port=s_port)

    def sealed_x(f: float, e: complex) -> complex:
        s = 1j * 2.0 * np.pi * f
        return (driver.bl / driver.re) * e / va.sealed_limit_delta(s)

    # Scale drive so both curves read sc.target_spl at the shared,
    # box/port-independent passband (top of the sub band).
    f_ref = sc.f_split
    p_ref_1v = physics.radiation_pressure_rms(
        f_ref, driver.sd * abs(sealed_x(f_ref, 1.0)), sc.r_listen)
    e = 10.0 ** ((sc.target_spl_for("sub") - physics.spl_from_pressure(p_ref_1v)) / 20.0)

    f = np.geomspace(sc.f_pz * 0.3, sc.f_split, 400)
    x_sealed = np.array([abs(sealed_x(x, e)) for x in f])
    x_vented = np.array([abs(va.x(1j * 2.0 * np.pi * x, e)) for x in f])
    u_vented = np.array([abs(va.u_total(1j * 2.0 * np.pi * x, e)) for x in f])

    spl_sealed = np.array([
        physics.spl_from_pressure(physics.radiation_pressure_rms(
            fi, driver.sd * xi, sc.r_listen))
        for fi, xi in zip(f, x_sealed)])
    spl_vented = np.array([
        physics.spl_from_pressure(physics.radiation_pressure_rms(
            fi, ui / (2.0 * np.pi * fi), sc.r_listen))
        for fi, ui in zip(f, u_vented)])

    ax_spl.plot(f, spl_sealed, lw=2, label=f"sealed, $Q_{{tc}}$={qtc:.2f}, "
                f"$F_c$={boxed.fc:.1f} Hz")
    ax_spl.plot(f, spl_vented, lw=2, ls="--",
                label=f"vented, $F_b$={fb:.1f} Hz")
    ax_spl.axhline(sc.target_spl_for("sub"), color="k", lw=1,
                   label=f"target {sc.target_spl_for('sub'):g} dB")
    ax_spl.set_ylabel(f"SPL at {sc.r_listen:g} m [dB]")
    ax_spl.set_title(f"{driver.label()}  |  {vb*1e3:.0f} L, sealed vs. vented")
    ax_spl.grid(True, which="both", alpha=0.3)
    ax_spl.legend(loc="lower right", fontsize=9)

    ax_x.plot(f, x_sealed * 1e3, lw=2, label="sealed |X(f)|")
    ax_x.plot(f, x_vented * 1e3, lw=2, ls="--", label="vented |X(f)|")
    ax_x.axhline(driver.xmax * 1e3, color="r", lw=1,
                 label=f"$X_{{max}}$={driver.xmax*1e3:g} mm")
    for x, name in ((sc.f_pz, "$f_{pz}$"), (boxed.fc, "$F_c$"), (fb, "$F_b$")):
        ax_x.axvline(x, color="grey", lw=0.8, ls=":")

    ax_x.set_xscale("log")
    ax_x.set_yscale("log")
    ax_x.set_xlabel("frequency [Hz]")
    ax_x.set_ylabel("cone excursion [mm, peak]")
    ax_x.grid(True, which="both", alpha=0.3)
    ax_x.legend(loc="upper right", fontsize=9)
    _style_log_xaxis(ax_spl)
    _style_log_xaxis(ax_x)
    return fig
