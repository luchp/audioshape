"""Matplotlib reporting layer.

Only this module (and `cli`) may import matplotlib; the core stays plot-free
so it can back a web service later.  Each function draws onto a provided or
fresh figure and returns it -- callers decide about show()/savefig().
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
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


def _annotate_notes(ax, ev: Evaluation) -> None:
    """Surface any non-blocking Evaluation.notes (e.g. the large-box + EQ
    fallback used when ideal Fc-at-target-corner alignment is geometrically
    unreachable, see MAX_VB_VAS_RATIO in ranking.py) as a small caveat text
    in the lower-left corner of the axes, distinct from the blocking
    `reasons` that make a driver infeasible."""
    if not ev.notes:
        return
    ax.text(0.01, 0.01, "note: " + " / ".join(ev.notes), transform=ax.transAxes,
             fontsize=7, color="tab:red", ha="left", va="bottom", wrap=True)


def spl_figure(ev: Evaluation, fig: Figure | None = None,
              show_power_axis: bool = False, f_min: float | None = None,
              crossover: float | None = None,
              crossover_shade: str | None = None,
              legend_loc: str = "lower right") -> Figure:
    """Achievable SPL at the listening position vs frequency.

    Curves: steady excursion ceiling, exact declared-burst excursion ceiling,
    and the finite electrical envelope, all on the same room/couch basis.

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

    # Excursion ceiling: SPL at which V_dem(f) = Vd_total.
    spl_sine = np.array([
        sc.target_spl_for(ev.role) + 20.0 * np.log10(boxed.vd_total / sc.demand_volume(x, ev.role))
        for x in f])

    # Finite steady electrical envelope, computed from the per-driver target
    # displacement rather than from coherent acoustic-power bookkeeping.
    electrical_gain = []
    sigma_total = boxed.wc / boxed.qtc
    for frequency in f:
        x_target = (
            sc.demand_volume(frequency, ev.role)
            / (boxed.n_units * d.sd)
        )
        power = physics.power_at_excursion_limit(
            frequency, d.mms, d.qes, d.fs, x_target,
            boxed.wc, d.sigma_m,
        )
        current = physics.current_at_excursion_limit(
            frequency, d.mms, d.qes, d.fs, d.re, x_target,
            boxed.wc, d.sigma_m,
        )
        voltage = physics.voltage_at_excursion_limit(
            frequency, d.mms, d.effective_bl, d.re, x_target,
            boxed.wc, sigma_total,
        )
        electrical_gain.append(min(
            sc.amplifier_voltage_rms / voltage,
            sc.amplifier_current_rms / current,
            math.sqrt(d.p_max / power),
            math.sqrt(sc.amplifier_power_continuous / power),
        ))
    spl_electrical = (
        sc.target_spl_for(ev.role)
        + 20.0 * np.log10(np.asarray(electrical_gain))
    )

    transient_f = np.geomspace(f[0], f[-1], 60)
    transient_ceiling = []
    for frequency in transient_f:
        burst = physics.sealed_burst_requirements(
            frequency,
            sc.demand_volume(frequency, ev.role)
            / (boxed.n_units * d.sd),
            boxed.fc,
            boxed.qtc,
            d.mms,
            d.re,
            d.effective_bl,
            cycles=sc.transient_cycles,
            window=sc.transient_window,
            phase_samples=sc.transient_phase_samples,
        )
        utilization = burst.displacement_peak / d.xmax
        transient_ceiling.append(
            sc.target_spl_for(ev.role) + 20.0 * np.log10(1.0 / utilization)
        )

    ax.plot(f, spl_sine, label="excursion ceiling, sine", lw=2)
    ax.plot(
        transient_f,
        transient_ceiling,
        label=(
            f"excursion ceiling, {sc.transient_cycles:g}-cycle "
            f"{sc.transient_window} burst"
        ),
        lw=2,
        ls="--",
    )
    ax.plot(
        f,
        spl_electrical,
        label="finite voltage/current/power envelope",
        lw=2,
        ls="-.",
    )
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
                 f"(Qtc={boxed.qtc:g}, Fc={boxed.fc:.1f} Hz)")
    ax.grid(True, which="both", alpha=0.3)
    _annotate_notes(ax, ev)

    if not show_power_axis:
        ax.legend(loc=legend_loc, fontsize=9)
        return fig

    # Secondary axis: per-unit electrical power needed to sit at the target.
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


def risk_figure(ev: Evaluation, fig: Figure | None = None,
                f_min: float | None = None, crossover: float | None = None,
                crossover_shade: str | None = None) -> Figure:
    """Separate risk indicators versus frequency at the target SPL.

    The curves retain their own physical meanings.  They are not added or
    labelled as a prediction of acoustic distortion.
    """
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    sc, d, boxed = ev.scenario, ev.driver, ev.boxed
    f = _freq_axis(ev, f_min=f_min)

    v_dem = np.array([sc.demand_volume(x, ev.role) for x in f])
    xi = v_dem / boxed.vd_total
    x1 = xi * d.xmax
    doppler = np.array([physics.doppler_im(ev.band_high, x) for x in x1])
    box = np.array([
        physics.box_spring_nonlinearity(
            v / boxed.n_units, boxed.vb, d.qts, boxed.qtc
        )
        for v in v_dem
    ])
    electrical = []
    sigma_total = boxed.wc / boxed.qtc
    for frequency, displacement in zip(
        f, v_dem / (boxed.n_units * d.sd)
    ):
        power = physics.power_at_excursion_limit(
            frequency, d.mms, d.qes, d.fs, displacement,
            boxed.wc, d.sigma_m,
        )
        current = physics.current_at_excursion_limit(
            frequency, d.mms, d.qes, d.fs, d.re, displacement,
            boxed.wc, d.sigma_m,
        )
        voltage = physics.voltage_at_excursion_limit(
            frequency, d.mms, d.effective_bl, d.re, displacement,
            boxed.wc, sigma_total,
        )
        electrical.append(max(
            power / d.p_max,
            power / sc.amplifier_power_continuous,
            current / sc.amplifier_current_rms,
            voltage / sc.amplifier_voltage_rms,
        ))

    transient_f = np.geomspace(f[0], f[-1], 60)
    transient_xi = []
    for frequency in transient_f:
        burst = physics.sealed_burst_requirements(
            frequency,
            sc.demand_volume(frequency, ev.role)
            / (boxed.n_units * d.sd),
            boxed.fc,
            boxed.qtc,
            d.mms,
            d.re,
            d.effective_bl,
            cycles=sc.transient_cycles,
            window=sc.transient_window,
            phase_samples=sc.transient_phase_samples,
        )
        transient_xi.append(burst.displacement_peak / d.xmax)

    ax.plot(f, xi, label="steady excursion utilization", lw=2)
    ax.plot(
        transient_f,
        transient_xi,
        label="transient excursion utilization",
        lw=2,
        ls="--",
    )
    ax.plot(
        f,
        doppler,
        label=f"Doppler sideband ratio onto {ev.band_high:g} Hz",
        lw=2,
        ls=":",
    )
    ax.plot(f, box, label="box-spring nonlinearity indicator",
            lw=2, ls="-.")
    ax.plot(f, electrical, label="maximum steady electrical utilization",
            lw=2, color="0.25")
    ax.axhline(1.0, color="r", lw=1, label="physical/electrical limit")
    ax.axhline(
        sc.preferred_excursion,
        color="tab:orange",
        lw=1,
        label=f"preferred excursion margin ({sc.preferred_excursion:.0%})",
    )
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
    ax.set_ylabel("dimensionless utilization / indicator")
    ax.set_title(
        f"{d.label()}  |  separate risk indicators, "
        f"{ev.boxed.n_units} unit(s)"
    )
    ax.grid(True, which="both", alpha=0.3)
    _annotate_notes(ax, ev)
    ax.legend(loc="upper right", fontsize=9)
    return fig


def demand_figure(sc: Scenario, fig: Figure | None = None,
                  role: str = "sub") -> Figure:
    """Required peak displaced volume V_dem(f) at the listening position
    (eq:demand), including the selected pressure-zone leakage model."""
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


# ----------------------------------------------------------------------
# Revised JAES Engineering Report figures (single canonical Scenario,
# aggregate-only private-database evidence).
# ----------------------------------------------------------------------

def architecture_figure(scenario: Scenario, sub_units: int = 4,
                        fig: Figure | None = None) -> Figure:
    """System topology and data flow: private-database screening feeding a
    role-based ranking, the fixed mono-sub / independent-stereo-upper-bass
    signal architecture, DSP crossover/EQ, and the prepared room/couch
    listening area.  Purely schematic; box text quotes this report's own
    canonical scenario numbers so the figure cannot silently drift from the
    rest of the pipeline.  ``sub_units`` must be even: an odd count cannot
    be arranged as symmetrically opposed pairs.  Such pairs permit
    first-order reaction-force cancellation under matched drive and
    mounting; mismatch leaves residual force.  The even count is a hard
    mechanical-layout constraint, not a ranking-policy preference.
    """
    if sub_units < 2 or sub_units % 2:
        raise ValueError(
            "sub_units must be an even count >= 2 for opposed pairing"
        )
    if fig is None:
        fig = Figure(figsize=(9.5, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, text: str,
            fc: str = "white", fontsize: float = 8.3) -> None:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="black",
                               lw=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
               fontsize=fontsize, zorder=3)

    def arrow(x0: float, y0: float, x1: float, y1: float,
              text: str | None = None) -> None:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                   arrowprops=dict(arrowstyle="-|>", lw=1.2, color="0.2"),
                   zorder=1)
        if text:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center",
                    va="bottom", fontsize=7.2, color="0.2")

    box(0.2, 5.6, 2.3, 1.1,
        "Private driver database\n(aggregate evidence only,\nnever redistributed)")
    box(3.0, 5.6, 2.7, 1.1, "Datasheet screening +\nrole-based ranking\n(this report)")
    arrow(2.5, 6.15, 3.0, 6.15)

    box(0.2, 3.5, 2.3, 1.0, "Stereo program\n(L, R)")
    box(3.0, 3.5, 2.7, 1.3,
        f"DSP crossover + EQ\nsplit at $f_{{sp}}$={scenario.f_split:g} Hz\n"
        f"band [{scenario.f_low:g}, {scenario.f_high:g}] Hz")
    arrow(2.5, 4.0, 3.0, 4.15)
    arrow(4.35, 5.6, 4.35, 4.8, "selection feeds\nrole targets")

    box(6.1, 4.3, 3.5, 1.7,
        f"Mono sub manifold\n{sub_units}x identical drivers, {sub_units // 2}"
        " opposed pairs\n(even count: hard constraint;\n"
        "first-order force cancellation\nwhen matched)\none amplifier channel each\n"
        f"aggregate target {scenario.sub_target_spl:g} dB total")
    arrow(5.7, 4.35, 6.1, 5.0, "mono, coherent\nL+R sum")

    box(6.1, 2.8, 1.7, 1.3,
        "Left channel\n1x driver, own\namplifier channel\n"
        f"target {scenario.attack_target_spl:g} dB")
    box(7.95, 2.8, 1.7, 1.3,
        "Right channel\n1x driver, own\namplifier channel\n"
        f"target {scenario.attack_target_spl:g} dB")
    arrow(5.7, 3.75, 6.1, 3.5, "L")
    arrow(5.7, 4.1, 7.95, 4.05, "R (routed above L)")
    ax.text(7.8, 2.55, "no stereo summing credit between L and R",
           ha="center", va="top", fontsize=7.5, color="tab:red", style="italic")

    box(4.6, 0.5, 5.0, 1.7,
        "Prepared room, half-space soffit mount\n"
        f"$V_{{room}}$={scenario.v_room:g} m$^3$, $L_{{max}}$={scenario.l_max:g} m\n"
        f"couch listening area, r={scenario.r_listen:g} m")
    arrow(6.7, 4.5, 6.7, 2.2)
    arrow(7.8, 2.8, 7.2, 2.2)

    ax.set_title("System topology and data flow", fontsize=11)
    return fig


def electrical_utilization_figure(ev_sub: Evaluation, ev_upper: Evaluation,
                                  fig: Figure | None = None) -> Figure:
    """2x2 panel: output margin above target, and normalized voltage/
    current/power amplifier utilization, for the selected sub and
    upper-bass drivers over each role's own complete operating band.

    Every curve uses the declared finite amplifier limits (no unconstrained
    amplifier); the excursion gate (Xmax) and the electrical gates are
    combined into one achievable-output margin in the top row.
    """
    if fig is None:
        fig = Figure(figsize=(11, 7), constrained_layout=True)
    axes = fig.subplots(2, 2)

    for col, ev, title in ((0, ev_sub, "sub manifold"),
                           (1, ev_upper, "upper-bass channel")):
        sc, d, boxed = ev.scenario, ev.driver, ev.boxed
        f = np.geomspace(ev.band_low, ev.band_high, 200)
        sigma_total = boxed.wc / boxed.qtc

        voltage = np.empty_like(f)
        current = np.empty_like(f)
        power = np.empty_like(f)
        excursion_util = np.empty_like(f)
        for i, freq in enumerate(f):
            x_target = sc.demand_volume(freq, ev.role) / (boxed.n_units * d.sd)
            excursion_util[i] = x_target / d.xmax
            power[i] = physics.power_at_excursion_limit(
                freq, d.mms, d.qes, d.fs, x_target, boxed.wc, d.sigma_m)
            current[i] = physics.current_at_excursion_limit(
                freq, d.mms, d.qes, d.fs, d.re, x_target, boxed.wc, d.sigma_m)
            voltage[i] = physics.voltage_at_excursion_limit(
                freq, d.mms, d.effective_bl, d.re, x_target, boxed.wc, sigma_total)

        voltage_util = voltage / sc.amplifier_voltage_rms
        current_util = current / sc.amplifier_current_rms
        power_util = power / min(d.p_max, sc.amplifier_power_continuous)
        electrical_gain = np.minimum.reduce([
            sc.amplifier_voltage_rms / voltage,
            sc.amplifier_current_rms / current,
            np.sqrt(d.p_max / power),
            np.sqrt(sc.amplifier_power_continuous / power),
        ])
        margin_db = 20.0 * np.log10(np.minimum(1.0 / excursion_util, electrical_gain))

        ax_top, ax_bot = axes[0, col], axes[1, col]
        ax_top.plot(f, margin_db, lw=2, color="tab:blue")
        ax_top.axhline(0.0, color="k", lw=1, ls="--", label="target")
        ax_top.set_title(f"{d.label()}  |  {title}")
        ax_top.set_ylabel("output margin above target [dB]")
        ax_top.grid(True, which="both", alpha=0.3)
        ax_top.legend(fontsize=8, loc="best")

        ax_bot.plot(f, voltage_util, lw=2, label="voltage / $V_{amp,max}$")
        ax_bot.plot(f, current_util, lw=2, ls="--", label="current / $I_{amp,max}$")
        ax_bot.plot(f, power_util, lw=2, ls="-.",
                   label=r"power / $\min(P_{driver}, P_{amp,cont})$")
        ax_bot.axhline(1.0, color="r", lw=1, label="limit")
        ax_bot.set_xlabel("frequency [Hz]")
        ax_bot.set_ylabel("normalized utilization")
        ax_bot.grid(True, which="both", alpha=0.3)
        ax_bot.legend(fontsize=8, loc="best")

        for ax in (ax_top, ax_bot):
            ax.set_xscale("log")
            _style_log_xaxis(ax)

    fig.suptitle("Output margin and finite amplifier utilization over each role's complete band")
    return fig


def transient_factor_figure(
    qtc_values: Sequence[float] = (0.50, 0.55, 0.61),
    windows: Sequence[str] = ("rectangular", "hann"),
    worked_points: Sequence[tuple[str, float, float]] = (),
    fig: Figure | None = None,
) -> Figure:
    """Numerical sealed-box ODE transient displacement factor (worst start
    phase, peak transient displacement / steady-sine peak at the same
    target) vs. f/Fc, for each declared Qtc and burst window.

    ``shape_factor`` is a linear-system scaling identity: it depends only on
    f/Fc, Qtc, and the declared burst (cycle count, window, phase sampling),
    not on the driver's Mms/Re/Bl, so one arbitrary unit driver stands in
    for every curve here. ``worked_points`` are optional
    (label, f/Fc, Qtc) markers for specific worked-example operating
    points, plotted at their own true Qtc even when it is not one of the
    three reference curves.
    """
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ratios = np.geomspace(0.1, 10.0, 120)
    fc = 100.0  # arbitrary reference corner [Hz]; shape_factor is scale-free
    mms, re, bl = 1.0, 1.0, 1.0
    styles = {"rectangular": "-", "hann": "--"}

    for qtc in qtc_values:
        for window in windows:
            shape = np.array([
                physics.sealed_burst_requirements(
                    f=ratio * fc, x_sine_peak=1e-3, fc=fc, qtc=qtc,
                    mms=mms, re=re, bl=bl, cycles=1.0, window=window,
                    phase_samples=8,
                ).shape_factor
                for ratio in ratios
            ])
            ax.plot(ratios, shape, lw=1.8, ls=styles[window],
                   label=f"$Q_{{tc}}$={qtc:.2f}, {window}")

    for label, f_over_fc, qtc in worked_points:
        shape = physics.sealed_burst_requirements(
            f=f_over_fc * fc, x_sine_peak=1e-3, fc=fc, qtc=qtc,
            mms=mms, re=re, bl=bl, cycles=1.0, window="rectangular",
            phase_samples=8,
        ).shape_factor
        ax.scatter([f_over_fc], [shape], marker="o", s=45, color="k", zorder=5)
        ax.annotate(label, (f_over_fc, shape), xytext=(5, 5),
                   textcoords="offset points", fontsize=8)

    ax.set_xscale("log")
    _style_log_xaxis(ax)
    ax.set_xlabel("$f / F_c$")
    ax.set_ylabel("transient displacement factor (worst phase)")
    ax.set_title("One-cycle burst transient factor vs. sealed-box alignment")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    return fig


def database_pareto_figure(
    vd_m3: np.ndarray,
    eta0_pmax: np.ndarray,
    selected: Sequence[tuple[str, float, float]] = (),
    fig: Figure | None = None,
) -> Figure:
    """Aggregate Vd vs. eta0*Pmax population evidence for the private
    database: hexbin density only, no labelled row-level scatter, plus the
    two independently public-datasheet-sourced drivers marked by name."""
    if fig is None:
        fig = Figure(figsize=(8, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(vd_m3 * 1e3, eta0_pmax, gridsize=40, xscale="log",
                   yscale="log", mincnt=1, cmap="Blues")
    fig.colorbar(hb, ax=ax, label="driver count per bin")

    for label, vd, ep in selected:
        ax.scatter([vd * 1e3], [ep], marker="*", s=200, color="tab:red",
                  edgecolor="k", zorder=5)
        ax.annotate(label, (vd * 1e3, ep), xytext=(6, 6),
                   textcoords="offset points", fontsize=8, color="tab:red")

    ax.set_xlabel(r"$V_d = S_d X_{\max}$ [L]")
    ax.set_ylabel(r"$\eta_0 P_{\max}$ [W$_{\mathrm{ac}}$]")
    ax.set_title("Displacement vs. thermal output capacity, database population")
    return fig


def corner_population_figure(
    corner_rate_hz: np.ndarray,
    stroke_mm: np.ndarray,
    selected: Sequence[tuple[str, float, float]] = (),
    fig: Figure | None = None,
) -> Figure:
    """Aggregate stroke (one-way Xmax) vs. corner-rate (Fs/Qts) population
    map for the private database: hexbin density only."""
    if fig is None:
        fig = Figure(figsize=(8, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    hb = ax.hexbin(corner_rate_hz, stroke_mm, gridsize=40, xscale="log",
                   yscale="log", mincnt=1, cmap="Oranges")
    fig.colorbar(hb, ax=ax, label="driver count per bin")

    for label, corner, stroke in selected:
        ax.scatter([corner], [stroke], marker="*", s=200, color="tab:blue",
                  edgecolor="k", zorder=5)
        ax.annotate(label, (corner, stroke), xytext=(6, 6),
                   textcoords="offset points", fontsize=8, color="tab:blue")

    ax.set_xlabel(r"corner rate $F_s/Q_{ts}$ [Hz]")
    ax.set_ylabel(r"stroke $X_{\max}$ [mm, one-way peak]")
    ax.set_title("Stroke vs. corner-rate population map")
    return fig


def rank_robustness_figure(
    variant_labels: Sequence[str],
    sub_overlap: Sequence[float],
    attack_overlap: Sequence[float],
    top_k: int,
    fig: Figure | None = None,
) -> Figure:
    """Top-k rank-stability overlap fraction under each declared scenario/
    policy variant, for the sub-manifold and upper-bass candidate pools
    reported separately (no combined score)."""
    if fig is None:
        fig = Figure(figsize=(9.5, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111)
    x = np.arange(len(variant_labels))
    width = 0.35
    ax.bar(x - width / 2, sub_overlap, width, label="sub manifold role",
          color="tab:blue")
    ax.bar(x + width / 2, attack_overlap, width, label="upper-bass role",
          color="tab:orange")
    ax.axhline(1.0, color="k", lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"top-{top_k} overlap fraction")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Rank stability under scenario/policy variants (top-{top_k} overlap)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    return fig


def room_sensitivity_figure(
    scenario: Scenario,
    leakage_variants: Sequence[float] = (0.0, 5.0, 10.0, 15.0),
    fig: Figure | None = None,
) -> Figure:
    """Required peak displaced volume V_dem(f) at the couch listening
    distance for the ideal pressure zone and several leakage-corner
    variants (eq:demand), all on the same room/target basis."""
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    f = np.geomspace(scenario.f_pz * 0.2, scenario.f_high, 400)

    for leak in leakage_variants:
        if leak <= 0.0:
            model, corner, label = "ideal_pressure_zone", 0.0, "ideal pressure zone"
        else:
            model, corner, label = "leaky_pressure_zone", leak, f"leaky, corner={leak:g} Hz"
        v_dem = np.array([
            physics.demand_volume(x, scenario.sub_target_spl, scenario.r_listen,
                                  scenario.v_room, scenario.l_max,
                                  room_model=model, leakage_corner_hz=corner) * 1e3
            for x in f
        ])
        ax.plot(f, v_dem, lw=2, label=label)

    ax.set_xscale("log")
    _style_log_xaxis(ax)
    ax.set_yscale("log")
    ax.axvline(scenario.f_pz, color="grey", lw=0.8, ls=":")
    ax.annotate("$f_{pz}$", (scenario.f_pz, ax.get_ylim()[0]),
               xytext=(2, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel(r"$V_{\mathrm{dem}}$ [L]")
    ax.set_title(f"Room-demand sensitivity to leakage corner, "
                f"{scenario.sub_target_spl:g} dB @ {scenario.r_listen:g} m, "
                f"3 m couch basis")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    return fig


def worked_system_figure(ev_sub: Evaluation, ev_upper: Evaluation,
                         fig: Figure | None = None) -> Figure:
    """Concise selected-pair summary: separate risk indicators for the sub
    manifold and upper-bass channel side by side.  No summed distortion
    figure is computed or implied; each bar keeps its own physical meaning.
    """
    if fig is None:
        fig = Figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111)

    rows = [
        ("steady excursion util.", "steady_excursion"),
        ("transient excursion util.", "transient_excursion"),
        ("Doppler sideband ratio", "doppler_modulation"),
        ("box-spring indicator", "box_spring"),
        ("driver power util.", "driver_power"),
        ("amplifier voltage util.", "amplifier_voltage"),
        ("amplifier current util.", "amplifier_current"),
        ("amplifier cont. power util.", "amplifier_continuous_power"),
        ("amplifier burst power util.", "amplifier_burst_power"),
    ]
    labels = [row[0] for row in rows]
    sub_values = [getattr(ev_sub.risk, row[1]) for row in rows]
    upper_values = [getattr(ev_upper.risk, row[1]) for row in rows]

    y = np.arange(len(rows))
    height = 0.35
    ax.barh(y + height / 2, sub_values, height, label="sub manifold",
           color="tab:blue")
    ax.barh(y - height / 2, upper_values, height, label="upper-bass channel",
           color="tab:orange")
    ax.axvline(1.0, color="r", lw=1, label="physical/electrical limit")
    ax.axvline(ev_sub.scenario.preferred_excursion, color="tab:green", lw=1,
              ls="--",
              label=f"preferred excursion margin "
                    f"({ev_sub.scenario.preferred_excursion:.0%})")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("dimensionless utilization / indicator (each reported separately)")
    ax.set_title(f"{ev_sub.driver.label()} (sub) vs. {ev_upper.driver.label()} "
                f"(upper) \u2014 worked selected pair")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    return fig
