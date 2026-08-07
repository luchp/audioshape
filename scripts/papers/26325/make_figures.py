"""Generate every figure and auto-generated table fragment used by
`sealed_driver_criteria.tex` (paper 26325), plus its `meta/metadata.tex`.

Single source of truth: run this whenever `physics.py`, `driver.py`,
`vented.py`, `plots.py`, or the worked-example driver parameters change,
then recompile the paper. Writes vector PDF figures into
`papers/26325/figures/`, `.tex` table fragments into `papers/26325/tables/`,
and `papers/26325/meta/metadata.tex`; all committed to git so the paper
stays buildable with `pdflatex` alone (no Python required for a plain
build).

Usage (from repo root):  scripts\\figures -p 26325
Or directly:              uv run python scripts/make_figures.py 26325
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

from audioshape import physics, plots, vented
from audioshape.driver import BoxedDriver, Driver
from audioshape.ranking import evaluate
from audioshape.scenario import Scenario

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ID = SCRIPT_DIR.name
BASEDIR = SCRIPT_DIR.parents[2]
PAPER_DIR = BASEDIR / "papers" / PAPER_ID
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
META_DIR = PAPER_DIR / "meta"
PAPER_METADATA_FILE = SCRIPT_DIR / "metadata.json"
SITE_METADATA_FILE = BASEDIR / "scripts" / "metadata.json"

# Add scripts dir to path for file_utils import
sys.path.insert(0, str(BASEDIR / "scripts"))
from file_utils import save_figure_checked, write_text_checked  # noqa: E402


def write_metadata() -> None:
    with open(SITE_METADATA_FILE) as f:
        smeta = json.load(f)
    with open(PAPER_METADATA_FILE) as f:
        ameta = json.load(f)
    author = f"{smeta['author_given_names']} {smeta['author_family_names']}"
    write_text_checked(
        META_DIR / "metadata.tex",
        f"""% Auto-generated DO NOT EDIT
    \\usepackage[
        pdfauthor={{{author}}},
        pdftitle={{{ameta['title']} v{ameta['version']}}},
        pdfsubject={{{ameta['summary']}}},
        pdfkeywords={{{ameta['keywords']}}},
    ]{{hyperref}}
    
    \\newcommand{{\\PaperTitle}}{{{ameta['title']}}}
    \\newcommand{{\\PaperVersion}}{{v{ameta['version']}}}
    \\newcommand{{\\PaperAuthor}}{{{author}}}
    \\newcommand{{\\PaperSummary}}{{{ameta['summary']}}}
    \\newcommand{{\\PaperCopyright}}{{{smeta['copyright']}}}
    \\newcommand{{\\PaperLicense}}{{{ameta['license']}}}
    \\newcommand{{\\PaperEmail}}{{{smeta['email']}}}
    \\newcommand{{\\PaperWebsite}}{{{smeta['website']}}}
    \\newcommand{{\\PaperAfiliation}}{{{smeta['affiliation']}}}
    """
    )


# ----------------------------------------------------------------------
# Worked-example drivers -- identical to tests/test_pair_ranking.py's
# fixtures, so the paper's numbers and the test suite's pinned numbers
# cannot silently diverge.
# ----------------------------------------------------------------------
DRIVER_S = Driver(
    manufacturer="Example", model="S18", size_in=18,
    fs=20.0, qes=0.543, qms=4.0, re=3.5, mms=0.400,
    sd=0.115, xmax=0.020, vas=0.297, p_max=600.0, bl=18.0, le=4e-3)
DRIVER_M = Driver(
    manufacturer="Example", model="M12", size_in=12,
    fs=48.0, qes=0.353, qms=6.0, re=5.2, mms=0.065,
    sd=0.052, xmax=0.008, vas=0.0648, p_max=350.0, bl=17.0, le=0.7e-3)

# Two reference scenarios, both the 60 m^3 / L_max=6 m room of the worked
# example (Sec. "Room closure"):
#  - r_listen = 1 m: the driver-comparison basis used throughout Secs. 3-6
#    and the worked-example table (A-list convention: compare drivers on
#    neutral ground before placing them in a room).
#  - r_listen = 3 m: the actual "couch" listening distance, used from
#    Sec. 7 on (Scenario's own default; self-consistent with
#    V_room ~ 2 r L_max^2/pi for L_max = 6 m, eq:demand).
# The two are NOT interchangeable and every use below is explicit about
# which one it needs (see Sec. "Room closure" for the paper-side discussion
# of why: the PZ branch of eq:demand is r-independent, but the free-field
# branch, and hence SPL/thermal figures, are not).
_BASE = Scenario(v_room=60.0, l_max=6.0, r_listen=1.0,
                 target_spl=110.0, distortion_budget=0.03,
                 qtc=0.55, f_low=15.0, f_split=80.0, f_high=250.0)
SC_SUB = replace(_BASE, qtc=0.71)   # S's sub-role alignment (250 L class)
SC_ATTACK = _BASE                    # M's attack-role alignment (0.55)
SC_ROOM = replace(_BASE, r_listen=3.0)  # couch distance, room-facing content


def _fmt_pct(v: float) -> str:
    """2 significant figures, matching the table's '0.43\\%'/'2.0\\%' style."""
    return f"{v:.2f}" if v < 1.0 else f"{v:.1f}"


def _ev_s():
    return evaluate(DRIVER_S, SC_SUB, n_units=2,
                    band_low=SC_SUB.f_low, band_high=SC_SUB.f_split,
                    doppler_ref=SC_SUB.f_split)


def _ev_m():
    return evaluate(DRIVER_M, SC_ATTACK, n_units=1,
                    band_low=SC_ATTACK.f_split, band_high=SC_ATTACK.f_high,
                    doppler_ref=SC_ATTACK.f_high)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def fig_spl_sub() -> None:
    save_figure_checked(plots.spl_figure(_ev_s()), FIGURES_DIR / "fig_spl_sub.pdf")


def fig_spl_attack() -> None:
    save_figure_checked(plots.spl_figure(_ev_m()), FIGURES_DIR / "fig_spl_attack.pdf")


def fig_distortion_sub() -> None:
    save_figure_checked(plots.distortion_figure(_ev_s()),
                        FIGURES_DIR / "fig_distortion_sub.pdf")


def fig_distortion_attack() -> None:
    save_figure_checked(plots.distortion_figure(_ev_m()),
                        FIGURES_DIR / "fig_distortion_attack.pdf")


def fig_demand() -> None:
    save_figure_checked(plots.demand_figure(SC_ROOM), FIGURES_DIR / "fig_demand.pdf")


def fig_vented_comparison() -> None:
    fig = plots.vented_comparison_figure(
        DRIVER_S, SC_ROOM, vb=BoxedDriver(DRIVER_S, qtc=0.71).vb, fb=23.0,
        s_port=0.008,
    )
    save_figure_checked(fig, FIGURES_DIR / "fig_vented_comparison.pdf")


# ----------------------------------------------------------------------
# Table: worked example (S vs M, datasheet + derived)
# ----------------------------------------------------------------------

def table_worked_example() -> None:
    s, m = DRIVER_S, DRIVER_M
    b55_s, b71_s = BoxedDriver(s, qtc=0.55), BoxedDriver(s, qtc=0.71)
    b55_m = BoxedDriver(m, qtc=0.55)

    fx_s = physics.regime_boundary_fx(s.fs, s.p_max, s.qes, s.mms, s.xmax)
    fx_m = physics.regime_boundary_fx(m.fs, m.p_max, m.qes, m.mms, m.xmax)

    pl0_s = physics.power_at_excursion_limit(
        0.0, s.mms, s.qes, s.fs, s.xmax, b71_s.wc, s.sigma_m)
    pl0_m = physics.power_at_excursion_limit(
        0.0, m.mms, m.qes, m.fs, m.xmax, b55_m.wc, m.sigma_m)

    # Motor materials bound (eq:thmbound): illustrative B=1.0 T (Sec.
    # "materials", "Numerical content" range 1.0-1.2 T), sub-class overhang
    # u=3 for S, a shallower u=1.5 for M's own smaller Xmax/coil -- both
    # comfortably inside the u<~1.8 attack-class bound at M's own EBP.
    ebp_u2_s = physics.motor_bound_ebp_u2(s.ebp, u=3.0)
    beta_s = physics.implied_coil_mass_fraction(s.ebp, u=3.0, b_field=1.0)
    ebp_u2_m = physics.motor_bound_ebp_u2(m.ebp, u=1.5)
    beta_m = physics.implied_coil_mass_fraction(m.ebp, u=1.5, b_field=1.0)

    rows = [
        (r"$F_s$ [Hz]", f"{s.fs:g}", f"{m.fs:g}"),
        (r"$\Mms$ [g]", f"{s.mms*1e3:g}", f"{m.mms*1e3:g}"),
        (r"$\Re$ [$\Omega$]", f"{s.re:g}", f"{m.re:g}"),
        (r"$\Bl$ [T\,m]", f"{s.bl:g}", f"{m.bl:g}"),
        (r"$\Qms$", f"{s.qms:g}", f"{m.qms:g}"),
        (r"$\Sd$ [m$^2$]", f"{s.sd:g}", f"{m.sd:g}"),
        (r"$\Xmax$ [mm]", f"{s.xmax*1e3:g}", f"{m.xmax*1e3:g}"),
        (r"$\Le$ [mH]", f"{s.le*1e3:g}", f"{m.le*1e3:g}"),
        (r"$\Pmax$ [W]", f"{s.p_max:g}", f"{m.p_max:g}"),
        ("MID", "", ""),
        (r"$\Qes$, $\Qts$", f"{s.qes:.3f},\\ {s.qts:.3f}",
         f"{m.qes:.3f},\\ {m.qts:.3f}"),
        (r"$\Vas$ [L]", f"{s.vas*1e3:.0f}", f"{m.vas*1e3:.0f}"),
        (r"$EBP=\se/2\pi$ [Hz]", f"{s.ebp:.0f}", f"{m.ebp:.0f}"),
        (r"$\sigma/2\pi=F_s/\Qts$ [Hz]",
         f"{s.corner_rate:.0f}", f"{m.corner_rate:.0f}"),
        (r"$\eta_0$ \eqref{eq:eta0}",
         f"{_fmt_pct(s.eta0*100)}\\%", f"{_fmt_pct(m.eta0*100)}\\%"),
        (r"$\Vd$ [L]", f"{s.vd*1e3:.2f}", f"{m.vd*1e3:.2f}"),
        (r"$\eta_0\Pmax$ [W$_{\mathrm{ac}}$]",
         f"{s.eta0*s.p_max:.2f}", f"{m.eta0*m.p_max:.2f}"),
        (r"$\mathrm{SPL_T}$ \eqref{eq:SPLT} [dB]",
         f"{physics.spl_thermal_ceiling(s.eta0, s.p_max, 1.0):.1f}",
         f"{physics.spl_thermal_ceiling(m.eta0, m.p_max, 1.0):.1f}"),
        (r"$\mathrm{SPL_L}(30/80\,\mathrm{Hz})$ \eqref{eq:SPLL} [dB]",
         f"{physics.spl_excursion_ceiling(30.0, s.vd, 1.0):.1f}",
         f"{physics.spl_excursion_ceiling(80.0, m.vd, 1.0):.1f}"),
        (r"$f_x$ \eqref{eq:fx} [Hz]", f"{fx_s:.0f}", f"{fx_m:.0f}"),
        (r"$\hat f_x$ \eqref{eq:fxburst} ($\kappa{=}4$, $C{=}2$) [Hz]",
         f"{physics.burst_boundary_fx(fx_s):.0f}",
         f"{physics.burst_boundary_fx(fx_m):.0f}"),
        (r"$f_L=\Re/2\pi\Le$ [Hz]",
         f"\\textbf{{{s.f_le:.0f}}}", f"{m.f_le:.0f}"),
        (r"box @ $\Qtc{=}0.55$: $\Vb$, $F_c$",
         f"{b55_s.vb*1e3:.0f} L, {b55_s.fc:.0f} Hz",
         f"{b55_m.vb*1e3:.0f} L, {b55_m.fc:.0f} Hz"),
        (r"box @ $\Qtc{=}0.71$: $\Vb$, $F_c$",
         f"{b71_s.vb*1e3:.0f} L, {b71_s.fc:.1f} Hz", "---"),
        (r"EQ tax $P_{\mathrm{L}}(0)$ \eqref{eq:PL0} (in that box) [W]",
         f"{pl0_s:.0f} ({b71_s.vb*1e3:.0f} L)",
         f"{pl0_m:.0f} ({b55_m.vb*1e3:.0f} L)"),
        (r"motor bound: $EBP\,u^2\;/\;$ implied $m_c$",
         f"{ebp_u2_s:.0f} Hz / {beta_s*s.mms*1e3:.0f} g",
         f"{ebp_u2_m:.0f} Hz / {beta_m*m.mms*1e3:.0f} g"),
    ]

    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r" & S: long-throw 18'' & M: high-efficiency 12''\\",
        r"\midrule",
    ]
    for label, vs, vm in rows:
        if label == "MID":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{3}{c}{\emph{derived}}\\")
            lines.append(r"\midrule")
            continue
        lines.append(f"{label} & {vs} & {vm}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    write_text_checked(TABLES_DIR / "table_worked_example.tex",
                       "\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Table: room sizing (V_dem^req, PZ branch, 3x3 grid)
# ----------------------------------------------------------------------

def table_room_sizing() -> None:
    spls = (105.0, 110.0, 115.0)
    v_rooms = (40.0, 60.0, 100.0)
    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$\Vd^{\mathrm{req}}$ [L] (PZ branch) & $L_t=105$ dB & 110 dB & 115 dB\\",
        r"\midrule",
    ]
    for i, vroom in enumerate(v_rooms):
        vals = [physics.demand_volume(15.0, spl, r=1.0, v_room=vroom, l_max=6.0)
                * 1e3 for spl in spls]
        prefix = r"$V_{\mathrm{room}}=40$ m$^3$" if i == 0 else f"${vroom:g}$ m$^3$"
        lines.append(f"{prefix} & " + " & ".join(f"{v:.1f}" for v in vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]

    write_text_checked(TABLES_DIR / "table_room_sizing.tex",
                       "\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Table: port air velocity (Sec. "Why sealed, not vented")
# ----------------------------------------------------------------------

def table_port_velocity() -> None:
    sc = SC_ROOM
    fb = 23.0
    p_t = sc.target_pressure
    area = vented.required_port_area(p_t, sc.r_listen, fb)
    diam = vented.port_diameter(area)
    typical_diam = 0.10
    typical_area = math.pi * (typical_diam / 2.0) ** 2
    v_typical = vented.port_velocity(p_t, sc.r_listen, fb, typical_area)

    lines = [
        r"\begin{center}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"quantity & value\\",
        r"\midrule",
        (rf"target & {sc.target_spl:g} dB, $r$={sc.r_listen:g} m, "
         rf"$F_b$={fb:g} Hz\\"),
        (r"turbulence onset $v_{\max}$ (Dickason) & "
         rf"{vented.TURBULENCE_VELOCITY_MAX:g} m/s\\"),
        (r"required port area / diameter for $v\le v_{\max}$ & "
         rf"{area*1e4:.0f} cm$^2$ / {diam*100:.1f} cm\\"),
        (rf"velocity of a {typical_diam*100:.0f} cm port at this target & "
         rf"{v_typical:.1f} m/s ($\approx${v_typical/vented.TURBULENCE_VELOCITY_MAX:.0f}$\times v_{{\max}}$)\\"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
    ]
    write_text_checked(TABLES_DIR / "table_port_velocity.tex",
                       "\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Console-only cross-check of the narrative (prose) numbers quoted in
# Secs. "Room closure" / "Worked example" / Appendix -- not written to
# any file, just printed so the LaTeX prose can be checked against code.
# ----------------------------------------------------------------------

def print_narrative_numbers() -> None:
    s = DRIVER_S
    vroom, lmax = 60.0, 6.0
    f_pz = physics.pressure_zone_frequency(lmax)
    v_dem = physics.demand_volume(15.0, 110.0, r=1.0, v_room=vroom, l_max=lmax)
    xi_1s = v_dem / s.vd
    xi_2s = v_dem / (2 * s.vd)
    x1_2s = min(xi_2s, 1.0) * s.xmax
    b71_s2 = BoxedDriver(s, qtc=0.71, n_units=2)
    box_hd_2s = physics.box_hd2(v_dem / 2, b71_s2.vb, s.qts, 0.71)

    p_t = physics.pressure_from_spl(110.0)
    w_ac = physics.acoustic_power_halfspace(p_t, 1.0)
    p_passband_per_unit = w_ac / (s.eta0 * 2 * 2)  # array-coherent scaling
    p_req = physics.eq_tax_power(max(15.0, f_pz), p_passband_per_unit,
                                 b71_s2.wc, s.sigma_m)
    xi_p_2s = p_req / s.p_max

    vd_s = s.vd
    spl_pz = 20 * math.log10(physics.RHO0 * physics.C_AIR ** 2 * vd_s
                             / (math.sqrt(2) * vroom * physics.P0))
    p_target = physics.pressure_from_spl(spl_pz)
    w_cross = math.sqrt(p_target * 2 * math.pi * 1.0 * math.sqrt(2)
                        / (physics.RHO0 * vd_s))
    f_cross = w_cross / (2 * math.pi)

    print("\n--- narrative cross-check (Sec. 'Room closure' / Appendix) ---")
    print(f"f_pz = {f_pz:.1f} Hz")
    print(f"V_dem(15 Hz, 110 dB, r=1, 60 m^3) = {v_dem*1e3:.2f} L")
    print(f"xi_x one S = {xi_1s:.2f}, two S = {xi_2s:.2f}")
    print(f"HD(two S) = {physics.harmonic_distortion(xi_2s)*100:.1f} %")
    print(f"X1(two S) = {x1_2s*1e3:.1f} mm")
    print(f"Doppler(two S, 80 Hz) = {physics.doppler_im(80.0, x1_2s)*100:.1f} %")
    print(f"Doppler(two S, 250 Hz) = {physics.doppler_im(250.0, x1_2s)*100:.1f} %")
    print(f"box_hd2(two S, 250 L, Qtc=0.71) = {box_hd_2s*100:.2f} %")
    print(f"xi_P(two S, array-coherent) = {xi_p_2s:.4f}")
    print(f"max_corner_rate(f_pz, 0.55) = {physics.max_corner_rate(f_pz, 0.55):.1f} Hz")
    print(f"SPL_pz(driver S, 60 m^3) = {spl_pz:.1f} dB")
    print(f"free-field crossing frequency = {f_cross:.1f} Hz")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        write_metadata,
        fig_spl_sub,
        fig_spl_attack,
        fig_distortion_sub,
        fig_distortion_attack,
        fig_demand,
        fig_vented_comparison,
        table_worked_example,
        table_room_sizing,
        table_port_velocity,
        print_narrative_numbers,
    ]
    if len(sys.argv) > 1:
        jobs = [job for job in jobs if job.__name__ in sys.argv[1:]]
    print("Starting make_figures, this can take a while, be patient...")
    for k, job in enumerate(jobs):
        t0 = time.time()
        job()
        print(f"Job {k} of {len(jobs)}: {job.__name__}: {time.time() - t0:.1f}s",
              flush=True)
    print(f"assets written to {FIGURES_DIR} and {TABLES_DIR}", flush=True)


if __name__ == "__main__":
    main()
