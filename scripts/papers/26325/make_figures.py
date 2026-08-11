"""Generate every figure and auto-generated table fragment for the revised
JAES Engineering Report (paper 26325), plus its `meta/metadata.tex`.

Single source of truth: run this whenever `physics.py`, `driver.py`,
`ranking.py`, `scenario.py`, or `plots.py` change, then recompile the paper.
Writes vector PDF figures into `papers/26325/figures/`, `.tex` table
fragments into `papers/26325/tables/`, a private-database evidence manifest
into `papers/26325/data/`, and `papers/26325/meta/metadata.tex`; all
committed to git so the paper stays buildable with `pdflatex` alone (no
Python required for a plain build).

This script uses **one canonical `Scenario`** (the defaults of
`audioshape.scenario.Scenario`, which already match the revised report's
architecture: 60 m^3 room, L_max 6 m, 3 m couch reference, mono sub target
110 dB total, upper-bass target 105 dB per stereo channel, 15/80/250 Hz
band split, leaky pressure zone with a 10 Hz corner, Qtc ceiling 0.55, box
cap min(10x Vas, 1.0 m^3 per role/channel), 80% preferred excursion with
Xmax=1.0 the only hard gate, a 90 V/15 A/500 W/1000 W amplifier per
physical driver, and a one-cycle rectangular burst sampled at 8 start
phases) and **two independently public-datasheet-sourced drivers** (never
rows copied from the private VituixCAD database):

- BMS 18N862: mono sub manifold, 4 identical units in an even,
  symmetrically opposed pairing (permitting first-order reaction-force
  cancellation under matched drive and mounting; the even count is a hard
  mechanical-layout constraint, not a ranking-policy preference; the
  candidate pool is capped at 18in), each assumed to have its own
  amplifier channel.
  https://bmsspeakers.com/product/18-neodymium-ultra-low-distortion-woofer-2/
- Eighteen Sound 12NTLW3500 (8 ohm): upper-bass, one unit per independent
  stereo channel.
  https://www.eighteensound.it/en/products/lf-driver/12-0/8/12ntlw3500

Aggregate private-database evidence (counts, hexbin/histogram population
plots, no row-level data) is produced from `data/VituixCAD_driver_db.txt`,
which is obtained/exported separately and never redistributed (see
`data/README.md`). Jobs that need it raise a clear `FileNotFoundError` if it
is absent; jobs that do not need it keep working from a fresh clone.

Usage (from repo root):  scripts\\figures -p 26325
Or directly:              uv run python scripts/make_figures.py -p 26325
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from audioshape import database, physics, plots
from audioshape.driver import Driver
from audioshape.ranking import Evaluation, evaluate, pareto_front, rank
from audioshape.scenario import Scenario

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_ID = SCRIPT_DIR.name
BASEDIR = SCRIPT_DIR.parents[2]
PAPER_DIR = BASEDIR / "papers" / PAPER_ID
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
META_DIR = PAPER_DIR / "meta"
DATA_DIR = PAPER_DIR / "data"
PAPER_METADATA_FILE = SCRIPT_DIR / "metadata.json"
SITE_METADATA_FILE = BASEDIR / "scripts" / "metadata.json"
DB_PATH = BASEDIR / "data" / "VituixCAD_driver_db.txt"

# Add scripts dir to path for file_utils import
sys.path.insert(0, str(BASEDIR / "scripts"))
from file_utils import save_figure_checked, write_text_checked  # noqa: E402


def _save_figure(fig, path: Path) -> None:
    """save_figure_checked with a fixed PDF CreationDate so re-running this
    script on unchanged inputs produces byte-identical, diff-quiet output
    (matplotlib's PDF backend otherwise embeds the wall-clock save time)."""
    save_figure_checked(fig, path, metadata={"CreationDate": None})


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
# Canonical scenario and the two independently public-datasheet-sourced
# drivers.  See module docstring for the source URLs.
# ----------------------------------------------------------------------

SCENARIO = Scenario()  # the canonical scenario *is* the Scenario default

# Final selection (confirmed): BMS 18N862, 4 identical units in two
# mechanically opposed pairs, sub candidate pool capped at 18in. At 3 units
# B&C 21IPAL had been the working pick, but moving to 4 units spreads the
# manifold's displacement/electrical demand over one more driver and made
# several 18in candidates newly feasible; BMS 18N862 was selected from that
# 18in-capped, 4-unit re-ranking (see docs/plans and the exploratory
# 18in-vs-21in comparison run alongside this change).
DRIVER_SUB = Driver(
    manufacturer="BMS", model="18N862", size_in=18,
    fs=25.1, qes=0.36, qms=6.75, re=5.56, mms=0.267,
    sd=0.1219, xmax=0.019, vas=0.312, p_max=1500.0,
    bl=25.52, le=0.00081,
)
DRIVER_UPPER = Driver(
    manufacturer="Eighteen Sound", model="12NTLW3500", size_in=12,
    fs=53.0, qes=0.35, qms=8.0, re=5.1, mms=0.080,
    sd=0.0531, xmax=0.0083, vas=0.045, p_max=900.0,
    bl=19.5, le=0.00046,
)

SUB_UNITS = 4     # mono manifold: an ODD unit count cannot be arranged as
                  # symmetrically opposed pairs, so it cannot cancel the
                  # reaction force each driver exerts on its shared
                  # enclosure/baffle; an EVEN, opposed-pair count is a hard
                  # mechanical-layout constraint here, not a ranking-policy
                  # preference (4 identical units, own amplifier channel
                  # each; 110 dB total is the whole manifold's target, not
                  # per driver).
UPPER_UNITS = 1   # one driver per independent stereo channel

# Physically-scoped role candidate pools, mirroring examples/example_recipe.toml:
# a mono opposed-pair manifold is only sensible for a 15in+ class driver,
# and the sub candidate pool is capped at 18in (confirmed selection); the
# upper-bass role is fixed at one 12in driver per channel.
SUB_SIZE_MIN = 15.0
SUB_SIZE_MAX = 18.0
UPPER_SIZE_MIN = UPPER_SIZE_MAX = 12.0

# VituixCAD "Type" column values that are bass-relevant (subwoofer/woofer/
# mid-woofer rows); used only for aggregate population evidence, never for
# row-level redistribution.
BASS_TYPE_CODES = ("S", "W", "WM")

ROBUSTNESS_TOP_K = 10


def _even_ceil(n: float) -> int:
    """Round a required sub-manifold driver count up to the nearest even
    integer (>= 2): an odd count cannot be arranged as symmetrically
    opposed pairs, so it cannot cancel each driver's reaction force on the
    shared enclosure/baffle -- a hard mechanical-layout constraint, not a
    ranking-policy rounding choice.
    """
    units = max(2, math.ceil(n))
    return units + (units % 2)


def _ev_sub(driver: Driver = DRIVER_SUB, scenario: Scenario = SCENARIO) -> Evaluation:
    return evaluate(driver, scenario, n_units=SUB_UNITS,
                    band_low=scenario.f_low, band_high=scenario.f_split,
                    doppler_ref=scenario.f_split, role="sub")


def _ev_upper(driver: Driver = DRIVER_UPPER, scenario: Scenario = SCENARIO) -> Evaluation:
    return evaluate(driver, scenario, n_units=UPPER_UNITS,
                    band_low=scenario.f_split, band_high=scenario.f_high,
                    doppler_ref=scenario.f_high, role="attack")


# ----------------------------------------------------------------------
# Private database access (aggregate evidence only -- see module
# docstring).  Cached per process since a full role ranking over the
# private database is the expensive step and several figures/tables reuse
# the same result.
# ----------------------------------------------------------------------

def _require_database() -> Path:
    if not DB_PATH.is_file():
        raise FileNotFoundError(
            f"local VituixCAD driver database not found at {DB_PATH}. It is "
            "obtained/exported separately and is never distributed with "
            "this repository (see data/README.md); place it there to "
            "regenerate the private-database figures, tables, and "
            "manifest. Already-committed assets remain valid for a clean "
            "clone."
        )
    return DB_PATH


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _database_result():
    return database.parse_database(_require_database())


@lru_cache(maxsize=1)
def _bass_population() -> tuple[Driver, ...]:
    return tuple(d for d in _database_result().drivers if d.type_code in BASS_TYPE_CODES)


@lru_cache(maxsize=1)
def _sub_candidates() -> tuple[Driver, ...]:
    return tuple(d for d in _database_result().drivers
                if SUB_SIZE_MIN <= d.size_in <= SUB_SIZE_MAX)


@lru_cache(maxsize=1)
def _upper_candidates() -> tuple[Driver, ...]:
    return tuple(d for d in _database_result().drivers
                if UPPER_SIZE_MIN <= d.size_in <= UPPER_SIZE_MAX)


def _rank_sub(drivers: tuple[Driver, ...] | None = None,
             scenario: Scenario = SCENARIO) -> list[Evaluation]:
    return rank(list(drivers) if drivers is not None else list(_sub_candidates()),
               scenario, n_units=SUB_UNITS, band_low=scenario.f_low,
               band_high=scenario.f_split, doppler_ref=scenario.f_split, role="sub")


def _rank_upper(drivers: tuple[Driver, ...] | None = None,
               scenario: Scenario = SCENARIO) -> list[Evaluation]:
    return rank(list(drivers) if drivers is not None else list(_upper_candidates()),
               scenario, n_units=UPPER_UNITS, band_low=scenario.f_split,
               band_high=scenario.f_high, doppler_ref=scenario.f_high, role="attack")


@lru_cache(maxsize=1)
def _sub_role_evaluations() -> tuple[Evaluation, ...]:
    return tuple(_rank_sub())


@lru_cache(maxsize=1)
def _upper_role_evaluations() -> tuple[Evaluation, ...]:
    return tuple(_rank_upper())


@lru_cache(maxsize=1)
def _sub_role_with_selected() -> tuple[Evaluation, ...]:
    """Sub-role ranking with the selected public driver appended, so its
    Pareto front/position can be reported against the frozen database."""
    return tuple(_rank_sub(_sub_candidates() + (DRIVER_SUB,)))


@lru_cache(maxsize=1)
def _upper_role_with_selected() -> tuple[Evaluation, ...]:
    return tuple(_rank_upper(_upper_candidates() + (DRIVER_UPPER,)))


def _find_evaluation(evals: tuple[Evaluation, ...], driver: Driver) -> Evaluation:
    for ev in evals:
        if ev.driver is driver:
            return ev
    raise ValueError(f"{driver.label()} not found in its own ranked candidate pool")


@lru_cache(maxsize=1)
def _dual_role_counts() -> tuple[int, int, int]:
    """Empirical database observation: among the full bass-relevant
    population (no per-role size restriction), how many rows independently
    satisfy the sub band's feasibility gates, the upper band's, and both at
    once.  Illustrative only -- the architecture requires physically
    distinct driver classes per role (a 15in+ manifold vs. a 12in
    per-channel driver), so this is not a dual-purpose recommendation.
    """
    population = _bass_population()
    sub_evals = _rank_sub(population)
    upper_evals = _rank_upper(population)
    sub_feasible = {id(ev.driver) for ev in sub_evals if ev.feasible}
    upper_feasible = {id(ev.driver) for ev in upper_evals if ev.feasible}
    return len(sub_feasible), len(upper_feasible), len(sub_feasible & upper_feasible)


# ----------------------------------------------------------------------
# Rank-stability (robustness) variants
# ----------------------------------------------------------------------

def _topk_labels(evals, k: int) -> set[str]:
    feasible = [ev for ev in evals if ev.feasible]
    return {ev.driver.label() for ev in feasible[:k]}


def _overlap(baseline: set[str], other: set[str], k: int) -> float:
    return len(baseline & other) / k


def _xmax_variant(drivers: tuple[Driver, ...], factor: float) -> tuple[Driver, ...]:
    return tuple(replace(d, xmax=d.xmax * factor) for d in drivers)


def _pmax_variant_db(drivers: tuple[Driver, ...], delta_db: float) -> tuple[Driver, ...]:
    factor = 10.0 ** (delta_db / 10.0)  # power ratio for a +/- dB rating shift
    return tuple(replace(d, p_max=d.p_max * factor) for d in drivers)


def _policy_variant_topk(evals, k: int) -> set[str]:
    """Illustrative, non-canonical re-ordering of the same feasible pool:
    amplifier headroom first instead of the canonical excursion-first
    policy key (`ranking.Evaluation.policy_key`).  Does not modify
    `ranking.py`; it only demonstrates one reasonable alternative ordering
    for the rank-stability check.
    """
    feasible = [ev for ev in evals if ev.feasible]
    reordered = sorted(
        feasible,
        key=lambda ev: (
            ev.pareto_rank,
            ev.amplifier_utilization,
            ev.xi_x,
            ev.driver.label(),
        ),
    )
    return {ev.driver.label() for ev in reordered[:k]}


def _rank_robustness_data() -> list[tuple[str, float, float]]:
    k = ROBUSTNESS_TOP_K
    base_sub = _sub_role_evaluations()
    base_upper = _upper_role_evaluations()
    base_sub_top = _topk_labels(base_sub, k)
    base_upper_top = _topk_labels(base_upper, k)

    variants: list[tuple[str, float, float]] = []

    def add(label: str, sub_evals, upper_evals) -> None:
        variants.append((
            label,
            _overlap(base_sub_top, _topk_labels(sub_evals, k), k),
            _overlap(base_upper_top, _topk_labels(upper_evals, k), k),
        ))

    add("Xmax +25%",
        _rank_sub(_xmax_variant(_sub_candidates(), 1.25)),
        _rank_upper(_xmax_variant(_upper_candidates(), 1.25)))
    add("Xmax -25%",
        _rank_sub(_xmax_variant(_sub_candidates(), 0.75)),
        _rank_upper(_xmax_variant(_upper_candidates(), 0.75)))
    add("Pmax +3 dB",
        _rank_sub(_pmax_variant_db(_sub_candidates(), 3.0)),
        _rank_upper(_pmax_variant_db(_upper_candidates(), 3.0)))
    add("Pmax -3 dB",
        _rank_sub(_pmax_variant_db(_sub_candidates(), -3.0)),
        _rank_upper(_pmax_variant_db(_upper_candidates(), -3.0)))
    add("leakage 5 Hz",
        _rank_sub(scenario=replace(SCENARIO, leakage_corner_hz=5.0)),
        _rank_upper(scenario=replace(SCENARIO, leakage_corner_hz=5.0)))
    add("leakage 15 Hz",
        _rank_sub(scenario=replace(SCENARIO, leakage_corner_hz=15.0)),
        _rank_upper(scenario=replace(SCENARIO, leakage_corner_hz=15.0)))
    variants.append((
        "policy: amplifier-first",
        _overlap(base_sub_top, _policy_variant_topk(base_sub, k), k),
        _overlap(base_upper_top, _policy_variant_topk(base_upper, k), k),
    ))
    return variants


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def fig_architecture() -> None:
    _save_figure(plots.architecture_figure(SCENARIO, sub_units=SUB_UNITS),
                FIGURES_DIR / "fig_architecture.pdf")


def fig_output_electrical() -> None:
    fig = plots.electrical_utilization_figure(_ev_sub(), _ev_upper())
    _save_figure(fig, FIGURES_DIR / "fig_output_electrical.pdf")


def fig_transient_factor() -> None:
    ev_s, ev_u = _ev_sub(), _ev_upper()
    worked_points = [
        (f"{DRIVER_SUB.model} worked point",
         ev_s.transient.shape_frequency / ev_s.boxed.fc, ev_s.boxed.qtc),
        (f"{DRIVER_UPPER.model} worked point",
         ev_u.transient.shape_frequency / ev_u.boxed.fc, ev_u.boxed.qtc),
    ]
    fig = plots.transient_factor_figure(worked_points=worked_points)
    _save_figure(fig, FIGURES_DIR / "fig_transient_factor.pdf")


def fig_database_pareto() -> None:
    population = _bass_population()
    vd = np.array([d.vd for d in population])
    eta0_pmax = np.array([d.eta0 * d.p_max for d in population])
    selected = [
        (f"{DRIVER_SUB.manufacturer} {DRIVER_SUB.model}",
         DRIVER_SUB.vd, DRIVER_SUB.eta0 * DRIVER_SUB.p_max),
        (f"{DRIVER_UPPER.manufacturer} {DRIVER_UPPER.model}",
         DRIVER_UPPER.vd, DRIVER_UPPER.eta0 * DRIVER_UPPER.p_max),
    ]
    fig = plots.database_pareto_figure(vd, eta0_pmax, selected=selected)
    _save_figure(fig, FIGURES_DIR / "fig_database_pareto.pdf")


def fig_corner_population() -> None:
    population = _bass_population()
    corner_rate = np.array([d.corner_rate for d in population])
    stroke_mm = np.array([d.xmax * 1e3 for d in population])
    selected = [
        (f"{DRIVER_SUB.manufacturer} {DRIVER_SUB.model}",
         DRIVER_SUB.corner_rate, DRIVER_SUB.xmax * 1e3),
        (f"{DRIVER_UPPER.manufacturer} {DRIVER_UPPER.model}",
         DRIVER_UPPER.corner_rate, DRIVER_UPPER.xmax * 1e3),
    ]
    fig = plots.corner_population_figure(corner_rate, stroke_mm, selected=selected)
    _save_figure(fig, FIGURES_DIR / "fig_corner_population.pdf")


def fig_rank_robustness() -> None:
    variants = _rank_robustness_data()
    fig = plots.rank_robustness_figure(
        [v[0] for v in variants], [v[1] for v in variants], [v[2] for v in variants],
        ROBUSTNESS_TOP_K,
    )
    _save_figure(fig, FIGURES_DIR / "fig_rank_robustness.pdf")


def fig_room_sensitivity() -> None:
    fig = plots.room_sensitivity_figure(SCENARIO, leakage_variants=(0.0, 5.0, 10.0, 15.0))
    _save_figure(fig, FIGURES_DIR / "fig_room_sensitivity.pdf")


def fig_worked_system() -> None:
    fig = plots.worked_system_figure(_ev_sub(), _ev_upper())
    _save_figure(fig, FIGURES_DIR / "fig_worked_system.pdf")


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def table_prior_art() -> None:
    """Established prior results this report integrates vs. its own
    methodological/policy contribution (claim-hierarchy Table T1)."""
    rows = [
        ("Sealed-box alignment, Thiele/Small closed-box theory",
         "established prior result",
         "reused unchanged as the sealed-box ODE and alignment line"),
        (r"Displacement-limited output ceiling ($V_d=S_dX_{\max}$)",
         "established prior result",
         "used as an exact descriptor, not a universal loudspeaker optimum"),
        (r"Thermal (dissipation-limited) output ceiling ($\eta_0P_{\max}$)",
         "established prior result",
         "used as an exact descriptor; never summed with the displacement ceiling"),
        (r"$X_{\max}$ as an IEC/Klippel excursion anchor",
         "established prior result",
         "used only as the clipping boundary; not converted into absolute THD"),
        ("Room pressure-zone compliance below the first mode",
         "established prior result",
         "extended with an explicit first-order leakage-corner term; used "
         "only as a first-pass sizing/limiting model, not a claim about "
         "the complete room"),
        ("Doppler (FM) sideband index",
         "established prior result",
         "reported as a separate, non-summed nonlinear-risk indicator"),
        ("Finite-band DSP room correction",
         "established prior result",
         "adopted as an explicit scope input rather than an unconstrained assumption"),
        ("Role-specific, non-compensatory ranking policy (mono sub "
         "manifold vs. independent stereo upper-bass channels)",
         "this report's contribution",
         "declared engineering policy; separate Pareto fronts per role, "
         "no cross-role or cross-channel summing credit"),
        ("Explicit finite-amplifier feasibility gates (voltage, current, "
         "continuous and burst power)",
         "this report's contribution",
         "integrated into every role evaluation and the worked example"),
        ("Numerical sealed-ODE transient (burst) displacement factor",
         "this report's contribution",
         r"replaces a constant free-mass burst factor near $F_c$"),
        ("Aggregate, hash-verified private-database screening evidence",
         "this report's contribution",
         "counts and hexbin/histogram population evidence only, no "
         "row-level redistribution"),
    ]
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Established prior results integrated by this report vs. "
        r"its own methodological and policy contribution.}",
        r"\label{tab:prior-art}",
        r"\begin{tabular}{p{5.1cm}p{2.6cm}p{6.7cm}}",
        r"\toprule",
        r"element & class & this report's treatment\\",
        r"\midrule",
    ]
    for element, cls, treatment in rows:
        lines.append(f"{element} & {cls} & {treatment}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_text_checked(TABLES_DIR / "table_prior_art.tex", "\n".join(lines) + "\n")


def table_database_summary() -> None:
    db_path = _require_database()
    parse_result = _database_result()
    sha = _sha256(db_path)
    sub_evals = _sub_role_evaluations()
    upper_evals = _upper_role_evaluations()
    sub_feasible_n = sum(ev.feasible for ev in sub_evals)
    upper_feasible_n = sum(ev.feasible for ev in upper_evals)
    sub_pareto_n = len(pareto_front(sub_evals))
    upper_pareto_n = len(pareto_front(upper_evals))
    _, _, dual_n = _dual_role_counts()

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Private driver-database evidence summary (aggregate "
        r"counts only. The accompanying manifest supplies the full source "
        r"hash, parser, and filter definitions).}",
        r"\label{tab:database-summary}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"quantity & value\\",
        r"\midrule",
        rf"source hash (SHA-256, first 12 hex) & \texttt{{{sha[:12]}}}\\",
        rf"retained rows & {len(parse_result.drivers)}\\",
        rf"skipped rows (missing required fields) & {len(parse_result.skipped)}\\",
        rf"sub-manifold role candidates ({SUB_SIZE_MIN:g}--{SUB_SIZE_MAX:g}in) & "
        rf"{len(_sub_candidates())}\\",
        rf"sub-manifold role feasible & {sub_feasible_n}\\",
        rf"sub-manifold role Pareto front & {sub_pareto_n}\\",
        rf"upper-bass role candidates ({UPPER_SIZE_MIN:g}in) & "
        rf"{len(_upper_candidates())}\\",
        rf"upper-bass role feasible & {upper_feasible_n}\\",
        rf"upper-bass role Pareto front & {upper_pareto_n}\\",
        rf"dual-role feasible (size-unrestricted, illustrative only) & {dual_n}\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    write_text_checked(TABLES_DIR / "table_database_summary.tex",
                       "\n".join(lines) + "\n")


def table_worked_pair() -> None:
    sc = SCENARIO
    ev_s = _find_evaluation(_sub_role_with_selected(), DRIVER_SUB)
    ev_u = _find_evaluation(_upper_role_with_selected(), DRIVER_UPPER)
    s, u = DRIVER_SUB, DRIVER_UPPER

    rows = [
        ("public source", r"BMS, bmsspeakers.com (official product page)",
         r"Eighteen Sound, eighteensound.it (8 ohm)"),
        ("role, units",
         f"mono sub manifold, {SUB_UNITS}x (even, opposed pairs for "
         f"first-order force cancellation when matched), aggregate target "
         f"{sc.sub_target_spl:g} dB for the manifold",
         f"stereo upper-bass, {UPPER_UNITS}x per channel, target "
         f"{sc.attack_target_spl:g} dB per channel"),
        (r"$V_d$ per driver [L]", f"{s.vd*1e3:.2f}", f"{u.vd*1e3:.2f}"),
        (r"$\eta_0 P_{\max}$ [W$_{\mathrm{ac}}$]",
         f"{s.eta0*s.p_max:.2f}", f"{u.eta0*u.p_max:.2f}"),
        (r"$F_s$ / $Q_{ts}$", f"{s.fs:g} Hz / {s.qts:.3f}",
         f"{u.fs:g} Hz / {u.qts:.3f}"),
        (r"$f_L=\Re/2\pi\Le$ [Hz]", f"{s.f_le:.0f}", f"{u.f_le:.0f}"),
        (r"box per driver: $V_b$, $F_c$",
         f"{ev_s.boxed.vb*1e3:.0f} L, {ev_s.boxed.fc:.1f} Hz",
         f"{ev_u.boxed.vb*1e3:.1f} L, {ev_u.boxed.fc:.1f} Hz"),
        (r"$Q_{tc}$", f"{ev_s.boxed.qtc:.3f}", f"{ev_u.boxed.qtc:.3f}"),
        ("worst steady excursion util.", f"{ev_s.xi_x:.2f}", f"{ev_u.xi_x:.2f}"),
        ("worst transient excursion util.",
         f"{ev_s.xi_x_transient:.2f}", f"{ev_u.xi_x_transient:.2f}"),
        ("worst voltage [V rms @ Hz]",
         f"{ev_s.electrical.voltage_rms:.1f} @ {ev_s.electrical.voltage_frequency:.1f}",
         f"{ev_u.electrical.voltage_rms:.1f} @ {ev_u.electrical.voltage_frequency:.1f}"),
        ("worst current [A rms @ Hz]",
         f"{ev_s.electrical.current_rms:.1f} @ {ev_s.electrical.current_frequency:.1f}",
         f"{ev_u.electrical.current_rms:.1f} @ {ev_u.electrical.current_frequency:.1f}"),
        ("worst power [W @ Hz]",
         f"{ev_s.electrical.coil_power_w:.0f} @ {ev_s.electrical.power_frequency:.1f}",
         f"{ev_u.electrical.coil_power_w:.0f} @ {ev_u.electrical.power_frequency:.1f}"),
        ("Pareto front / status",
         f"{ev_s.pareto_rank} / {'feasible' if ev_s.feasible else 'flagged'}",
         f"{ev_u.pareto_rank} / {'feasible' if ev_u.feasible else 'flagged'}"),
    ]

    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Worked selected pair: BMS 18N862 (mono sub "
        rf"manifold, {SUB_UNITS} identical units in an even, symmetrically "
        r"opposed pairing -- permitting first-order reaction-force "
        r"cancellation under matched drive and mounting; the even count is "
        r"a hard mechanical-layout constraint, not a ranking-policy "
        r"preference -- "
        r"each with its own amplifier channel, aggregate target "
        rf"{sc.sub_target_spl:g} dB for the manifold, not per driver) and "
        r"Eighteen Sound 12NTLW3500 (upper-bass, one unit "
        rf"per independent stereo channel, target {sc.attack_target_spl:g}"
        r" dB per channel). Pareto front number is one-based and is against "
        r"this report's frozen private-database candidate pool for each role.}",
        r"\label{tab:worked-pair}",
        r"\small",
        r"\begin{tabular}{p{4.0cm}p{5.1cm}p{5.1cm}}",
        r"\toprule",
        r" & BMS 18N862 (sub) & 18 Sound 12NTLW3500 (upper)\\",
        r"\midrule",
    ]
    for label, vs, vu in rows:
        lines.append(f"{label} & {vs} & {vu}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_text_checked(TABLES_DIR / "table_worked_pair.tex", "\n".join(lines) + "\n")


def table_room_sensitivity() -> None:
    sc = SCENARIO
    leakage_variants = (0.0, 5.0, 10.0, 15.0)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{Room-demand sensitivity to the leakage corner at the "
        rf"worst (lowest) sub-band frequency, {sc.sub_target_spl:g} dB "
        rf"target at the {sc.r_listen:g} m couch basis "
        rf"($V_{{\mathrm{{room}}}}$={sc.v_room:g} m$^3$, "
        rf"$L_{{\max}}$={sc.l_max:g} m). Required units use the selected "
        r"BMS 18N862 sub driver's own $V_d$, rounded up to the nearest "
        r"even count (opposed pairs permit first-order reaction-force "
        r"cancellation under matched drive and mounting; the even count is "
        r"a hard mechanical-layout constraint).}",
        r"\label{tab:room-sensitivity}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"leakage corner & worst $V_{\mathrm{dem}}$ [L] & units at "
        r"$X_{\max}$ & units at 80\% preferred\\",
        r"\midrule",
    ]
    for leak in leakage_variants:
        model = "ideal_pressure_zone" if leak <= 0.0 else "leaky_pressure_zone"
        label = "ideal (0 Hz)" if leak <= 0.0 else f"{leak:g} Hz"
        v_dem = physics.demand_volume(
            sc.f_low, sc.sub_target_spl, sc.r_listen, sc.v_room, sc.l_max,
            room_model=model, leakage_corner_hz=leak,
        )
        units_xmax = _even_ceil(v_dem / DRIVER_SUB.vd)
        units_preferred = _even_ceil(
            v_dem / (DRIVER_SUB.vd * sc.preferred_excursion)
        )
        lines.append(
            f"{label} & {v_dem*1e3:.2f} & {units_xmax} & {units_preferred}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_text_checked(TABLES_DIR / "table_room_sensitivity.tex",
                       "\n".join(lines) + "\n")


def write_driver_database_manifest() -> None:
    db_path = _require_database()
    parse_result = _database_result()
    population = _bass_population()
    sub_evals = _sub_role_evaluations()
    upper_evals = _upper_role_evaluations()
    sub_feasible_n = sum(ev.feasible for ev in sub_evals)
    upper_feasible_n = sum(ev.feasible for ev in upper_evals)
    sub_pareto_n = len(pareto_front(sub_evals))
    upper_pareto_n = len(pareto_front(upper_evals))
    sub_dual_n, upper_dual_n, dual_n = _dual_role_counts()

    manifest = {
        "source": {
            "description": (
                "VituixCAD community driver database (tab-separated "
                "manufacturer-datasheet export), obtained/exported "
                "separately and never redistributed with this repository."
            ),
            "local_path": "data/VituixCAD_driver_db.txt",
            "retrieval_date": None,
            "retrieval_date_note": (
                "Not recorded; the source was already held locally when "
                "the publication manifest was created."
            ),
            "sha256": _sha256(db_path),
        },
        "parser": {
            "module": "audioshape.database.parse_database",
            "required_datasheet_fields": [
                "Size [in]", "Re [ohm]", "fs [Hz]", "Qms", "Qes",
                "Mms [g]", "Vas [l]", "Sd [cm2]", "Pmax [W]", "Xmax [mm]",
            ],
            "retained_rows": len(parse_result.drivers),
            "skipped_rows": len(parse_result.skipped),
        },
        "filters": {
            "bass_relevant_type_codes": list(BASS_TYPE_CODES),
            "bass_relevant_population": len(population),
            "sub_role_candidate_size_in_min": SUB_SIZE_MIN,
            "sub_role_candidate_size_in_max": SUB_SIZE_MAX,
            "sub_role_candidate_population": len(_sub_candidates()),
            "upper_role_candidate_size_in_min": UPPER_SIZE_MIN,
            "upper_role_candidate_size_in_max": UPPER_SIZE_MAX,
            "upper_role_candidate_population": len(_upper_candidates()),
        },
        "role_feasibility": {
            "sub_manifold_units": SUB_UNITS,
            "sub_manifold_layout_constraint": (
                "even, symmetrically opposed pairs, permitting first-order "
                "reaction-force cancellation under matched drive and "
                "mounting; the even count is a hard mechanical-layout "
                "constraint, not a ranking-policy preference -- an odd "
                "unit count is not considered. The 110 dB sub target is "
                "the total manifold output, not per driver."
            ),
            "sub_role_feasible": sub_feasible_n,
            "sub_role_pareto_front": sub_pareto_n,
            "upper_channel_units": UPPER_UNITS,
            "upper_role_feasible": upper_feasible_n,
            "upper_role_pareto_front": upper_pareto_n,
            "dual_role_note": (
                "size-unrestricted count over the full bass-relevant "
                "population, illustrative only: the architecture requires "
                "physically distinct driver classes per role (15in+ "
                "manifold vs. 12in per-channel), so this is not a "
                "dual-purpose recommendation."
            ),
            "dual_role_sub_band_feasible": sub_dual_n,
            "dual_role_upper_band_feasible": upper_dual_n,
            "dual_role_both_feasible": dual_n,
        },
        "canonical_scenario": asdict(SCENARIO),
        "selected_public_datasheet_records": {
            "sub": {
                "role": (
                    "four-driver mono manifold in two mechanically opposed "
                    "pairs; 110 dB is the complete-manifold target"
                ),
                "manufacturer": DRIVER_SUB.manufacturer,
                "model": DRIVER_SUB.model,
                "source_url": (
                    "https://bmsspeakers.com/product/"
                    "18-neodymium-ultra-low-distortion-woofer-2/"
                ),
                "parameters_si": asdict(DRIVER_SUB),
            },
            "upper_bass": {
                "role": (
                    "one driver per independent stereo channel; "
                    "105 dB is the per-channel target"
                ),
                "manufacturer": DRIVER_UPPER.manufacturer,
                "model": "12NTLW3500-8",
                "source_url": (
                    "https://www.eighteensound.it/en/products/lf-driver/"
                    "12-0/8/12ntlw3500"
                ),
                "parameters_si": asdict(DRIVER_UPPER),
                "parameter_note": (
                    "The public manufacturer Xmax value used here is "
                    "8.3 mm one-way."
                ),
            },
        },
    }
    write_text_checked(
        DATA_DIR / "driver_database_manifest.json",
        json.dumps(manifest, indent=2) + "\n",
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        write_metadata,
        fig_architecture,
        fig_output_electrical,
        fig_transient_factor,
        fig_room_sensitivity,
        fig_worked_system,
        fig_database_pareto,
        fig_corner_population,
        fig_rank_robustness,
        table_prior_art,
        table_room_sensitivity,
        table_worked_pair,
        table_database_summary,
        write_driver_database_manifest,
    ]
    if len(sys.argv) > 1:
        jobs = [job for job in jobs if job.__name__ in sys.argv[1:]]
    print("Starting make_figures, this can take a while, be patient...")
    for k, job in enumerate(jobs):
        t0 = time.time()
        job()
        print(f"Job {k} of {len(jobs)}: {job.__name__}: {time.time() - t0:.1f}s",
              flush=True)
    print(f"assets written to {FIGURES_DIR}, {TABLES_DIR}, and {DATA_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
