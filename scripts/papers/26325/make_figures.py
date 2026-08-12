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
band split, leaky pressure zone with a 10 Hz corner, preferred alignment
Qtc 0.55, box cap 4x Vas per physical driver, 80% preferred excursion with
Xmax=1.0 the only hard gate, a 90 V/15 A/500 W/1000 W amplifier per
physical driver, and a one-cycle rectangular burst sampled at 8 start
phases) and **two independently public-datasheet-sourced drivers** (never
rows copied from the private VituixCAD database):

- Dayton Audio UMII18-22: mono sub manifold, 2 identical units in one
  symmetrically opposed pair (permitting first-order reaction-force
  cancellation under matched drive and mounting; the even count is a hard
  mechanical-layout constraint), each with its own amplifier channel.
  https://www.daytonaudio.com/images/resources/
  295-718--dayton-audio-UMII18-22-spec-sheet.pdf
- FaitalPRO 12HP1030 (8 ohm): upper-bass, one unit per independent stereo
  channel.
  https://faitalpro.com/en/products/LF_Loudspeakers/product_details/
  index.php?id=201050130

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
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from audioshape import architecture, database, physics, plots
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

# Revised public worked pair after making Qtc an alignment target rather than
# a hard gate and replacing the 1 m3 role cap with 4x Vas per driver. The
# high-Qts/high-stroke Ultimax is now charged for its large enclosure and EQ
# demand rather than rejected geometrically.
DRIVER_SUB = Driver(
    manufacturer="Dayton Audio", model="UMII18-22", size_in=18,
    fs=22.0, qes=0.67, qms=2.53, re=4.2, mms=0.420,
    sd=0.1184, xmax=0.028, vas=0.2482, p_max=1200.0,
    bl=19.2, le=0.00115,
)
DRIVER_UPPER = Driver(
    manufacturer="FaitalPRO", model="12HP1030", size_in=12,
    fs=45.0, qes=0.31, qms=13.8, re=5.0, mms=0.1305,
    sd=0.0518, xmax=0.01245, vas=0.0359, p_max=1000.0,
    bl=24.3, le=0.00135,
)

SUB_UNITS = 2     # mono manifold: an ODD unit count cannot be arranged as
                  # symmetrically opposed pairs, so it cannot cancel the
                  # reaction force each driver exerts on its shared
                  # enclosure/baffle; an EVEN, opposed-pair count is a hard
                  # mechanical-layout constraint here, not a ranking-policy
                  # preference (2 identical units, own amplifier channel
                  # each; 110 dB total is the whole manifold's target, not
                  # per driver).
UPPER_UNITS = 1   # one driver per independent stereo channel

# Physically scoped role pools. Diameter is a selection output inside these
# user-declared packaging ranges, not a value fixed to the worked example.
SUB_SIZE_MIN, SUB_SIZE_MAX = 15.0, 21.0
UPPER_SIZE_MIN, UPPER_SIZE_MAX = 8.0, 15.0

# VituixCAD "Type" column values that are bass-relevant (subwoofer/woofer/
# mid-woofer rows); used only for aggregate population evidence, never for
# row-level redistribution.
BASS_TYPE_CODES = ("S", "W", "WM")

ROBUSTNESS_TOP_K = 10

# Architecture-comparison envelope requested after the first revision.
# Room pairs are explicit because volume alone does not determine f_pz; the
# 90 m^3 / 9 m case intentionally reaches f_pz < 20 Hz.
ARCHITECTURE_ROOM_CASES = (
    (40.0, 5.0),
    (60.0, 6.0),
    (90.0, 9.0),
)
# The multidimensional factorial uses four crossover levels; a separate
# canonical 5 Hz sweep resolves the 60--120 Hz crossover axis in detail.
ARCHITECTURE_SPLITS_HZ = (60.0, 80.0, 100.0, 120.0)
ARCHITECTURE_HIGH_EDGES_HZ = (200.0, 250.0, 350.0)
ARCHITECTURE_SUB_TARGETS_DB = (105.0, 110.0, 115.0)
ARCHITECTURE_BASE_SUB_TARGET_DB = min(ARCHITECTURE_SUB_TARGETS_DB)
ARCHITECTURE_BASE_UPPER_TARGET_DB = ARCHITECTURE_BASE_SUB_TARGET_DB - 5.0
ARCHITECTURE_SUB_UNIT_OPTIONS = (2, 4, 6)
ARCHITECTURE_UPPER_UNIT_OPTIONS = (1, 2)
ARCHITECTURE_SINGLE_UNIT_OPTIONS = (1, 2)
ARCHITECTURE_SUB_SIZE_RANGE = (15.0, 21.0)
ARCHITECTURE_UPPER_SIZE_RANGE = (8.0, 15.0)
ARCHITECTURE_SINGLE_SIZE_RANGE = (10.0, 18.0)


@dataclass(frozen=True)
class ArchitectureStudyPoint:
    """Aggregate-only result for one architecture-comparison scenario."""

    room_volume_m3: float
    longest_dimension_m: float
    pressure_zone_hz: float
    split_hz: float
    high_edge_hz: float
    sub_target_db: float
    upper_target_db: float
    manifold_compatible: bool
    require_reported_inductance: bool
    sub_feasible_records: int
    upper_feasible_records: int
    single_feasible_records: int
    split_feasible: bool
    single_feasible: bool
    combined_front_split_designs: int
    combined_front_single_designs: int
    driver_only_outcome: str
    manifold_outcome: str


SPLIT_SWEEP_HZ = tuple(float(value) for value in range(60, 121, 5))


@dataclass(frozen=True)
class SplitSweepPoint:
    split_hz: float
    manifold_compatible: bool
    sub_feasible_records: int
    sub_preferred_records: int
    upper_feasible_records: int
    upper_preferred_records: int
    worked_sub_excursion: float
    worked_upper_excursion: float
    worked_upper_doppler: float
    worked_upper_amplifier: float
    worked_upper_box_l: float
    worked_pair_driver_feasible: bool
    worked_pair_preferred: bool


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
    return tuple(d for d in _bass_population()
                if SUB_SIZE_MIN <= d.size_in <= SUB_SIZE_MAX)


@lru_cache(maxsize=1)
def _upper_candidates() -> tuple[Driver, ...]:
    return tuple(d for d in _bass_population()
                if UPPER_SIZE_MIN <= d.size_in <= UPPER_SIZE_MAX)


@lru_cache(maxsize=1)
def _architecture_sub_candidates() -> tuple[Driver, ...]:
    low, high = ARCHITECTURE_SUB_SIZE_RANGE
    return tuple(
        driver for driver in _bass_population()
        if low <= driver.size_in <= high
    )


@lru_cache(maxsize=1)
def _architecture_upper_candidates() -> tuple[Driver, ...]:
    low, high = ARCHITECTURE_UPPER_SIZE_RANGE
    return tuple(
        driver for driver in _bass_population()
        if low <= driver.size_in <= high
    )


@lru_cache(maxsize=1)
def _architecture_single_candidates() -> tuple[Driver, ...]:
    low, high = ARCHITECTURE_SINGLE_SIZE_RANGE
    return tuple(
        driver for driver in _bass_population()
        if low <= driver.size_in <= high
    )


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
    once at the worked counts. Illustrative only: this driver-level overlap
    does not evaluate the piecewise full-band target or the architecture
    count/packaging search.
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


def _overlap(baseline: set[str], other: set[str]) -> float:
    """Fraction of the available baseline shortlist retained by a variant."""
    if not baseline:
        raise ValueError("rank robustness requires a non-empty baseline shortlist")
    return len(baseline & other) / len(baseline)


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
            _overlap(base_sub_top, _topk_labels(sub_evals, k)),
            _overlap(base_upper_top, _topk_labels(upper_evals, k)),
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
        _overlap(base_sub_top, _policy_variant_topk(base_sub, k)),
        _overlap(base_upper_top, _policy_variant_topk(base_upper, k)),
    ))
    return variants


# ----------------------------------------------------------------------
# Split-radiator versus stereo full-band architecture comparison
# ----------------------------------------------------------------------

def _architecture_scenario(
    room_volume_m3: float,
    longest_dimension_m: float,
    split_hz: float,
    high_edge_hz: float,
    *,
    stereo_summing_db: float = 6.0,
) -> Scenario:
    return replace(
        SCENARIO,
        v_room=room_volume_m3,
        l_max=longest_dimension_m,
        f_split=split_hz,
        f_high=high_edge_hz,
        sub_target_spl=ARCHITECTURE_BASE_SUB_TARGET_DB,
        attack_target_spl=ARCHITECTURE_BASE_UPPER_TARGET_DB,
        stereo_low_bass_summing_db=stereo_summing_db,
    )


def _evaluate_architecture_pool(
    drivers: tuple[Driver, ...],
    scenario: Scenario,
    role: str,
) -> tuple[Evaluation, ...]:
    if role == "sub":
        band_low, band_high, doppler_ref = (
            scenario.f_low,
            scenario.f_split,
            scenario.f_split,
        )
    elif role == "attack":
        band_low, band_high, doppler_ref = (
            scenario.f_split,
            scenario.f_high,
            scenario.f_high,
        )
    elif role == "full":
        band_low, band_high, doppler_ref = (
            scenario.f_low,
            scenario.f_high,
            scenario.f_high,
        )
    else:
        raise ValueError(f"unknown architecture-study role {role!r}")
    return tuple(
        evaluate(
            driver,
            scenario,
            n_units=1,
            band_low=band_low,
            band_high=band_high,
            doppler_ref=doppler_ref,
            role=role,
        )
        for driver in drivers
    )


@lru_cache(maxsize=None)
def _architecture_sub_evaluations(
    room_volume_m3: float,
    longest_dimension_m: float,
    split_hz: float,
) -> tuple[Evaluation, ...]:
    scenario = _architecture_scenario(
        room_volume_m3,
        longest_dimension_m,
        split_hz,
        max(ARCHITECTURE_HIGH_EDGES_HZ),
    )
    return _evaluate_architecture_pool(
        _architecture_sub_candidates(), scenario, "sub"
    )


@lru_cache(maxsize=None)
def _architecture_upper_evaluations(
    split_hz: float,
    high_edge_hz: float,
) -> tuple[Evaluation, ...]:
    scenario = _architecture_scenario(
        SCENARIO.v_room,
        SCENARIO.l_max,
        split_hz,
        high_edge_hz,
    )
    return _evaluate_architecture_pool(
        _architecture_upper_candidates(), scenario, "attack"
    )


def _architecture_full_evaluations(
    room_volume_m3: float,
    longest_dimension_m: float,
    split_hz: float,
    high_edge_hz: float,
    *,
    stereo_summing_db: float = 6.0,
) -> tuple[Evaluation, ...]:
    scenario = _architecture_scenario(
        room_volume_m3,
        longest_dimension_m,
        split_hz,
        high_edge_hz,
        stereo_summing_db=stereo_summing_db,
    )
    return _evaluate_architecture_pool(
        _architecture_single_candidates(), scenario, "full"
    )


def _architecture_role_designs(
    evaluations: tuple[Evaluation, ...],
    unit_options: tuple[int, ...],
    source_count: int,
    level_delta_db: float,
    *,
    require_reported_inductance: bool,
) -> list[architecture.RoleDesign]:
    return [
        design
        for evaluation in evaluations
        for units in unit_options
        if (
            design := architecture.scale_role_evaluation(
                evaluation,
                units_per_source=units,
                source_count=source_count,
                level_delta_db=level_delta_db,
                require_reported_inductance=require_reported_inductance,
            )
        ).feasible
    ]


def _dedupe_pareto_values(candidates):
    """Drop objective-identical records so database duplicates do not inflate
    architecture-front counts. Architecture identity is retained for systems.
    """
    unique = {}
    for candidate in candidates:
        key = tuple(round(value, 12) for value in candidate.pareto_values())
        if isinstance(candidate, architecture.SystemDesign):
            key += (candidate.architecture,)
        unique.setdefault(key, candidate)
    return list(unique.values())


def _architecture_outcome(
    front: list[architecture.SystemDesign],
) -> str:
    architectures = {design.architecture for design in front}
    if not architectures:
        return "none"
    if architectures == {"split"}:
        return "split_only"
    if architectures == {"single"}:
        return "single_only"
    return "mixed"


def _analyze_architecture_point(
    *,
    room_volume_m3: float,
    longest_dimension_m: float,
    split_hz: float,
    high_edge_hz: float,
    sub_target_db: float,
    sub_evaluations: tuple[Evaluation, ...],
    upper_evaluations: tuple[Evaluation, ...],
    full_evaluations: tuple[Evaluation, ...],
    require_reported_inductance: bool,
) -> ArchitectureStudyPoint:
    level_delta_db = sub_target_db - ARCHITECTURE_BASE_SUB_TARGET_DB
    sub_designs = _architecture_role_designs(
        sub_evaluations,
        ARCHITECTURE_SUB_UNIT_OPTIONS,
        source_count=1,
        level_delta_db=level_delta_db,
        require_reported_inductance=require_reported_inductance,
    )
    upper_designs = _architecture_role_designs(
        upper_evaluations,
        ARCHITECTURE_UPPER_UNIT_OPTIONS,
        source_count=2,
        level_delta_db=level_delta_db,
        require_reported_inductance=require_reported_inductance,
    )
    full_designs = _architecture_role_designs(
        full_evaluations,
        ARCHITECTURE_SINGLE_UNIT_OPTIONS,
        source_count=2,
        level_delta_db=level_delta_db,
        require_reported_inductance=require_reported_inductance,
    )

    sub_front = _dedupe_pareto_values(
        architecture.pareto_front(sub_designs)
    )
    upper_front = _dedupe_pareto_values(
        architecture.pareto_front(upper_designs)
    )
    single_front = _dedupe_pareto_values(
        architecture.pareto_front(full_designs)
    )

    split_systems = [
        architecture.split_system(sub, upper)
        for sub in sub_front
        for upper in upper_front
    ]
    split_front = _dedupe_pareto_values(
        architecture.pareto_front(split_systems)
    )
    single_systems = [
        architecture.single_system(full) for full in single_front
    ]
    combined_front = _dedupe_pareto_values(
        architecture.pareto_front(split_front + single_systems)
    )

    manifold_systems = [
        design
        for design in split_front
        if design.manifold_compatible
    ] + single_systems
    manifold_front = _dedupe_pareto_values(
        architecture.pareto_front(manifold_systems)
    )

    return ArchitectureStudyPoint(
        room_volume_m3=room_volume_m3,
        longest_dimension_m=longest_dimension_m,
        pressure_zone_hz=physics.pressure_zone_frequency(longest_dimension_m),
        split_hz=split_hz,
        high_edge_hz=high_edge_hz,
        sub_target_db=sub_target_db,
        upper_target_db=sub_target_db - 5.0,
        manifold_compatible=(
            split_hz <= SCENARIO.manifold_crossover_ceiling_hz
        ),
        require_reported_inductance=require_reported_inductance,
        sub_feasible_records=len({design.label for design in sub_designs}),
        upper_feasible_records=len(
            {design.label for design in upper_designs}
        ),
        single_feasible_records=len(
            {design.label for design in full_designs}
        ),
        split_feasible=bool(split_systems),
        single_feasible=bool(single_systems),
        combined_front_split_designs=sum(
            design.architecture == "split" for design in combined_front
        ),
        combined_front_single_designs=sum(
            design.architecture == "single" for design in combined_front
        ),
        driver_only_outcome=_architecture_outcome(combined_front),
        manifold_outcome=_architecture_outcome(manifold_front),
    )


@lru_cache(maxsize=1)
def _architecture_study_data() -> tuple[ArchitectureStudyPoint, ...]:
    points: list[ArchitectureStudyPoint] = []
    total_geometries = (
        len(ARCHITECTURE_ROOM_CASES)
        * len(ARCHITECTURE_SPLITS_HZ)
        * len(ARCHITECTURE_HIGH_EDGES_HZ)
    )
    geometry_index = 0
    for room_volume_m3, longest_dimension_m in ARCHITECTURE_ROOM_CASES:
        for split_hz in ARCHITECTURE_SPLITS_HZ:
            sub_evaluations = _architecture_sub_evaluations(
                room_volume_m3, longest_dimension_m, split_hz
            )
            for high_edge_hz in ARCHITECTURE_HIGH_EDGES_HZ:
                geometry_index += 1
                print(
                    "architecture study geometry "
                    f"{geometry_index}/{total_geometries}: "
                    f"{room_volume_m3:g} m3, split {split_hz:g} Hz, "
                    f"f2 {high_edge_hz:g} Hz",
                    flush=True,
                )
                upper_evaluations = _architecture_upper_evaluations(
                    split_hz, high_edge_hz
                )
                full_evaluations = _architecture_full_evaluations(
                    room_volume_m3,
                    longest_dimension_m,
                    split_hz,
                    high_edge_hz,
                )
                for sub_target_db in ARCHITECTURE_SUB_TARGETS_DB:
                    points.append(_analyze_architecture_point(
                        room_volume_m3=room_volume_m3,
                        longest_dimension_m=longest_dimension_m,
                        split_hz=split_hz,
                        high_edge_hz=high_edge_hz,
                        sub_target_db=sub_target_db,
                        sub_evaluations=sub_evaluations,
                        upper_evaluations=upper_evaluations,
                        full_evaluations=full_evaluations,
                        require_reported_inductance=True,
                    ))
    return tuple(points)


@lru_cache(maxsize=1)
def _canonical_architecture_sensitivities() -> dict[str, ArchitectureStudyPoint]:
    room_volume_m3, longest_dimension_m = 60.0, 6.0
    split_hz, high_edge_hz, sub_target_db = 80.0, 250.0, 110.0
    sub_evaluations = _architecture_sub_evaluations(
        room_volume_m3, longest_dimension_m, split_hz
    )
    upper_evaluations = _architecture_upper_evaluations(
        split_hz, high_edge_hz
    )
    full_evaluations = _architecture_full_evaluations(
        room_volume_m3, longest_dimension_m, split_hz, high_edge_hz
    )
    full_evaluations_3db = _architecture_full_evaluations(
        room_volume_m3,
        longest_dimension_m,
        split_hz,
        high_edge_hz,
        stereo_summing_db=3.0,
    )

    common = dict(
        room_volume_m3=room_volume_m3,
        longest_dimension_m=longest_dimension_m,
        split_hz=split_hz,
        high_edge_hz=high_edge_hz,
        sub_target_db=sub_target_db,
        sub_evaluations=sub_evaluations,
        upper_evaluations=upper_evaluations,
    )
    return {
        "reported_inductance_6db_summing": _analyze_architecture_point(
            **common,
            full_evaluations=full_evaluations,
            require_reported_inductance=True,
        ),
        "permissive_inductance_6db_summing": _analyze_architecture_point(
            **common,
            full_evaluations=full_evaluations,
            require_reported_inductance=False,
        ),
        "reported_inductance_3db_summing": _analyze_architecture_point(
            **common,
            full_evaluations=full_evaluations_3db,
            require_reported_inductance=True,
        ),
        "permissive_inductance_3db_summing": _analyze_architecture_point(
            **common,
            full_evaluations=full_evaluations_3db,
            require_reported_inductance=False,
        ),
    }


@lru_cache(maxsize=1)
def _split_sweep_data() -> tuple[SplitSweepPoint, ...]:
    points: list[SplitSweepPoint] = []
    for split_hz in SPLIT_SWEEP_HZ:
        scenario = replace(SCENARIO, f_split=split_hz)
        sub_evaluations = _rank_sub(scenario=scenario)
        upper_evaluations = _rank_upper(scenario=scenario)
        worked_sub = _ev_sub(scenario=scenario)
        worked_upper = _ev_upper(scenario=scenario)
        worked_sub_excursion = max(
            worked_sub.xi_x, worked_sub.xi_x_transient
        )
        worked_upper_excursion = max(
            worked_upper.xi_x, worked_upper.xi_x_transient
        )
        points.append(SplitSweepPoint(
            split_hz=split_hz,
            manifold_compatible=scenario.is_manifold_crossover_valid,
            sub_feasible_records=sum(
                evaluation.feasible for evaluation in sub_evaluations
            ),
            sub_preferred_records=sum(
                evaluation.feasible
                and evaluation.is_preferred_excursion
                for evaluation in sub_evaluations
            ),
            upper_feasible_records=sum(
                evaluation.feasible for evaluation in upper_evaluations
            ),
            upper_preferred_records=sum(
                evaluation.feasible
                and evaluation.is_preferred_excursion
                for evaluation in upper_evaluations
            ),
            worked_sub_excursion=worked_sub_excursion,
            worked_upper_excursion=worked_upper_excursion,
            worked_upper_doppler=worked_upper.doppler_im,
            worked_upper_amplifier=worked_upper.amplifier_utilization,
            worked_upper_box_l=worked_upper.risk.box_volume_m3 * 1e3,
            worked_pair_driver_feasible=(
                worked_sub.feasible and worked_upper.feasible
            ),
            worked_pair_preferred=(
                worked_sub.feasible
                and worked_upper.feasible
                and worked_sub.is_preferred_excursion
                and worked_upper.is_preferred_excursion
            ),
        ))
    return tuple(points)


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


def fig_split_sensitivity() -> None:
    points = _split_sweep_data()
    fig = plots.split_sensitivity_figure(
        [point.split_hz for point in points],
        [point.sub_feasible_records for point in points],
        [point.sub_preferred_records for point in points],
        [point.upper_feasible_records for point in points],
        [point.upper_preferred_records for point in points],
        [point.worked_sub_excursion for point in points],
        [point.worked_upper_excursion for point in points],
        manifold_ceiling_hz=SCENARIO.manifold_crossover_ceiling_hz,
        preferred_excursion=SCENARIO.preferred_excursion,
        sub_population=len(_sub_candidates()),
        upper_population=len(_upper_candidates()),
    )
    _save_figure(fig, FIGURES_DIR / "fig_split_sensitivity.pdf")


def _outcome_counts(
    points: tuple[ArchitectureStudyPoint, ...],
    target_db: float,
    field: str,
) -> dict[str, int]:
    counts = {
        "split_only": 0,
        "mixed": 0,
        "single_only": 0,
        "none": 0,
    }
    for point in points:
        if point.sub_target_db == target_db:
            counts[getattr(point, field)] += 1
    return counts


def fig_architecture_comparison() -> None:
    points = _architecture_study_data()
    driver_only = [
        _outcome_counts(points, target, "driver_only_outcome")
        for target in ARCHITECTURE_SUB_TARGETS_DB
    ]
    manifold = [
        _outcome_counts(points, target, "manifold_outcome")
        for target in ARCHITECTURE_SUB_TARGETS_DB
    ]
    cases_per_target = (
        len(ARCHITECTURE_ROOM_CASES)
        * len(ARCHITECTURE_SPLITS_HZ)
        * len(ARCHITECTURE_HIGH_EDGES_HZ)
    )
    fig = plots.architecture_outcome_figure(
        ARCHITECTURE_SUB_TARGETS_DB,
        driver_only,
        manifold,
        cases_per_target=cases_per_target,
    )
    _save_figure(fig, FIGURES_DIR / "fig_architecture_comparison.pdf")


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
        r"\caption{Private driver-database evidence summary at the worked "
        r"unit counts (aggregate counts only. The accompanying manifest "
        r"supplies the full source hash, parser, and filter definitions).}",
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
        rf"sub-manifold role feasible ({SUB_UNITS} total) & {sub_feasible_n}\\",
        rf"sub-manifold role Pareto front & {sub_pareto_n}\\",
        rf"upper-bass role candidates ({UPPER_SIZE_MIN:g}--{UPPER_SIZE_MAX:g}in) & "
        rf"{len(_upper_candidates())}\\",
        rf"upper-bass role feasible ({UPPER_UNITS}/channel) & {upper_feasible_n}\\",
        rf"upper-bass role Pareto front & {upper_pareto_n}\\",
        rf"both role tests feasible at worked counts (size-unrestricted) & {dual_n}\\",
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
        ("public source",
         r"Dayton Audio official UMII18-22 specification sheet",
         r"FaitalPRO official 12HP1030 product data (8 ohm)"),
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
        r"\caption{Worked selected pair: Dayton Audio UMII18-22 (mono sub "
        rf"manifold, {SUB_UNITS} identical units in an even, symmetrically "
        r"opposed pairing -- permitting first-order reaction-force "
        r"cancellation under matched drive and mounting; the even count is "
        r"a hard mechanical-layout constraint, not a ranking-policy "
        r"preference -- "
        r"each with its own amplifier channel, aggregate target "
        rf"{sc.sub_target_spl:g} dB for the manifold, not per driver) and "
        r"FaitalPRO 12HP1030 (upper-bass, one unit "
        rf"per independent stereo channel, target {sc.attack_target_spl:g}"
        r" dB per channel). Pareto front number is one-based and is against "
        r"this report's frozen private-database candidate pool for each role.}",
        r"\label{tab:worked-pair}",
        r"\small",
        r"\begin{tabular}{p{4.0cm}p{5.1cm}p{5.1cm}}",
        r"\toprule",
        r" & Dayton UMII18-22 (sub) & FaitalPRO 12HP1030 (upper)\\",
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
        rf"target at the {sc.r_listen:g} m listening-area basis "
        rf"($V_{{\mathrm{{room}}}}$={sc.v_room:g} m$^3$, "
        rf"$L_{{\max}}$={sc.l_max:g} m). Required units use the selected "
        r"Dayton Audio UMII18-22 sub driver's own $V_d$, rounded up to the nearest "
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


def table_split_sensitivity() -> None:
    points = _split_sweep_data()
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Crossover sensitivity with every other scenario input "
        r"fixed. Candidate entries are hard-feasible/preferred-margin record "
        r"counts in the declared role pools. The public worked pair is "
        r"re-aligned and its enclosure recalculated at every crossover. "
        r"Points above the declared manifold ceiling are driver-only "
        r"counterfactuals (CF), not valid manifold recommendations. "
        r"$F/P$ denotes hard-feasible/preferred-margin records.}",
        r"\label{tab:split-sensitivity}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{rrrrrrrl}",
        r"\toprule",
        r"$f_{\mathrm{sp}}$ [Hz] & valid & sub $F/P$ & upper $F/P$ & "
        r"worked $\xi_s/\xi_u$ & $V_{b,u}$ [L] & $d_{\mathrm D,u}$ & status\\",
        r"\midrule",
    ]
    for point in points:
        if not point.worked_pair_driver_feasible:
            status = "reject"
        elif point.worked_pair_preferred:
            status = "preferred"
        else:
            status = "feasible"
        if not point.manifold_compatible and status != "reject":
            status += "; CF"
        lines.append(
            f"{point.split_hz:.0f} & "
            f"{'yes' if point.manifold_compatible else 'no'} & "
            f"{point.sub_feasible_records}/{point.sub_preferred_records} & "
            f"{point.upper_feasible_records}/{point.upper_preferred_records} & "
            f"{point.worked_sub_excursion:.2f}/{point.worked_upper_excursion:.2f} & "
            f"{point.worked_upper_box_l:.1f} & "
            f"{point.worked_upper_doppler:.3f} & {status}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_text_checked(
        TABLES_DIR / "table_split_sensitivity.tex",
        "\n".join(lines) + "\n",
    )


def table_architecture_comparison() -> None:
    points = _architecture_study_data()
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Architecture comparison over the declared domestic/"
        r"mastering factorial. Each target has 36 cases: three room/"
        r"longest-dimension pairs, four crossovers, and three upper-band "
        r"edges. Upper-bass target is 5 dB below the listed low-bass target. "
        r"Upper/full candidates must report $L_e$. ``Separate radiators'' "
        r"compares the two-role architecture at every tested crossover; "
        r"``80-Hz manifold'' removes split designs above the declared "
        r"manifold ceiling. Counts are scenario cases, not combinatorial "
        r"driver-pair counts.}",
        r"\label{tab:architecture-comparison}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"basis & $L_{\mathrm{sub}}$ [dB] & split feas. & full feas. & "
        r"split only & mixed & full only & neither\\",
        r"\midrule",
    ]
    for target in ARCHITECTURE_SUB_TARGETS_DB:
        selected = [
            point for point in points if point.sub_target_db == target
        ]
        driver_counts = _outcome_counts(
            points, target, "driver_only_outcome"
        )
        manifold_counts = _outcome_counts(
            points, target, "manifold_outcome"
        )
        lines.append(
            f"separate radiators & {target:.0f} & "
            f"{sum(point.split_feasible for point in selected)} & "
            f"{sum(point.single_feasible for point in selected)} & "
            f"{driver_counts['split_only']} & {driver_counts['mixed']} & "
            f"{driver_counts['single_only']} & {driver_counts['none']}\\\\"
        )
        lines.append(
            f"80-Hz manifold & {target:.0f} & "
            f"{sum(point.split_feasible and point.manifold_compatible for point in selected)} & "
            f"{sum(point.single_feasible for point in selected)} & "
            f"{manifold_counts['split_only']} & {manifold_counts['mixed']} & "
            f"{manifold_counts['single_only']} & {manifold_counts['none']}\\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_text_checked(
        TABLES_DIR / "table_architecture_comparison.tex",
        "\n".join(lines) + "\n",
    )


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
    split_sweep = _split_sweep_data()
    architecture_points = _architecture_study_data()
    architecture_sensitivities = _canonical_architecture_sensitivities()

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
                "driver-level, size-unrestricted count over the full "
                "bass-relevant population at the worked unit counts; "
                "illustrative only, not the piecewise full-band "
                "architecture test. The canonical worked example uses an "
                "18in manifold record and a 12in per-channel record, but "
                "size and unit count are selection outputs within "
                "user-declared packaging ranges."
            ),
            "dual_role_sub_band_feasible": sub_dual_n,
            "dual_role_upper_band_feasible": upper_dual_n,
            "dual_role_both_feasible": dual_n,
        },
        "split_sensitivity": {
            "definition": {
                "split_hz": list(SPLIT_SWEEP_HZ),
                "all_other_inputs": "canonical scenario",
                "upper_alignment": (
                    "reoptimized at each split; this is a design sweep, "
                    "not a fixed-enclosure crossover-only sweep"
                ),
                "manifold_crossover_ceiling_hz": (
                    SCENARIO.manifold_crossover_ceiling_hz
                ),
                "above_ceiling_treatment": (
                    "driver-only counterfactual; excluded from manifold "
                    "recommendations"
                ),
            },
            "aggregate_points": [asdict(point) for point in split_sweep],
        },
        "architecture_comparison": {
            "definition": {
                "primary_data_policy": (
                    "upper/full-band records must report Le; the canonical "
                    "permissive sensitivity also reports results when "
                    "missing Le is retained as unresolved"
                ),
                "room_cases": [
                    {
                        "volume_m3": volume,
                        "longest_dimension_m": length,
                        "pressure_zone_hz": (
                            physics.pressure_zone_frequency(length)
                        ),
                    }
                    for volume, length in ARCHITECTURE_ROOM_CASES
                ],
                "split_hz": list(ARCHITECTURE_SPLITS_HZ),
                "upper_edge_hz": list(ARCHITECTURE_HIGH_EDGES_HZ),
                "sub_target_db": list(ARCHITECTURE_SUB_TARGETS_DB),
                "upper_target_offset_db": -5.0,
                "single_architecture_low_bass_summing_db": 6.0,
                "split_sub_size_in": list(ARCHITECTURE_SUB_SIZE_RANGE),
                "split_sub_unit_options_total": list(
                    ARCHITECTURE_SUB_UNIT_OPTIONS
                ),
                "split_upper_size_in": list(
                    ARCHITECTURE_UPPER_SIZE_RANGE
                ),
                "split_upper_unit_options_per_channel": list(
                    ARCHITECTURE_UPPER_UNIT_OPTIONS
                ),
                "single_size_in": list(ARCHITECTURE_SINGLE_SIZE_RANGE),
                "single_unit_options_per_channel": list(
                    ARCHITECTURE_SINGLE_UNIT_OPTIONS
                ),
                "objectives": [
                    "steady excursion utilization",
                    "transient excursion utilization",
                    "Doppler first-sideband ratio",
                    "amplifier utilization",
                    "total enclosure volume",
                    "physical driver count",
                ],
                "count_note": (
                    "published counts are aggregate record/scenario counts, "
                    "not row-level identities or combinatorial pair totals"
                ),
            },
            "aggregate_points": [
                asdict(point) for point in architecture_points
            ],
            "canonical_sensitivities": {
                key: asdict(value)
                for key, value in architecture_sensitivities.items()
            },
        },
        "canonical_scenario": asdict(SCENARIO),
        "selected_public_datasheet_records": {
            "sub": {
                "role": (
                    "two-driver mono manifold in one mechanically opposed "
                    "pair; 110 dB is the complete-manifold target"
                ),
                "manufacturer": DRIVER_SUB.manufacturer,
                "model": DRIVER_SUB.model,
                "source_url": (
                    "https://www.daytonaudio.com/images/resources/"
                    "295-718--dayton-audio-UMII18-22-spec-sheet.pdf"
                ),
                "parameters_si": asdict(DRIVER_SUB),
                "parameter_note": (
                    "The official manufacturer one-way Xmax value is "
                    "28 mm; 22 in the model name denotes the dual 2-ohm "
                    "voice-coil configuration, not excursion."
                ),
            },
            "upper_bass": {
                "role": (
                    "one driver per independent stereo channel; "
                    "105 dB is the per-channel target"
                ),
                "manufacturer": DRIVER_UPPER.manufacturer,
                "model": "12HP1030-8",
                "source_url": (
                    "https://faitalpro.com/en/products/LF_Loudspeakers/"
                    "product_details/index.php?id=201050130"
                ),
                "parameters_si": asdict(DRIVER_UPPER),
                "parameter_note": (
                    "The public manufacturer Xmax value used here is "
                    "12.45 mm one-way and the continuous rating is the "
                    "1000 W AES value, not the 2000 W maximum figure."
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
        fig_split_sensitivity,
        fig_architecture_comparison,
        fig_database_pareto,
        fig_corner_population,
        fig_rank_robustness,
        table_prior_art,
        table_room_sensitivity,
        table_worked_pair,
        table_split_sensitivity,
        table_architecture_comparison,
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
