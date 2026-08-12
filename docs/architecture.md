# Architecture

Sealed-box bass driver selection tool for a compact living-room stereo
listening area. It applies exact output and finite-amplifier gates, then
uses separate risk indicators, Pareto fronts, and declared role policies.

> This package was migrated from the working repo `driver_criteria` into
> `audioshape` to prepare the paper for publication (paper id `26325`),
> following the structure and release tooling of the sibling `mimoshape`
> repository.

## Layout

```
src/audioshape/
    physics.py     pure physics functions, SI units, no I/O (web-safe core)
    driver.py      Driver / BoxedDriver dataclasses + derived T-S quantities
    scenario.py    Scenario: room/leakage, targets, amplifier, transient test
    vented.py      minimal vented-box (Small alignment) physics, comparison-
                   only: not used by ranking/cli, backs the paper's
                   "why sealed, not vented" section and its figure/table
    recipe.py      Recipe dataclass + load_recipe(): TOML config, read-only
    database.py    VituixCAD TSV parser -> list[Driver] + skipped-row report
    ranking.py     evaluate/rank (single driver, role-restricted band) and
                   PairEvaluation/pair_rank (independent sub+attack ranking)
    architecture.py architecture-level unit scaling, split/single system
                   assembly, and non-compensatory system Pareto comparison
    plots.py       matplotlib layer (only module besides cli importing it)
    vituixcad.py   export a driver selection to a VituixCAD .vxp project +
                   driver-database TSV (string/XML building, no file I/O)
    cli.py         argparse entry point: `rank`, `pair`, `plot`,
                   `export-vituixcad` subcommands
tests/             pytest; physics pinned to the paper's worked example
data/              local driver-database location; the VituixCAD database is
                   intentionally not distributed and is ignored by Git
examples/          example_recipe.toml (sample recipe config)
docs/plans/        design notes for multi-step features (e.g. pair_ranking.md)
scripts/           repo-level release tooling (mirrors mimoshape):
                   make_figures.py dispatches to scripts/papers/<id>/
                   make_figures.py, the single source of truth for every
                   figure/table embedded in that paper's .tex
papers/26325/      the paper bundle (see below)
```

### Paper bundle (`papers/26325/`)

```
papers/26325/
    sealed_driver_criteria.tex          main paper (preprint build)
    sealed_driver_criteria_journal.tex  thin driver: sets \ELSVIERBUILD,
                                        \input{sealed_driver_criteria.tex}
    refs.bib                            bibliography (natbib, unsrtnat)
    frontmatter/                        preprint/journal header+titlepage
    meta/metadata.tex                   AUTO-GENERATED, do not hand-edit
    figures/*.pdf                       generated PDF figures (committed)
    tables/*.tex                        generated LaTeX table fragments
                                         (committed)
```

`figures/` and `tables/` are committed (not gitignored) so the paper builds
from a fresh clone without a Python environment; treat every file in them as
generated -- fix `scripts/papers/26325/make_figures.py` (or the physics it
calls) and re-run, never hand-edit a figure/table in place.

## Data flow

`cli` loads a `Recipe` (TOML) -> `Scenario` + `database.parse_database` ->
`ranking.rank` (single role) or `ranking.pair_rank` (sub+attack) -> printed
table, or `plots.*_figure` -> PNG/show. All user-tunable parameters live in
the recipe file, not CLI flags (`--recipe path.toml` is the only required
input besides `--role`/`--driver`).

Publication sensitivity studies may additionally pass one-driver
`Evaluation` objects to `architecture.scale_role_evaluation()`. Because the
box cap is per physical driver, repeated identical units leave the alignment
unchanged: displacement, voltage, current, and Doppler scale as `1/N`, while
power scales as `1/N^2`. `split_system()` and `single_system()` then compare
complete architectures on separate steady excursion, transient excursion,
Doppler, amplifier, enclosure-volume, and physical-driver-count objectives.
`split_system()` enforces one mono source with an even manifold count plus
two upper-bass sources; `single_system()` enforces two full-band stereo
sources.

## Two-role architecture (sub + attack)

The tool assumes the physical layout from the paper's Part II: a mono bass
**sub** (e.g. a manifold of identical drivers behind a soffit wall) covering
`[f_low, f_split]`, and a stereo **attack**/mid-woofer covering
`[f_split, f_high]`. These are evaluated as two *independent* roles, not a
single driver spanning the whole range:

- `evaluate(driver, scenario, band_low=..., band_high=..., doppler_ref=...,
  role=...)` restricts the demand curve, inductance gate and Doppler
  sideband reference tone to a role's own band, and aligns the box to that role's own
  corner target (`Scenario.target_corner_hz(role)`: `f_pz` for sub/full,
  `f_split` for attack) rather than a single fixed Qtc -- see "Qtc is an
  alignment target, not a hard gate" below. When omitted, `doppler_ref` defaults
  to the evaluated role's upper band edge.
- `cli._ROLE_BANDS` maps `--role {sub,attack,full}` to
  `(band_low, band_high, doppler_ref)` Scenario attribute names. `role` is
  also what `target_corner_hz`/the attack-only budget gates key off, so a
  caller scoring a driver in the attack band must pass `role="attack"`
  explicitly (it is not inferred from `band_low`).
- `pair_rank(drivers, scenario, sub_units, attack_units, top_k_each,
  require_even_sub_units=...)` ranks
  the sub band and attack band **separately** (own excursion/Doppler
  reference, own feasibility gates), takes each side's top-K, and returns
  the cross product without adding unlike role risks.
  `Scenario.manifold_crossover_ceiling_hz` is the highest nominal crossover
  accepted for this manifold architecture. `Recipe` and `pair_rank` reject a
  higher `f_split`; role-only `evaluate`/`rank` calls may still inspect such
  points as explicitly labelled driver-only counterfactuals. The ceiling is
  not a brick-wall crossover edge: the implemented low-pass must adequately
  suppress manifold output above its validated acoustic band.
  The canonical recipe uses two sub drivers and requires an even count so
  the manifold can be built as opposed pairs. Such pairs cancel reaction
  force only to first order under matched drive and mounting.
  There is **no cross-driver coupling term**: sub and attack are physically
  separate enclosures sharing only the baffle plane, so each Doppler
  sideband indicator is intrinsic to its own band (see
  `papers/26325/sealed_driver_criteria.tex` "Scope of the tool" note).
  The T/S model does not predict manifold cavity modes, path-length
  interference, or compression; its user-supplied crossover ceiling is
  therefore an architectural constraint that must ultimately come from
  geometry, simulation, or measurement.

### Qtc is an alignment target, not a hard gate

`Scenario.alignment_qtc` is the preferred alignment used when it is
physically reachable inside the configured box cap. The implementation first
targets
`min(alignment_qtc, f_target / driver.corner_rate)`, where `f_target` is
`f_pz` for sub/full and `f_split` for attack. If that target would require an
impossible or oversized box, the driver receives the maximum allowed
per-driver volume (default `4 * Vas`) and its actual `Qtc`/`Fc` are retained.
An actual `Qtc` above the preference is **not** an automatic rejection:
equalizing a physical corner above `f_pz` consumes excursion, voltage,
current, power, and transient margin, and those explicit gates decide
feasibility. This permits high-Qts, high-stroke specialists to compete while
making their enclosure volume and correction cost visible.

The optional absolute cap is per physical driver; adding drivers no longer
silently shrinks each enclosure. Total system volume is a Pareto objective
and selection output. `boxed.qtc` is the realized physical alignment;
`scenario.alignment_qtc` is only the preferred target.

### Excursion clipping and preferred margin

Published one-way `Xmax` is the sole excursion rejection boundary:
`xi_x > 1` clips. `Scenario.preferred_excursion` defaults to 0.80 and is
reported as a design-margin warning and preferred unit count, not as an
acoustic-THD prediction or a second feasibility gate. The 110 dB mono-sub
and 105 dB-per-upper-bass-channel targets are occasional maximum-output
conditions. Exposure limits may show why such levels are not sustained
listening conditions, but they do not establish a mechanical threshold.

## Scope assumptions (locked in; see `docs/plans/pair_ranking.md`)

1. Half-space (2 pi, soffit-wall) radiation for every driver/role.
2. A compact couch or mastering area represented by a declared reference position or
   small measurement set. The intended optimization preserves independent
   stereo-channel timing and direct/early-field information; it is not a
   diffuse-field or multi-seat SPL optimization.
3. Below `f_pz`, the prepared room is represented by an acoustic compliance
   plus a declared leakage corner. This is a low-frequency limiting model,
   not a claim that the complete room is one second-order system.
4. The system is stereo. `Scenario.sub_target_spl`/`attack_target_spl`
   use explicit reference bases:
   - The sub is a single shared mono manifold -> must hit `sub_target_spl`
     directly (no summing ambiguity at all).
   - The attack/tower is genuinely stereo (L/R can carry independent
     content) -> each channel must hit `attack_target_spl` **on its own**.
   - In the full-band architecture baseline, each stereo channel receives
     `sub_target_spl - stereo_low_bass_summing_db` below `f_split` and the
     independent `attack_target_spl` above it. The primary study uses a
     favorable 6 dB mono-bass credit and reports a 3 dB sensitivity.
   - A design using two separate mono subs instead would gain +3 to +6 dB
     from correlated/decorrelated summing, but that architecture is out of
     scope here (see `scenario.py`'s `sub_target_spl`/`attack_target_spl`
     docstrings, and `target_spl_for(role)` which selects between them).
5. Configuration is via a hand-authored recipe TOML file
   (`recipe.load_recipe`), not many CLI flags.

## Design decisions

- **Core is plot-free and I/O-free.**  `physics`, `driver`, `scenario`,
  `ranking` import neither matplotlib nor files, so the same code can back a
  web service later.  `plots` builds on `matplotlib.figure.Figure` directly
  (backend-free); only `cli` touches `pyplot`. `vituixcad` follows the same
  rule for exports: it returns strings (TSV text, XML text) and never
  touches the filesystem -- only `cli` (and a caller such as a web app)
  writes the files.
- **Half-space (2 pi) radiation everywhere** (soffit/wall mounting), matching
  the constants in the paper (108.5 dB and 112.2 dB reference levels).
- **Room model**: above `f_pz = c/(2 L_max)`, demand follows half-space
  radiation. Below it, the compliance demand is multiplied by
  `sqrt(1 + (f_leak/f)^2)` for the configured leakage corner. The ideal
  pressure-zone model remains available for sensitivity analysis.
- **Finite electrical model**: target displacement is divided among the
  physical drivers in one manifold/channel, then voltage, current, and coil
  power are computed per driver across the complete role band. This avoids
  an acoustic-power `N^2` shortcut. Amplifier limits are per physical driver.
- **Transient model**: the sealed ODE is integrated for the declared finite
  voltage burst and several start phases. Peak displacement includes
  ring-down; voltage/current/power RMS values cover the active burst.
- **Ranking policy**: hard gates first; Pareto fronts over `RiskVector`;
  role-specific lexicographic ordering within each one-based front number.
  Excursion, Doppler sideband, box-spring, thermal, electrical, inductive,
  and size indicators remain separate.
- **Parser tolerance**: the community database is patchy; rows missing
  required parameters are skipped and listed in `ParseResult.skipped`
  (reported, never silent).
- **VituixCAD export (`vituixcad.py`) is two files, not one**: the `.vxp`
  project's `<DRIVER>` block only carries Model/SPL/Z/response-file fields --
  VituixCAD stores Thiele/Small parameters exclusively in the separate
  driver-database TSV, matched by "Manufacturer Model" name. So a selection
  is exported as the `.vxp` project *and* a driver-database TSV snippet
  (reusing `database.COLUMN_MAP` for the inverse SI -> VituixCAD-units
  mapping) that the user imports into their local VituixCAD driver database
  if the selected driver isn't already in it. `HalfSpace` is forced `True`
  (overriding VituixCAD's own default) since the physics model is always
  2 pi half-space; `DrvN` per role is `BoxedDriver.n_units`; driver `SPL`/`Z`
  are computed placeholders (documented in code), not measurements. This
  schema was confirmed by driving the actually-installed VituixCAD3.exe
  (Save-As to capture a real default `.vxp`, then loading a generated file
  back in), not by decompiling the app.

## Paper figure/table generation

`scripts/papers/26325/make_figures.py` is the **single source of truth** for
every generated number, figure, and table in both manuscripts. Worked
system-feasibility assets use one canonical 3 m listening-area scenario; a
1 m value may appear only as a clearly labelled intrinsic descriptor. The script
calls `plots.spl_figure`/`risk_figure`/`demand_figure`/
`vented_comparison_figure` and writes plain-LaTeX `tabular` fragments into
`papers/26325/figures/` and `papers/26325/tables/` respectively, and
`meta/metadata.tex` via `write_metadata()`. Run it (via the top-level
dispatcher) and re-run `pdflatex`/`bibtex` whenever `physics.py`/`vented.py`/
the example driver/scenario parameters change:

```
scripts\figures -p 26325
```

The script also prints a console-only "narrative cross-check" block (not
written to any file) with every number quoted in the paper's prose, so a
stale figure caption or hand-calc can be caught by diffing that printout
against the `.tex` text.

## Developer workflow

- Install/run: `uv sync`, `uv run audioshape ...`
- Tests: `uv run pytest -q`
- Scratch output (plots etc.) goes in `dev/` (not checked in).
- Build the paper: `pdflatex -interaction=nonstopmode -halt-on-error
  sealed_driver_criteria.tex`, then `bibtex sealed_driver_criteria`, then
  `pdflatex` twice more (bibliography/cross-references need the extra
  passes), then remove `*.aux`/`*.log`/`*.out`/`*.toc`/`*.bbl`/`*.blg`
  (gitignored). See `scripts/publish_release.py` for the automated version.

Example commands:

```
uv run audioshape rank --recipe examples/example_recipe.toml --role sub
uv run audioshape rank --recipe examples/example_recipe.toml --role attack
uv run audioshape pair --recipe examples/example_recipe.toml --top 5
uv run audioshape plot --recipe examples/example_recipe.toml --role sub \
    --driver "UMII18" --save dev/out
```
