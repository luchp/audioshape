# Architecture

Sealed-box bass driver selection tool: ranks drivers from the VituixCAD
driver database using the first-principles criteria derived in
`papers/26325/sealed_driver_criteria.tex`, and plots SPL ceilings /
non-correctable distortion for a chosen configuration.

> This package was migrated from the working repo `driver_criteria` into
> `audioshape` to prepare the paper for publication (paper id `26325`),
> following the structure and release tooling of the sibling `mimoshape`
> repository. The Python API and CLI are otherwise unchanged.

## Layout

```
src/audioshape/
    physics.py     pure physics functions, SI units, no I/O (web-safe core)
    driver.py      Driver / BoxedDriver dataclasses + derived T-S quantities
    scenario.py    Scenario dataclass: room, distance, target SPL, budgets
    vented.py      minimal vented-box (Small alignment) physics, comparison-
                   only: not used by ranking/cli, backs the paper's
                   "why sealed, not vented" section and its figure/table
    recipe.py      Recipe dataclass + load_recipe(): TOML config, read-only
    database.py    VituixCAD TSV parser -> list[Driver] + skipped-row report
    ranking.py     evaluate/rank (single driver, role-restricted band) and
                   PairEvaluation/pair_rank (independent sub+attack ranking)
    plots.py       matplotlib layer (only module besides cli importing it)
    vituixcad.py   export a driver selection to a VituixCAD .vxp project +
                   driver-database TSV (string/XML building, no file I/O)
    cli.py         argparse entry point: `rank`, `pair`, `plot`,
                   `export-vituixcad` subcommands
tests/             pytest; physics pinned to the paper's worked example
data/              VituixCAD_driver_db.txt (driver database)
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

## Two-role architecture (sub + attack)

The tool assumes the physical layout from the paper's Part II: a mono bass
**sub** (e.g. a manifold of identical drivers behind a soffit wall) covering
`[f_low, f_split]`, and a stereo **attack**/mid-woofer covering
`[f_split, f_high]`. These are evaluated as two *independent* roles, not a
single driver spanning the whole range:

- `evaluate(driver, scenario, band_low=..., band_high=..., doppler_ref=...)`
  restricts the demand curve, corner-rate gate, inductance gate and Doppler
  reference tone to a role's own band. Defaults (`band_low=f_low,
  band_high=f_high, doppler_ref=f_split`) reproduce the original
  whole-range single-driver evaluation.
- `cli._ROLE_BANDS` maps `--role {sub,attack,full}` to
  `(band_low, band_high, doppler_ref)` Scenario attribute names.
- `pair_rank(drivers, scenario, sub_units, attack_units, top_k_each)` ranks
  the sub band and attack band **separately** (own excursion/Doppler
  reference, own feasibility gates), takes each side's top-K, and returns
  the cross product sorted by combined (feasible-first, total distortion).
  There is **no cross-driver coupling term**: sub and attack are physically
  separate enclosures sharing only the baffle plane, so each self-Doppler
  reference is intrinsic to its own band (see
  `papers/26325/sealed_driver_criteria.tex` "Scope of the tool" note).

## Scope assumptions (locked in; see `docs/plans/pair_ranking.md`)

1. Half-space (2 pi, soffit-wall) radiation for every driver/role.
2. Single fixed listening position ("the couch"); DSP linear-phase/EQ
   alignment is assumed, so only *non-correctable* (nonlinear) distortion
   is scored — never distortion that a FIR filter could fix.
3. Room gain modeled only via the adiabatic pressure-zone corner `f_pz`.
4. The system is stereo, but `Scenario.target_spl` is defined as the level
   required from **one mono source/channel**, with no automatic stereo
   summing credit applied:
   - The sub is a single shared mono manifold -> must hit `target_spl`
     directly (no summing ambiguity at all).
   - The attack/tower is genuinely stereo (L/R can carry independent
     content) -> each channel must hit `target_spl` **on its own**.
   - A design using two separate mono subs instead would gain +3 to +6 dB
     from correlated/decorrelated summing, but that architecture is out of
     scope here (see `scenario.py`'s `target_spl` docstring).
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
- **Room model**: pressure-zone corner `f_pz = c/(2 L_max)`; the demand curve
  `V_dem(f)` falls 12 dB/oct above `f_pz` (radiation) and is flat below
  (adiabatic room pressurization).  All ceilings and distortion curves are
  expressed at the listening position through this demand curve.
- **Ranking = predicted non-correctable distortion at the target** (motor HD
  from excursion utilization, Doppler IM, box air-spring HD2), per the
  paper's equivalence proposition.  Feasibility criteria (Qts < Qtc,
  corner-rate rule Fs/Qts <= f_pz/Qtc, inductance corner f_L above the band,
  excursion/thermal clip) are reported as flags; flagged drivers sort after
  clean ones but are shown by default.
- **Distortion law anchoring**: `D(xi) = 0.05 xi + 0.05 xi^2` so that
  D(1) = 10 % at the IEC 62458 Xmax.  Ranking-grade, not absolute THD.
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
every number, figure, and table in `sealed_driver_criteria.tex`: it defines
the worked-example drivers S/M and the running-example scenarios (`SC_SUB`,
`SC_ATTACK`, `SC_ROOM`; `SC_SUB` and `SC_ROOM` both use the self-consistent
`r_listen = L_max/2 = 3 m` for the Sec. "Room closure" worked example, while
`SC_ATTACK` keeps the `r_listen = 1 m` driver-comparison basis of Sec. "Scope
of the tool" for driver-ranking figures -- this is why driver M's own
`r=1 m` SPL/distortion figures still show a step at `f_pz`, while driver S's
figures (evaluated at the room's own `r_listen`) do not), then calls
`plots.spl_figure`/`distortion_figure`/`demand_figure`/
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
