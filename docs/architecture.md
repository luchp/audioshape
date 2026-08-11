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
  `f_split` for attack) rather than a single fixed Qtc -- see "Qtc is a
  ceiling, not a fixed target" below. When omitted, `doppler_ref` defaults
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
  The canonical recipe uses four sub drivers and requires an even count so
  the manifold can be built as opposed pairs. Such pairs cancel reaction
  force only to first order under matched drive and mounting.
  There is **no cross-driver coupling term**: sub and attack are physically
  separate enclosures sharing only the baffle plane, so each Doppler
  sideband indicator is intrinsic to its own band (see
  `papers/26325/sealed_driver_criteria.tex` "Scope of the tool" note).

### Qtc is a ceiling, not a fixed target

`Scenario.qtc` is a **ceiling**: `ranking.evaluate()` never boxes a driver
above it, but a driver whose corner rate `Fs/Qts` would overshoot its own
role's corner target at that ceiling gets a *lower* Qtc (bigger box)
instead of being rejected outright --
`physics.qtc_for_target_corner(driver.corner_rate, f_target, sc.qtc) =
min(sc.qtc, f_target / driver.corner_rate)`, where `f_target =
sc.target_corner_hz(role)`. This mirrors the paper's own undershoot/
overshoot asymmetry after `eq:Fsrule` (undershoot is a free EQ cut,
overshoot is a taxed boost that costs excursion) applied to *both* roles'
corners, not just the sub's `f_pz`. If the target is unreachable in any
finite box (`qtc <= Qts`), evaluation uses an explicit `10 * Vas` fallback
and reports the EQ caveat; excursion and electrical demand, rather than the
geometry alone, decide feasibility. `boxed.qtc` (not `sc.qtc`) is therefore
the value to read/print/plot for a specific driver.

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
2. A compact couch area represented by a declared reference position or
   small measurement set. The intended optimization preserves independent
   stereo-channel timing and direct/early-field information; it is not a
   diffuse-field or multi-seat SPL optimization.
3. Below `f_pz`, the prepared room is represented by an acoustic compliance
   plus a declared leakage corner. This is a low-frequency limiting model,
   not a claim that the complete room is one second-order system.
4. The system is stereo, but `Scenario.sub_target_spl`/`attack_target_spl`
   are each defined as the level required from **one mono source/channel**
   of that role, with no automatic stereo summing credit applied:
   - The sub is a single shared mono manifold -> must hit `sub_target_spl`
     directly (no summing ambiguity at all).
   - The attack/tower is genuinely stereo (L/R can carry independent
     content) -> each channel must hit `attack_target_spl` **on its own**.
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
every number, figure, and table in `sealed_driver_criteria.tex`. All
system-feasibility assets use one canonical 3 m couch-area scenario; a 1 m
value may appear only as a clearly labelled intrinsic descriptor. The script
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
