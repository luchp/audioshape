# audioshape

First-principles sealed-box bass driver selection using public datasheet
fields and an optional separately obtained local candidate database. It
backs *Datasheet-Based Ranking of Sealed-Box Bass Drivers for a Small-Area
DSP-Integrated Stereo System* (paper id `26325`). See
`papers/26325/sealed_driver_criteria.tex` for the full derivation.

The tool applies hard excursion and finite-amplifier feasibility gates,
then ranks the surviving drivers with separate risk indicators, Pareto
fronts, and role-specific policy keys. Excursion utilization, transient
excursion, the first-order Doppler sideband ratio, box-spring nonlinearity,
voltage, current, power, inductive risk, and box volume are never added into
a synthetic "total distortion" value.

Published one-way `Xmax` is the clipping boundary. The default 80% value is
only a preferred design margin and unit-count guide, leaving about 1.9 dB of
excursion headroom. The 110 dB mono-sub and 105 dB-per-stereo-channel targets
are occasional maximum-output conditions, not continuous listening levels.
Hearing-exposure guidance can contextualize how extreme sustained levels
would be, but does not define the mechanical screening limit.

## Layout

- `src/audioshape/physics.py` — pure physics functions, SI units, no I/O
- `src/audioshape/driver.py` — `Driver`/`BoxedDriver` dataclasses + derived T-S quantities
- `src/audioshape/scenario.py` — `Scenario` dataclass: room, distance, target SPL, budgets
- `src/audioshape/vented.py` — vented-box comparison model (backs the paper's "why sealed, not vented" section only; not part of ranking/CLI)
- `src/audioshape/recipe.py` — `Recipe` dataclass + `load_recipe()`: TOML config
- `src/audioshape/database.py` — VituixCAD TSV parser
- `src/audioshape/ranking.py` — `evaluate`/`rank` and `PairEvaluation`/`pair_rank`
- `src/audioshape/plots.py` — matplotlib figure functions
- `src/audioshape/vituixcad.py` — export a driver selection to a VituixCAD `.vxp` project + driver-database TSV
- `src/audioshape/cli.py` — `rank`, `pair`, `plot`, `export-vituixcad` subcommands
- `tests/` — pytest, physics pinned to the paper's worked example
- `data/` — local driver-database location; the VituixCAD database is not
  distributed by this repository
- `examples/` — `example_recipe.toml` sample recipe config
- `scripts/` — regenerates every figure and table in the papers; pass `-p <paperid>`, the directory name in `papers/`: `figures -p 26325`
- `papers/` — LaTeX source of the papers

See `docs/architecture.md` for the full design.

## Quick start

Obtain/export the VituixCAD driver database under its applicable terms and
place the local file at `data/VituixCAD_driver_db.txt`. The path is ignored by
Git and must not be committed.

```
uv sync
uv run audioshape rank --recipe examples/example_recipe.toml --role sub
uv run audioshape pair --recipe examples/example_recipe.toml --top 5
uv run audioshape plot --recipe examples/example_recipe.toml --role sub \
    --driver "UMII18" --save dev/out
uv run audioshape export-vituixcad --recipe examples/example_recipe.toml \
    --sub-driver "UMII18" --attack-driver "TD15S" --save dev/out
```

## Development

```
uv sync --group dev
uv run pytest -q
```

## Generate figures and tables for paper 26325

```
scripts\figures -p 26325
```

## Build the paper

```
cd papers/26325
pdflatex -interaction=nonstopmode -halt-on-error sealed_driver_criteria.tex
bibtex sealed_driver_criteria
pdflatex -interaction=nonstopmode -halt-on-error sealed_driver_criteria.tex
pdflatex -interaction=nonstopmode -halt-on-error sealed_driver_criteria.tex
```

Or use the release pipeline: `scripts/publish_release.py` (see `scripts/release.cmd`).

## License

This reference implementation is released under the **MIT License**. The
underlying paper is licensed under **CC BY 4.0**.

## How to Cite

See `CITATION.cff`.
