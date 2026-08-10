# audioshape

First-principles sealed-box bass driver selection: ranking and plots over
the VituixCAD driver database, backing the paper *Sealed-Box Driver
Selection Criteria for High-SPL Monitoring* (paper id `26325`). See
`papers/26325/sealed_driver_criteria.tex` for the full derivation.

The tool ranks drivers by predicted non-correctable (nonlinear) distortion
at a target SPL/room/listening-position scenario — motor harmonic
distortion from excursion utilization, Doppler intermodulation, and box
air-spring second harmonic — subject to feasibility gates (Qts vs. target
Qtc, the corner-rate room-closure rule, inductance corner, excursion/
thermal clipping). It supports both single-role ranking and a two-role
(sub + attack) pair ranking for the paper's Part II architecture.

`distortion_budget` (`D*`, set in the recipe) is a **selection ceiling at
the target SPL**, not an operating point: the target itself is typically a
brief reference-level peak (105 dB per channel, coherently stereo-summed
to ~111 dB at the seat, is only safely sustainable for about a minute per
NIOSH exposure limits), so a driver picked to just clear `D*` there will,
in ordinary continuous listening, run at distortion levels well below it.
`audioshape plot ... distortion` plots the total-distortion-vs-frequency
curve both at the target and 20 dB down (a proxy for normal listening
level) against the same `D*` line, so you can see how much margin actually
remains day-to-day.

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
- `data/` — `VituixCAD_driver_db.txt` driver database
- `examples/` — `example_recipe.toml` sample recipe config
- `scripts/` — regenerates every figure and table in the papers; pass `-p <paperid>`, the directory name in `papers/`: `figures -p 26325`
- `papers/` — LaTeX source of the papers

See `docs/architecture.md` for the full design.

## Quick start

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
