"""Parser for the VituixCAD driver database (tab-separated text export).

The file is a TSV with a header row; columns include (units in brackets):
Manufacturer, Model, Type, Size [in], Re [ohm], fs [Hz], Qms, Qes, Qts,
Mms [g], Cms [mm/N], Vas [l], Sd [cm2], BL [Tm], Pmax [W], Xmax [mm],
Le [mH], ...

Rows missing any parameter required for the criteria are skipped and reported
in `ParseResult.skipped` (no hidden failures, but a community database is
expected to be patchy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from audioshape.driver import Driver

# header name -> (Driver field, scale factor to SI)
_COLUMN_MAP = {
    "Size [in]": ("size_in", 1.0),
    "Re [ohm]": ("re", 1.0),
    "fs [Hz]": ("fs", 1.0),
    "Qms": ("qms", 1.0),
    "Qes": ("qes", 1.0),
    "Mms [g]": ("mms", 1e-3),
    "Vas [l]": ("vas", 1e-3),
    "Sd [cm2]": ("sd", 1e-4),
    "BL [Tm]": ("bl", 1.0),
    "Pmax [W]": ("p_max", 1.0),
    "Xmax [mm]": ("xmax", 1e-3),
    "Le [mH]": ("le", 1e-3),
}

_REQUIRED = ("size_in", "re", "fs", "qms", "qes", "mms", "vas", "sd",
             "p_max", "xmax")


@dataclass(frozen=True)
class ParseResult:
    drivers: list[Driver]
    skipped: list[str]  # "Manufacturer Model: reason" for each rejected row


def parse_database(path: str | Path) -> ParseResult:
    """Parse a VituixCAD driver database file."""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"empty database file: {path}")

    header = lines[0].split("\t")
    try:
        col_index = {name: header.index(name) for name in _COLUMN_MAP}
        i_manufacturer = header.index("Manufacturer")
        i_model = header.index("Model")
        i_type = header.index("Type")
    except ValueError as exc:
        raise ValueError(f"unrecognized database header in {path}: {exc}") from exc

    drivers: list[Driver] = []
    skipped: list[str] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        label = f"{_cell(cells, i_manufacturer)} {_cell(cells, i_model)}".strip()

        fields: dict[str, float] = {}
        for name, (attr, scale) in _COLUMN_MAP.items():
            raw = _cell(cells, col_index[name])
            if raw:
                try:
                    fields[attr] = float(raw.replace(",", ".")) * scale
                except ValueError:
                    skipped.append(f"{label}: bad value {raw!r} in {name}")
                    break
        else:
            missing = [a for a in _REQUIRED
                       if a not in fields or fields[a] <= 0.0]
            if missing:
                skipped.append(f"{label}: missing {', '.join(missing)}")
                continue
            drivers.append(Driver(
                manufacturer=_cell(cells, i_manufacturer),
                model=_cell(cells, i_model),
                type_code=_cell(cells, i_type),
                **fields,
            ))
    return ParseResult(drivers=drivers, skipped=skipped)


def _cell(cells: list[str], index: int) -> str:
    return cells[index].strip() if index < len(cells) else ""
