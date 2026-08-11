"""Export a driver selection to VituixCAD: a `.vxp` crossover project and a
driver-database TSV snippet.

Two output formats, matching what a user can hand straight to VituixCAD:

- `driver_database_tsv` -- rows in VituixCAD's own driver-database TSV
  schema (the schema `database.parse_database` reads). The website's driver
  picker uses a separately obtained local database; a user's VituixCAD
  install may use a different snapshot and not have a selected driver.
  Import/merge this file with VituixCAD's
  Driver Database Manager first so name-based lookups (Enclosure tool,
  "Copy T/S from database") resolve for it.
- `project_xml` -- a `.vxp` crossover project. Its schema was confirmed by
  inspecting a real "File > New" project saved by VituixCAD3 (Save As,
  Ctrl+Shift+S): the `<DRIVER>` entry only carries a name/SPL/Z/response
  files, never Fs/Qts/Vas/etc. (those live in the driver database above),
  so the sealed-box/risk results computed here have no native field
  to go into -- they are written into the project `Description` instead,
  the one free-text field meant for exactly this. One `Driver` PART is
  added per role, each wired to its own Generator/Ground pair like
  VituixCAD's own blank template (a real "File > New" project leaves
  Generator and Driver unconnected too -- the user wires up a crossover
  from there), with   `DrvN` set to that role's physical unit count, and `HalfSpace` on to match
  this tool's 2 pi (soffit-wall) radiation model. Electrical feasibility in
  audioshape remains a per-driver calculation.

Pure string building, no file I/O (AGENTS.md: core stays I/O-free) -- the
caller (CLI, web backend) decides where the bytes end up.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from audioshape import physics
from audioshape.database import COLUMN_MAP
from audioshape.driver import Driver
from audioshape.ranking import Evaluation

# Exact header of VituixCAD's own driver database
# (Documents\VituixCAD\Enclosure\VituixCAD_Drivers.txt) -- columns beyond
# COLUMN_MAP's are written blank, matching how the community database itself
# leaves optional columns blank for most rows.
_DB_HEADER = [
    "Manufacturer", "Model", "Type", "Status", "Size [in]", "Re [ohm]",
    "fs [Hz]", "Qms", "Qes", "Qts", "Rms [Ns/m]", "Mms [g]", "Cms [mm/N]",
    "Vas [l]", "Sd [cm2]", "BL [Tm]", "Pmax [W]", "Xmax [mm]", "Beta",
    "Z1k [ohm]", "Z10k [ohm]", "Le [mH]", "Leb [mH]", "Ke [sH]", "Rss [ohm]",
    "USPL [dB]", "BL\u00b2/Re", "Revision", "Updated",
]


@dataclass(frozen=True)
class RoleSelection:
    """One role's chosen driver result, for VituixCAD export.

    `role` is a display label only (e.g. "sub", "attack"); `band_low`/
    `band_high` are whatever band `evaluation` was actually computed over
    (an `Evaluation` does not remember its own band, so the caller -- which
    chose it -- passes it through).
    """

    role: str
    evaluation: Evaluation
    band_low: float
    band_high: float


def _fmt(x: float) -> str:
    return f"{x:.6g}"


# ----------------------------------------------------------------------
# Driver-database TSV (round-trips through database.parse_database)
# ----------------------------------------------------------------------

def driver_database_tsv(drivers: Sequence[Driver]) -> str:
    """VituixCAD driver-database TSV text for `drivers` (header + one row
    each, de-duplicated by label), ready to merge into a user's local
    ``VituixCAD_Drivers.txt`` via the Driver Database Manager."""
    lines = ["\t".join(_DB_HEADER)]
    seen: set[str] = set()
    for d in drivers:
        if d.label() in seen:
            continue
        seen.add(d.label())
        lines.append(_driver_row(d))
    return "\n".join(lines) + "\n"


def _driver_row(d: Driver) -> str:
    cells = dict.fromkeys(_DB_HEADER, "")
    cells["Manufacturer"] = d.manufacturer
    cells["Model"] = d.model
    cells["Type"] = d.type_code
    for name, (attr, scale) in COLUMN_MAP.items():
        value = getattr(d, attr)
        if math.isfinite(value):
            cells[name] = _fmt(value / scale)
    # Qts has no COLUMN_MAP entry (Driver.qts is derived from Qes/Qms, not a
    # parsed field) but is an exact, not estimated, T/S value -- fill it in
    # since a database row with Qes/Qms and a blank Qts looks incomplete.
    cells["Qts"] = _fmt(d.qts)
    return "\t".join(cells[name] for name in _DB_HEADER)


# ----------------------------------------------------------------------
# .vxp crossover project
# ----------------------------------------------------------------------

# Global project settings: VituixCAD's own "File > New" defaults, except
# HalfSpace (this tool models 2 pi / soffit-wall radiation everywhere, not
# VituixCAD's own full-space default).
_SPEAKER_DEFAULTS = [
    ("ReferenceAngle", "0"), ("DualPlane", "True"), ("KeywordHor", "hor"),
    ("KeywordVer", "ver"), ("AngleMultiplier", "1"), ("XMin", "20"),
    ("XMax", "20000"), ("Interpolate", "True"),
    ("UserAnglesHor", None), ("UserAnglesVer", None),
    ("IntensitySphere", "True"), ("IntensityCylinder", "False"),
    ("IncludeHor", "True"), ("IncludeVer", "True"),
    ("HalfSpace", "True"), ("Corner", "False"),
    ("LiswinDI", "True"), ("CTA2034Aweights", "True"),
    ("AngleStep", "10"), ("FrontWall", "False"), ("FrontWallZ", "1000"),
    ("LeftWall", "False"), ("LeftWallX", "-1000"), ("Ceiling", "False"),
    ("CeilingY", "1500"), ("Floor", "False"), ("FloorY", "-1000"),
    ("Toein", "25"), ("AbsorpWall", "2"), ("AbsorpCeil", "2"),
    ("AbsorpFloor", "2"), ("ReferDistance", "2000"), ("PlaneRotation", "0"),
    ("DrvOffsetX", "0"), ("DrvOffsetY", "0"),
]


def _reference_spl(d: Driver) -> float:
    """Estimated half-space sensitivity at 2.83 V / 1 m, from this driver's
    own Thiele/Small parameters (P = 2.83^2/Re at its own Re;
    `physics.spl_thermal_ceiling`'s 112.2 dB constant already is this tool's
    half-space reference). A computed placeholder, not a measurement --
    callers should say so wherever this ends up (e.g. a project
    description)."""
    p_2v83 = 2.83 * 2.83 / d.re
    return physics.spl_thermal_ceiling(d.eta0, p_2v83, 1.0)


def _add_param(parent: ET.Element, pi: int, name: str, value: float,
              unit: str, vmin: float, vmax: float) -> None:
    p = ET.SubElement(parent, "PARAM")
    p.set("pi", str(pi))
    ET.SubElement(p, "Name").text = name
    ET.SubElement(p, "Value").text = _fmt(value)
    ET.SubElement(p, "Unit").text = unit
    ET.SubElement(p, "Optimize").text = "False"
    ET.SubElement(p, "Expression")
    ET.SubElement(p, "Min").text = _fmt(vmin)
    ET.SubElement(p, "Max").text = _fmt(vmax)
    ET.SubElement(p, "OptiBlock").text = "False"


def _add_wire(parent: ET.Element, wi: int, x: float, y: float) -> None:
    w = ET.SubElement(parent, "WIRE")
    w.set("wi", str(wi))
    ET.SubElement(w, "X").text = _fmt(x)
    ET.SubElement(w, "Y").text = _fmt(y)


def _add_target(parent: ET.Element, tag: str, freq_min: float, freq_max: float,
                spl: float, drv_n: int = 1) -> None:
    t = ET.SubElement(parent, tag)
    ET.SubElement(t, "FreqMin").text = _fmt(freq_min)
    ET.SubElement(t, "FreqMax").text = _fmt(freq_max)
    ET.SubElement(t, "SPL").text = _fmt(spl)
    ET.SubElement(t, "Tilt").text = "0.0"
    ET.SubElement(t, "DrvN").text = str(drv_n)
    ET.SubElement(t, "Invert").text = "False"
    ET.SubElement(t, "FreeLF").text = "False"
    ET.SubElement(t, "FreeHF").text = "False"


def _add_driver_entry(parent: ET.Element, di: int, model: str, spl: float,
                      z: float) -> None:
    d = ET.SubElement(parent, "DRIVER")
    d.set("di", str(di))
    ET.SubElement(d, "Model").text = model
    ET.SubElement(d, "SPL").text = _fmt(spl)
    ET.SubElement(d, "Z").text = _fmt(z)
    ET.SubElement(d, "ExtendedData").text = "False"
    ET.SubElement(d, "ResponseDirectory")
    ET.SubElement(d, "ResponseScale").text = "1"
    ET.SubElement(d, "ResponseDelay").text = "0"
    ET.SubElement(d, "ResponseInvert").text = "False"
    ET.SubElement(d, "ResponseMute").text = "False"
    ET.SubElement(d, "MinimumPhase").text = "False"
    ET.SubElement(d, "ResponseSmooth").text = "None"
    ET.SubElement(d, "ImpedanceFile")
    ET.SubElement(d, "ImpedanceScale").text = "1"
    r = ET.SubElement(d, "RESPONSE")
    r.set("ri", "0")
    ET.SubElement(r, "FileName")
    ET.SubElement(r, "Hor").text = "0"
    ET.SubElement(r, "Ver").text = "0"


def _add_generator(parent: ET.Element, xi: int, cx: float, cy: float,
                   part_id: str) -> int:
    p = ET.SubElement(parent, "PART")
    p.set("xi", str(xi))
    ET.SubElement(p, "Type").text = "Generator"
    ET.SubElement(p, "CenX").text = _fmt(cx)
    ET.SubElement(p, "CenY").text = _fmt(cy)
    ET.SubElement(p, "PartID").text = part_id
    ET.SubElement(p, "GUID")
    _add_param(p, 0, "Eg", 2.83, "V", 0.01, 400)
    _add_param(p, 1, "Tg", 0, "us", -50000, 50000)
    _add_param(p, 2, "Rg", 0.001, "\u03a9", 0.001, 1000)
    _add_wire(p, 0, cx, cy - 3)
    _add_wire(p, 1, cx, cy + 3)
    return xi + 1


def _add_ground(parent: ET.Element, xi: int, cx: float, cy: float) -> int:
    """Ground for the component whose row center is `(cx, cy)` -- its own
    center sits one grid unit below the shared lead at `cy + 3`."""
    p = ET.SubElement(parent, "PART")
    p.set("xi", str(xi))
    ET.SubElement(p, "Type").text = "Ground"
    ET.SubElement(p, "CenX").text = _fmt(cx)
    ET.SubElement(p, "CenY").text = _fmt(cy + 4)
    ET.SubElement(p, "Open").text = "False"
    ET.SubElement(p, "Rotated").text = "False"
    ET.SubElement(p, "GUID")
    _add_wire(p, 0, cx, cy + 3)
    return xi + 1


def _add_driver_part(parent: ET.Element, xi: int, cx: float, cy: float,
                     model: str, part_id: str, band_low: float,
                     band_high: float, target_spl: float, drv_n: int) -> int:
    p = ET.SubElement(parent, "PART")
    p.set("xi", str(xi))
    ET.SubElement(p, "Type").text = "Driver"
    ET.SubElement(p, "CenX").text = _fmt(cx)
    ET.SubElement(p, "CenY").text = _fmt(cy)
    ET.SubElement(p, "Model").text = model
    ET.SubElement(p, "Open").text = "False"
    ET.SubElement(p, "Shorted").text = "False"
    ET.SubElement(p, "Muted").text = "False"
    ET.SubElement(p, "Hidden").text = "False"
    ET.SubElement(p, "Inverted").text = "False"
    ET.SubElement(p, "PartID").text = part_id
    ET.SubElement(p, "GUID")
    _add_target(p, "DriverTarget", band_low, band_high, target_spl, drv_n)
    _add_target(p, "FilterTarget", band_low, band_high, 0.0, drv_n)
    _add_param(p, 0, "X", 0, "mm", -2000, 2000)
    _add_param(p, 1, "Y", 0, "mm", -5000, 5000)
    _add_param(p, 2, "Z", 0, "mm", -2000, 2000)
    _add_param(p, 3, "R", 0, "deg", -180, 180)
    _add_param(p, 4, "T", 0, "deg", -180, 180)
    _add_wire(p, 0, cx - 1, cy - 3)
    _add_wire(p, 1, cx - 1, cy + 3)
    return xi + 1


def project_xml(selections: Sequence[RoleSelection], description: str = "") -> str:
    """Build a VituixCAD `.vxp` crossover-project (as XML text): one Driver
    part per `(role, Evaluation, band)` in `selections`, each wired to its
    own Generator/Ground pair, `DrvN` set to that role's unit count, and
    `description` written verbatim into the project's `Description` field
    (see module docstring for why: no other field fits box/risk
    results)."""
    if not selections:
        raise ValueError("need at least one RoleSelection")

    root = ET.Element("SPEAKER")
    ET.SubElement(root, "Description").text = description or None
    for name, value in _SPEAKER_DEFAULTS:
        ET.SubElement(root, name).text = value

    sc = selections[0].evaluation.scenario
    # VituixCAD exposes only one global system target, but this architecture
    # has different role bands and levels. Per-driver DriverTarget elements
    # below carry the unambiguous role-specific values instead.

    for di, sel in enumerate(selections):
        d = sel.evaluation.driver
        _add_driver_entry(root, di, d.label(), spl=_reference_spl(d),
                          z=round(d.re, 1))

    ET.SubElement(root, "Variant").text = "0"

    xover = ET.SubElement(root, "CROSSOVER")
    ET.SubElement(xover, "DSP").text = "Analog"
    ET.SubElement(xover, "SampleRate").text = "96000"
    ET.SubElement(xover, "DSPSettings")
    ET.SubElement(xover, "DSPTemplate")

    xi = _add_generator(xover, 0, cx=3, cy=9, part_id="G1")
    xi = _add_ground(xover, xi, cx=3, cy=9)
    for row, sel in enumerate(selections):
        cy = 9 + 20 * row
        ev = sel.evaluation
        xi = _add_driver_part(xover, xi, cx=33, cy=cy, model=ev.driver.label(),
                              part_id=f"D{row + 1}", band_low=sel.band_low,
                              band_high=sel.band_high,
                              target_spl=sc.target_spl_for(ev.role),
                              drv_n=ev.boxed.n_units)
        xi = _add_ground(xover, xi, cx=32, cy=cy)

    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="utf-8"?>\n'
            "<!--VituixCAD PROJECT-->\n"
            "<!--Version 2-->\n" + body + "\n")
