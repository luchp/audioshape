"""Sealed-box bass driver selection from first principles.

Package layout (keep core importable from a web backend later):

- physics    -- pure functions, SI units, no I/O
- driver     -- Driver / BoxedDriver dataclasses with derived parameters
- database   -- VituixCAD driver-database (TSV) parser
- scenario   -- user scenario (room, listening distance, target SPL, ...)
- ranking    -- per-driver evaluation and sorting
- plots      -- matplotlib reporting layer
- cli        -- argparse entry point
"""

from audioshape.driver import BoxedDriver, Driver
from audioshape.scenario import Scenario

__all__ = ["Driver", "BoxedDriver", "Scenario"]
