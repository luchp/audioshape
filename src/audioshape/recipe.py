"""Recipe files: a TOML config bundling a `Scenario` with the pair-ranking
parameters (driver database path, sub/attack size windows and unit counts).

Replaces long CLI flag lists (AGENTS.md: configuration in config files, not
hardcoded / not endless flags).  Read-only: a recipe is meant to be hand
authored and version-controlled next to a project, not written by the tool.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from audioshape.scenario import Scenario

_SCENARIO_FIELDS = {f.name for f in fields(Scenario)}


@dataclass(frozen=True)
class Recipe:
    """A scenario plus everything needed to run `pair_rank` on a database."""

    scenario: Scenario
    db: Path
    sub_units: int = 2
    attack_units: int = 1
    require_even_sub_units: bool = True
    sub_size_min: float = 0.0
    sub_size_max: float = 21.0
    attack_size_min: float = 0.0
    attack_size_max: float = float("inf")
    top_k_each: int = 15

    def __post_init__(self) -> None:
        self.scenario.require_valid_manifold_crossover()
        if self.sub_units < 1 or self.attack_units < 1:
            raise ValueError("role unit counts must be positive")
        if self.require_even_sub_units and self.sub_units % 2:
            raise ValueError(
                "opposed-pair sub manifolds require an even unit count"
            )


def load_recipe(path: str | Path) -> Recipe:
    """Load a `Recipe` from a TOML file.

    Expected shape::

        # Local database obtained/exported separately; not distributed.
        db = "../data/VituixCAD_driver_db.txt"   # relative to the recipe file

        [scenario]
        v_room = 60.0
        l_max = 6.0
        r_listen = 3.0
        sub_target_spl = 110.0
        attack_target_spl = 105.0
        stereo_low_bass_summing_db = 6.0
        room_model = "leaky_pressure_zone"
        leakage_corner_hz = 10.0
        alignment_qtc = 0.55
        max_box_vas_ratio = 4.0
        f_low = 15.0
        f_split = 80.0
        manifold_crossover_ceiling_hz = 80.0
        f_high = 250.0
        preferred_excursion = 0.80
        amplifier_voltage_rms = 90.0
        amplifier_current_rms = 15.0
        amplifier_power_continuous = 500.0
        amplifier_power_burst = 1000.0

        [pair]
        sub_units = 2
        attack_units = 1
        require_even_sub_units = true
        sub_size_min = 15
        sub_size_max = 21
        attack_size_min = 8
        attack_size_max = 15
        top_k_each = 15
    """
    path = Path(path)
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    if "db" not in data:
        raise ValueError(f"recipe {path} is missing a top-level 'db' path")
    db = (path.parent / data["db"]).resolve()

    scenario_data = data.get("scenario", {})
    unknown = set(scenario_data) - _SCENARIO_FIELDS
    if unknown:
        raise ValueError(f"recipe {path}: unknown [scenario] keys {unknown}")
    scenario = Scenario(**scenario_data)

    pair_data = dict(data.get("pair", {}))
    known_pair_fields = {f.name for f in fields(Recipe)} - {"scenario", "db"}
    unknown_pair = set(pair_data) - known_pair_fields
    if unknown_pair:
        raise ValueError(f"recipe {path}: unknown [pair] keys {unknown_pair}")

    return Recipe(scenario=scenario, db=db, **pair_data)
