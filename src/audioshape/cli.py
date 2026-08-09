"""Command-line interface.

    audioshape pair --recipe example_recipe.toml
    audioshape rank --recipe example_recipe.toml --role sub
    audioshape plot --recipe example_recipe.toml --driver "TD15H" \
        --role sub --save dev/out
    audioshape export-vituixcad --recipe example_recipe.toml \
        --sub-driver "TD15H" --attack-driver "TD15S" --save dev/out

Thin layer: argument parsing, table printing, figure saving.  All computation
lives in `ranking` / `physics`.  Configuration lives in a recipe TOML file
(`recipe.load_recipe`), not in long flag lists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from audioshape.database import parse_database
from audioshape.driver import Driver
from audioshape.ranking import Evaluation, evaluate, pair_rank, rank
from audioshape.recipe import load_recipe
from audioshape.scenario import Scenario

_ROLE_BANDS = {
    # role -> (band_low, band_high, doppler_ref) as Scenario attribute names
    "sub": ("f_low", "f_split", "f_split"),
    "attack": ("f_split", "f_high", "f_high"),
    "full": ("f_low", "f_high", "f_split"),  # single driver covers everything
}


def _band_for_role(sc: Scenario, role: str) -> tuple[float, float, float]:
    lo_name, hi_name, ref_name = _ROLE_BANDS[role]
    return getattr(sc, lo_name), getattr(sc, hi_name), getattr(sc, ref_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audioshape",
        description="Sealed-box bass driver selection from first principles. "
                    "target_spl in the recipe is the level required from a "
                    "single source (mono sub manifold, or one stereo tower "
                    "channel on its own) -- no stereo summing correction is "
                    "applied.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_recipe_arg(p: argparse.ArgumentParser) -> None:
        p.add_argument("--recipe", type=Path, required=True,
                       help="recipe TOML file (see example_recipe.toml)")

    p_rank = sub.add_parser("rank", help="rank drivers for one role")
    add_recipe_arg(p_rank)
    p_rank.add_argument("--role", choices=list(_ROLE_BANDS), default="sub",
                        help="sub, attack, or full-range (default: sub)")
    p_rank.add_argument("--feasible-only", action="store_true",
                        help="hide drivers with feasibility flags")
    p_rank.add_argument("--top", type=int, default=25,
                        help="rows to print (default 25)")

    p_pair = sub.add_parser("pair", help="jointly rank sub + attack pairs")
    add_recipe_arg(p_pair)
    p_pair.add_argument("--top", type=int, default=15,
                        help="pairs to print (default 15)")

    p_plot = sub.add_parser("plot", help="SPL and distortion plots for one driver")
    add_recipe_arg(p_plot)
    p_plot.add_argument("--driver", required=True,
                        help="substring of 'Manufacturer Model' (case-insensitive)")
    p_plot.add_argument("--role", choices=list(_ROLE_BANDS), default="sub",
                        help="sub, attack, or full-range (default: sub)")
    p_plot.add_argument("--units", type=int, default=None,
                        help="override recipe's unit count for this role")
    p_plot.add_argument("--save", type=Path, default=None,
                        help="directory for PNGs (default: show interactively)")

    p_export = sub.add_parser(
        "export-vituixcad",
        help="export a sub/attack driver selection as a VituixCAD project "
             "+ driver-database TSV")
    add_recipe_arg(p_export)
    p_export.add_argument("--sub-driver", default=None,
                          help="substring of 'Manufacturer Model' for the sub role")
    p_export.add_argument("--attack-driver", default=None,
                          help="substring of 'Manufacturer Model' for the attack role")
    p_export.add_argument("--save", type=Path, required=True,
                          help="output directory for the .vxp project + driver TSV")
    return parser


def cmd_rank(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    sc = recipe.scenario
    result = parse_database(recipe.db)
    band_low, band_high, doppler_ref = _band_for_role(sc, args.role)
    n_units = recipe.sub_units if args.role == "sub" else recipe.attack_units
    size_min, size_max = ((recipe.sub_size_min, recipe.sub_size_max)
                          if args.role == "sub"
                          else (recipe.attack_size_min, recipe.attack_size_max))

    evals = rank(result.drivers, sc, n_units=n_units,
                min_size_in=size_min, max_size_in=size_max,
                band_low=band_low, band_high=band_high, doppler_ref=doppler_ref)
    if args.feasible_only:
        evals = [e for e in evals if e.feasible]

    print(f"Parsed {len(result.drivers)} drivers "
          f"({len(result.skipped)} skipped for missing data).")
    print(f"Role '{args.role}': band [{band_low:g}, {band_high:g}] Hz, "
          f"{n_units} unit(s), target {sc.target_spl:g} dB @ {sc.r_listen:g} m "
          f"(this source's own level; no stereo summing applied)")
    print(f"Room {sc.v_room:g} m^3 (f_pz={sc.f_pz:.1f} Hz), "
          f"D*={sc.distortion_budget:.1%}, Qtc={sc.qtc:g}\n")

    _print_rank_table(evals[:args.top])
    return 0


def _print_rank_table(evals: list[Evaluation]) -> None:
    header = (f"{'#':>3} {'driver':<38} {'in':>4} {'Vd[L]':>6} {'Vb[L]':>6} "
              f"{'Fc':>5} {'xi_x':>5} {'D%':>6} {'Dop%':>5} {'box%':>5} "
              f"{'xi_P':>5} {'N*':>3} flags")
    print(header)
    print("-" * len(header))
    for i, ev in enumerate(evals, 1):
        print(_format_row(i, ev))


def _format_row(i: int, ev: Evaluation) -> str:
    d = ev.driver
    flags = "" if ev.feasible else "; ".join(ev.reasons)
    return (f"{i:>3} {d.label():<38.38} {d.size_in:>4.0f} {d.vd*1e3:>6.2f} "
            f"{ev.boxed.vb*1e3:>6.0f} {ev.boxed.fc:>5.1f} {ev.xi_x:>5.2f} "
            f"{100*ev.hd:>6.2f} {100*ev.doppler_im:>5.2f} "
            f"{100*ev.box_hd2:>5.2f} {ev.xi_p:>5.2f} "
            f"{ev.n_units_required:>3} {flags}")


def cmd_pair(args: argparse.Namespace) -> int:
    recipe = load_recipe(args.recipe)
    sc = recipe.scenario
    result = parse_database(recipe.db)
    pairs = pair_rank(result.drivers, sc,
                      sub_units=recipe.sub_units, attack_units=recipe.attack_units,
                      sub_size_min=recipe.sub_size_min, sub_size_max=recipe.sub_size_max,
                      attack_size_min=recipe.attack_size_min,
                      attack_size_max=recipe.attack_size_max,
                      top_k_each=recipe.top_k_each)

    print(f"Parsed {len(result.drivers)} drivers "
          f"({len(result.skipped)} skipped for missing data).")
    print(f"Target {sc.target_spl:g} dB @ {sc.r_listen:g} m (this source's own "
          f"level; no stereo summing applied), room {sc.v_room:g} m^3 "
          f"(f_pz={sc.f_pz:.1f} Hz), D*={sc.distortion_budget:.1%}, "
          f"Qtc={sc.qtc:g}, split {sc.f_split:g} Hz\n")

    for i, pe in enumerate(pairs[:args.top], 1):
        print(f"--- pair #{i} "
              f"(total distortion {100*pe.total_distortion:.2f} %, "
              f"{'feasible' if pe.feasible else 'FLAGGED'}) ---")
        print(f"  sub    ({recipe.sub_units}x): " + _format_row(1, pe.sub).lstrip("1 "))
        print(f"  attack ({recipe.attack_units}x): "
              + _format_row(1, pe.attack).lstrip("1 "))
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    from audioshape import plots  # keep matplotlib import lazy

    recipe = load_recipe(args.recipe)
    sc = recipe.scenario
    result = parse_database(recipe.db)
    needle = args.driver.lower()
    matches = [d for d in result.drivers if needle in d.label().lower()]
    if not matches:
        print(f"no driver matching {args.driver!r}", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"{len(matches)} matches; using first: "
              + ", ".join(d.label() for d in matches[:5]), file=sys.stderr)
    driver = matches[0]

    band_low, band_high, doppler_ref = _band_for_role(sc, args.role)
    default_units = recipe.sub_units if args.role == "sub" else recipe.attack_units
    n_units = args.units if args.units is not None else default_units

    ev = evaluate(driver, sc, n_units=n_units,
                 band_low=band_low, band_high=band_high, doppler_ref=doppler_ref)
    if not ev.feasible:
        print("note: " + "; ".join(ev.reasons), file=sys.stderr)

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        stem = driver.label().replace(" ", "_").replace("/", "-")
        for maker, kind in ((plots.spl_figure, "spl"),
                            (plots.distortion_figure, "distortion")):
            out = args.save / f"{stem}_{args.role}_{kind}.png"
            maker(ev).savefig(out, dpi=150)
            print(f"wrote {out}")
    else:
        import matplotlib.pyplot as plt
        for maker in (plots.spl_figure, plots.distortion_figure):
            fig = plt.figure(figsize=(9, 6), layout="constrained")
            maker(ev, fig=fig)
        plt.show()
    return 0


def _match_one(drivers: list[Driver], needle: str, role: str) -> Driver | None:
    lo = needle.lower()
    matches = [d for d in drivers if lo in d.label().lower()]
    if not matches:
        print(f"no driver matching {needle!r} for role {role!r}", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"{len(matches)} matches for role {role!r}; using first: "
              + ", ".join(d.label() for d in matches[:5]), file=sys.stderr)
    return matches[0]


def cmd_export(args: argparse.Namespace) -> int:
    from audioshape import vituixcad  # keep import lazy, mirrors cmd_plot

    recipe = load_recipe(args.recipe)
    sc = recipe.scenario
    result = parse_database(recipe.db)

    role_requests = [
        ("sub", args.sub_driver, recipe.sub_units),
        ("attack", args.attack_driver, recipe.attack_units),
    ]
    selections: list[vituixcad.RoleSelection] = []
    for role, needle, n_units in role_requests:
        if not needle:
            continue
        driver = _match_one(result.drivers, needle, role)
        if driver is None:
            return 1
        band_low, band_high, doppler_ref = _band_for_role(sc, role)
        ev = evaluate(driver, sc, n_units=n_units, band_low=band_low,
                     band_high=band_high, doppler_ref=doppler_ref)
        selections.append(vituixcad.RoleSelection(role, ev, band_low, band_high))

    if not selections:
        print("pass --sub-driver and/or --attack-driver", file=sys.stderr)
        return 1

    args.save.mkdir(parents=True, exist_ok=True)
    project_path = args.save / "driver_selection.vxp"
    project_path.write_bytes(vituixcad.project_xml(selections).encode("utf-8-sig"))
    tsv_path = args.save / "VituixCAD_Drivers_selection.txt"
    tsv_path.write_text(
        vituixcad.driver_database_tsv([s.evaluation.driver for s in selections]),
        encoding="utf-8")
    print(f"wrote {project_path}")
    print(f"wrote {tsv_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "rank":
        return cmd_rank(args)
    if args.command == "pair":
        return cmd_pair(args)
    if args.command == "export-vituixcad":
        return cmd_export(args)
    return cmd_plot(args)


if __name__ == "__main__":
    sys.exit(main())
