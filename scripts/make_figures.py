#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Dispatch figure generation to scripts/papers/<paper_id>/make_figures.py"
    )
    parser.add_argument(
        "-p",
        "--paper-id",
        required=True,
        help="Paper identifier, for example: 26293",
    )
    return parser.parse_known_args()


def load_dispatch_target(paper_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", paper_id):
        raise SystemExit(f"Invalid paper id: {paper_id!r}")
    module_name = f"scripts.papers.{paper_id}.make_figures"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import figure script for paper '{paper_id}' ({module_name})."
        ) from exc
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        raise SystemExit(
            f"Module '{module_name}' does not provide a callable main() function."
        )
    return main_fn


def main() -> None:
    args, passthrough_args = parse_args()
    dispatch_main = load_dispatch_target(args.paper_id)
    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0], *passthrough_args]
        dispatch_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
