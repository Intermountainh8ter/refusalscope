"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from refusalscope import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refusalscope",
        description=(
            "Compare stock and ablated language models on structured tool calling and "
            "indirect prompt injection."
        ),
    )
    parser.add_argument("--version", action="version", version=f"refusalscope {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    parser.print_help(sys.stderr)
    del args
    return 2
