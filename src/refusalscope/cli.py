"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from refusalscope import __version__
from refusalscope.config import ConfigError, load_config, pair_warnings


def cmd_check_config(args: argparse.Namespace) -> int:
    """Validate a configuration file and print the pairs it defines."""
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"study: {config.study}  (config sha256 {config.digest()[:12]})")
    for pair_id in config.pair_ids():
        stock, ablated = config.pair(pair_id)
        print(
            f"pair {pair_id}: stock={stock.name} @ {stock.base_url}  ablated={ablated.name} @ {ablated.base_url}"
        )
    warnings = pair_warnings(config)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if config.is_mock:
        print("MOCK configuration: every output will be labelled MOCK")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refusalscope",
        description=(
            "Compare stock and ablated language models on structured tool calling and "
            "indirect prompt injection."
        ),
    )
    parser.add_argument("--version", action="version", version=f"refusalscope {__version__}")
    sub = parser.add_subparsers(dest="command")
    check = sub.add_parser("check-config", help="validate a study configuration")
    check.add_argument("--config", required=True)
    check.set_defaults(func=cmd_check_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    return func(args)
