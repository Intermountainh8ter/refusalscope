#!/usr/bin/env python3
"""Build report tables and figures from results/*.json.

This script reads result files and nothing else. It refuses to render when none exist, and it
refuses MOCK result files unless --allow-mock is given (the output is then stamped MOCK).

    python scripts/make_report.py --results results --out report
    python scripts/make_report.py --update-readme README.md   # real results only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import refusalscope  # noqa: F401
except ImportError:  # running from a checkout that was not pip-installed
    sys.path.insert(0, str(ROOT / "src"))

from refusalscope.report import ReportError, make_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results", default=str(ROOT / "results"), help="directory with results/*.json"
    )
    parser.add_argument(
        "--out", default=str(ROOT / "report"), help="where to write tables and figures"
    )
    parser.add_argument(
        "--allow-mock", action="store_true", help="render MOCK results, stamped MOCK"
    )
    parser.add_argument("--update-readme", help="README.md whose results block should be filled")
    args = parser.parse_args(argv)
    try:
        written = make_report(
            args.results, args.out, allow_mock=args.allow_mock, readme=args.update_readme
        )
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
