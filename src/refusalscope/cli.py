"""Command line entry point: ``refusalscope <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from refusalscope import __version__
from refusalscope.config import ConfigError, StudyConfig, load_config, pair_warnings


def _load(path: str) -> StudyConfig | None:
    try:
        return load_config(path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return None


def cmd_check_config(args: argparse.Namespace) -> int:
    """Validate a configuration file and print the pairs it defines."""
    config = _load(args.config)
    if config is None:
        return 2
    print(f"study: {config.study}  (config sha256 {config.digest()[:12]})")
    for pair_id in config.pair_ids():
        stock, ablated = config.pair(pair_id)
        print(
            f"pair {pair_id}: stock={stock.name} @ {stock.base_url}  "
            f"ablated={ablated.name} @ {ablated.base_url}"
        )
    for warning in pair_warnings(config):
        print(f"warning: {warning}", file=sys.stderr)
    if config.is_mock:
        print("MOCK configuration: every output will be labelled MOCK")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Write the default suites as JSON Lines snapshots."""
    from refusalscope.suites import write_snapshots

    counts = write_snapshots(args.out, args.only)
    for name, count in counts.items():
        print(
            f"wrote {count} rows for {name} to {Path(args.out) / (name if name != 'kl' else 'kl_prompts')}.jsonl"
        )
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check every endpoint in the configuration and explain how to fix failures."""
    from refusalscope.doctor import run_doctor

    config = _load(args.config)
    if config is None:
        return 2
    checks = run_doctor(config)
    for check in checks:
        print(check.line())
    failed = sum(1 for c in checks if c.status == "fail")
    warned = sum(1 for c in checks if c.status == "warn")
    print(f"\n{len(checks)} checks: {failed} failed, {warned} warnings")
    return 1 if failed else 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run (or resume) the suites against every model in the configuration."""
    from refusalscope.runner import RunError, Runner, resolve_suites

    config = _load(args.config)
    if config is None:
        return 2
    try:
        suites = resolve_suites(args.suite)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        run_id = args.run_id or f"{'MOCK-' if config.is_mock else ''}{config.study}-{stamp}"
        if config.is_mock and not run_id.startswith("MOCK"):
            run_id = f"MOCK-{run_id}"
        run_dir = Path(args.out or config.output_dir) / run_id
        runner = Runner(config, run_dir, workers=args.workers, run_id=run_id)
        for note in runner.data.notes:
            print(f"note: {note}", file=sys.stderr)
        runner.check_mock_flags()
        summary = runner.run(suites, models=args.model or None, limit=args.limit, force=args.force)
    except (RunError, ValueError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"run {run_id}: {summary.executed} executed, {summary.already_done} already done, "
        f"{summary.errors} with request errors, {summary.elapsed_s:.1f}s"
    )
    print(f"trajectories: {run_dir / 'trajectories.jsonl'}")
    if summary.aborted:
        print(f"error: {summary.aborted}", file=sys.stderr)
        return 3
    print(f"next: refusalscope analyze --run {run_dir}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Re-grade a run and compute the pre-registered paired statistics."""
    from refusalscope.analysis import analyze_run
    from refusalscope.runner import RunError

    try:
        results, path = analyze_run(args.run, args.out, n_boot=args.bootstrap)
    except (RunError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"analyzed {results['n_trajectories']} trajectories ({results['label']})")
    for pair_id, pair in results["pairs"].items():
        print(f"  pair {pair_id}: {pair['status']}  - {pair['manipulation_check']['reason']}")
    for entry in results["hypotheses"]["family"]:
        print(
            f"  {entry['id']} [{entry['pair']}]: {entry['label']} (Holm p = {entry['p_holm']:.4f})"
        )
    print(f"wrote {path}")
    if results["mock"]:
        print("MOCK run: these numbers say nothing about any real model")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Build Markdown tables and SVG figures from results/*.json."""
    from refusalscope.report import ReportError, make_report

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


def cmd_audit(args: argparse.Namespace) -> int:
    """Print a random sample of graded trajectories for manual checking."""
    from refusalscope.analysis import audit_samples
    from refusalscope.runner import RunError

    try:
        blocks = audit_samples(args.run, args.suite, args.n, args.seed)
    except (RunError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        ("\n\n" + "-" * 72 + "\n").join(blocks)
        if blocks
        else f"no {args.suite} trajectories in {args.run}"
    )
    return 0


def cmd_power(args: argparse.Namespace) -> int:
    """Print the exact McNemar power table used in the pre-registration."""
    from refusalscope.stats import mcnemar_power

    header = "n".rjust(6) + "  p_disc" + "".join(f"  rd={rd:.2f}".rjust(10) for rd in args.rd)
    print(header)
    for n in args.n:
        for p_d in args.p_discordant:
            cells = [
                f"{mcnemar_power(n, p_d, rd, args.alpha):.3f}".rjust(10)
                if rd <= p_d
                else "n/a".rjust(10)
                for rd in args.rd
            ]
            print(f"{n:>6}  {p_d:>6.2f}" + "".join(cells))
    return 0


def cmd_mock_serve(args: argparse.Namespace) -> int:  # pragma: no cover - blocks until interrupted
    """Run the MOCK server in the foreground."""
    from refusalscope.mock import PROFILES, serve_forever
    from refusalscope.suites import mock_provider
    from refusalscope.suites.refusal import load_behaviors

    harmful = None
    if args.harmful_file:
        items, _ = load_behaviors(args.harmful_file)
        harmful = [i["prompt"] for i in items]
    if args.profile not in PROFILES:
        print(
            f"error: unknown profile {args.profile!r}; choose from {', '.join(PROFILES)}",
            file=sys.stderr,
        )
        return 2
    serve_forever(args.profile, args.host, args.port, mock_provider(harmful))
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

    generate = sub.add_parser("generate", help="write the default suites as JSONL snapshots")
    generate.add_argument("--out", default="suites")
    generate.add_argument("--only", nargs="*", help="suite names to write")
    generate.set_defaults(func=cmd_generate)

    doctor = sub.add_parser("doctor", help="preflight checks for every endpoint")
    doctor.add_argument("--config", required=True)
    doctor.set_defaults(func=cmd_doctor)

    run = sub.add_parser("run", help="run or resume the suites")
    run.add_argument("--config", required=True)
    run.add_argument("--suite", nargs="+", default=["all"], help="all, ablation, or suite names")
    run.add_argument(
        "--out", help="parent directory for run directories (default: config output_dir)"
    )
    run.add_argument("--run-id", help="name of the run directory; reuse it to resume")
    run.add_argument("--workers", type=int, help="parallel requests (default: config workers)")
    run.add_argument("--limit", type=int, help="run an evenly spaced subset of N cases per suite")
    run.add_argument("--model", action="append", help="only this model (repeatable)")
    run.add_argument(
        "--force", action="store_true", help="accept a changed configuration on resume"
    )
    run.set_defaults(func=cmd_run)

    analyze = sub.add_parser("analyze", help="paired statistics for a run directory")
    analyze.add_argument("--run", required=True)
    analyze.add_argument("--out", default="results")
    analyze.add_argument("--bootstrap", type=int, default=10_000)
    analyze.set_defaults(func=cmd_analyze)

    report = sub.add_parser("report", help="tables and figures from results/*.json")
    report.add_argument("--results", default="results")
    report.add_argument("--out", default="report")
    report.add_argument(
        "--allow-mock", action="store_true", help="render MOCK results (stamped MOCK)"
    )
    report.add_argument("--update-readme", help="README.md whose results block should be filled")
    report.set_defaults(func=cmd_report)

    audit = sub.add_parser("audit", help="print graded samples for manual review")
    audit.add_argument("--run", required=True)
    audit.add_argument("--suite", required=True)
    audit.add_argument("--n", type=int, default=10)
    audit.add_argument("--seed", type=int, default=0)
    audit.set_defaults(func=cmd_audit)

    power = sub.add_parser("power", help="exact McNemar power table")
    power.add_argument("--n", type=int, nargs="+", default=[374, 378])
    power.add_argument("--p-discordant", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    power.add_argument("--rd", type=float, nargs="+", default=[0.03, 0.05, 0.08, 0.10])
    power.add_argument("--alpha", type=float, default=0.05)
    power.set_defaults(func=cmd_power)

    serve = sub.add_parser("mock-serve", help="run the MOCK server (tests the harness only)")
    serve.add_argument("--profile", default="reference")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8101)
    serve.add_argument("--harmful-file", help="placeholder behaviours file the mock should refuse")
    serve.set_defaults(func=cmd_mock_serve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help(sys.stderr)
        return 2
    return func(args)
