"""A self-contained end-to-end run against the MOCK server (``refusalscope mock-demo``).

It starts two mock servers in-process, runs the real runner, analysis and report generator, and
writes everything into one directory. Every file it creates is named ``MOCK-...``. The numbers
come from scripted answers with arbitrary error rates and say nothing about any real model; the
point is to prove that the pipeline works on a machine with no GPU and no weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from refusalscope.analysis import analyze_run
from refusalscope.config import parse_config
from refusalscope.mock import PROFILES, MockServer
from refusalscope.report import make_report
from refusalscope.runner import Runner, resolve_suites
from refusalscope.suites import mock_provider

PLACEHOLDER_BEHAVIOURS = [
    f"Placeholder request {i:03d} for harness testing only. It is not a real behaviour."
    for i in range(1, 61)
]
RUN_ID = "MOCK-demo"


def run_demo(
    out_dir: str | Path,
    limit: int | None = 40,
    workers: int = 4,
    n_boot: int = 2000,
    suites: str = "all",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full pipeline on mock servers; return the paths and the results document."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda message: None)
    behaviours = root / "MOCK-placeholder-behaviours.jsonl"
    behaviours.write_text(
        "\n".join(json.dumps({"prompt": text}) for text in PLACEHOLDER_BEHAVIOURS) + "\n",
        encoding="utf-8",
    )
    provider = mock_provider(PLACEHOLDER_BEHAVIOURS)
    with (
        MockServer(profile=PROFILES["reference"], provider=provider) as stock,
        MockServer(profile=PROFILES["degraded"], provider=provider) as ablated,
    ):
        raw = {
            "study": "mock-demo",
            "output_dir": str(root / "runs"),
            "workers": workers,
            "models": [
                {"name": "mock-reference", "base_url": stock.base_url, "role": "stock", "pair_id": "mock-pair",
                 "params_b": 4.0, "quant": "mock", "mock": True},
                {"name": "mock-degraded", "base_url": ablated.base_url, "role": "ablated", "pair_id": "mock-pair",
                 "params_b": 4.0, "quant": "mock", "mock": True},
            ],
            "sampling": {"temperatures": [0.0], "seeds": [1234], "max_tokens": 256, "timeout_s": 30, "retries": 1},
            "suites": {"refusal": {"harmful_path": str(behaviours)}},
        }  # fmt: skip
        (root / "MOCK-config.json").write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        config = parse_config(raw)
        runner = Runner(
            config, root / "runs" / RUN_ID, workers=workers, progress=say, run_id=RUN_ID
        )
        runner.check_mock_flags()
        summary = runner.run(resolve_suites(suites), limit=limit)
    results, results_path = analyze_run(root / "runs" / RUN_ID, root / "results", n_boot=n_boot)
    reports = make_report(root / "results", root / "report", allow_mock=True)
    return {
        "run_dir": root / "runs" / RUN_ID,
        "results": results,
        "results_path": results_path,
        "reports": reports,
        "summary": summary,
    }
