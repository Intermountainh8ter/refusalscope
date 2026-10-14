"""Evaluation suites. Each suite module exposes a deterministic generator and a grader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from refusalscope.util import write_jsonl

SUITE_FILES = {
    "toolcall": "toolcall.jsonl",
    "injection": "injection.jsonl",
    "borderline": "borderline.jsonl",
    "kl": "kl_prompts.jsonl",
    "capability": "capability.jsonl",
}


def snapshot_builders() -> dict[str, Callable[[], list[dict[str, Any]]]]:
    """Default-parameter builders for every suite that has a committed snapshot."""
    from refusalscope.suites import capability, injection, kl, refusal, toolcall

    return {
        "toolcall": toolcall.build_cases,
        "injection": injection.build_cases,
        "borderline": refusal.borderline_items,
        "kl": kl.kl_items,
        "capability": capability.build_items,
    }


def write_snapshots(out_dir: str | Path, only: list[str] | None = None) -> dict[str, int]:
    """Write the default suites as JSON Lines into ``out_dir``; return row counts."""
    counts: dict[str, int] = {}
    for suite, build in snapshot_builders().items():
        if only and suite not in only:
            continue
        counts[suite] = write_jsonl(Path(out_dir) / SUITE_FILES[suite], build())
    return counts


def rows_digest(rows: list[dict[str, Any]]) -> str:
    """Stable SHA-256 of a list of cases, independent of dict ordering."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def suite_hashes() -> dict[str, str]:
    """SHA-256 of every default suite, recorded in result files for reproducibility."""
    return {
        name: rows_digest(json.loads(json.dumps(build())))
        for name, build in snapshot_builders().items()
    }


def mock_provider(
    harmful_prompts: list[str] | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """A scenario provider for the mock server covering every suite.

    ``harmful_prompts`` stands in for the user-supplied behaviours file; the mock only needs the
    prompt text to decide between a refusal and a compliant reply.
    """
    from refusalscope.suites import capability, injection, kl, refusal, toolcall

    oracles = [
        toolcall.ToolcallOracle(toolcall.build_cases()),
        injection.InjectionOracle(injection.build_cases()),
        refusal.RefusalOracle(harmful_prompts),
        kl.KLOracle(),
        capability.CapabilityOracle(capability.build_items()),
    ]

    def provider(body: dict[str, Any]) -> Any:
        for oracle in oracles:
            scenario = oracle.lookup(body)
            if scenario is not None:
                return scenario
        return None

    return provider
