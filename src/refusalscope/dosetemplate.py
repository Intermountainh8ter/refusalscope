"""MOCK dose-response with a chat-template mismatch between the stock and ablated arms.

Simulation only: no model, no server. Extends dosetool: the ablated arm's benign tool success is
lowered by a constant MISMATCH at every strength, as if the two arms were served under different
chat templates. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dosetool import (
    ALPHA,
    BASE_TOOL,
    FLIP,
    GRID,
    N_PROMPTS,
    SESOI,
    TARGET_POWER,
    first,
    tool_success,
)
from .mdd import mcnemar_exact_p

SCHEMA = "refusalscope.dosetemplate/1"
MISMATCH = 0.04
MIN_GAP = 0.10
FP_FLOOR = 0.50


def power_at(seed: int, idx: int, s: float, n: int, draws: int, mismatch: float, stub: str | None = None) -> float:
    pa = {"always_refuse": 0.0, "always_comply": 1.0}.get(stub, BASE_TOOL)
    pb = pa if stub else tool_success(s) - mismatch
    rng = np.random.default_rng([seed, idx, n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        a = (latent < pa) ^ (rng.random(n) < FLIP)
        b = (latent < pb) ^ (rng.random(n) < FLIP)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def detect(seed: int, grid: list[float], n: int, draws: int, mismatch: float, offset: int) -> tuple[dict, float | None]:
    power = {str(s): power_at(seed, offset + i, s, n, draws, mismatch) for i, s in enumerate(grid)}
    found = first(grid, lambda s: BASE_TOOL - tool_success(s) + mismatch > SESOI and power[str(s)] >= TARGET_POWER)
    return power, found


def run(seed: int, draws: int, grid: list[float] | None = None, n: int = N_PROMPTS) -> dict:
    grid = grid or GRID
    p_match, m_match = detect(seed, grid, n, draws, 0.0, 0)
    p_mis, m_mis = detect(seed, grid, n, draws, MISMATCH, 0)
    stubs = {
        name: first(
            grid,
            lambda s, name=name: power_at(seed, 1000 + grid.index(s), s, n, draws, MISMATCH, name) >= TARGET_POWER,
        )
        for name in ("always_refuse", "always_comply")
    }
    fp0 = power_at(seed, 2000, 0.0, n, draws, MISMATCH)
    gap = None if m_match is None or m_mis is None else round(m_match - m_mis, 2)
    return {
        "template": {
            "min_strength_matched": m_match,
            "min_strength_mismatch": m_mis,
            "gap_matched_minus_mismatch": gap,
            "significant_fraction_at_strength_0_mismatch": fp0,
            "power_matched": p_match,
            "power_mismatch": p_mis,
            "degenerate_stub_detected_strength": stubs,
        },
        "setup": {
            "base_tool_success": BASE_TOOL,
            "template_mismatch_tool_success_drop": MISMATCH,
            "batch_flip_floor": FLIP,
            "smallest_effect_of_interest": SESOI,
            "n_prompts": n,
            "target_power": TARGET_POWER,
            "alpha": ALPHA,
            "grid": grid,
            "n_draws": draws,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.draws)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK dose-response with chat-template mismatch (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope dosetemplate --seed {args.seed} --draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
