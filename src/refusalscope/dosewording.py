"""MOCK dose-response with a prompt-wording perturbation floor on the unablated arm.

Simulation only: no model, no server. Extends dosetool: the unablated arm's tool success varies
across paraphrases by up to WORDING; a strength only counts as detected when the true loss exceeds
the SESOI plus that measured floor. Not evidence about any model.
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

SCHEMA = "refusalscope.dosewording/1"
WORDING = 0.04
PARAPHRASES = 5
MIN_GAP = 0.05


def wording_floor(paraphrases: int, wording: float) -> float:
    """Largest drop from baseline across paraphrases of the unablated arm."""
    rates = [BASE_TOOL - wording * k / max(paraphrases - 1, 1) for k in range(paraphrases)]
    return round(BASE_TOOL - min(rates), 10)


def power_at(seed: int, idx: int, s: float, n: int, draws: int, stub: str | None = None) -> float:
    pa = {"always_refuse": 0.0, "always_comply": 1.0}.get(stub, BASE_TOOL)
    pb = pa if stub else tool_success(s)
    rng = np.random.default_rng([seed, idx, n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        a = (latent < pa) ^ (rng.random(n) < FLIP)
        b = (latent < pb) ^ (rng.random(n) < FLIP)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def run(seed: int, draws: int, grid: list[float] | None = None, n: int = N_PROMPTS, paraphrases: int = PARAPHRASES) -> dict:
    grid = grid or GRID
    floor = wording_floor(paraphrases, WORDING)
    power = {str(s): power_at(seed, i, s, n, draws) for i, s in enumerate(grid)}
    free = first(grid, lambda s: BASE_TOOL - tool_success(s) > SESOI and power[str(s)] >= TARGET_POWER)
    floored = first(grid, lambda s: BASE_TOOL - tool_success(s) > SESOI + floor and power[str(s)] >= TARGET_POWER)
    stubs = {
        name: first(grid, lambda s, name=name: power_at(seed, 1000 + grid.index(s), s, n, draws, name) >= TARGET_POWER)
        for name in ("always_refuse", "always_comply")
    }
    gap = None if free is None or floored is None else round(floored - free, 2)
    return {
        "dose_response": {
            "min_strength_wordingfree": free,
            "min_strength_wording_floor": floored,
            "gap_floor_minus_free": gap,
            "measured_wording_floor": floor,
            "power": power,
            "degenerate_stub_detected_strength": stubs,
        },
        "setup": {
            "base_tool_success": BASE_TOOL,
            "wording_floor_param": WORDING,
            "paraphrases": paraphrases,
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
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--studies", type=int, default=400)
    ap.add_argument("--paraphrases", type=int, default=PARAPHRASES)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.studies, None, args.prompts, args.paraphrases)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK dose-response with wording floor (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope dose-wording --seed {args.seed} --prompts {args.prompts} --studies {args.studies} --paraphrases {args.paraphrases} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
