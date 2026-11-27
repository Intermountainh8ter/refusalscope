"""MOCK exact-McNemar power versus paired-prompt count for a between-stack refusal shift.

Simulation only: no model, no server. Per simulated study, n synthetic items get a latent draw;
each stack labels an item refused when the latent is below its refusal probability, then every
label flips independently with the repeat-run jitter probability. Power is the fraction of studies
with a two-sided exact McNemar p-value below alpha. Required prompts is the smallest grid point
with power at least the target, or null. Output is an estimator check, not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.mdd/1"
JITTER = 0.06
ALPHA = 0.05
TARGET_POWER = 0.80
GRID = [50, 100, 150, 200, 300, 400, 600]
STUBS = {
    "template_shift": (0.30, 0.40),
    "identical_stack": (0.30, 0.30),
    "always_refuse": (1.0, 1.0),
    "always_comply": (0.0, 0.0),
}


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _labels(latent: np.ndarray, p: float, rng: np.random.Generator, degenerate: bool) -> np.ndarray:
    lab = latent < p
    if not degenerate:
        lab = lab ^ (rng.random(latent.size) < JITTER)
    return lab


def power_at(seed: int, stub: str, n: int, draws: int) -> float:
    pa, pb = STUBS[stub]
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub), n])
    degenerate = pa in (0.0, 1.0)
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        a = _labels(latent, pa, rng, degenerate)
        b = _labels(latent, pb, rng, degenerate)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def run(seed: int, draws: int) -> dict:
    power = {s: {str(n): power_at(seed, s, n, draws) for n in GRID} for s in STUBS}
    required = {
        s: next((n for n in GRID if power[s][str(n)] >= TARGET_POWER), None) for s in STUBS
    }
    return {
        "mdd": {
            "power": power,
            "required_prompts": required,
            "false_positive_rate": {"identical_stack": power["identical_stack"]},
        },
        "injected": {
            "stubs": {k: list(v) for k, v in STUBS.items()},
            "jitter": JITTER,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "grid": GRID,
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
            "label": "MOCK exact-McNemar minimum-prompts estimator check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope mdd --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
