"""MOCK exact-McNemar power for a rare (0.01) batch-induced refusal shift over a large prompt grid.

Simulation only: no model, no server. Reuses the mdd estimator (latent draw per item, label flips
with the repeat-run jitter, two-sided exact McNemar). Output is an estimator check, not evidence
about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mdd import ALPHA, JITTER, mcnemar_exact_p

SCHEMA = "refusalscope.rareflip/1"
GRID = [100, 300, 1000, 3000, 10000]
STUBS = {
    "rare_batch": (0.30, 0.31),
    "identical_stack": (0.30, 0.30),
    "always_refuse": (1.0, 1.0),
    "always_comply": (0.0, 0.0),
}


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


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    power = {s: {str(n): power_at(seed, s, n, draws) for n in grid} for s in STUBS}
    return {
        "rare_flip": {
            "power": power,
            "false_positive_rate": {"identical_stack": power["identical_stack"]},
        },
        "injected": {
            "stubs": {k: list(v) for k, v in STUBS.items()},
            "jitter": JITTER,
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
            "label": "MOCK rare-batch-flip exact-McNemar power check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope rareflip --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
