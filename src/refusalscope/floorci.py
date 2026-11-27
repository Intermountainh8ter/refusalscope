"""MOCK pilot-size check for the identical-stack repeat-run flip floor (Wilson interval).

Simulation only: no model, no server. Two repeat runs of an identical-stack stub are compared on
a pilot of n prompts; the floor is the disagreement fraction with a Wilson 95% interval. Not
evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mdd import JITTER

SCHEMA = "refusalscope.floorci/1"
GRID = [30, 100, 300]
Z = 1.959964
STUBS = {"identical_stack": 0.30, "always_refuse": 1.0, "always_comply": 0.0}
TRUE_FLOOR = 2 * JITTER * (1 - JITTER)


def wilson(k: int, n: int) -> tuple[float, float]:
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def at(seed: int, stub: str, n: int, draws: int) -> dict:
    p = STUBS[stub]
    degenerate = p in (0.0, 1.0)
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub), n])
    cover = 0
    widths = []
    ests = []
    for _ in range(draws):
        latent = rng.random(n)
        runs = []
        for _r in range(2):
            lab = latent < p
            if not degenerate:
                lab = lab ^ (rng.random(n) < JITTER)
            runs.append(lab)
        k = int((runs[0] != runs[1]).sum())
        lo, hi = wilson(k, n)
        cover += lo <= TRUE_FLOOR <= hi
        widths.append(hi - lo)
        ests.append(k / n)
    return {
        "coverage": cover / draws,
        "mean_width": float(np.mean(widths)),
        "mean_floor": float(np.mean(ests)),
        "max_floor": float(np.max(ests)),
    }


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    res = {s: {str(n): at(seed, s, n, draws) for n in grid} for s in STUBS}
    ident = res["identical_stack"]
    return {
        "floor_ci": {
            "coverage": {"identical_stack": {k: v["coverage"] for k, v in ident.items()}},
            "mean_width": {"identical_stack": {k: v["mean_width"] for k, v in ident.items()}},
            "mean_floor": {"identical_stack": {k: v["mean_floor"] for k, v in ident.items()}},
            "degenerate_max_floor": {
                s: {k: v["max_floor"] for k, v in res[s].items()}
                for s in ("always_refuse", "always_comply")
            },
        },
        "injected": {
            "stubs": dict(STUBS),
            "jitter": JITTER,
            "true_floor": TRUE_FLOOR,
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
            "label": "MOCK flip-floor Wilson pilot-size check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope floorci --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [
                    {"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}
                ],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
