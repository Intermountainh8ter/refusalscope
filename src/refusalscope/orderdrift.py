"""MOCK run-order drift: fixed A-then-B versus AB/BA counterbalanced exact-McNemar false positives.

Simulation only: no model, no server. Identical-stack stubs get a refusal-rate drift on whichever
run comes second; the estimator is the mdd exact McNemar. Not evidence about any model.
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

SCHEMA = "refusalscope.orderdrift/1"
GRID = [300, 1000]
DRIFT = 0.02
STUBS = {
    "identical_stack": 0.30,
    "always_refuse": 1.0,
    "always_comply": 0.0,
}
DESIGNS = ("fixed_order", "counterbalanced")


def _labels(latent: np.ndarray, p: float, rng: np.random.Generator, degenerate: bool) -> np.ndarray:
    lab = latent < p
    if not degenerate:
        lab = lab ^ (rng.random(latent.size) < JITTER)
    return lab


def fp_at(seed: int, stub: str, design: str, n: int, draws: int) -> float:
    p = STUBS[stub]
    degenerate = p in (0.0, 1.0)
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub), DESIGNS.index(design), n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        first = _labels(latent, p, rng, degenerate)
        second = _labels(latent, p if degenerate else p + DRIFT, rng, degenerate)
        if design == "fixed_order":
            a, b = first, second
        else:
            a_first = np.arange(n) % 2 == 0
            a = np.where(a_first, first, second)
            b = np.where(a_first, second, first)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    fp = {
        d: {s: {str(n): fp_at(seed, s, d, n, draws) for n in grid} for s in STUBS} for d in DESIGNS
    }
    return {
        "order_drift": {
            "false_positive_rate": {
                d: dict(fp[d]["identical_stack"]) for d in DESIGNS
            },
            "degenerate_false_positive_rate": {
                d: {s: fp[d][s] for s in ("always_refuse", "always_comply")} for d in DESIGNS
            },
        },
        "injected": {
            "stubs": dict(STUBS),
            "drift": DRIFT,
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
            "label": "MOCK run-order drift false-positive check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope orderdrift --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
