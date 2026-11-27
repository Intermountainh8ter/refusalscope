"""MOCK stacked serving-noise sources: exact-McNemar power with jitter alone vs two cluster shocks.

Simulation only: no model, no server. Run B's refusal threshold gets two independent zero-mean
normal shocks (one per consecutive cluster of each size); degenerate stubs are unshifted; the
estimator is the mdd exact McNemar. Not evidence about any model.
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

SCHEMA = "refusalscope.stacksources/1"
GRID = [1000]
SHOCK_SD = 0.05
CLUSTERS = (20, 100)
BASE = 0.30
SHIFT = 0.04
STUBS = ("shifted", "identical_stack", "always_refuse", "always_comply")
MODES = ("jitter_only", "stacked")


def _labels(latent, p, rng, degenerate):
    lab = latent < p
    if not degenerate:
        lab = lab ^ (rng.random(latent.size) < JITTER)
    return lab


def rate_at(seed: int, stub: str, mode: str, n: int, draws: int) -> float:
    degenerate = stub.startswith("always")
    pa = {"always_refuse": 1.0, "always_comply": 0.0}.get(stub, BASE)
    pb = pa + (SHIFT if stub == "shifted" else 0.0)
    rng = np.random.default_rng([seed, STUBS.index(stub), MODES.index(mode), n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        first = _labels(latent, pa, rng, degenerate)
        thr = np.full(n, pb)
        if mode == "stacked" and not degenerate:
            for c in CLUSTERS:
                thr = thr + np.repeat(rng.normal(0.0, SHOCK_SD, -(-n // c)), c)[:n]
        second = _labels(latent, thr, rng, degenerate)
        if mcnemar_exact_p(int((first & ~second).sum()), int((~first & second).sum())) < ALPHA:
            hits += 1
    return hits / draws


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    r = {m: {s: {str(n): rate_at(seed, s, m, n, draws) for n in grid} for s in STUBS} for m in MODES}
    return {
        "stacked_sources": {
            "power": {m: {"shift_0.04": dict(r[m]["shifted"])} for m in MODES},
            "false_positive_rate": {m: dict(r[m]["identical_stack"]) for m in MODES},
            "degenerate_rate": {
                m: {s: r[m][s] for s in ("always_refuse", "always_comply")} for m in MODES
            },
        },
        "injected": {
            "base_refusal_probability": BASE,
            "shift": SHIFT,
            "shock_sd": SHOCK_SD,
            "clusters": list(CLUSTERS),
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
            "label": "MOCK stacked-shock power check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope stacksources --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
