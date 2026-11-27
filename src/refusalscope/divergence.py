"""MOCK text-vs-tool refusal divergence: power of a paired difference-in-drops z-test.

Simulation only: no model, no server. Each prompt has a text latent and a tool latent (Gaussian
copula, correlation rho) shared by the stock and ablated stub; labels flip with repeat-run jitter.
The statistic is the per-prompt difference between the text drop and the tool drop. Not evidence
about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from math import erf, sqrt
from pathlib import Path

import numpy as np

from .mdd import ALPHA, JITTER

SCHEMA = "refusalscope.divergence/1"
GRID = [100, 300]
RHOS = (0.3, 0.8)
DIVERGENCES = (0.0, 0.15)
STOCK = 0.90
TEXT_ABLATED = 0.20
TOOL_ABLATED_BASE = 0.20
STUBS = ("graded", "always_refuse", "always_comply")
Z_CRIT = 1.959963984540054


def _phi(x):
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _labels(u, p, rng, degenerate):
    lab = u < p
    if not degenerate:
        lab = lab ^ (rng.random(u.size) < JITTER)
    return lab


def rejects(d: np.ndarray) -> bool:
    sd = d.std(ddof=1)
    if sd == 0:
        return False
    return abs(d.mean() / (sd / sqrt(d.size))) > Z_CRIT


def rate_at(seed: int, stub: str, rho: float, div: float, n: int, draws: int) -> float:
    degenerate = stub != "graded"
    if stub == "always_refuse":
        ps = pt_a = pc_a = 1.0
    elif stub == "always_comply":
        ps = pt_a = pc_a = 0.0
    else:
        ps, pt_a, pc_a = STOCK, TEXT_ABLATED, TOOL_ABLATED_BASE + div
    rng = np.random.default_rng([seed, STUBS.index(stub), RHOS.index(rho), DIVERGENCES.index(div), n])
    hits = 0
    for _ in range(draws):
        zt = rng.standard_normal(n)
        zc = rho * zt + sqrt(1 - rho * rho) * rng.standard_normal(n)
        ut, uc = _phi(zt), _phi(zc)
        drop_t = _labels(ut, ps, rng, degenerate).astype(int) - _labels(ut, pt_a, rng, degenerate).astype(int)
        drop_c = _labels(uc, ps, rng, degenerate).astype(int) - _labels(uc, pc_a, rng, degenerate).astype(int)
        if rejects(drop_t - drop_c):
            hits += 1
    return hits / draws


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    def cell(stub, rho, div):
        return {str(n): rate_at(seed, stub, rho, div, n, draws) for n in grid}
    power, fpr = {}, {}
    for rho in RHOS:
        power[f"rho_{rho}"] = {"divergence_0.15": cell("graded", rho, 0.15)}
        fpr[f"rho_{rho}"] = cell("graded", rho, 0.0)
    degenerate = {
        s: {f"rho_{rho}": {f"divergence_{d}": cell(s, rho, d) for d in DIVERGENCES} for rho in RHOS}
        for s in ("always_refuse", "always_comply")
    }
    return {
        "divergence": {"power": power, "false_positive_rate": fpr, "degenerate_rate": degenerate},
        "injected": {
            "stock_refusal": STOCK,
            "text_ablated_refusal": TEXT_ABLATED,
            "tool_ablated_refusal_zero_divergence": TOOL_ABLATED_BASE,
            "divergences": list(DIVERGENCES),
            "rhos": list(RHOS),
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
            "label": "MOCK text-vs-tool divergence power check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope divergence --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
