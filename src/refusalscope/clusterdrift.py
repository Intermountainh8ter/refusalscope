"""MOCK batch-clustered refusal-threshold shocks: exact-McNemar false positives by cluster size.

Simulation only: no model, no server. Identical-stack stubs get a zero-mean normal shift of the
refusal threshold per cluster of consecutive items on run B (degenerate stubs are unshifted); the estimator is the mdd exact McNemar. Not evidence about any model.
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

SCHEMA = "refusalscope.clusterdrift/1"
GRID = [1000]
SHIFT_SD = 0.10
CLUSTERS = (1, 20)
STUBS = {
    "identical_stack": 0.30,
    "always_refuse": 1.0,
    "always_comply": 0.0,
}


def _labels(latent: np.ndarray, p, rng: np.random.Generator, degenerate: bool) -> np.ndarray:
    lab = latent < p
    if not degenerate:
        lab = lab ^ (rng.random(latent.size) < JITTER)
    return lab


def fp_at(seed: int, stub: str, cluster: int, n: int, draws: int) -> float:
    p = STUBS[stub]
    degenerate = p in (0.0, 1.0)
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub), cluster, n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        first = _labels(latent, p, rng, degenerate)
        if degenerate:
            thr = np.full(n, p)
        else:
            thr = p + np.repeat(rng.normal(0.0, SHIFT_SD, -(-n // cluster)), cluster)[:n]
        second = _labels(latent, thr, rng, degenerate)
        a, b = first, second
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    fp = {
        c: {s: {str(n): fp_at(seed, s, c, n, draws) for n in grid} for s in STUBS} for c in CLUSTERS
    }
    return {
        "cluster_drift": {
            "false_positive_rate": {
                f"cluster_{c}": dict(fp[c]["identical_stack"]) for c in CLUSTERS
            },
            "degenerate_false_positive_rate": {
                f"cluster_{c}": {s: fp[c][s] for s in ("always_refuse", "always_comply")} for c in CLUSTERS
            },
        },
        "injected": {
            "stubs": dict(STUBS),
            "shift_sd": SHIFT_SD, "clusters": list(CLUSTERS),
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
            "label": "MOCK clustered-shock false-positive check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope clusterdrift --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
