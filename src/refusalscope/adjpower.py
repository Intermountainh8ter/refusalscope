"""MOCK adjudication power: prompts needed for exact-McNemar power 0.80, raw vs majority-adjudicated labels.

Simulation only: no model, no server. Two stacks share a latent per prompt; each run's label flips
independently with the jitter probability. Adjudicated labels are the majority of several reruns.
Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mdd import ALPHA, mcnemar_exact_p

SCHEMA = "refusalscope.adjpower/1"
GRID = list(range(100, 2001, 50))
P_A, P_B = 0.30, 0.33
JITTER = 0.06
RERUNS = 5
TARGET_POWER = 0.80


def labels(rng, latent: np.ndarray, runs: int) -> np.ndarray:
    flips = rng.random((runs, latent.size)) < JITTER
    votes = (latent[None, :] ^ flips).sum(axis=0)
    return votes * 2 > runs


def power_at(seed: int, n: int, draws: int, runs: int, stub: str | None = None) -> float:
    rng = np.random.default_rng([seed, n, runs, 0 if stub is None else 1 + len(stub)])
    pa, pb = {"always_refuse": (1.0, 1.0), "always_comply": (0.0, 0.0)}.get(stub, (P_A, P_B))
    hits = 0
    for _ in range(draws):
        u = rng.random(n)
        a = labels(rng, u < pa, runs)
        b = labels(rng, u < pb, runs)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def first(power: dict[str, float]) -> int | None:
    return next((n for n in GRID if power[str(n)] >= TARGET_POWER), None)


def run(seed: int, draws: int) -> dict:
    raw = {str(n): power_at(seed, n, draws, 1) for n in GRID}
    adj = {str(n): power_at(seed, n, draws, RERUNS) for n in GRID}
    stubs = {
        name: {str(n): power_at(seed, n, draws, RERUNS, name) for n in GRID}
        for name in ("always_refuse", "always_comply")
    }
    mr, ma = first(raw), first(adj)
    return {
        "power": {
            "min_prompts_raw": mr,
            "min_prompts_adjudicated": ma,
            "raw_over_adjudicated": None if mr is None or ma is None else round(mr / ma, 3),
            "raw": raw,
            "adjudicated": adj,
            "degenerate_stub_min_prompts": {k: first(v) for k, v in stubs.items()},
            "degenerate_stub_max_power": {k: max(v.values()) for k, v in stubs.items()},
        },
        "setup": {
            "refusal_prob_stack_a": P_A,
            "refusal_prob_stack_b": P_B,
            "repeat_run_jitter": JITTER,
            "adjudication_reruns": RERUNS,
            "target_power": TARGET_POWER,
            "alpha": ALPHA,
            "grid": [GRID[0], GRID[-1], 50],
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
            "label": "MOCK adjudication power check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope adjpower --seed {args.seed} --draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
