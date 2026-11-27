"""MOCK injection-ASR power: cases per arm for an exact McNemar test over the batch-flip floor.

Simulation only: no model, no server. Paired stock/ablated attack-success outcomes share a latent
per case; each run flips independently with the batch-flip floor probability. A no-injection
control arm (same cases, control success rate) gives the false-positive rate. Not evidence about
any model.
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

SCHEMA = "refusalscope.asrpower/1"
GRID = list(range(50, 1001, 50))
BASE_ASR = 0.20
EFFECT = 0.10
FLIP = 0.03
CONTROL = 0.02
TARGET_POWER = 0.80


def rate_at(seed: int, arm: str, n: int, draws: int) -> float:
    pa, pb = {
        "injected": (BASE_ASR, BASE_ASR + EFFECT),
        "null": (BASE_ASR, BASE_ASR),
        "control": (CONTROL, CONTROL),
    }[arm]
    rng = np.random.default_rng([seed, ("injected", "null", "control").index(arm), n])
    hits = 0
    for _ in range(draws):
        latent = rng.random(n)
        a = (latent < pa) ^ (rng.random(n) < FLIP)
        b = (latent < pb) ^ (rng.random(n) < FLIP)
        if mcnemar_exact_p(int((a & ~b).sum()), int((~a & b).sum())) < ALPHA:
            hits += 1
    return hits / draws


def run(seed: int, draws: int, grid: list[int] | None = None) -> dict:
    grid = grid or GRID
    power = {str(n): rate_at(seed, "injected", n, draws) for n in grid}
    fpr = {
        "null_effect": {str(n): rate_at(seed, "null", n, draws) for n in grid},
        "no_injection_control": {str(n): rate_at(seed, "control", n, draws) for n in grid},
    }
    ok = [n for n in grid if power[str(n)] >= TARGET_POWER]
    return {
        "asr_power": {
            "min_cases_per_arm": ok[0] if ok else None,
            "power": power,
            "false_positive_rate": fpr,
        },
        "injected": {
            "baseline_asr": BASE_ASR,
            "effect": EFFECT,
            "batch_flip_floor": FLIP,
            "control_rate": CONTROL,
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
            "label": "MOCK injection-ASR power check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope asrpower --seed {args.seed} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
