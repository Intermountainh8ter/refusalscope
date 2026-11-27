"""MOCK batch-composition noise floor on a refusal rate and the smallest detectable dose step.

Simulation only: no model, no server. Each prompt has a latent refusal margin; batch composition
adds independent Gaussian noise to it, and a dose step is a uniform margin shift. Reports the
refusal-rate change from batching alone and the smallest step exact McNemar detects against it.
Degenerate stubs never flip. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np

from .mdd import mcnemar_exact_p

SCHEMA = "refusalscope.batchnoise/1"
BASE_RATE = 0.30
MU = NormalDist().inv_cdf(BASE_RATE)  # margin mean so that P(margin > 0) = BASE_RATE
GRID = [round(0.01 * k, 2) for k in range(1, 16)]
ALPHA = 0.05
TARGET_POWER = 0.80
STUBS = ("noisy_stub", "always_refuse", "always_comply")


def _labels(margin: np.ndarray, noise: np.ndarray, stub: str) -> np.ndarray:
    if stub == "always_refuse":
        return np.ones(margin.shape, dtype=bool)
    if stub == "always_comply":
        return np.zeros(margin.shape, dtype=bool)
    return margin + noise > 0


def _shift(step: float) -> float:
    return NormalDist().inv_cdf(min(BASE_RATE + step, 0.999)) - MU


def simulate(seed: int, stub: str, prompts: int, studies: int, noise_sd: float) -> dict:
    rng = np.random.default_rng([seed, STUBS.index(stub)])
    pcache: dict[tuple[int, int], float] = {}

    def sig(b: int, c: int) -> bool:
        key = (b, c)
        if key not in pcache:
            pcache[key] = mcnemar_exact_p(b, c)
        return pcache[key] < ALPHA

    diffs, flips = [], 0
    for _ in range(studies):
        m = rng.normal(MU, 1.0, prompts)
        a = _labels(m, rng.normal(0, noise_sd, prompts), stub)
        b = _labels(m, rng.normal(0, noise_sd, prompts), stub)
        diffs.append(float(a.mean() - b.mean()))
        flips += int((a != b).sum())
    power = {}
    for step in GRID:
        hits = 0
        delta = _shift(step)
        for _ in range(studies):
            m = rng.normal(MU, 1.0, prompts)
            a = _labels(m, rng.normal(0, noise_sd, prompts), stub)
            b = _labels(m + delta, rng.normal(0, noise_sd, prompts), stub)
            up = int((~a & b).sum())
            down = int((a & ~b).sum())
            hits += sig(up, down)
        power[f"{step:.2f}"] = hits / studies
    detected = [s for s in GRID if power[f"{s:.2f}"] >= TARGET_POWER]
    return {
        "mean_batch_only_diff": float(np.mean(diffs)),
        "mean_batch_only_flips_per_study": flips / studies,
        "power_by_step": power,
        "min_detectable_step": detected[0] if detected else None,
    }


def run(seed: int, prompts: int, studies: int, noise_sd: float) -> dict:
    sims = {s: simulate(seed, s, prompts, studies, noise_sd) for s in STUBS}
    primary = sims["noisy_stub"]
    return {
        "batch_noise": {
            "min_detectable_step": primary["min_detectable_step"],
            "mean_batch_only_diff": primary["mean_batch_only_diff"],
            "mean_batch_only_flips_per_study": primary["mean_batch_only_flips_per_study"],
            "power_by_step": primary["power_by_step"],
            "stubs": {s: sims[s] for s in STUBS if s != "noisy_stub"},
        },
        "injected": {
            "base_rate": BASE_RATE,
            "noise_sd": noise_sd,
            "prompts": prompts,
            "studies": studies,
            "grid": GRID,
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", type=int, default=300)
    ap.add_argument("--studies", type=int, default=400)
    ap.add_argument("--noise-sd", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.prompts, args.studies, args.noise_sd)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK batch-noise floor and detectable dose step (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope batch-noise --seed {args.seed} "
                f"--prompts {args.prompts} --studies {args.studies} "
                f"--noise-sd {args.noise_sd} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
