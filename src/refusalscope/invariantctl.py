"""MOCK refusal-label flip rate between two batch compositions, with a batch-invariant control.

Simulation only: no model, no server. Each prompt has a refusal margin; a batch composition adds
independent Gaussian numeric noise to it and the label is refused when margin plus noise exceeds
zero. The control arm uses noise sd 0 (batch-invariant kernels). The sampling-alone arm flips
labels with two independent draws of a per-prompt Bernoulli whose probability is the logistic of
the margin. Degenerate stubs never flip. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.invariantctl/1"
MARGIN_MEAN = -0.5244
STUBS = ("margin_stub", "always_refuse", "always_comply")


def _margins(rng: np.random.Generator, stub: str, prompts: int) -> np.ndarray:
    if stub == "always_refuse":
        return np.full(prompts, 1e9)
    if stub == "always_comply":
        return np.full(prompts, -1e9)
    return rng.normal(MARGIN_MEAN, 1.0, prompts)


def simulate(seed: int, stub: str, prompts: int, studies: int, noise_sd: float) -> dict:
    rng = np.random.default_rng([seed, STUBS.index(stub)])
    batch, control, sampling = [], [], []
    for _ in range(studies):
        m = _margins(rng, stub, prompts)
        a = m + rng.normal(0, noise_sd, prompts) > 0
        b = m + rng.normal(0, noise_sd, prompts) > 0
        batch.append(float((a != b).mean()))
        # control: zero noise, so both compositions see identical margins
        control.append(float(((m > 0) != (m > 0)).mean()))
        p = 1.0 / (1.0 + np.exp(-np.clip(m, -50, 50)))
        s1, s2 = rng.random(prompts) < p, rng.random(prompts) < p
        sampling.append(float((s1 != s2).mean()))
    return {
        "flip_rate_batch": float(np.mean(batch)),
        "flip_rate_control": float(np.mean(control)),
        "flip_rate_sampling": float(np.mean(sampling)),
        "flip_rate_batch_sd_across_studies": float(np.std(batch)),
    }


def run(seed: int, prompts: int, studies: int, noise_sd: float) -> dict:
    sims = {s: simulate(seed, s, prompts, studies, noise_sd) for s in STUBS}
    return {
        "batch_invariant": {
            **sims["margin_stub"],
            "stubs": {s: sims[s] for s in STUBS if s != "margin_stub"},
        },
        "injected": {
            "margin_mean": MARGIN_MEAN,
            "margin_sd": 1.0,
            "prompts": prompts,
            "studies": studies,
            "noise_sd": noise_sd,
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
            "label": "MOCK batch-invariant control vs noisy-batch refusal flip rate (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope invariant-ctl --seed {args.seed} "
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
