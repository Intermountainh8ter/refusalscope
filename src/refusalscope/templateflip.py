"""MOCK refusal-label flip rate from a chat-template-omission margin shift vs sampling alone.

Simulation only: no model, no server. Each prompt has a refusal margin; omitting the chat template
shifts it by a constant. The template flip rate is the share of prompts whose deterministic label
(margin > 0) differs between the two arms. The sampling-alone flip rate uses two independent
Bernoulli draws with logistic probability of the margin. Degenerate stubs never flip and their
ratio is null. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.templateflip/1"
MARGIN_MEAN = -0.5244
STUBS = ("margin_stub", "always_refuse", "always_comply")


def _margins(rng: np.random.Generator, stub: str, prompts: int) -> np.ndarray:
    if stub == "always_refuse":
        return np.full(prompts, 1e9)
    if stub == "always_comply":
        return np.full(prompts, -1e9)
    return rng.normal(MARGIN_MEAN, 1.0, prompts)


def simulate(seed: int, stub: str, prompts: int, studies: int, shift: float) -> dict:
    rng = np.random.default_rng([seed, STUBS.index(stub)])
    tmpl, sampling = [], []
    for _ in range(studies):
        m = _margins(rng, stub, prompts)
        tmpl.append(float(((m > 0) != (m + shift > 0)).mean()))
        p = 1.0 / (1.0 + np.exp(-np.clip(m, -50, 50)))
        s1, s2 = rng.random(prompts) < p, rng.random(prompts) < p
        sampling.append(float((s1 != s2).mean()))
    t, s = float(np.mean(tmpl)), float(np.mean(sampling))
    return {
        "flip_rate_template": t,
        "flip_rate_sampling": s,
        "flip_rate_vs_sampling_ratio": (t / s) if s > 0 else None,
    }


def run(seed: int, prompts: int, studies: int, shift: float) -> dict:
    sims = {s: simulate(seed, s, prompts, studies, shift) for s in STUBS}
    return {
        "template": {
            **sims["margin_stub"],
            "stubs": {s: sims[s] for s in STUBS if s != "margin_stub"},
        },
        "injected": {
            "margin_mean": MARGIN_MEAN,
            "margin_sd": 1.0,
            "prompts": prompts,
            "studies": studies,
            "shift": shift,
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", type=int, default=300)
    ap.add_argument("--studies", type=int, default=400)
    ap.add_argument("--shift", type=float, default=0.2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.prompts, args.studies, args.shift)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK chat-template-omission vs sampling-alone refusal flip rate (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope templateflip --seed {args.seed} "
                f"--prompts {args.prompts} --studies {args.studies} "
                f"--shift {args.shift} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
