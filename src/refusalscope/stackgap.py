"""MOCK share of a between-stack refusal-rate gap explained by repeat-run label-flip noise.

Simulation only: no model, no server. Seeded stub responders refuse a synthetic item set with a
fixed per-item latent draw; every run additionally flips each item's label with a small jitter
probability. Per bootstrap draw, the observed gap is |rate(stack B) - rate(stack A run 1)| and the
noise gap is |rate(stack A run 2) - rate(stack A run 1)|. Explained fraction is the ratio of the
means. For the identical-stack stub, stack B is the A repeat run, so the two gaps coincide.
Output is an estimator check, not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.stackgap/1"
JITTER = 0.06
# (refusal probability on stack A, on stack B, B is the repeat run of A)
STUBS = {
    "template_shift": (0.30, 0.40, False),
    "identical_stack": (0.30, 0.30, True),
    "always_refuse": (1.0, 1.0, False),
    "always_comply": (0.0, 0.0, False),
}


def _run(latent: np.ndarray, p: float, rng: np.random.Generator, jitter: float, deterministic: bool) -> float:
    labels = latent < p
    if not deterministic:
        labels = labels ^ (rng.random(latent.size) < jitter)
    return float(labels.mean())


def simulate(seed: int, stub: str, n_items: int, n_draws: int) -> dict:
    pa, pb, repeat = STUBS[stub]
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub)])
    latent = rng.random(n_items)
    degenerate = pa in (0.0, 1.0)
    obs, noise = [], []
    for _ in range(n_draws):
        a1 = _run(latent, pa, rng, JITTER, degenerate)
        a2 = _run(latent, pa, rng, JITTER, degenerate)
        b = a2 if repeat else _run(latent, pb, rng, JITTER, degenerate)
        obs.append(abs(b - a1))
        noise.append(abs(a2 - a1))
    return {"observed_gap": float(np.mean(obs)), "noise_gap": float(np.mean(noise))}


def run(seed: int, n_items: int, n_draws: int) -> dict:
    sims = {s: simulate(seed, s, n_items, n_draws) for s in STUBS}
    frac = {s: (v["noise_gap"] / v["observed_gap"] if v["observed_gap"] > 0 else None) for s, v in sims.items()}
    return {
        "noise_floor": {
            "observed_gap": {s: v["observed_gap"] for s, v in sims.items()},
            "noise_gap": {s: v["noise_gap"] for s, v in sims.items()},
            "explained_fraction": frac,
        },
        "injected": {"stubs": {k: list(v) for k, v in STUBS.items()}, "jitter": JITTER, "n_items": n_items, "n_draws": n_draws},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.items, args.draws)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK between-stack gap vs repeat-run noise estimator check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope noise-floor --seed {args.seed} "
                f"--items {args.items} --draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
