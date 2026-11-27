"""MOCK 2x2x2 factorial variance decomposition of a refusal rate (template, batch slots, cache).

Simulation only: no model, no server. A seeded stub refuses a synthetic item set with a fixed
per-item latent draw; its refusal probability depends on the template and slots factors only, and
every run flips each item's label with a small jitter probability. For each of the eight cells,
one refusal rate per draw. Each factor's share is its main-effect sum of squares over the total
sum of squares across all cell-draw rates. Degenerate stubs have total variance exactly 0 and
report null shares. Output is an estimator check, not evidence about any model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.varshare/1"
JITTER = 0.06
FACTORS = ("template", "slots", "cache")
# stub -> (base p, template B shift, slots 4 shift, cache on shift, deterministic)
STUBS = {
    "factorial_stub": (0.30, 0.10, 0.02, 0.0, False),
    "always_refuse": (1.0, 0.0, 0.0, 0.0, True),
    "always_comply": (0.0, 0.0, 0.0, 0.0, True),
}


def simulate(seed: int, stub: str, n_items: int, n_draws: int) -> dict:
    base, d_t, d_s, d_c, deterministic = STUBS[stub]
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub)])
    latent = rng.random(n_items)
    cells = list(itertools.product((0, 1), repeat=3))
    rates = np.zeros((8, n_draws))
    for ci, (t, s, c) in enumerate(cells):
        p = min(1.0, max(0.0, base + d_t * t + d_s * s + d_c * c))
        for d in range(n_draws):
            labels = latent < p
            if not deterministic:
                labels = labels ^ (rng.random(n_items) < JITTER)
            rates[ci, d] = labels.mean()
    grand = rates.mean()
    ss_total = float(((rates - grand) ** 2).sum())
    out = {"total_variance": float(rates.var())}
    shares = {}
    for fi, name in enumerate(FACTORS):
        if ss_total == 0.0:
            shares[name] = None
            continue
        ss = 0.0
        for level in (0, 1):
            mask = np.array([cell[fi] == level for cell in cells])
            ss += rates[mask].size * (rates[mask].mean() - grand) ** 2
        shares[name] = float(ss / ss_total)
    out["shares"] = shares
    return out


def run(seed: int, n_items: int, n_draws: int) -> dict:
    sims = {s: simulate(seed, s, n_items, n_draws) for s in STUBS}
    return {
        "variance_share": {
            f: {s: v["shares"][f] for s, v in sims.items()} for f in FACTORS
        }
        | {"total_variance": {s: v["total_variance"] for s, v in sims.items()}},
        "injected": {
            "stubs": {k: list(v) for k, v in STUBS.items()},
            "jitter": JITTER,
            "n_items": n_items,
            "n_draws": n_draws,
        },
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
            "label": "MOCK 2x2x2 factorial variance-share estimator check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope variance-share --seed {args.seed} "
                f"--items {args.items} --draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
