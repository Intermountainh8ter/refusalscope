"""MOCK raw-to-genuine flip ratio under a 5-rerun majority adjudication.

Simulation only: no model, no server. A raw candidate is an item whose single-run labels differ
between stacks; it is genuine when the 5-rerun majority labels also differ. Output is an estimator
check, not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .mdd import JITTER

SCHEMA = "refusalscope.adjudicate/1"
N_ITEMS = 1000
RERUNS = 5
STUBS = {
    "rare_batch": (0.30, 0.31),
    "template_shift": (0.30, 0.40),
    "identical_stack": (0.30, 0.30),
    "always_refuse": (1.0, 1.0),
    "always_comply": (0.0, 0.0),
}


def _noisy(lab: np.ndarray, rng: np.random.Generator, degenerate: bool) -> np.ndarray:
    return lab if degenerate else lab ^ (rng.random(lab.shape) < JITTER)


def flips_for(seed: int, stub: str, draws: int, n: int = N_ITEMS) -> tuple[list[int], list[int]]:
    pa, pb = STUBS[stub]
    rng = np.random.default_rng([seed, sorted(STUBS).index(stub)])
    degenerate = pa in (0.0, 1.0)
    raw, genuine = [], []
    for _ in range(draws):
        latent = rng.random(n)
        side = []
        for p in (pa, pb):
            base = latent < p
            single = _noisy(base, rng, degenerate)
            reruns = _noisy(np.broadcast_to(base, (RERUNS, n)), rng, degenerate)
            side.append((single, reruns.sum(axis=0) * 2 > RERUNS))
        cand = side[0][0] != side[1][0]
        raw.append(int(cand.sum()))
        genuine.append(int((cand & (side[0][1] != side[1][1])).sum()))
    return raw, genuine


def run(seed: int, draws: int, n: int = N_ITEMS) -> dict:
    ratio, stats = {}, {}
    for s in STUBS:
        raw, gen = flips_for(seed, s, draws, n)
        tr, tg = sum(raw), sum(gen)
        ratio[s] = (tr / tg) if tg else None
        stats[s] = {
            "total_raw": tr,
            "total_genuine": tg,
            "mean_raw_per_study": tr / draws,
            "mean_genuine_per_study": tg / draws,
        }
    return {
        "adjudication": {"raw_to_genuine_ratio": ratio, "flips": stats},
        "injected": {
            "stubs": {k: list(v) for k, v in STUBS.items()},
            "jitter": JITTER,
            "reruns": RERUNS,
            "n_items": n,
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
            "label": "MOCK raw-to-genuine flip adjudication check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": f"python3 -m refusalscope adjudicate --seed {args.seed} "
                f"--draws {args.draws} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
