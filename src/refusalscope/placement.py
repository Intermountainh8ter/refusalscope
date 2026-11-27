"""MOCK instruction-placement spread of stub responders against a bootstrap interval.

Simulation only: no model, no server. Seeded stub responders answer a synthetic
shutdown-compliance item set with the same instruction placed in the system prompt or in the
user prompt. The placement score spread is compared with the half-width of the 95% bootstrap
interval of a single placement. Output is an estimator check, not evidence about any model.

    PYTHONPATH=src python3 -m refusalscope.placement --seed 7 --out results/611-placement-flip-floor-stub.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .util import stable_rng

SCHEMA = "refusalscope.placement/1"
PLACEMENTS = ("system", "user")
# Compliance probability per placement for each stub responder.
STUBS = {
    "positional_bias": {"system": 0.70, "user": 0.55},
    "always_wrong": {"system": 0.0, "user": 0.0},
    "coin_flip": {"system": 0.5, "user": 0.5},
}


def compliance(seed: int, stub: str, placement: str, n_items: int) -> list[bool]:
    """Per-item compliance. The item draw is shared across placements (paired design)."""
    p = STUBS[stub][placement]
    return [stable_rng("placement", seed, stub, i).random() < p for i in range(n_items)]


def half_width(x: list[bool], n_boot: int, seed: int) -> float:
    a = np.asarray(x, dtype=float)
    idx = np.random.default_rng(seed).integers(0, a.size, size=(n_boot, a.size))
    s = a[idx].mean(axis=1)
    return float((np.percentile(s, 97.5) - np.percentile(s, 2.5)) / 2)


def run(seed: int, n_items: int, n_boot: int) -> dict:
    per_stub = {}
    ratios = {}
    for stub in STUBS:
        scores = {pl: float(np.mean(compliance(seed, stub, pl, n_items))) for pl in PLACEMENTS}
        spread = abs(scores["system"] - scores["user"])
        hw = half_width(compliance(seed, stub, "system", n_items), n_boot, seed)
        # No sampling variation and no spread: nothing to separate, ratio is 0 by definition.
        ratio = spread / hw if hw > 0 else 0.0
        per_stub[stub] = {"score": scores, "spread": spread, "interval_half_width": hw}
        ratios[stub] = ratio
    return {
        "placement": {"spread_over_interval": ratios, "per_stub": per_stub},
        "injected": {"stubs": STUBS, "n_items": n_items, "n_boot": n_boot},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.items, args.n_boot)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK instruction-placement spread estimator check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": "PYTHONPATH=src python3 -m refusalscope.placement "
                f"--seed {args.seed} --out {args.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded stub responders, no external data", "license": "MIT (repository licence)"}],
            },
        }
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
