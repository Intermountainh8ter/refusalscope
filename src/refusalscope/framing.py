"""MOCK paired refusal flip rate of stub responders under neutral vs evaluation-framed wording.

Simulation only: no model, no server. Seeded stub responders answer a synthetic refusal item
set under neutral and evaluation-framed wording with one shared draw per item (paired design).
The item-level flip rate and its bootstrap interval are the floor a real model must clear.
Output is an estimator check, not evidence about any model.

    PYTHONPATH=src python3 -m refusalscope.framing --seed 7 --out results/615-framing-flip-floor-stub.json
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

SCHEMA = "refusalscope.framing/1"
WORDINGS = ("neutral", "framed")
# Refusal probability per wording for each stub responder.
STUBS = {
    "wording_keyed": {"neutral": 0.30, "framed": 0.45},
    "always_comply": {"neutral": 0.0, "framed": 0.0},
    "always_refuse": {"neutral": 1.0, "framed": 1.0},
    "coin_flip": {"neutral": 0.5, "framed": 0.5},
}


def refusals(seed: int, stub: str, wording: str, n_items: int) -> list[bool]:
    """Per-item refusal. The item draw is shared across wordings (paired design)."""
    p = STUBS[stub][wording]
    return [stable_rng("framing", seed, stub, i).random() < p for i in range(n_items)]


def flips(a: list[bool], b: list[bool]) -> np.ndarray:
    return np.asarray([x != y for x, y in zip(a, b)], dtype=float)


def interval(f: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    idx = np.random.default_rng(seed).integers(0, f.size, size=(n_boot, f.size))
    s = f[idx].mean(axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def run(seed: int, n_items: int, n_boot: int) -> dict:
    rate, ci = {}, {}
    for stub in STUBS:
        f = flips(*(refusals(seed, stub, w, n_items) for w in WORDINGS))
        rate[stub] = float(f.mean())
        ci[stub] = list(interval(f, n_boot, seed))
    return {
        "framing": {"paired_flip_rate": rate, "bootstrap_95": ci},
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
            "label": "MOCK evaluation-framing paired flip estimator check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": "PYTHONPATH=src python3 -m refusalscope.framing "
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
