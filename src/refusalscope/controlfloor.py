"""MOCK random-direction control floor for dose-response claims.

Simulation only: no model, no server. A designed direction has a fixed true effect; matched random
directions have true effects drawn from a normal; every estimate adds independent measurement
noise. The metric is the fraction of random directions whose measured effect reaches the designed
direction's measured effect. A zero-effect stub reports exactly 0.0 effects and a null fraction.
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

SCHEMA = "refusalscope.controlfloor/1"
DESIGNED_EFFECT = 1.0
RANDOM_MEAN = 0.2
RANDOM_SD = 0.3
NOISE_SD = 0.15


def simulate(seed, directions, replicates):
    rng = np.random.default_rng([seed, 0])
    fracs, designed, random_mean = [], [], []
    for _ in range(replicates):
        d = DESIGNED_EFFECT + rng.normal(0.0, NOISE_SD)
        r = rng.normal(RANDOM_MEAN, RANDOM_SD, directions) + rng.normal(0.0, NOISE_SD, directions)
        fracs.append(float((r >= d).mean()))
        designed.append(d)
        random_mean.append(float(r.mean()))
    return {
        "random_reaches_designed_fraction": float(np.mean(fracs)),
        "designed_effect_measured": float(np.mean(designed)),
        "random_effect_measured": float(np.mean(random_mean)),
    }


def zero_stub():
    return {"designed_effect": 0.0, "random_effect": 0.0, "random_reaches_designed_fraction": None}


def run(seed, directions, replicates):
    return {
        "control_floor": {**simulate(seed, directions, replicates), "stubs": {"zero_effect": zero_stub()}},
        "injected": {
            "designed_effect": DESIGNED_EFFECT,
            "random_mean": RANDOM_MEAN,
            "random_sd": RANDOM_SD,
            "noise_sd": NOISE_SD,
            "directions": directions,
            "replicates": replicates,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--directions", type=int, default=64)
    ap.add_argument("--replicates", type=int, default=400)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(a.seed, a.directions, a.replicates)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK random-direction control floor (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [a.seed],
                "command": f"python3 -m refusalscope controlfloor --seed {a.seed} --directions {a.directions} "
                f"--replicates {a.replicates} --out {a.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
