"""MOCK dose-response estimator check: tool-call validity slope against ablation dose.

Simulation only: no model, no server. A seeded stub whose validity falls linearly with dose is
sampled on a dose grid; the least-squares slope and its bootstrap interval are compared with
two scripted floors (always-refuse, never-call) whose validity and slope are exactly zero.
Output is an estimator check, not evidence about any model.

    PYTHONPATH=src python3 -m refusalscope.dose_validity --seed 7 --out results/624-dose-validity-flip-floor-mock.json
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

SCHEMA = "refusalscope.dose_validity/1"
N_DOSES = 11
INTERCEPT = 0.90
TRUE_SLOPE = -0.30
DOSES = [i / (N_DOSES - 1) for i in range(N_DOSES)]


def p_valid(stub: str, dose: float) -> float:
    if stub == "linear_decay":
        return INTERCEPT + TRUE_SLOPE * dose
    return 0.0  # always_refuse and never_call never emit a valid call


STUBS = ("linear_decay", "always_refuse", "never_call")


def draws(seed: int, stub: str, n_draws: int) -> np.ndarray:
    """Shape (n_doses, n_draws) of 0/1 validity outcomes."""
    out = np.zeros((N_DOSES, n_draws))
    for k, d in enumerate(DOSES):
        p = p_valid(stub, d)
        out[k] = [stable_rng("dose_validity", seed, stub, k, j).random() < p for j in range(n_draws)]
    return out


def slope(means: np.ndarray) -> float:
    return float(np.polyfit(np.asarray(DOSES), means, 1)[0])


def bootstrap(y: np.ndarray, n_boot: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = y.shape[1]
    s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=(N_DOSES, n))
        s.append(slope(np.take_along_axis(y, idx, axis=1).mean(axis=1)))
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def run(seed: int, n_draws: int, n_boot: int) -> dict:
    validity, slopes, ci = {}, {}, {}
    for stub in STUBS:
        y = draws(seed, stub, n_draws)
        m = y.mean(axis=1)
        validity[stub] = [float(v) for v in m]
        slopes[stub] = slope(m)
        ci[stub] = bootstrap(y, n_boot, seed)
    return {
        "dose_response": {
            "tool_call_validity_slope": slopes["linear_decay"],
            "slope_bootstrap_95": ci["linear_decay"],
            "floor_slopes": {k: slopes[k] for k in STUBS if k != "linear_decay"},
            "validity_by_dose": validity,
            "doses": DOSES,
        },
        "injected": {"intercept": INTERCEPT, "true_slope": TRUE_SLOPE, "n_draws": n_draws, "n_boot": n_boot},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(args.seed, args.draws, args.n_boot)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK tool-call validity dose-response estimator check (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [args.seed],
                "command": "PYTHONPATH=src python3 -m refusalscope.dose_validity "
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
