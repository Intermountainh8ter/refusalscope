"""MOCK wording vs environment variance shares of cell refusal rates.

Simulation only: no model, no server. W prompt wordings are crossed with E environments; the
logit of the refusal probability is the base plus an additive wording effect and an additive
environment effect (no interaction). Each cell refusal rate is a binomial proportion. Shares are
sums of squares about the grand mean divided by the total. Constant responders have total sum of
squares 0 and null shares. Not evidence about any model.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA = "refusalscope.wordenv/1"
BASE_P = 0.30
WORDING_SD = 0.5
ENV_SD = 0.25
STUBS = ("effects_stub", "always_refuse", "always_comply")


def _cell_rates(rng, stub, wordings, envs, draws):
    if stub == "always_refuse":
        return np.ones((wordings, envs))
    if stub == "always_comply":
        return np.zeros((wordings, envs))
    base = np.log(BASE_P / (1 - BASE_P))
    w = rng.normal(0.0, WORDING_SD, wordings)
    e = rng.normal(0.0, ENV_SD, envs)
    p = 1.0 / (1.0 + np.exp(-(base + w[:, None] + e[None, :])))
    return rng.binomial(draws, p) / draws


def simulate(seed, stub, wordings, envs, draws, replicates):
    rng = np.random.default_rng([seed, STUBS.index(stub)])
    ws, es, tots, wins = [], [], [], 0
    for _ in range(replicates):
        y = _cell_rates(rng, stub, wordings, envs, draws)
        g = y.mean()
        total = float(((y - g) ** 2).sum())
        tots.append(total)
        if total > 0:
            ssw = float(envs * ((y.mean(axis=1) - g) ** 2).sum())
            sse = float(wordings * ((y.mean(axis=0) - g) ** 2).sum())
            ws.append(ssw / total)
            es.append(sse / total)
            wins += ssw > sse
    n = len(ws)
    return {
        "total_ss": float(np.mean(tots)),
        "wording_share": float(np.mean(ws)) if n else None,
        "environment_share": float(np.mean(es)) if n else None,
        "wording_exceeds_env_fraction": (wins / n) if n else None,
    }


def run(seed, wordings, envs, draws, replicates):
    sims = {s: simulate(seed, s, wordings, envs, draws, replicates) for s in STUBS}
    return {
        "wordenv": {
            **sims["effects_stub"],
            "stubs": {s: sims[s] for s in STUBS if s != "effects_stub"},
        },
        "injected": {
            "base_p": BASE_P,
            "wording_sd": WORDING_SD,
            "env_sd": ENV_SD,
            "wordings": wordings,
            "envs": envs,
            "draws": draws,
            "replicates": replicates,
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--wordings", type=int, default=6)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t0 = time.time()
    result = run(a.seed, a.wordings, a.envs, a.draws, a.replicates)
    result.update(
        {
            "schema": SCHEMA,
            "mock": True,
            "label": "MOCK wording vs environment variance share of refusal rates (stub responders)",
            "provenance": {
                "started_at": started,
                "seeds": [a.seed],
                "command": f"python3 -m refusalscope wordenv --seed {a.seed} --wordings {a.wordings} "
                f"--envs {a.envs} --draws {a.draws} --replicates {a.replicates} --out {a.out}",
                "hardware": f"{platform.machine()} cpu, {platform.system()}",
                "wall_s": round(time.time() - t0, 3),
                "data_sources": [{"url": "synthetic: seeded in-process stub responders, no external data", "license": "MIT (repository)"}],
            },
        }
    )
    Path(a.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0
