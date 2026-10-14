"""Statistics for paired stock-versus-ablated comparisons (stdlib + numpy only).

Every function here is covered by tests against hand-computed or textbook values. Conventions:

* ``b`` counts cases where the **stock** arm succeeded and the **ablated** arm failed;
  ``c`` counts the opposite. Risk difference is ``rate(ablated) - rate(stock)``.
* Confidence level defaults to 0.95; bootstrap resampling is over cases (the cluster) with a
  seeded ``numpy.random.default_rng``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import permutations
from statistics import NormalDist
from typing import Any

import numpy as np

DEFAULT_BOOTSTRAP = 10_000
SEED_PAIRED = 424_242
SEED_DID = 424_243
SESOI = 0.05
ALPHA = 0.05

# --------------------------------------------------------------------------- binomial


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def binom_pmf(k: int, n: int, p: float) -> float:
    """P(X = k) for X ~ Binomial(n, p)."""
    if k < 0 or k > n:
        return 0.0
    if p <= 0.0:
        return 1.0 if k == 0 else 0.0
    if p >= 1.0:
        return 1.0 if k == n else 0.0
    return math.exp(_log_choose(n, k) + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return min(1.0, math.fsum(binom_pmf(i, n, p) for i in range(k + 1)))


def _z(conf: float) -> float:
    return NormalDist().inv_cdf(0.5 + conf / 2.0)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval. Returns ``(proportion, lower, upper)``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    z = _z(conf)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact binomial confidence interval, found by bisection on the binomial CDF."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    alpha = 1.0 - conf

    def solve(target: float, upper_tail: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            value = (1.0 - binom_cdf(k - 1, n, mid)) if upper_tail else binom_cdf(k, n, mid)
            # upper_tail: P(X >= k) increases with p; lower_tail: P(X <= k) decreases with p
            if upper_tail:
                lo, hi = (mid, hi) if value < target else (lo, mid)
            else:
                lo, hi = (lo, mid) if value < target else (mid, hi)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(alpha / 2, upper_tail=True)
    upper = 1.0 if k == n else solve(alpha / 2, upper_tail=False)
    return lower, upper


# --------------------------------------------------------------------------- paired binary


def paired_counts(stock: Sequence[bool], ablated: Sequence[bool]) -> dict[str, int]:
    """2x2 paired table: ``both``, ``b`` (stock only), ``c`` (ablated only), ``neither``."""
    if len(stock) != len(ablated):
        raise ValueError("arms must have the same number of cases")
    both = sum(1 for s, a in zip(stock, ablated) if s and a)
    b = sum(1 for s, a in zip(stock, ablated) if s and not a)
    c = sum(1 for s, a in zip(stock, ablated) if not s and a)
    return {"both": both, "b": b, "c": c, "neither": len(stock) - both - b - c}


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    """Exact two-sided McNemar test: ``min(1, 2 * P(X <= min(b, c)))``, ``X ~ Bin(b + c, 1/2)``."""
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "p_value": 1.0}
    p = min(1.0, 2.0 * binom_cdf(min(b, c), n, 0.5))
    return {"b": b, "c": c, "n_discordant": n, "p_value": p}


def paired_odds_ratio(b: int, c: int, conf: float = 0.95) -> dict[str, Any]:
    """Conditional odds ratio ``c / b`` (ablated success odds relative to stock).

    The interval transforms the exact Clopper-Pearson interval of ``c / (b + c)`` to the odds
    scale. With a zero discordant count the point estimate uses the Haldane-Anscombe 0.5
    correction and an infinite bound is reported as ``None``.
    """
    n = b + c
    if n == 0:
        return {"or": None, "lo": None, "hi": None, "haldane": False}
    lo_p, hi_p = clopper_pearson(c, n, conf)

    def odds(p: float) -> float | None:
        return None if p >= 1.0 else p / (1.0 - p)

    haldane = b == 0 or c == 0
    point = (c + 0.5) / (b + 0.5) if haldane else c / b
    return {"or": point, "lo": odds(lo_p), "hi": odds(hi_p), "haldane": haldane}


def odds_ratio_woolf(s1: int, f1: int, s0: int, f0: int, conf: float = 0.95) -> dict[str, Any]:
    """Marginal odds ratio of success for arm 1 versus arm 0 with a Woolf (log) interval.

    ``s``/``f`` are success and failure counts. Any zero cell adds 0.5 to every cell.
    """
    cells = [s1, f1, s0, f0]
    if min(cells) < 0:
        raise ValueError("counts must be non-negative")
    haldane = 0 in cells
    if haldane:
        cells = [x + 0.5 for x in cells]
    a, b, c, d = cells
    log_or = math.log((a * d) / (b * c))
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    z = _z(conf)
    return {
        "or": math.exp(log_or),
        "lo": math.exp(log_or - z * se),
        "hi": math.exp(log_or + z * se),
        "haldane": haldane,
    }


# --------------------------------------------------------------------------- bootstrap


def _percentile_p(samples: np.ndarray, null: float = 0.0) -> float:
    """Two-sided percentile-bootstrap p-value (interval inversion), resolution 1 / (B + 1)."""
    n = len(samples)
    below = (int(np.sum(samples <= null)) + 1) / (n + 1)
    above = (int(np.sum(samples >= null)) + 1) / (n + 1)
    return min(1.0, 2.0 * min(below, above))


def _interval(samples: np.ndarray, conf: float) -> tuple[float, float]:
    tail = (1.0 - conf) / 2.0 * 100.0
    lo, hi = np.percentile(samples, [tail, 100.0 - tail])
    return float(lo), float(hi)


def bootstrap_paired_diff(
    stock: Sequence[float],
    ablated: Sequence[float],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = SEED_PAIRED,
    conf: float = 0.95,
) -> dict[str, Any]:
    """Paired percentile bootstrap of ``mean(ablated) - mean(stock)`` resampling cases."""
    x = np.asarray(stock, dtype=float)
    y = np.asarray(ablated, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("arms must be non-empty and the same length")
    diffs = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    samples = diffs[idx].mean(axis=1)
    lo, hi = _interval(samples, conf)
    return {
        "mean_diff": float(diffs.mean()),
        "lo": lo,
        "hi": hi,
        "p_value": _percentile_p(samples),
        "n": int(diffs.size),
        "n_boot": n_boot,
    }


def bootstrap_did(
    stock_a: Sequence[float],
    ablated_a: Sequence[float],
    stock_b: Sequence[float],
    ablated_b: Sequence[float],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = SEED_DID,
    conf: float = 0.95,
) -> dict[str, Any]:
    """Difference of two paired mean differences, resampling each case set independently.

    Returns ``(mean(ablated_a) - mean(stock_a)) - (mean(ablated_b) - mean(stock_b))``.
    """
    da = np.asarray(ablated_a, dtype=float) - np.asarray(stock_a, dtype=float)
    db = np.asarray(ablated_b, dtype=float) - np.asarray(stock_b, dtype=float)
    if da.size == 0 or db.size == 0:
        raise ValueError("both case sets must be non-empty")
    rng = np.random.default_rng(seed)
    first = da[rng.integers(0, da.size, size=(n_boot, da.size))].mean(axis=1)
    second = db[rng.integers(0, db.size, size=(n_boot, db.size))].mean(axis=1)
    samples = first - second
    lo, hi = _interval(samples, conf)
    return {
        "did": float(da.mean() - db.mean()),
        "lo": lo,
        "hi": hi,
        "p_value": _percentile_p(samples),
        "n_a": int(da.size),
        "n_b": int(db.size),
        "n_boot": n_boot,
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = SEED_PAIRED,
    conf: float = 0.95,
) -> dict[str, Any]:
    """Percentile bootstrap interval for a mean."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("values must be non-empty")
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, data.size, size=(n_boot, data.size))].mean(axis=1)
    lo, hi = _interval(samples, conf)
    return {"mean": float(data.mean()), "lo": lo, "hi": hi, "n": int(data.size)}


def cohens_dz(differences: Sequence[float]) -> float | None:
    """Standardised paired mean difference (mean / sample standard deviation of differences)."""
    data = np.asarray(differences, dtype=float)
    if data.size < 2:
        return None
    sd = float(data.std(ddof=1))
    return None if sd == 0.0 else float(data.mean() / sd)


# --------------------------------------------------------------------------- multiplicity


def holm_bonferroni(p_values: Sequence[float], alpha: float = ALPHA) -> dict[str, list]:
    """Holm step-down adjustment. Returns adjusted p-values and rejection flags in input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return {"adjusted": adjusted, "reject": [a < alpha for a in adjusted]}


# --------------------------------------------------------------------------- rank correlation


def _ranks(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = arr.argsort(kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman rank correlation (average ranks for ties); ``None`` if undefined."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need two equal-length sequences of at least two values")
    rx, ry = _ranks(x), _ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def spearman_test(
    x: Sequence[float], y: Sequence[float], max_exact_n: int = 9, n_mc: int = 100_000, seed: int = 7
) -> dict[str, Any]:
    """Two-sided permutation test for the Spearman correlation (exact for ``n <= 9``)."""
    n = len(x)
    rho = spearman(x, y)
    if rho is None:
        return {"rho": None, "p_value": None, "exact": True, "n": n}
    rx, ry = _ranks(x), _ranks(y)
    rxc, ryc = rx - rx.mean(), ry - ry.mean()
    scale = n * rx.std() * ry.std()
    observed = abs(float(rxc @ ryc)) / scale
    if n <= max_exact_n:
        orders = np.array(list(permutations(range(n))))
        stats = np.abs(ryc[orders] @ rxc) / scale
        return {
            "rho": rho,
            "p_value": float(np.mean(stats >= observed - 1e-12)),
            "exact": True,
            "n": n,
        }
    rng = np.random.default_rng(seed)
    orders = np.argsort(rng.random((n_mc, n)), axis=1)
    stats = np.abs(ryc[orders] @ rxc) / scale
    hits = int(np.sum(stats >= observed - 1e-12))
    return {"rho": rho, "p_value": (hits + 1) / (n_mc + 1), "exact": False, "n": n}


# --------------------------------------------------------------------------- power


def mcnemar_power(n: int, p_discordant: float, rd: float, alpha: float = ALPHA) -> float:
    """Exact power of the two-sided McNemar test by enumeration.

    ``D ~ Binomial(n, p_discordant)`` pairs disagree; each favours the second arm with
    probability ``(p_discordant + rd) / (2 * p_discordant)``.
    """
    if not 0 < p_discordant <= 1:
        raise ValueError("p_discordant must be in (0, 1]")
    q = (p_discordant + rd) / (2.0 * p_discordant)
    if not 0.0 <= q <= 1.0:
        raise ValueError("rd must satisfy |rd| <= p_discordant")
    total = 0.0
    for d in range(n + 1):
        weight = binom_pmf(d, n, p_discordant)
        if weight < 1e-14:
            continue
        reject = math.fsum(
            binom_pmf(c, d, q) for c in range(d + 1) if mcnemar_exact(d - c, c)["p_value"] < alpha
        )
        total += weight * reject
    return total


# --------------------------------------------------------------------------- decision rule


def decision_label(
    rd: float,
    ci: tuple[float, float],
    p_adjusted: float,
    direction: int,
    sesoi: float = SESOI,
    alpha: float = ALPHA,
) -> str:
    """Pre-registered decision label (``docs/PREREGISTRATION.md`` section 7).

    ``direction`` is +1 if the hypothesis predicts a positive effect and -1 for a negative one.
    Order: supported, contrary, equivalent, inconclusive.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    significant = p_adjusted < alpha
    if significant and rd * direction >= sesoi:
        return "supported"
    if significant and -rd * direction >= sesoi:
        return "contrary"
    if ci[0] > -sesoi and ci[1] < sesoi:
        return "equivalent"
    return "inconclusive"
