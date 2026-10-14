"""Every statistic is checked against a hand-computed or textbook value."""

import math

import numpy as np
import pytest

from refusalscope import stats
from refusalscope.stats import (
    binom_cdf,
    binom_pmf,
    bootstrap_did,
    bootstrap_mean_ci,
    bootstrap_paired_diff,
    clopper_pearson,
    cohens_dz,
    decision_label,
    holm_bonferroni,
    mcnemar_exact,
    mcnemar_power,
    odds_ratio_woolf,
    paired_counts,
    paired_odds_ratio,
    spearman,
    spearman_test,
    wilson_ci,
)

# ------------------------------------------------------------------ binomial


def test_binom_pmf_hand_values():
    assert binom_pmf(0, 3, 0.5) == pytest.approx(0.125)
    assert binom_pmf(1, 3, 0.5) == pytest.approx(0.375)
    assert binom_pmf(2, 5, 0.3) == pytest.approx(10 * 0.09 * 0.343)
    assert binom_pmf(7, 5, 0.3) == 0.0 and binom_pmf(-1, 5, 0.3) == 0.0
    assert binom_pmf(0, 4, 0.0) == 1.0 and binom_pmf(1, 4, 0.0) == 0.0
    assert binom_pmf(4, 4, 1.0) == 1.0 and binom_pmf(3, 4, 1.0) == 0.0


def test_binom_cdf_hand_values():
    assert binom_cdf(2, 10, 0.5) == pytest.approx(56 / 1024)
    assert binom_cdf(-1, 10, 0.5) == 0.0
    assert binom_cdf(10, 10, 0.5) == 1.0
    assert binom_cdf(0, 3, 0.5) == pytest.approx(0.125)


def test_pmf_sums_to_one():
    assert math.fsum(binom_pmf(k, 40, 0.37) for k in range(41)) == pytest.approx(1.0)


# ------------------------------------------------------------------ intervals


def test_wilson_textbook_values():
    p, lo, hi = wilson_ci(8, 10)
    assert (
        p == 0.8 and lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)
    )
    _, lo0, hi0 = wilson_ci(0, 10)
    assert lo0 == 0.0 and hi0 == pytest.approx(0.2775, abs=1e-4)
    _, lo1, hi1 = wilson_ci(10, 10)
    assert hi1 == 1.0 and lo1 == pytest.approx(0.7225, abs=1e-4)


def test_wilson_is_narrower_for_larger_samples_and_validates():
    narrow = wilson_ci(80, 100)
    wide = wilson_ci(8, 10)
    assert (narrow[2] - narrow[1]) < (wide[2] - wide[1])
    with pytest.raises(ValueError):
        wilson_ci(1, 0)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    assert wilson_ci(5, 10, conf=0.90)[1] > wilson_ci(5, 10, conf=0.99)[1]


def test_clopper_pearson_table_values():
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.1871, abs=1e-4) and hi == pytest.approx(0.8129, abs=1e-4)
    lo0, hi0 = clopper_pearson(0, 10)
    assert lo0 == 0.0 and hi0 == pytest.approx(1 - 0.025 ** (1 / 10), abs=1e-6)
    assert hi0 == pytest.approx(0.3085, abs=1e-4)
    lo1, hi1 = clopper_pearson(1, 10)
    assert lo1 == pytest.approx(0.0025, abs=1e-4) and hi1 == pytest.approx(0.4450, abs=1e-4)
    lo10, hi10 = clopper_pearson(10, 10)
    assert hi10 == 1.0 and lo10 == pytest.approx(0.6915, abs=1e-4)
    with pytest.raises(ValueError):
        clopper_pearson(1, 0)
    with pytest.raises(ValueError):
        clopper_pearson(-1, 10)


def test_clopper_pearson_contains_wilson_centre_and_is_wider():
    cp = clopper_pearson(30, 100)
    _, wl, wh = wilson_ci(30, 100)
    assert cp[0] <= wl + 1e-9 and cp[1] >= wh - 1e-9


# ------------------------------------------------------------------ paired binary


def test_paired_counts():
    stock = [True, True, False, False, True]
    ablated = [True, False, True, False, False]
    assert paired_counts(stock, ablated) == {"both": 1, "b": 2, "c": 1, "neither": 1}
    with pytest.raises(ValueError):
        paired_counts([True], [True, False])


def test_mcnemar_hand_computed_cases():
    assert mcnemar_exact(2, 8)["p_value"] == pytest.approx(112 / 1024)
    assert mcnemar_exact(8, 2)["p_value"] == pytest.approx(112 / 1024)
    assert mcnemar_exact(5, 5)["p_value"] == 1.0
    assert mcnemar_exact(0, 0)["p_value"] == 1.0
    assert mcnemar_exact(0, 10)["p_value"] == pytest.approx(2 / 1024)
    assert mcnemar_exact(1, 9)["n_discordant"] == 10
    assert mcnemar_exact(0, 1)["p_value"] == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_mcnemar_is_symmetric_and_monotone():
    assert mcnemar_exact(3, 12)["p_value"] == mcnemar_exact(12, 3)["p_value"]
    assert mcnemar_exact(3, 20)["p_value"] < mcnemar_exact(3, 12)["p_value"]


def test_paired_odds_ratio():
    result = paired_odds_ratio(2, 8)
    assert result["or"] == 4.0 and not result["haldane"]
    lo_p, hi_p = clopper_pearson(8, 10)
    assert result["lo"] == pytest.approx(lo_p / (1 - lo_p))
    assert result["hi"] == pytest.approx(hi_p / (1 - hi_p))
    assert result["lo"] < 4.0 < result["hi"]
    assert paired_odds_ratio(8, 2)["or"] == 0.25


def test_paired_odds_ratio_zero_cells_use_haldane_and_open_bound():
    only_c = paired_odds_ratio(0, 5)
    assert only_c["haldane"] and only_c["or"] == pytest.approx(5.5 / 0.5) and only_c["hi"] is None
    only_b = paired_odds_ratio(5, 0)
    assert only_b["haldane"] and only_b["or"] == pytest.approx(0.5 / 5.5) and only_b["lo"] == 0.0
    none = paired_odds_ratio(0, 0)
    assert none["or"] is None and none["lo"] is None


def test_woolf_odds_ratio_hand_computed():
    result = odds_ratio_woolf(10, 20, 5, 25)
    assert result["or"] == pytest.approx(2.5)
    se = math.sqrt(1 / 10 + 1 / 20 + 1 / 5 + 1 / 25)
    assert result["lo"] == pytest.approx(math.exp(math.log(2.5) - 1.959964 * se), rel=1e-5)
    assert result["hi"] == pytest.approx(math.exp(math.log(2.5) + 1.959964 * se), rel=1e-5)
    assert result["lo"] == pytest.approx(0.7351, abs=1e-3) and result["hi"] == pytest.approx(
        8.5019, abs=1e-3
    )
    assert not result["haldane"]


def test_woolf_zero_cell_gets_haldane_correction():
    result = odds_ratio_woolf(5, 0, 2, 3)
    assert result["haldane"]
    assert result["or"] == pytest.approx((5.5 * 3.5) / (0.5 * 2.5))
    with pytest.raises(ValueError):
        odds_ratio_woolf(-1, 1, 1, 1)


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_paired_constant_difference_is_degenerate():
    x = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    y = [v + 0.25 for v in x]
    result = bootstrap_paired_diff(x, y, n_boot=500)
    assert result["mean_diff"] == pytest.approx(0.25)
    assert result["lo"] == pytest.approx(0.25) and result["hi"] == pytest.approx(0.25)
    assert result["n"] == 6 and result["n_boot"] == 500


def test_bootstrap_is_reproducible_and_seed_dependent():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, 80).astype(float)
    y = rng.integers(0, 2, 80).astype(float)
    a = bootstrap_paired_diff(x, y, n_boot=1000, seed=1)
    b = bootstrap_paired_diff(x, y, n_boot=1000, seed=1)
    c = bootstrap_paired_diff(x, y, n_boot=1000, seed=2)
    assert a == b
    assert (a["lo"], a["hi"], a["p_value"]) != (c["lo"], c["hi"], c["p_value"])
    assert a["lo"] <= a["mean_diff"] <= a["hi"]


def test_bootstrap_interval_matches_normal_approximation():
    rng = np.random.default_rng(3)
    x = np.zeros(400)
    y = (rng.random(400) < 0.3).astype(float)
    result = bootstrap_paired_diff(x, y, n_boot=4000)
    se = math.sqrt(0.3 * 0.7 / 400)
    assert result["hi"] - result["lo"] == pytest.approx(2 * 1.96 * se, rel=0.12)
    assert result["p_value"] < 0.001


def test_bootstrap_interval_coverage_is_close_to_nominal():
    rng = np.random.default_rng(11)
    covered = 0
    trials = 150
    for t in range(trials):
        x = (rng.random(60) < 0.5).astype(float)
        y = (rng.random(60) < 0.6).astype(float)
        r = bootstrap_paired_diff(x, y, n_boot=400, seed=t)
        covered += r["lo"] <= 0.1 <= r["hi"]
    assert 0.86 <= covered / trials <= 0.995


def test_bootstrap_null_gives_large_p_value():
    rng = np.random.default_rng(5)
    x = (rng.random(300) < 0.5).astype(float)
    result = bootstrap_paired_diff(x, x.copy(), n_boot=500)
    assert result["p_value"] == 1.0


def test_bootstrap_input_validation():
    with pytest.raises(ValueError):
        bootstrap_paired_diff([], [])
    with pytest.raises(ValueError):
        bootstrap_paired_diff([1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError):
        bootstrap_did([], [], [1.0], [1.0])


def test_bootstrap_did_recovers_known_difference():
    n = 400
    stock_a = np.ones(n)
    ablated_a = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])  # RD_a = -0.5
    stock_b = np.ones(n)
    ablated_b = np.ones(n)  # RD_b = 0
    result = bootstrap_did(stock_a, ablated_a, stock_b, ablated_b, n_boot=1000)
    assert result["did"] == pytest.approx(-0.5)
    assert result["hi"] < -0.4 and result["p_value"] < 0.01
    assert result["n_a"] == n and result["n_b"] == n


def test_bootstrap_did_with_equal_effects_is_centered_on_zero():
    rng = np.random.default_rng(9)
    x = (rng.random(300) < 0.7).astype(float)
    y = (rng.random(300) < 0.6).astype(float)
    result = bootstrap_did(x, y, x, y, n_boot=1000)
    assert result["did"] == pytest.approx(0.0)
    assert result["lo"] < 0 < result["hi"]


def test_bootstrap_mean_ci():
    result = bootstrap_mean_ci([1.0] * 10, n_boot=200)
    assert result["mean"] == 1.0 and result["lo"] == 1.0 and result["hi"] == 1.0
    rng = np.random.default_rng(1)
    data = rng.normal(5, 1, 200)
    r = bootstrap_mean_ci(data, n_boot=2000)
    assert r["lo"] < 5 < r["hi"] and r["n"] == 200


def test_cohens_dz():
    assert cohens_dz([1.0, 3.0]) == pytest.approx(2.0 / math.sqrt(2.0))
    assert cohens_dz([1.0, 1.0, 1.0]) is None
    assert cohens_dz([1.0]) is None
    d = [0.5, 1.5, 1.0, 2.0]
    assert cohens_dz(d) == pytest.approx(1.25 / np.std(d, ddof=1))


# ------------------------------------------------------------------ multiplicity


def test_holm_textbook_example():
    result = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
    assert result["adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert result["reject"] == [True, False, False, True]


def test_holm_single_and_capping_and_monotonicity():
    assert holm_bonferroni([0.03])["adjusted"] == [0.03]
    assert holm_bonferroni([0.9, 0.8])["adjusted"] == [1.0, 1.0]
    result = holm_bonferroni([0.001, 0.002, 0.003, 0.9])
    assert result["adjusted"] == pytest.approx([0.004, 0.006, 0.006, 0.9])
    assert holm_bonferroni([])["adjusted"] == []
    assert holm_bonferroni([0.04], alpha=0.01)["reject"] == [False]


def test_holm_never_less_conservative_than_raw_p():
    rng = np.random.default_rng(2)
    p = rng.random(20).tolist()
    adjusted = holm_bonferroni(p)["adjusted"]
    assert all(a >= raw for a, raw in zip(adjusted, p))


# ------------------------------------------------------------------ Spearman


def test_spearman_hand_computed():
    assert spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]) == pytest.approx(0.8208, abs=1e-3)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    with pytest.raises(ValueError):
        spearman([1], [1])
    with pytest.raises(ValueError):
        spearman([1, 2], [1, 2, 3])


def test_spearman_ties_use_average_ranks():
    assert spearman([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)


@pytest.mark.parametrize(("n", "p"), [(3, 2 / 6), (4, 2 / 24), (5, 2 / 120), (6, 2 / 720)])
def test_spearman_exact_p_for_perfect_monotone_data(n, p):
    result = spearman_test(list(range(n)), list(range(n, 0, -1)))
    assert result["exact"] and result["rho"] == pytest.approx(-1.0)
    assert result["p_value"] == pytest.approx(p)


def test_spearman_not_significant_below_five_pairs():
    for n in (3, 4):
        assert spearman_test(list(range(n)), list(range(n)))["p_value"] > 0.05
    assert spearman_test(list(range(5)), list(range(5)))["p_value"] < 0.05


def test_spearman_exact_for_nine_and_monte_carlo_for_larger():
    nine = spearman_test(list(range(9)), list(range(9)))
    assert nine["exact"] and nine["p_value"] == pytest.approx(2 / math.factorial(9))
    ten = spearman_test(list(range(10)), list(range(10)), n_mc=20000)
    assert not ten["exact"] and ten["p_value"] < 0.001


def test_spearman_undefined_input():
    result = spearman_test([1, 1, 1], [1, 2, 3])
    assert result["rho"] is None and result["p_value"] is None


def test_spearman_unrelated_data_is_not_significant():
    result = spearman_test([1, 2, 3, 4, 5, 6], [2, 6, 1, 5, 3, 4])
    assert result["p_value"] > 0.1


# ------------------------------------------------------------------ power (reproduces the pre-registration)


@pytest.mark.parametrize(
    ("n", "p_d", "rd", "expected"),
    [
        (374, 0.10, 0.05, 0.847),
        (374, 0.20, 0.05, 0.539),
        (374, 0.20, 0.08, 0.928),
        (374, 0.05, 0.03, 0.698),
        (378, 0.10, 0.05, 0.852),
        (378, 0.20, 0.08, 0.931),
        (378, 0.30, 0.03, 0.162),
        (378, 0.30, 0.10, 0.940),
    ],
)
def test_power_reproduces_the_preregistered_table(n, p_d, rd, expected):
    assert mcnemar_power(n, p_d, rd) == pytest.approx(expected, abs=0.001)


def test_power_with_holm_threshold_matches_preregistration():
    assert mcnemar_power(378, 0.10, 0.05, alpha=0.05 / 6) == pytest.approx(0.642, abs=0.001)
    assert mcnemar_power(378, 0.20, 0.08, alpha=0.05 / 6) == pytest.approx(0.786, abs=0.001)


def test_power_properties():
    assert mcnemar_power(100, 0.2, 0.0) < 0.05  # no effect: rejection rate stays near alpha
    assert (
        mcnemar_power(300, 0.2, 0.1) > mcnemar_power(300, 0.2, 0.05) > mcnemar_power(300, 0.2, 0.02)
    )
    assert mcnemar_power(600, 0.1, 0.05) > mcnemar_power(300, 0.1, 0.05)
    with pytest.raises(ValueError):
        mcnemar_power(10, 0.0, 0.0)
    with pytest.raises(ValueError):
        mcnemar_power(10, 0.1, 0.5)


# ------------------------------------------------------------------ decision rule


@pytest.mark.parametrize(
    ("rd", "ci", "p", "direction", "label"),
    [
        (-0.08, (-0.12, -0.04), 0.001, -1, "supported"),
        (-0.05, (-0.09, -0.01), 0.01, -1, "supported"),
        (
            -0.04,
            (-0.07, -0.01),
            0.01,
            -1,
            "inconclusive",
        ),  # significant but below SESOI, CI crosses margin
        (0.07, (0.03, 0.11), 0.002, -1, "contrary"),
        (0.01, (-0.03, 0.04), 0.6, -1, "equivalent"),
        (-0.02, (-0.045, 0.004), 0.2, -1, "equivalent"),
        (0.0, (-0.08, 0.08), 0.9, -1, "inconclusive"),
        (0.08, (0.04, 0.12), 0.001, 1, "supported"),
        (-0.08, (-0.12, -0.04), 0.001, 1, "contrary"),
        (-0.08, (-0.12, -0.04), 0.2, -1, "inconclusive"),  # not significant after correction
        (-0.03, (-0.0499, 0.0499), 0.5, -1, "equivalent"),
        (
            -0.03,
            (-0.05, 0.04),
            0.5,
            -1,
            "inconclusive",
        ),  # CI touches the margin: not strictly inside
    ],
)
def test_decision_labels(rd, ci, p, direction, label):
    assert decision_label(rd, ci, p, direction) == label


def test_decision_label_rejects_bad_direction_and_uses_constants():
    with pytest.raises(ValueError):
        decision_label(0.1, (0.0, 0.2), 0.01, 0)
    assert stats.SESOI == 0.05 and stats.ALPHA == 0.05 and stats.DEFAULT_BOOTSTRAP == 10_000
    assert stats.SEED_PAIRED == 424242 and stats.SEED_DID == 424243
