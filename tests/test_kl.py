import math

import pytest

from refusalscope.client import ChatClient
from refusalscope.mock import MockProfile, MockServer
from refusalscope.suites.kl import (
    HARMLESS_PROMPTS,
    KLOracle,
    first_token_top,
    grade_kl_probe,
    kl_divergence_from_top,
    kl_items,
    logprob_warnings,
)


def top(**probs):
    return [{"token": t, "logprob": math.log(p)} for t, p in probs.items()]


def test_sixty_unique_harmless_prompts():
    assert len(HARMLESS_PROMPTS) == 60 == len(set(HARMLESS_PROMPTS))
    assert kl_items()[0] == {"id": "K-000", "prompt": HARMLESS_PROMPTS[0]}


def test_identical_distributions_have_zero_divergence():
    p = top(a=0.5, b=0.3, c=0.1)
    assert kl_divergence_from_top(p, p) == pytest.approx(0.0, abs=1e-12)


def test_textbook_two_outcome_example():
    p, q = top(a=0.5, b=0.5), top(a=0.9, b=0.1)
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert expected == pytest.approx(0.5108256, abs=1e-6)
    assert kl_divergence_from_top(p, q) == pytest.approx(expected, abs=1e-9)
    reverse = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl_divergence_from_top(q, p) == pytest.approx(reverse, abs=1e-9)
    assert kl_divergence_from_top(p, q) != pytest.approx(kl_divergence_from_top(q, p), abs=1e-3)


def test_truncated_lists_with_partial_overlap_follow_the_documented_formula():
    p = top(a=0.6, b=0.3)  # tail 0.1
    q = top(a=0.5, c=0.4)  # tail 0.1
    floor_p = min(0.3, 0.1)
    floor_q = min(0.4, 0.1)
    expected = (
        0.6 * math.log(0.6 / 0.5)
        + 0.3 * math.log(0.3 / floor_q)
        + floor_p * math.log(floor_p / 0.4)
        + 0.1 * math.log(0.1 / 0.1)
    )
    assert kl_divergence_from_top(p, q) == pytest.approx(expected, abs=1e-9)


def test_tail_mass_difference_contributes():
    p = top(a=0.9)  # tail 0.1
    q = top(a=0.5)  # tail 0.5
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert kl_divergence_from_top(p, q) == pytest.approx(expected, abs=1e-9)


def test_result_is_never_negative():
    p = top(a=0.5, b=0.5)
    q = [{"token": "a", "logprob": math.log(0.5) + 1e-15}, {"token": "b", "logprob": math.log(0.5)}]
    assert kl_divergence_from_top(p, q) >= 0.0


def test_empty_lists_are_rejected():
    with pytest.raises(ValueError):
        kl_divergence_from_top([], top(a=1.0))
    with pytest.raises(ValueError):
        kl_divergence_from_top(top(a=1.0), [])


def test_duplicate_tokens_are_merged():
    split = [{"token": "a", "logprob": math.log(0.25)}, {"token": "a", "logprob": math.log(0.25)},
             {"token": "b", "logprob": math.log(0.5)}]  # fmt: skip
    merged = top(a=0.5, b=0.5)
    assert kl_divergence_from_top(split, merged) == pytest.approx(0.0, abs=1e-9)


def test_logprob_warnings():
    assert logprob_warnings(None, 5) == ["no logprobs returned"]
    assert logprob_warnings(top(a=0.4, b=0.3), 5) == []
    over = logprob_warnings(top(a=0.9, b=0.9), 5)
    assert any("> 1" in w for w in over)
    truncated = logprob_warnings(top(a=0.7, b=0.3), 5)
    assert any("post-sampling" in w for w in truncated)
    assert logprob_warnings(top(a=0.7, b=0.3), 2) == []


def test_first_token_top_and_grade_probe():
    logprobs = [{"token": "A", "logprob": -0.1, "top": top(A=0.9, B=0.05)}]
    assert first_token_top(logprobs)[0]["token"] == "A"
    assert (
        first_token_top(None) is None
        and first_token_top([{"token": "x", "logprob": 0, "top": []}]) is None
    )
    graded = grade_kl_probe(logprobs, 2)
    assert graded["has_logprobs"] and graded["first_token"] == "A" and graded["warnings"] == []
    missing = grade_kl_probe(None, 2)
    assert not missing["has_logprobs"] and missing["warnings"]
    assert grade_kl_probe(logprobs, 2, ok=False)["warnings"] == ["request failed"]


def probe(profile, prompt, k=20):
    oracle = KLOracle()
    with MockServer(profile=profile, provider=oracle.lookup) as server:
        client = ChatClient(server.base_url, "m", retries=0)
        return client.chat(
            [{"role": "user", "content": prompt}], logprobs=True, top_logprobs=k, max_tokens=1
        )


def test_mock_probe_gives_zero_kl_for_identical_profiles_and_positive_for_shifted():
    same, shifted = [], []
    for prompt in HARMLESS_PROMPTS[:10]:
        a = probe(MockProfile(kl_shift=0.0), prompt).logprobs[0]["top"]
        b = probe(MockProfile(kl_shift=0.0), prompt).logprobs[0]["top"]
        c = probe(MockProfile(kl_shift=0.8), prompt).logprobs[0]["top"]
        same.append(kl_divergence_from_top(a, b))
        shifted.append(kl_divergence_from_top(a, c))
    assert max(same) == pytest.approx(0.0, abs=1e-12)
    assert min(shifted) > 0.0


def test_larger_shift_gives_larger_average_kl():
    def mean_kl(shift):
        values = []
        for prompt in HARMLESS_PROMPTS[:20]:
            a = probe(MockProfile(kl_shift=0.0), prompt).logprobs[0]["top"]
            b = probe(MockProfile(kl_shift=shift), prompt).logprobs[0]["top"]
            values.append(kl_divergence_from_top(a, b))
        return sum(values) / len(values)

    assert mean_kl(0.2) < mean_kl(0.6) < mean_kl(1.2)


def test_truncated_top_k_still_orders_correctly():
    prompt = HARMLESS_PROMPTS[0]
    a = probe(MockProfile(), prompt, k=5).logprobs[0]["top"]
    b = probe(MockProfile(kl_shift=1.0), prompt, k=5).logprobs[0]["top"]
    assert len(a) == 5 and kl_divergence_from_top(a, b) > 0.0


def test_oracle_requires_logprobs_and_known_prompt():
    oracle = KLOracle()
    known = {"messages": [{"role": "user", "content": HARMLESS_PROMPTS[0]}], "logprobs": True}
    assert oracle.lookup(known) is not None
    assert oracle.lookup({**known, "logprobs": False}) is None
    assert oracle.lookup({**known, "tools": [{"x": 1}]}) is None
    assert (
        oracle.lookup({"messages": [{"role": "user", "content": "other"}], "logprobs": True})
        is None
    )
    assert oracle.lookup({"messages": [], "logprobs": True}) is None
