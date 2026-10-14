import json
from pathlib import Path

import pytest

from refusalscope.client import ChatClient
from refusalscope.mock import MockProfile, MockServer
from refusalscope.suites import (
    SUITE_FILES,
    mock_provider,
    rows_digest,
    snapshot_builders,
    suite_hashes,
)
from refusalscope.suites.capability import build_items
from refusalscope.suites.refusal import BORDERLINE_BENIGN, classify_refusal
from refusalscope.util import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDERS = [f"Test fixture request number {i} (not a real behaviour)" for i in range(30)]


def ask(client, text):
    return client.chat([{"role": "user", "content": text}], seed=1).content


def test_refusal_probe_separates_a_refusing_mock_from_a_complying_one():
    provider = mock_provider(PLACEHOLDERS)
    refusing = MockProfile(p_refuse_harmful=1.0, p_refuse_benign=0.0)
    complying = MockProfile(p_refuse_harmful=0.0, p_refuse_benign=0.0)
    rates = {}
    for name, profile in (("refusing", refusing), ("complying", complying)):
        with MockServer(profile=profile, provider=provider) as server:
            client = ChatClient(server.base_url, "m", retries=0)
            harmful = [classify_refusal(ask(client, p))["label"] == "refusal" for p in PLACEHOLDERS]
            benign = [
                classify_refusal(ask(client, p))["label"] == "refusal" for p in BORDERLINE_BENIGN
            ]
        rates[name] = (sum(harmful) / len(harmful), sum(benign) / len(benign))
    assert rates["refusing"] == (1.0, 0.0)
    assert rates["complying"] == (0.0, 0.0)


def test_over_refusal_is_visible_on_the_benign_set():
    provider = mock_provider(PLACEHOLDERS)
    with MockServer(profile=MockProfile(p_refuse_benign=1.0), provider=provider) as server:
        client = ChatClient(server.base_url, "m", retries=0)
        flags = [classify_refusal(ask(client, p))["label"] == "refusal" for p in BORDERLINE_BENIGN]
    assert all(flags)


def test_one_provider_serves_all_suites():
    provider = mock_provider()
    items = build_items()
    assert provider({"messages": [{"role": "user", "content": items[0]["prompt"]}]}) is not None
    assert provider({"messages": [{"role": "user", "content": "unknown"}]}) is None


def test_snapshots_exist_and_match_generators():
    assert set(SUITE_FILES) == set(snapshot_builders())
    for name, build in snapshot_builders().items():
        committed = list(read_jsonl(ROOT / "suites" / SUITE_FILES[name]))
        assert rows_digest(committed) == rows_digest(json.loads(json.dumps(build()))), name


def test_suite_hashes_are_stable_hex_digests():
    first = suite_hashes()
    assert set(first) == set(SUITE_FILES)
    assert all(len(h) == 64 for h in first.values())
    assert suite_hashes() == first


@pytest.mark.parametrize("name", ["borderline", "kl", "capability"])
def test_suite_c_snapshots_have_expected_sizes(name):
    expected = {"borderline": 40, "kl": 60, "capability": 100}[name]
    assert len(list(read_jsonl(ROOT / "suites" / SUITE_FILES[name]))) == expected
