from __future__ import annotations

from collections.abc import Callable

import pytest

from refusalscope.client import ChatClient
from refusalscope.mock import MockProfile, MockServer, Scenario


@pytest.fixture
def make_server():
    """Factory for mock servers that are always stopped at teardown."""
    started: list[MockServer] = []

    def factory(
        profile: MockProfile | None = None,
        scenarios: dict[str, Scenario] | None = None,
        provider: Callable[[dict], Scenario | None] | None = None,
        **kwargs,
    ) -> MockServer:
        table = scenarios or {}

        def lookup(body: dict) -> Scenario | None:
            for message in body.get("messages", []):
                if message.get("role") == "user":
                    return table.get(message.get("content"))
            return None

        server = MockServer(profile=profile, provider=provider or lookup, **kwargs).start()
        started.append(server)
        return server

    yield factory
    for server in started:
        server.stop()


@pytest.fixture
def make_client():
    def factory(server: MockServer, **kwargs) -> ChatClient:
        kwargs.setdefault("sleep", lambda _s: None)
        kwargs.setdefault("retries", 0)
        return ChatClient(server.base_url, "mock-model", **kwargs)

    return factory


# --------------------------------------------------------------------------- study helpers

import json  # noqa: E402

from refusalscope.analysis import analyze_run  # noqa: E402
from refusalscope.mock import PROFILES  # noqa: E402
from refusalscope.runner import Runner, resolve_suites  # noqa: E402
from refusalscope.suites import mock_provider  # noqa: E402
from tests.helpers import PLACEHOLDERS, make_study  # noqa: E402


@pytest.fixture(scope="session")
def provider():
    """One scenario provider for all suites, with placeholder text standing in for the behaviours file."""
    return mock_provider(PLACEHOLDERS)


@pytest.fixture(scope="session")
def behaviours_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("behaviours") / "placeholders.jsonl"
    path.write_text(
        "\n".join(json.dumps({"prompt": p}) for p in PLACEHOLDERS) + "\n", encoding="utf-8"
    )
    return path


@pytest.fixture
def run_study(tmp_path, provider, behaviours_file):
    """Run a pair of mock profiles through the real runner and analysis; return everything."""
    servers: list[MockServer] = []

    def factory(
        stock="reference", ablated="degraded", limit=60, suites="all", n_boot=500, **cfg_overrides
    ):
        profiles = {"stock": PROFILES[stock] if isinstance(stock, str) else stock,
                    "ablated": PROFILES[ablated] if isinstance(ablated, str) else ablated}  # fmt: skip
        started = {}
        for arm, profile in profiles.items():
            server = MockServer(profile=profile, provider=provider).start()
            servers.append(server)
            started[arm] = server
        config = make_study(
            {"stock-m": started["stock"].base_url, "ablated-m": started["ablated"].base_url},
            behaviours_file,
            **cfg_overrides,
        )
        run_dir = tmp_path / "runs" / "MOCK-unit"
        runner = Runner(config, run_dir, workers=4, progress=lambda m: None, run_id="MOCK-unit")
        summary = runner.run(resolve_suites(suites), limit=limit)
        results, path = analyze_run(run_dir, tmp_path / "results", n_boot=n_boot)
        return {"results": results, "path": path, "run_dir": run_dir, "config": config,
                "summary": summary, "servers": started, "runner": runner}  # fmt: skip

    yield factory
    for server in servers:
        server.stop()
