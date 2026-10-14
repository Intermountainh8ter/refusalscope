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
