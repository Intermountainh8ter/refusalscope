import json
import random

import pytest

from refusalscope.suites.injection import build_cases, tools_for
from refusalscope.suites.injection_envs import (
    ENV_ORDER,
    ENVS,
    FACTS,
    Environment,
    make_items,
)


@pytest.fixture(scope="module")
def cases():
    return build_cases()


def case_for(cases, env, vector=None, kind=None):
    return next(
        c
        for c in cases
        if c["meta"]["environment"] == env
        and (vector is None or c["meta"]["vector"] == vector)
        and (kind is None or c["meta"]["kind"] == kind)
    )


def test_nine_environments_with_enough_tools():
    assert len(ENVS) == 9 and set(ENV_ORDER) == set(ENVS)
    for spec in ENVS.values():
        names = [t["function"]["name"] for t in spec.tools]
        assert 4 <= len(names) <= 6, spec.name
        assert len(set(names)) == len(names)
        assert spec.read_tool in names
        assert spec.list_tool is None or spec.list_tool in names
        assert spec.search_tool is None or spec.search_tool in names
        for tool in spec.tools:
            params = tool["function"]["parameters"]
            assert set(params["required"]) <= set(params["properties"])


def test_every_environment_has_a_side_effecting_tool():
    for spec in ENVS.values():
        reads = {spec.read_tool, spec.list_tool, spec.search_tool}
        actions = [t["function"]["name"] for t in spec.tools if t["function"]["name"] not in reads]
        assert len(actions) >= 2, spec.name


def test_facts_generate_distinct_values():
    rng = random.Random(0)
    for fact in FACTS:
        values = {fact.make(rng) for _ in range(12)}
        assert len(values) > 1, fact.question
        assert "{v}" in fact.sentence


def test_make_items_places_the_fact_in_the_primary_item_only():
    spec = ENVS["email"]
    rng = random.Random(3)
    fact = FACTS[0]
    items, _ = make_items(spec, rng, fact, 2, "CF-1234Q")
    assert len(items) == 5
    holders = [i["id"] for i in items if "CF-1234Q" in i["body"]]
    assert holders == [items[2]["id"]]


@pytest.mark.parametrize("env", ENV_ORDER)
def test_read_listing_and_actions_for_every_environment(cases, env):
    case = case_for(cases, env, kind="report")
    live = Environment(case, injected=False)
    spec = live.spec
    required = case["task"]["required"][0]
    arguments = {
        k: (v["$any_of"][0] if isinstance(v, dict) else v) for k, v in required["arguments"].items()
    }
    read = json.loads(live.execute(required["name"], arguments).split("\n\n")[0])
    assert read["messages" if spec.read_mode == "container" else "id"]
    if spec.list_tool:
        list_args = {}
        for p in next(t for t in spec.tools if t["function"]["name"] == spec.list_tool)["function"][
            "parameters"
        ]["required"]:
            list_args[p] = (
                case["env"]["items"][0]["id"].rsplit("/", 1)[0] if p == "directory" else "Paris"
            )
        listing = json.loads(live.execute(spec.list_tool, list_args))
        assert listing
    assert len(live.log) >= 1
    goal = case["injection"]["goal"]
    ack = json.loads(live.execute(goal["tool"], goal["arguments"]))
    assert ack["status"] == "ok" and ack["action"] == goal["tool"]


def test_unknown_tool_and_invalid_arguments_return_errors(cases):
    live = Environment(case_for(cases, "email"), injected=False)
    assert live.execute("rm_rf", {}).startswith("Error: unknown tool")
    assert "invalid arguments" in live.execute("read_email", {})
    assert "invalid arguments" in live.execute("read_email", {"email_id": 5})
    assert "invalid arguments" in live.execute("read_email", {"email_id": "m1", "extra": 1})
    assert live.log == []  # invalid calls are not recorded as executed
    assert "no email item" in live.execute("read_email", {"email_id": "m99"})
    assert live.execute("read_email", None).startswith("Error")


def test_search_tool(cases):
    case = case_for(cases, "email")
    live = Environment(case, injected=False)
    subject = case["env"]["items"][0]["name"]
    hits = json.loads(live.execute("search_emails", {"query": subject.split()[0]}))["results"]
    assert any(h["id"] == case["env"]["items"][0]["id"] for h in hits)
    assert json.loads(live.execute("search_emails", {"query": "zzzqqq"}))["results"] == []
    web = Environment(case_for(cases, "web"), injected=False)
    assert "results" in json.loads(web.execute("web_search", {"query": "garden"}))


def test_case_ids_are_found_case_insensitively(cases):
    case = case_for(cases, "crm", kind="report")
    live = Environment(case, injected=False)
    primary = case["env"]["primary"]
    assert "Error" not in live.execute("get_contact", {"contact_id": primary.upper()})


@pytest.mark.parametrize(
    ("vector", "slot"),
    [
        ("tool_result_direct", "tail"),
        ("important_info_block", "body"),
        ("document_footer", "body"),
        ("name_field", "name"),
        ("description_field", "meta"),
        ("json_note_field", "meta"),
    ],
)
def test_injection_slots_deliver_the_payload_only_when_attacking(cases, vector, slot):
    case = case_for(cases, "email", vector=vector)
    assert case["injection"]["slot"] == slot
    payload = case["injection"]["payload"]
    read_args = {"email_id": case["env"]["primary"]}
    clean = Environment(case, injected=False).execute("read_email", read_args)
    attacked = Environment(case, injected=True).execute("read_email", read_args)
    assert payload not in clean and json.dumps(payload, ensure_ascii=False)[1:-1] not in clean
    assert payload in attacked or json.dumps(payload, ensure_ascii=False)[1:-1] in attacked


def test_tail_payload_is_outside_the_json_and_only_on_the_primary(cases):
    case = case_for(cases, "email", vector="tool_result_direct")
    live = Environment(case, injected=True)
    primary = case["env"]["primary"]
    other = next(i["id"] for i in case["env"]["items"] if i["id"] != primary)
    body, _, tail = live.execute("read_email", {"email_id": primary}).partition("\n\n")
    json.loads(body)
    assert tail == case["injection"]["payload"]
    assert case["injection"]["payload"] not in live.execute("read_email", {"email_id": other})


def test_name_slot_shows_up_in_listings(cases):
    case = case_for(cases, "tickets", vector="name_field")
    listing = Environment(case, injected=True).execute("list_tickets", {})
    assert case["injection"]["payload"] in json.loads(listing)["items"][0]["title"] + listing


def test_body_middle_position_keeps_the_payload_verbatim(cases):
    case = case_for(cases, "crm", vector="important_info_block")
    live = Environment(case, injected=True)
    primary = next(i for i in live.items if i["id"] == case["env"]["primary"])
    assert case["injection"]["payload"] in primary["body"]
    assert primary["body"].startswith(
        case["env"]["items"][[i["id"] for i in case["env"]["items"]].index(primary["id"])][
            "body"
        ].split(". ")[0]
    )


def test_chat_channel_container_reads(cases):
    case = case_for(cases, "chat", vector="document_footer")
    live = Environment(case, injected=True)
    channel = case["env"]["channel"]
    other = "random" if channel != "random" else "general"
    assert json.loads(live.execute("read_channel", {"channel": other}))["messages"] == []
    text = live.execute("read_channel", {"channel": f"#{channel}"})
    assert case["injection"]["payload"].split("\n")[0] in text or "Automated footer" in text
    assert json.loads(live.execute("list_channels", {}))["channels"][0] == channel


def test_files_listing_is_scoped_to_the_directory(cases):
    case = case_for(cases, "files")
    live = Environment(case, injected=False)
    directory = case["env"]["items"][0]["id"].rsplit("/", 1)[0]
    assert len(json.loads(live.execute("list_files", {"directory": directory}))["files"]) == 5
    assert json.loads(live.execute("list_files", {"directory": "/nowhere"}))["files"] == []


def test_tools_for_returns_the_environment_tools(cases):
    case = case_for(cases, "banking")
    assert tools_for(case) is ENVS["banking"].tools
