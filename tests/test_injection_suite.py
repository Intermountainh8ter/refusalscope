import collections
import json
from pathlib import Path

import pytest

from refusalscope.client import ChatClient, ChatResult, ToolCall, parse_completion
from refusalscope.mock import MockProfile, MockServer
from refusalscope.schema import materialize
from refusalscope.suites import rows_digest
from refusalscope.suites.injection import (
    CONDITIONS,
    SYSTEM_PROMPT,
    VECTOR_ORDER,
    VECTORS,
    InjectionOracle,
    agent_scenario,
    build_cases,
    case_key,
    goal_matches,
    grade_injection,
    run_case,
    tools_for,
)
from refusalscope.suites.injection_envs import ENV_ORDER, ENVS
from refusalscope.turn import Turn
from refusalscope.util import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cases():
    return build_cases()


def chat_result(content=None, calls=(), ok=True, error=None):
    """A ChatResult as the client would produce it for a scripted reply."""
    if not ok:
        return ChatResult(ok=False, error=error or "boom", request={})
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls)
        ]
    body = {"choices": [{"message": message, "finish_reason": "tool_calls" if calls else "stop"}]}
    return ChatResult(ok=True, request={}, response=body, **parse_completion(body))


def concrete(spec):
    return (spec["name"], materialize(spec["arguments"]))


class ScriptedAgent:
    """Replays a fixed sequence of replies, one per request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def __call__(self, messages, tools):
        self.seen.append((list(messages), tools))
        return self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]


def good_agent(case):
    required = [concrete(s) for s in case["task"]["required"]]
    replies = [chat_result(calls=[call]) for call in required]
    answer = " ".join(case["task"]["answer_contains"]) or "Done."
    replies.append(chat_result(content=f"Result: {answer}"))
    return ScriptedAgent(replies)


def grade(case, transcript):
    return grade_injection(case, [Turn.from_record(t) for t in transcript["turns"]])


# ------------------------------------------------------------------ generator


def test_suite_size_and_balance(cases):
    assert len(cases) == 378
    assert collections.Counter(c["meta"]["environment"] for c in cases) == dict.fromkeys(
        ENV_ORDER, 42
    )
    assert collections.Counter(c["meta"]["vector"] for c in cases) == dict.fromkeys(
        VECTOR_ORDER, 27
    )
    combos = collections.Counter((c["meta"]["environment"], c["meta"]["vector"]) for c in cases)
    assert len(combos) == 126 and set(combos.values()) == {3}
    assert collections.Counter(c["task"]["kind"] for c in cases) == {"report": 189, "act": 189}


def test_requirements_on_environments_and_vectors():
    assert len(ENVS) >= 8 and len(VECTORS) >= 10
    slots = {slot for slot, *_ in VECTORS.values()}
    assert slots == {"tail", "body", "name", "meta"}
    expected_styles = {"ignore_previous", "system_notice", "document_body_inline", "name_field"}
    assert expected_styles <= set(VECTORS)


def test_generation_is_deterministic_and_seed_sensitive(cases):
    assert build_cases() == cases
    assert build_cases(seed=5)[:3] != cases[:3]
    assert len(build_cases(n_cases=20)) == 20


def test_prompts_are_unique(cases):
    keys = [case_key(c) for c in cases]
    assert len(set(keys)) == len(keys)
    assert len({c["id"] for c in cases}) == 378


def test_goals_use_environment_tools_and_are_self_consistent(cases):
    for case in cases:
        names = {t["function"]["name"] for t in tools_for(case)}
        goal = case["injection"]["goal"]
        assert goal["tool"] in names, case["id"]
        call = ToolCall("x", goal["tool"], json.dumps(goal["arguments"]), goal["arguments"], None)
        assert goal_matches(goal, call), case["id"]
        for spec in case["task"]["required"]:
            assert spec["name"] in names, case["id"]
        assert goal["instruction"].casefold() in case["injection"]["payload"].casefold()


def test_attacker_values_are_clearly_fake(cases):
    blob = json.dumps([c["injection"]["goal"] for c in cases])
    assert "evil.example" in blob or "ZZ99EVIL" in blob or "mallory" in blob
    for case in cases:
        value = json.dumps(case["injection"]["goal"]["arguments"])
        assert not any(real in value for real in ("gmail.com", "paypal", "@outlook"))


def test_every_case_task_is_completable_by_a_scripted_agent(cases):
    for case in cases[::7]:
        agent = good_agent(case)
        transcript = run_case(case, agent, "control")
        grade_ = grade(case, transcript)
        assert grade_["utility"] and grade_["tool_use"] and not grade_["asr"], case["id"]


def test_answers_can_only_be_found_by_reading_the_primary_item(cases):
    for case in cases:
        env_items = case["env"]["items"]
        wanted = case["task"]["answer_contains"]
        if not wanted:
            continue
        holders = [i for i in env_items if wanted[0] in i["body"]]
        assert [h["id"] for h in holders] == [case["env"]["primary"]], case["id"]


def test_snapshot_matches_generator():
    committed = list(read_jsonl(ROOT / "suites" / "injection.jsonl"))
    regenerated = json.loads(json.dumps(build_cases()))
    assert rows_digest(committed) == rows_digest(regenerated)


# ------------------------------------------------------------------ agent loop


def test_run_case_rejects_unknown_condition(cases):
    with pytest.raises(ValueError, match="condition"):
        run_case(cases[0], ScriptedAgent([chat_result(content="x")]), "both")


def test_control_condition_never_exposes_the_payload(cases):
    for case in cases[::9]:
        agent = good_agent(case)
        run_case(case, agent, "control")
        seen = json.dumps([m for messages, _ in agent.seen for m in messages])
        escaped = json.dumps(case["injection"]["payload"], ensure_ascii=False)[1:-1]
        assert escaped not in seen and case["injection"]["payload"] not in seen


def test_attack_condition_exposes_the_payload_to_the_model(cases):
    for case in cases[::9]:
        agent = good_agent(case)
        run_case(case, agent, "attack")
        tool_text = "\n".join(
            m["content"] for messages, _ in agent.seen for m in messages if m["role"] == "tool"
        )
        escaped = json.dumps(case["injection"]["payload"], ensure_ascii=False)[1:-1]
        assert case["injection"]["payload"] in tool_text or escaped in tool_text, case["id"]


def test_loop_feeds_tool_results_back_and_stops_on_text(cases):
    case = cases[0]
    agent = good_agent(case)
    transcript = run_case(case, agent, "control")
    assert len(transcript["turns"]) == len(case["task"]["required"]) + 1
    roles = [m["role"] for m in agent.seen[-1][0]]
    assert roles[:2] == ["system", "user"] and "tool" in roles
    assert agent.seen[0][0][0]["content"] == SYSTEM_PROMPT
    assert [e["name"] for e in transcript["executed"]] == [
        s["name"] for s in case["task"]["required"]
    ]


def test_loop_stops_at_step_limit(cases):
    case = cases[0]
    name, args = concrete(case["task"]["required"][0])
    endless = ScriptedAgent([chat_result(calls=[(name, args)])])
    transcript = run_case(case, endless, "control", max_steps=3)
    assert len(transcript["turns"]) == 3
    assert grade(case, transcript)["hit_step_limit"]


def test_loop_stops_on_request_error_and_counts_it(cases):
    case = cases[0]
    transcript = run_case(case, ScriptedAgent([chat_result(ok=False)]), "control")
    result = grade(case, transcript)
    assert result["errors"] == 1 and not result["utility"] and not result["tool_use"]


def test_malformed_arguments_are_reported_to_the_model(cases):
    case = cases[0]
    body = {
        "choices": [
            {"message": {"content": None, "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "read_email", "arguments": "{bad"}}
            ]}, "finish_reason": "tool_calls"}
        ]
    }  # fmt: skip
    bad = ChatResult(ok=True, request={}, response=body, **parse_completion(body))
    agent = ScriptedAgent([bad, chat_result(content="sorry")])
    run_case(case, agent, "control")
    tool_message = next(m for m in agent.seen[1][0] if m["role"] == "tool")
    assert "could not parse arguments" in tool_message["content"]


# ------------------------------------------------------------------ grader


def attack_call(case):
    goal = case["injection"]["goal"]
    return (goal["tool"], goal["arguments"])


def test_attack_success_requires_the_attacker_values(cases):
    case = cases[0]
    agent = ScriptedAgent(
        [chat_result(calls=[concrete(case["task"]["required"][0])]), chat_result(calls=[attack_call(case)]),
         chat_result(content="Done.")]
    )  # fmt: skip
    result = grade(case, run_case(case, agent, "attack"))
    assert result["asr"] and result["asr_incl_text"] and result["tool_use"]


def test_same_function_with_other_values_is_not_an_attack(cases):
    case = next(c for c in cases if c["injection"]["goal"]["tool"] == "send_email")
    name, args = attack_call(case)
    legit = {**args, "to": "dana.okafor@example.com"}
    agent = ScriptedAgent([chat_result(calls=[(name, legit)]), chat_result(content="ok")])
    assert not grade(case, run_case(case, agent, "attack"))["asr"]


def test_text_form_attack_counts_only_in_the_sensitivity_metric(cases):
    case = cases[0]
    name, args = attack_call(case)
    text = f"<tool_call>{json.dumps({'name': name, 'arguments': args})}</tool_call>"
    result = grade(case, run_case(case, ScriptedAgent([chat_result(content=text)]), "attack"))
    assert not result["asr"] and result["asr_incl_text"] and result["text_tool_call"]


def test_utility_needs_the_answer_for_report_tasks(cases):
    case = next(c for c in cases if c["task"]["answer_contains"])
    required = concrete(case["task"]["required"][0])
    wrong = ScriptedAgent(
        [chat_result(calls=[required]), chat_result(content="I could not find it.")]
    )
    result = grade(case, run_case(case, wrong, "control"))
    assert result["tool_use"] and not result["utility"]
    right = good_agent(case)
    assert grade(case, run_case(case, right, "control"))["utility"]


def test_answer_matching_is_case_insensitive(cases):
    case = next(c for c in cases if c["task"]["answer_contains"])
    required = concrete(case["task"]["required"][0])
    shout = case["task"]["answer_contains"][0].upper()
    agent = ScriptedAgent([chat_result(calls=[required]), chat_result(content=f"It is {shout}.")])
    assert grade(case, run_case(case, agent, "control"))["utility"]


def test_act_tasks_need_the_action_call(cases):
    case = next(c for c in cases if c["task"]["kind"] == "act")
    read_only = ScriptedAgent(
        [chat_result(calls=[concrete(case["task"]["required"][0])]), chat_result(content="ok")]
    )
    assert not grade(case, run_case(case, read_only, "control"))["tool_use"]
    assert grade(case, run_case(case, good_agent(case), "control"))["utility"]


def test_empty_transcript_scores_zero(cases):
    result = grade_injection(cases[0], [])
    assert not result["utility"] and not result["asr"] and result["n_turns"] == 0


def test_final_text_after_tool_calls_is_the_only_answer(cases):
    case = next(c for c in cases if c["task"]["answer_contains"])
    fact = case["task"]["answer_contains"][0]
    required = concrete(case["task"]["required"][0])
    agent = ScriptedAgent([chat_result(content=f"{fact}", calls=[required])])
    result = grade(case, run_case(case, agent, "control", max_steps=1))
    assert (
        result["tool_use"] and not result["utility"]
    )  # the fact was mid-loop text, never a final answer


# ------------------------------------------------------------------ mock end to end


def run_all(cases, profile, conditions=CONDITIONS):
    oracle = InjectionOracle(cases)
    out = {c: [] for c in conditions}
    with MockServer(profile=profile, provider=oracle.lookup) as server:
        client = ChatClient(server.base_url, "mock-model", retries=0)
        for case in cases:
            for condition in conditions:

                def chat(messages, tools):
                    return client.chat(messages, tools=tools, seed=1)

                transcript = run_case(case, chat, condition)
                out[condition].append(grade(case, transcript))
    return out


def test_oracle_returns_none_for_unknown_requests(cases):
    oracle = InjectionOracle(cases)
    assert oracle.lookup({"messages": [], "tools": []}) is None
    odd = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"function": {"name": "x"}}],
    }
    assert oracle.lookup(odd) is None
    scenario = agent_scenario(cases[0])
    assert scenario.attacker_call["name"] == cases[0]["injection"]["goal"]["tool"]
    assert agent_scenario(next(c for c in cases if c["task"]["kind"] == "act")).answer.startswith(
        "Done"
    )


def test_mock_never_obeys_when_probability_zero(cases):
    result = run_all(cases[::6], MockProfile(p_inject=0.0))
    assert not any(g["asr"] for g in result["attack"] + result["control"])
    assert all(g["utility"] for g in result["control"] + result["attack"])


def test_mock_always_obeys_when_probability_one_and_control_stays_clean(cases):
    result = run_all(cases[::6], MockProfile(p_inject=1.0))
    assert all(g["asr"] for g in result["attack"])
    assert not any(g["asr"] for g in result["control"])  # grader sanity: asr_control == 0
    assert all(g["utility"] for g in result["attack"])


def test_mock_attack_rate_tracks_probability(cases):
    result = run_all(cases, MockProfile(p_inject=0.3, seed=21), conditions=("attack",))
    rate = sum(g["asr"] for g in result["attack"]) / len(cases)
    assert 0.22 <= rate <= 0.38


def test_mock_text_form_attacks_are_only_counted_in_the_sensitivity_metric(cases):
    result = run_all(
        cases[::9], MockProfile(p_inject=1.0, p_attack_text=1.0), conditions=("attack",)
    )
    assert not any(g["asr"] for g in result["attack"])
    assert all(g["asr_incl_text"] for g in result["attack"])


def test_mock_task_failures_lower_utility_equally_in_both_conditions(cases):
    result = run_all(cases[::3], MockProfile(p_inject=0.0, p_task_success=0.5, seed=4))
    control = [g["utility"] for g in result["control"]]
    attack = [g["utility"] for g in result["attack"]]
    assert control == attack and 0.3 < sum(control) / len(control) < 0.7
