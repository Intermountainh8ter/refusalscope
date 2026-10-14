import json

from refusalscope.textcalls import detect_text_calls


def test_plain_text_has_no_calls():
    assert detect_text_calls("The capital of France is Paris.") == []
    assert detect_text_calls("") == []
    assert detect_text_calls(None) == []


def test_tool_call_tags():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    calls = detect_text_calls(text)
    assert [(c.name, c.arguments) for c in calls] == [("get_weather", {"city": "Paris"})]
    assert calls[0].id.startswith("text_")


def test_multiple_tags():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call> and then '
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    assert [c.name for c in detect_text_calls(text)] == ["a", "b"]


def test_tag_with_json_array_and_noise():
    text = '<tool_call>[{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]</tool_call>'
    assert [c.name for c in detect_text_calls(text)] == ["a", "b"]
    noisy = '<tool_call>sure thing {"name": "a", "arguments": {"q": "}"}} done</tool_call>'
    assert detect_text_calls(noisy)[0].arguments == {"q": "}"}


def test_tag_with_unparseable_body_is_ignored():
    assert detect_text_calls("<tool_call>not json at all</tool_call>") == []


def test_fenced_json_and_bare_json():
    fenced = 'Here you go:\n```json\n{"name": "read_file", "parameters": {"path": "/a"}}\n```'
    assert detect_text_calls(fenced)[0].arguments == {"path": "/a"}
    bare = 'I will call {"name": "read_file", "arguments": {"path": "/b"}} now.'
    assert detect_text_calls(bare)[0].name == "read_file"


def test_openai_style_function_wrapper_and_string_arguments():
    wrapped = json.dumps({"function": {"name": "f", "arguments": '{"x": 1}'}})
    assert detect_text_calls(wrapped)[0].arguments == {"x": 1}
    broken = json.dumps({"name": "f", "arguments": "{oops"})
    call = detect_text_calls(broken)[0]
    assert call.arguments is None and call.parse_error is not None


def test_json_without_call_shape_is_not_a_call():
    assert detect_text_calls('{"status": "ok", "count": 3}') == []
    assert detect_text_calls('{"name": "Dana"}') == []
    assert detect_text_calls('{"name": 5, "arguments": {}}') == []


def test_python_style_calls_need_known_names():
    text = '[get_weather(city="Paris", days=3)]'
    assert detect_text_calls(text) == []
    calls = detect_text_calls(text, {"get_weather"})
    assert calls[0].arguments == {"city": "Paris", "days": 3}
    assert detect_text_calls("unknown_fn(a=1)", {"get_weather"}) == []
    assert detect_text_calls("get_weather(1, 2)", {"get_weather"}) == []
    assert detect_text_calls("get_weather(city=undefined_name)", {"get_weather"}) == []
    assert detect_text_calls("get_weather(city=", {"get_weather"}) == []


def test_string_aware_brace_matching():
    text = '{"name": "w", "arguments": {"text": "a } b { c"}}'
    assert detect_text_calls(text)[0].arguments == {"text": "a } b { c"}
    escaped = r'{"name": "w", "arguments": {"text": "say \"hi\" }"}}'
    assert detect_text_calls(escaped)[0].arguments == {"text": 'say "hi" }'}
