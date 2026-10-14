import copy
import json

import pytest

from refusalscope.config import (
    ConfigError,
    SamplingConfig,
    load_config,
    pair_warnings,
    parse_config,
)

BASE = {
    "study": "t",
    "models": [
        {
            "name": "m-stock",
            "base_url": "http://127.0.0.1:8080/v1",
            "role": "stock",
            "pair_id": "p1",
            "params_b": 4,
            "quant": "Q4_K_M",
        },
        {
            "name": "m-abl",
            "base_url": "http://127.0.0.1:8081/v1",
            "role": "ablated",
            "pair_id": "p1",
            "params_b": 4,
            "quant": "Q4_K_M",
        },
    ],
}


def cfg(**patch):
    raw = copy.deepcopy(BASE)
    raw.update(patch)
    return raw


def test_parse_minimal_config_uses_defaults():
    config = parse_config(cfg())
    assert config.study == "t"
    assert config.sampling.temperatures == (0.0,)
    assert config.sampling.seeds == (1234,)
    assert config.suites.toolcall_per_category == 22
    assert config.suites.injection_cases == 378
    assert config.workers == 1
    assert config.pair_ids() == ["p1"]
    stock, ablated = config.pair("p1")
    assert (stock.role, ablated.role) == ("stock", "ablated")
    assert stock.served_name == "m-stock"
    assert config.model("m-abl").params_b == 4.0


def test_served_name_prefers_model_field():
    raw = cfg()
    raw["models"][0]["model"] = "qwen3-4b"
    assert parse_config(raw).models[0].served_name == "qwen3-4b"


def test_model_lookup_unknown_raises():
    with pytest.raises(KeyError):
        parse_config(cfg()).model("nope")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c.update(bogus=1), "unknown key"),
        (lambda c: c.update(models=[]), "non-empty list"),
        (lambda c: c.update(study=""), "study"),
        (lambda c: c.update(output_dir=""), "output_dir"),
        (lambda c: c.update(workers=0), "workers"),
        (lambda c: c["models"][0].update(role="base"), "role must be one of"),
        (lambda c: c["models"][0].update(base_url="localhost:8080"), "base_url"),
        (lambda c: c["models"][0].update(bogus=1), "unknown key"),
        (lambda c: c["models"][0].pop("name"), "'name' is required"),
        (lambda c: c["models"][0].update(params_b=-1), "params_b"),
        (lambda c: c["models"][0].update(params_b="big"), "expected a number"),
        (lambda c: c["models"][0].update(extra_body=[1]), "extra_body"),
        (lambda c: c["models"][0].update(mock="yes"), "mock"),
        (lambda c: c["models"][0].update(quant=3), "quant"),
        (lambda c: c["models"][0].update(price_per_1k_prompt=-1), "price_per_1k_prompt"),
        (lambda c: c["models"][1].update(name="m-stock"), "duplicate model name"),
        (lambda c: c["models"][1].update(role="stock"), "exactly one stock and one ablated"),
        (lambda c: c["models"][1].update(pair_id="p2"), "exactly one stock and one ablated"),
        (lambda c: c.update(sampling=[1]), "sampling"),
        (lambda c: c.update(sampling={"bogus": 1}), "unknown key"),
        (lambda c: c.update(sampling={"temperatures": []}), "temperatures"),
        (lambda c: c.update(sampling={"temperatures": [-0.1]}), "temperatures"),
        (lambda c: c.update(sampling={"seeds": []}), "seeds"),
        (lambda c: c.update(sampling={"seeds": [1.5]}), "integer"),
        (
            lambda c: c.update(sampling={"seeds": [1], "n_samples": 2}),
            "exceeds the number of seeds",
        ),
        (lambda c: c.update(sampling={"max_tokens": 0}), "max_tokens"),
        (lambda c: c.update(sampling={"parallel_tool_calls": "yes"}), "parallel_tool_calls"),
        (lambda c: c.update(suites=[1]), "suites"),
        (lambda c: c.update(suites={"nope": {}}), "unknown key"),
        (lambda c: c.update(suites={"toolcall": 3}), "suites.toolcall"),
        (lambda c: c.update(suites={"toolcall": {"bogus": 1}}), "unknown key"),
        (lambda c: c.update(suites={"refusal": {"harmful_path": 3}}), "harmful_path"),
        (lambda c: c.update(suites={"injection": {"n_cases": 0}}), "n_cases"),
    ],
)
def test_invalid_configs_are_rejected(mutate, message):
    raw = cfg()
    mutate(raw)
    with pytest.raises(ConfigError, match=message):
        parse_config(raw)


def test_top_level_must_be_object():
    with pytest.raises(ConfigError, match="JSON object"):
        parse_config([1, 2])


def test_sampling_and_suite_overrides():
    raw = cfg(
        sampling={
            "temperatures": [0.0, 0.7],
            "seeds": [11, 22, 33],
            "n_samples": 2,
            "max_tokens": 256,
            "parallel_tool_calls": None,
        },
        suites={
            "toolcall": {"seed": 5, "n_per_category": 3},
            "injection": {"seed": 6, "n_cases": 14, "max_steps": 4},
            "refusal": {"harmful_path": "data/h.jsonl", "max_items": 50},
            "kl": {"top_logprobs": 10},
            "capability": {"seed": 7, "n_items": 20},
        },
        workers=4,
    )
    config = parse_config(raw)
    assert config.sampling.parallel_tool_calls is None
    assert config.sampling.max_tokens == 256
    assert config.suites.toolcall_seed == 5
    assert config.suites.injection_max_steps == 4
    assert config.suites.refusal_path == "data/h.jsonl"
    assert config.suites.refusal_max_items == 50
    assert config.suites.kl_top_logprobs == 10
    assert config.suites.capability_items == 20
    assert config.workers == 4


def test_seeds_for_greedy_uses_one_sample():
    sampling = SamplingConfig(temperatures=(0.0, 0.7), seeds=(11, 22, 33))
    assert sampling.seeds_for(0.0) == (11,)
    assert sampling.seeds_for(0.7) == (11, 22, 33)
    assert SamplingConfig(seeds=(1, 2, 3), n_samples=2).seeds_for(0.7) == (1, 2)
    assert sampling.primary_temperature == 0.0


def test_digest_is_stable_and_sensitive():
    first = parse_config(cfg())
    assert first.digest() == parse_config(cfg()).digest()
    other = parse_config(cfg(workers=2))
    assert first.digest() != other.digest()


def test_to_dict_is_json_serialisable_and_has_no_key_material(monkeypatch):
    monkeypatch.setenv("SECRET_ENV", "super-secret-value")
    raw = cfg()
    raw["models"][0]["api_key_env"] = "SECRET_ENV"
    dumped = json.dumps(parse_config(raw).to_dict())
    assert "SECRET_ENV" in dumped
    assert "super-secret-value" not in dumped


def test_api_key_resolution(monkeypatch):
    raw = cfg()
    raw["models"][0]["api_key_env"] = "RS_TEST_KEY"
    model = parse_config(raw).models[0]
    monkeypatch.delenv("RS_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="RS_TEST_KEY"):
        model.api_key()
    monkeypatch.setenv("RS_TEST_KEY", "abc")
    assert model.api_key() == "abc"
    assert parse_config(cfg()).models[0].api_key() is None


def test_pair_warnings_detect_confounds():
    raw = cfg()
    raw["models"][1]["quant"] = "Q8_0"
    raw["models"][1]["extra_body"] = {"x": 1}
    raw["models"][1]["params_b"] = 8
    text = " | ".join(pair_warnings(parse_config(raw)))
    assert "quantisation differs" in text
    assert "extra_body differs" in text
    assert "params_b differs" in text


def test_pair_warnings_missing_quant_and_mock_mismatch():
    raw = cfg()
    raw["models"][0]["quant"] = ""
    raw["models"][0]["mock"] = True
    text = " | ".join(pair_warnings(parse_config(raw)))
    assert "quant is not recorded" in text
    assert "only one arm is marked mock" in text
    assert pair_warnings(parse_config(cfg())) == []


def test_is_mock_requires_all_models_mock():
    raw = cfg()
    assert parse_config(raw).is_mock is False
    for model in raw["models"]:
        model["mock"] = True
    assert parse_config(raw).is_mock is True


def test_load_config_errors(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(bad)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(cfg()), encoding="utf-8")
    assert load_config(good).study == "t"
