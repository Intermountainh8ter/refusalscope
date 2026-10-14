"""Study configuration: models, pairs, sampling and suite settings.

The configuration is a JSON file. Unknown keys are rejected so a typo cannot silently fall back
to a default. API keys are never stored; a model names an environment variable instead.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from refusalscope.util import canonical_json, sha256_text

ROLES = ("stock", "ablated")
SUITE_IDS = ("toolcall", "injection", "refusal", "kl", "capability")


class ConfigError(ValueError):
    """The configuration file is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    role: str
    pair_id: str
    model: str = ""
    api_key_env: str = ""
    params_b: float | None = None
    quant: str = ""
    extra_body: dict[str, Any] = field(default_factory=dict)
    mock: bool = False
    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0
    notes: str = ""

    @property
    def served_name(self) -> str:
        return self.model or self.name

    def api_key(self) -> str | None:
        """Resolve the API key from the environment; ``None`` if no variable is configured."""
        if not self.api_key_env:
            return None
        value = os.environ.get(self.api_key_env)
        if not value:
            raise ConfigError(
                f"model {self.name!r}: environment variable {self.api_key_env!r} is not set"
            )
        return value


@dataclass(frozen=True)
class SamplingConfig:
    temperatures: tuple[float, ...] = (0.0,)
    seeds: tuple[int, ...] = (1234,)
    n_samples: int | None = None
    max_tokens: int = 1024
    top_p: float = 1.0
    timeout_s: float = 120.0
    retries: int = 3
    parallel_tool_calls: bool | None = True

    def seeds_for(self, temperature: float) -> tuple[int, ...]:
        """Seeds to run at a temperature. Greedy decoding gets exactly one sample."""
        count = self.n_samples if self.n_samples is not None else len(self.seeds)
        chosen = self.seeds[:count]
        if temperature <= 0.0:
            return chosen[:1]
        return chosen

    @property
    def primary_temperature(self) -> float:
        return min(self.temperatures)


@dataclass(frozen=True)
class SuiteSettings:
    toolcall_seed: int = 1001
    toolcall_per_category: int = 22
    injection_seed: int = 2002
    injection_cases: int = 378
    injection_max_steps: int = 8
    refusal_path: str = ""
    refusal_max_items: int | None = None
    refusal_max_tokens: int = 300
    kl_top_logprobs: int = 20
    capability_seed: int = 3003
    capability_items: int = 100
    capability_max_tokens: int = 200


@dataclass(frozen=True)
class StudyConfig:
    study: str
    models: tuple[ModelConfig, ...]
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    suites: SuiteSettings = field(default_factory=SuiteSettings)
    output_dir: str = "runs"
    workers: int = 1

    def pair_ids(self) -> list[str]:
        seen: list[str] = []
        for model in self.models:
            if model.pair_id not in seen:
                seen.append(model.pair_id)
        return seen

    def pair(self, pair_id: str) -> tuple[ModelConfig, ModelConfig]:
        """Return ``(stock, ablated)`` for a pair id."""
        stock = next(m for m in self.models if m.pair_id == pair_id and m.role == "stock")
        ablated = next(m for m in self.models if m.pair_id == pair_id and m.role == "ablated")
        return stock, ablated

    def model(self, name: str) -> ModelConfig:
        for entry in self.models:
            if entry.name == name:
                return entry
        raise KeyError(name)

    @property
    def is_mock(self) -> bool:
        return bool(self.models) and all(m.mock for m in self.models)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict. Contains no secrets: only environment variable names."""
        data = asdict(self)
        data["sampling"]["temperatures"] = list(self.sampling.temperatures)
        data["sampling"]["seeds"] = list(self.sampling.seeds)
        data["models"] = [asdict(m) for m in self.models]
        data["suites"] = {
            section: {key: getattr(self.suites, attr) for key, attr in mapping.items()}
            for section, mapping in _SUITE_SECTION_KEYS.items()
        }
        return data

    def digest(self) -> str:
        """Hash of the whole configuration (no secrets)."""
        return sha256_text(canonical_json(self.to_dict()))

    def fingerprint(self) -> str:
        """Hash of only the settings that change results.

        Endpoints, timeouts, retries, worker counts, prices and notes are left out so a study can
        be resumed after a server moves to another port or machine.
        """
        data = self.to_dict()
        keep = ("name", "model", "role", "pair_id", "params_b", "quant", "extra_body", "mock")
        sampling = {k: v for k, v in data["sampling"].items() if k not in ("timeout_s", "retries")}
        suites = json.loads(json.dumps(data["suites"]))
        suites["refusal"].pop("harmful_path", None)
        payload = {
            "models": [{k: m[k] for k in keep} for m in data["models"]],
            "sampling": sampling,
            "suites": suites,
        }
        return sha256_text(canonical_json(payload))


# --------------------------------------------------------------------------- parsing

_MODEL_KEYS = {f for f in ModelConfig.__dataclass_fields__}
_SAMPLING_KEYS = {
    "temperatures",
    "seeds",
    "n_samples",
    "max_tokens",
    "top_p",
    "timeout_s",
    "retries",
    "parallel_tool_calls",
}
_SUITE_SECTION_KEYS: dict[str, dict[str, str]] = {
    "toolcall": {"seed": "toolcall_seed", "n_per_category": "toolcall_per_category"},
    "injection": {
        "seed": "injection_seed",
        "n_cases": "injection_cases",
        "max_steps": "injection_max_steps",
    },
    "refusal": {
        "harmful_path": "refusal_path",
        "max_items": "refusal_max_items",
        "max_tokens": "refusal_max_tokens",
    },
    "kl": {"top_logprobs": "kl_top_logprobs"},
    "capability": {
        "seed": "capability_seed",
        "n_items": "capability_items",
        "max_tokens": "capability_max_tokens",
    },
}
_TOP_KEYS = {"study", "models", "sampling", "suites", "output_dir", "workers"}


def _reject_unknown(section: str, given: dict[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(given) - allowed)
    if unknown:
        raise ConfigError(
            f"{section}: unknown key(s) {', '.join(map(repr, unknown))}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )


def _number(section: str, value: Any, *, minimum: float | None = None, integer: bool = False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}: expected a number, got {value!r}")
    if integer and int(value) != value:
        raise ConfigError(f"{section}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{section}: must be >= {minimum}, got {value!r}")
    return int(value) if integer else float(value)


def _parse_model(raw: Any, index: int) -> ModelConfig:
    where = f"models[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected an object")
    _reject_unknown(where, raw, _MODEL_KEYS)
    for required in ("name", "base_url", "role", "pair_id"):
        if not isinstance(raw.get(required), str) or not raw[required].strip():
            raise ConfigError(f"{where}: {required!r} is required and must be a non-empty string")
    if raw["role"] not in ROLES:
        raise ConfigError(f"{where}: role must be one of {ROLES}, got {raw['role']!r}")
    if not raw["base_url"].startswith(("http://", "https://")):
        raise ConfigError(f"{where}: base_url must start with http:// or https://")
    kwargs = dict(raw)
    if kwargs.get("params_b") is not None:
        kwargs["params_b"] = _number(f"{where}.params_b", kwargs["params_b"], minimum=1e-9)
    if "extra_body" in kwargs and not isinstance(kwargs["extra_body"], dict):
        raise ConfigError(f"{where}.extra_body: expected an object")
    for key in ("mock",):
        if key in kwargs and not isinstance(kwargs[key], bool):
            raise ConfigError(f"{where}.{key}: expected true or false")
    for key in ("price_per_1k_prompt", "price_per_1k_completion"):
        if key in kwargs:
            kwargs[key] = _number(f"{where}.{key}", kwargs[key], minimum=0.0)
    for key in ("model", "api_key_env", "quant", "notes"):
        if key in kwargs and not isinstance(kwargs[key], str):
            raise ConfigError(f"{where}.{key}: expected a string")
    return ModelConfig(**kwargs)


def _parse_sampling(raw: Any) -> SamplingConfig:
    if raw is None:
        return SamplingConfig()
    if not isinstance(raw, dict):
        raise ConfigError("sampling: expected an object")
    _reject_unknown("sampling", raw, _SAMPLING_KEYS)
    kwargs: dict[str, Any] = {}
    if "temperatures" in raw:
        temps = raw["temperatures"]
        if not isinstance(temps, list) or not temps:
            raise ConfigError("sampling.temperatures: expected a non-empty list")
        kwargs["temperatures"] = tuple(
            _number("sampling.temperatures", t, minimum=0.0) for t in temps
        )
    if "seeds" in raw:
        seeds = raw["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ConfigError("sampling.seeds: expected a non-empty list")
        kwargs["seeds"] = tuple(_number("sampling.seeds", s, integer=True) for s in seeds)
    if raw.get("n_samples") is not None:
        kwargs["n_samples"] = _number(
            "sampling.n_samples", raw["n_samples"], minimum=1, integer=True
        )
    if "max_tokens" in raw:
        kwargs["max_tokens"] = _number(
            "sampling.max_tokens", raw["max_tokens"], minimum=1, integer=True
        )
    if "top_p" in raw:
        kwargs["top_p"] = _number("sampling.top_p", raw["top_p"], minimum=0.0)
    if "timeout_s" in raw:
        kwargs["timeout_s"] = _number("sampling.timeout_s", raw["timeout_s"], minimum=0.001)
    if "retries" in raw:
        kwargs["retries"] = _number("sampling.retries", raw["retries"], minimum=0, integer=True)
    if "parallel_tool_calls" in raw:
        value = raw["parallel_tool_calls"]
        if value is not None and not isinstance(value, bool):
            raise ConfigError("sampling.parallel_tool_calls: expected true, false or null")
        kwargs["parallel_tool_calls"] = value
    sampling = SamplingConfig(**kwargs)
    if sampling.n_samples is not None and sampling.n_samples > len(sampling.seeds):
        raise ConfigError(
            f"sampling.n_samples ({sampling.n_samples}) exceeds the number of seeds "
            f"({len(sampling.seeds)}); add seeds"
        )
    return sampling


def _parse_suites(raw: Any) -> SuiteSettings:
    if raw is None:
        return SuiteSettings()
    if not isinstance(raw, dict):
        raise ConfigError("suites: expected an object")
    _reject_unknown("suites", raw, set(_SUITE_SECTION_KEYS))
    kwargs: dict[str, Any] = {}
    for section, mapping in _SUITE_SECTION_KEYS.items():
        body = raw.get(section)
        if body is None:
            continue
        if not isinstance(body, dict):
            raise ConfigError(f"suites.{section}: expected an object")
        _reject_unknown(f"suites.{section}", body, set(mapping))
        for key, attr in mapping.items():
            if key not in body or body[key] is None:
                continue
            value = body[key]
            if attr == "refusal_path":
                if not isinstance(value, str):
                    raise ConfigError("suites.refusal.harmful_path: expected a string")
                kwargs[attr] = value
            else:
                minimum = (
                    1
                    if attr not in ("toolcall_seed", "injection_seed", "capability_seed")
                    else None
                )
                kwargs[attr] = _number(
                    f"suites.{section}.{key}", value, minimum=minimum, integer=True
                )
    return SuiteSettings(**kwargs)


def _validate_pairs(models: tuple[ModelConfig, ...]) -> None:
    names = [m.name for m in models]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ConfigError(f"duplicate model name(s): {', '.join(duplicates)}")
    by_pair: dict[str, list[ModelConfig]] = {}
    for model in models:
        by_pair.setdefault(model.pair_id, []).append(model)
    for pair_id, members in by_pair.items():
        roles = sorted(m.role for m in members)
        if roles != ["ablated", "stock"]:
            raise ConfigError(
                f"pair {pair_id!r} must have exactly one stock and one ablated model, "
                f"found roles {roles}"
            )


def parse_config(raw: Any) -> StudyConfig:
    """Validate a decoded JSON document and build a :class:`StudyConfig`."""
    if not isinstance(raw, dict):
        raise ConfigError("the configuration must be a JSON object")
    _reject_unknown("config", raw, _TOP_KEYS)
    study = raw.get("study", "study")
    if not isinstance(study, str) or not study.strip():
        raise ConfigError("study: expected a non-empty string")
    raw_models = raw.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("models: expected a non-empty list")
    models = tuple(_parse_model(m, i) for i, m in enumerate(raw_models))
    _validate_pairs(models)
    output_dir = raw.get("output_dir", "runs")
    if not isinstance(output_dir, str) or not output_dir:
        raise ConfigError("output_dir: expected a non-empty string")
    workers = _number("workers", raw.get("workers", 1), minimum=1, integer=True)
    return StudyConfig(
        study=study,
        models=models,
        sampling=_parse_sampling(raw.get("sampling")),
        suites=_parse_suites(raw.get("suites")),
        output_dir=output_dir,
        workers=workers,
    )


def load_config(path: str | os.PathLike[str]) -> StudyConfig:
    """Read and validate a configuration file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
    return parse_config(raw)


def pair_warnings(config: StudyConfig) -> list[str]:
    """Differences between the two arms of a pair that could confound the comparison."""
    warnings: list[str] = []
    for pair_id in config.pair_ids():
        stock, ablated = config.pair(pair_id)
        if stock.quant != ablated.quant:
            warnings.append(
                f"pair {pair_id}: quantisation differs (stock {stock.quant!r}, "
                f"ablated {ablated.quant!r})"
            )
        if stock.extra_body != ablated.extra_body:
            warnings.append(f"pair {pair_id}: extra_body differs between arms")
        if stock.params_b != ablated.params_b:
            warnings.append(f"pair {pair_id}: params_b differs between arms")
        if stock.mock != ablated.mock:
            warnings.append(f"pair {pair_id}: only one arm is marked mock")
        if not stock.quant or not ablated.quant:
            warnings.append(f"pair {pair_id}: quant is not recorded; add it for the result file")
    return warnings
