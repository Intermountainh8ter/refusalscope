"""Resumable study runner with full JSONL trajectory logging.

Every finished job appends one line to ``<run_dir>/trajectories.jsonl`` containing the complete
request and response bodies of every request it made, latencies, token usage and the grade
computed at run time. A crash can lose at most the job in flight: on restart the runner drops a
truncated final line, skips jobs that already finished and retries jobs whose requests failed.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from refusalscope import __version__
from refusalscope.client import MOCK_HEADER, ChatClient
from refusalscope.config import ModelConfig, StudyConfig
from refusalscope.schema import LENIENT, STRICT
from refusalscope.suites import capability, injection, kl, refusal, suite_hashes, toolcall
from refusalscope.turn import Turn
from refusalscope.util import stable_hash, stable_rng, utc_now_iso

SUITES = ("toolcall", "injection", "refusal", "kl", "capability")
SUITE_ALIASES: dict[str, tuple[str, ...]] = {"all": SUITES, "ablation": ("refusal", "kl")}
TRAJECTORY_FILE = "trajectories.jsonl"
RUN_FILE = "run.json"
SCHEMA = 1
BREAKER_DEFAULT = 10


class RunError(RuntimeError):
    """The run cannot proceed (mismatched configuration, unreachable model, corrupt log)."""


def resolve_suites(requested: str | list[str]) -> list[str]:
    """Expand ``all``/``ablation`` aliases into an ordered, duplicate-free suite list."""
    names = [requested] if isinstance(requested, str) else list(requested)
    out: list[str] = []
    for name in names:
        for suite in SUITE_ALIASES.get(name, (name,)):
            if suite not in SUITES:
                raise RunError(
                    f"unknown suite {name!r}; choose from {', '.join([*SUITES, *SUITE_ALIASES])}"
                )
            if suite not in out:
                out.append(suite)
    return out


@dataclass(frozen=True)
class Job:
    """One unit of work: a case, run on a model, at a temperature and seed, in a condition."""

    job_id: str
    suite: str
    case_id: str
    model: str
    temperature: float
    seed: int
    condition: str


@dataclass
class SuiteData:
    """All cases for a study, built from the configuration's suite settings."""

    toolcall: dict[str, dict[str, Any]]
    injection: dict[str, dict[str, Any]]
    refusal: dict[str, dict[str, Any]]
    kl: dict[str, dict[str, Any]]
    capability: dict[str, dict[str, Any]]
    harmful: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: StudyConfig) -> SuiteData:
        s = config.suites
        notes: list[str] = []
        items: dict[str, dict[str, Any]] = {}
        harmful: dict[str, Any] | None = None
        if s.refusal_path:
            loaded, harmful = refusal.load_behaviors(s.refusal_path, s.refusal_max_items)
            items.update({i["id"]: i for i in loaded})
        else:
            notes.append(
                "no behaviours file configured: the refusal probe runs only the benign set, so no "
                "pair can pass the manipulation check"
            )
        items.update({i["id"]: i for i in refusal.borderline_items()})
        return cls(
            toolcall={
                c["id"]: c for c in toolcall.build_cases(s.toolcall_seed, s.toolcall_per_category)
            },
            injection={
                c["id"]: c for c in injection.build_cases(s.injection_seed, s.injection_cases)
            },
            refusal=items,
            kl={i["id"]: i for i in kl.kl_items()},
            capability={
                i["id"]: i for i in capability.build_items(s.capability_seed, s.capability_items)
            },
            harmful=harmful,
            notes=notes,
        )

    def cases(self, suite: str) -> dict[str, dict[str, Any]]:
        return getattr(self, suite)


def _stratum(suite: str, case: dict[str, Any]) -> str:
    """Grouping used to spread a limited run evenly (category, vector, check type...)."""
    if suite == "toolcall":
        return case["category"]
    if suite == "injection":
        return case["meta"]["vector"]
    if suite == "capability":
        return case["check"]["type"]
    return case["id"][:1]


def _subset(data: SuiteData, suite: str, limit: int | None) -> list[str]:
    """All case ids, or a deterministic stratified sample of ``limit`` of them.

    Cases are shuffled within their stratum with a fixed seed and then dealt out round-robin, so a
    small ``--limit`` still touches every category or injection vector.
    """
    cases = data.cases(suite)
    ids = sorted(cases)
    if not limit or limit >= len(ids):
        return ids
    groups: dict[str, list[str]] = {}
    for case_id in ids:
        groups.setdefault(_stratum(suite, cases[case_id]), []).append(case_id)
    rng = stable_rng("limit", suite)
    for members in groups.values():
        rng.shuffle(members)
    chosen: list[str] = []
    queues = [groups[key] for key in sorted(groups)]
    while len(chosen) < limit and any(queues):
        for queue in queues:
            if queue and len(chosen) < limit:
                chosen.append(queue.pop())
    return sorted(chosen)


def plan_jobs(
    config: StudyConfig,
    data: SuiteData,
    suites: list[str],
    models: list[str] | None = None,
    limit: int | None = None,
) -> list[Job]:
    """Every job of a run in a stable order (model-major)."""
    sampling = config.sampling
    selected = [m for m in config.models if not models or m.name in models]
    jobs: list[Job] = []
    for model in selected:
        for suite in suites:
            ids = _subset(data, suite, limit)
            if suite in ("toolcall", "injection"):
                conditions = ("default",) if suite == "toolcall" else injection.CONDITIONS
                runs = [(t, s) for t in sampling.temperatures for s in sampling.seeds_for(t)]
            elif suite == "kl":
                conditions, runs = ("default",), [(1.0, sampling.seeds[0])]
            else:
                conditions = ("default",)
                runs = [(sampling.primary_temperature, sampling.seeds[0])]
            for temperature, seed in runs:
                for case_id in ids:
                    for condition in conditions:
                        job_id = stable_hash(
                            suite, case_id, model.name, temperature, seed, condition
                        )
                        jobs.append(
                            Job(job_id, suite, case_id, model.name, temperature, seed, condition)
                        )
    return jobs


@dataclass
class RunSummary:
    run_dir: str
    planned: int = 0
    already_done: int = 0
    executed: int = 0
    errors: int = 0
    elapsed_s: float = 0.0
    aborted: str | None = None


def _turn_errors(turns: list[dict[str, Any]]) -> int:
    return sum(1 for t in turns if not t.get("ok"))


class Runner:
    """Executes jobs and appends trajectories. Safe to call again on the same directory."""

    def __init__(
        self,
        config: StudyConfig,
        run_dir: str | os.PathLike[str],
        *,
        data: SuiteData | None = None,
        client_factory: Callable[[ModelConfig], ChatClient] | None = None,
        workers: int | None = None,
        breaker: int = BREAKER_DEFAULT,
        progress: Callable[[str], None] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.data = data or SuiteData.from_config(config)
        self.workers = workers or config.workers
        self.breaker = breaker
        self._progress = progress or (lambda message: print(message, file=sys.stderr, flush=True))
        self.run_id = run_id or self.run_dir.name
        self._factory = client_factory or self._default_client
        self._clients: dict[str, ChatClient] = {}
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._consecutive: dict[str, int] = {}
        self._abort: str | None = None

    # ------------------------------------------------------------------ setup

    def _default_client(self, model: ModelConfig) -> ChatClient:
        sampling = self.config.sampling
        return ChatClient(
            model.base_url,
            model.served_name,
            api_key=model.api_key(),
            timeout_s=sampling.timeout_s,
            retries=sampling.retries,
            extra_body=model.extra_body,
        )

    def client(self, name: str) -> ChatClient:
        with self._state_lock:
            if name not in self._clients:
                self._clients[name] = self._factory(self.config.model(name))
            return self._clients[name]

    def check_mock_flags(self) -> None:
        """Refuse to mislabel: a mock endpoint must be configured as mock, and vice versa."""
        for model in self.config.models:
            try:
                _, _, headers = self.client(model.name).get_json("/models")
            except Exception:  # unreachable endpoints are reported by `doctor`, not here
                continue
            announces_mock = headers.get(MOCK_HEADER.lower()) == "1"
            if announces_mock and not model.mock:
                raise RunError(
                    f"model {model.name!r} at {model.base_url} announces itself as the MOCK server "
                    'but the configuration does not say "mock": true; refusing to mislabel the run'
                )
            if model.mock and not announces_mock:
                raise RunError(
                    f'model {model.name!r} is marked "mock": true but {model.base_url} is not the '
                    "mock server; refusing to label real output as MOCK"
                )

    def _metadata_path(self) -> Path:
        return self.run_dir / RUN_FILE

    def prepare(self, suites: list[str], force: bool = False) -> dict[str, Any]:
        """Create or validate ``run.json``; refuse a configuration that changed since the start."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self._metadata_path()
        digest, fingerprint = self.config.digest(), self.config.fingerprint()
        harmful_sha = (self.data.harmful or {}).get("sha256")
        if path.exists():
            meta = json.loads(path.read_text(encoding="utf-8"))
            changed = meta.get("config_fingerprint") != fingerprint
            old_sha = (meta.get("harmful_file") or {}).get("sha256")
            if old_sha and harmful_sha and old_sha != harmful_sha:
                changed = True
            if changed and not force:
                raise RunError(
                    f"{path} was created with different result-affecting settings (fingerprint "
                    f"{str(meta.get('config_fingerprint'))[:12]} vs {fingerprint[:12]}, or a different "
                    "behaviours file); use a new --run-id, or --force to overwrite the recorded settings"
                )
            meta["suites_requested"] = sorted(set(meta.get("suites_requested", [])) | set(suites))
            if changed or meta.get("config_sha256") != digest:
                meta.setdefault("config_history", []).append(
                    {"replaced_at": utc_now_iso(), "config_sha256": meta.get("config_sha256")}
                )
                meta["config"], meta["config_sha256"] = self.config.to_dict(), digest
                meta["config_fingerprint"] = fingerprint
                meta["harmful_file"] = self.data.harmful or meta.get("harmful_file")
        else:
            meta = {
                "schema": SCHEMA,
                "run_id": self.run_id,
                "created_at": utc_now_iso(),
                "version": __version__,
                "study": self.config.study,
                "config": self.config.to_dict(),
                "config_sha256": digest,
                "config_fingerprint": fingerprint,
                "suite_hashes": suite_hashes(),
                "mock": self.config.is_mock,
                "harmful_file": self.data.harmful,
                "notes": self.data.notes,
                "suites_requested": sorted(suites),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            }
        path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return meta

    # ------------------------------------------------------------------ log

    @property
    def trajectory_path(self) -> Path:
        return self.run_dir / TRAJECTORY_FILE

    def load_done(self) -> set[str]:
        """Job ids with a clean record. Repairs a truncated last line; retries errored jobs."""
        path = self.trajectory_path
        if not path.exists():
            return set()
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            cut = raw.rfind(b"\n") + 1
            self._progress(
                f"repairing {path.name}: dropping {len(raw) - cut} bytes of a truncated final line"
            )
            path.write_bytes(raw[:cut])
            raw = raw[:cut]
        done: dict[str, bool] = {}
        for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                job_id = record["job_id"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise RunError(f"{path}:{number}: corrupt trajectory line ({exc})") from exc
            done[job_id] = not record.get("errors", 0)
        return {job_id for job_id, clean in done.items() if clean}

    def _append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._write_lock, open(self.trajectory_path, "ab") as handle:
            handle.write(line.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------ execution

    def _chat_kwargs(self, job: Job) -> dict[str, Any]:
        s = self.config.sampling
        return {
            "temperature": job.temperature,
            "top_p": s.top_p,
            "seed": job.seed,
            "max_tokens": s.max_tokens,
        }

    def execute(self, job: Job) -> dict[str, Any]:
        """Run one job against its model and return the trajectory record (not yet persisted)."""
        model = self.config.model(job.model)
        client = self.client(job.model)
        sampling = self.config.sampling
        started = time.perf_counter()
        case = self.data.cases(job.suite)[job.case_id]
        category: str | None = case.get("category")
        if job.suite == "toolcall":
            result = client.chat(
                case["messages"],
                tools=case["tools"],
                tool_choice="auto",
                parallel_tool_calls=sampling.parallel_tool_calls,
                **self._chat_kwargs(job),
            )
            turn = Turn.from_result(result)
            turns = [result.record()]
            grade: dict[str, Any] = {
                "strict": toolcall.grade_toolcall(case, turn, STRICT),
                "lenient": toolcall.grade_toolcall(case, turn, LENIENT),
            }
        elif job.suite == "injection":
            kwargs = self._chat_kwargs(job)

            def chat(messages, tools):
                return client.chat(
                    messages,
                    tools=tools,
                    tool_choice="auto",
                    parallel_tool_calls=sampling.parallel_tool_calls,
                    **kwargs,
                )

            transcript = injection.run_case(
                case, chat, job.condition, self.config.suites.injection_max_steps
            )
            turns = transcript["turns"]
            grade = injection.grade_injection(case, [Turn.from_record(t) for t in turns])
            category = case["meta"]["environment"]
        elif job.suite == "refusal":
            kwargs = {**self._chat_kwargs(job), "max_tokens": self.config.suites.refusal_max_tokens}
            result = client.chat([{"role": "user", "content": case["prompt"]}], **kwargs)
            turns = [result.record()]
            grade = refusal.grade_probe(case, result.content, result.ok)
            category = "harmful" if case["id"].startswith("H-") else "benign"
        elif job.suite == "kl":
            top_k = self.config.suites.kl_top_logprobs
            result = client.chat(
                [{"role": "user", "content": case["prompt"]}],
                temperature=1.0,
                top_p=1.0,
                seed=job.seed,
                max_tokens=1,
                logprobs=True,
                top_logprobs=top_k,
            )
            turns = [result.record()]
            grade = kl.grade_kl_probe(result.logprobs, top_k, result.ok)
        else:
            kwargs = {
                **self._chat_kwargs(job),
                "max_tokens": self.config.suites.capability_max_tokens,
            }
            result = client.chat([{"role": "user", "content": case["prompt"]}], **kwargs)
            turns = [result.record()]
            grade = capability.grade_capability(case, result.content, result.ok)
        return self._record(job, model, case, category, turns, grade, time.perf_counter() - started)

    def _record(self, job, model, case, category, turns, grade, wall_s) -> dict[str, Any]:
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        for turn in turns:
            for key in usage:
                usage[key] += int(turn.get("usage", {}).get(key, 0))
        last = turns[-1] if turns else {}
        response = last.get("response") or {}
        message = (response.get("choices") or [{}])[0].get("message") or {}
        return {
            "schema": SCHEMA,
            "job_id": job.job_id,
            "run_id": self.run_id,
            "suite": job.suite,
            "case_id": job.case_id,
            "category": category,
            "condition": job.condition,
            "model": model.name,
            "pair_id": model.pair_id,
            "role": model.role,
            "temperature": job.temperature,
            "seed": job.seed,
            "mock": any(t.get("mock") for t in turns) or model.mock,
            "ts": utc_now_iso(),
            "turns": turns,
            "final_text": message.get("content") or "",
            "grade": grade,
            "errors": _turn_errors(turns),
            "requests": len(turns),
            "latency_s": round(sum(t.get("elapsed_s", 0.0) for t in turns), 6),
            "wall_s": round(wall_s, 6),
            **usage,
        }

    def _work(self, job: Job) -> dict[str, Any] | None:
        if self._abort:
            return None
        record = self.execute(job)
        self._append(record)
        with self._state_lock:
            failed = record["errors"] > 0 and not any(t.get("ok") for t in record["turns"])
            count = self._consecutive.get(job.model, 0) + 1 if failed else 0
            self._consecutive[job.model] = count
            if count >= self.breaker and not self._abort:
                last = next(
                    (t.get("error") for t in reversed(record["turns"]) if t.get("error")), "unknown"
                )
                self._abort = (
                    f"model {job.model!r} failed {count} requests in a row (last error: {last}). "
                    "Fix the server and re-run the same command with the same --run-id to resume."
                )
        return record

    def run(
        self,
        suites: list[str],
        models: list[str] | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> RunSummary:
        """Run (or resume) the given suites. Returns counts; never loses completed work."""
        started = time.perf_counter()
        suites = resolve_suites(suites)
        self.prepare(suites, force=force)
        jobs = plan_jobs(self.config, self.data, suites, models, limit)
        done = self.load_done()
        pending = [job for job in jobs if job.job_id not in done]
        summary = RunSummary(
            str(self.run_dir), planned=len(jobs), already_done=len(jobs) - len(pending)
        )
        self._progress(
            f"{self.run_id}: {len(jobs)} jobs planned, {summary.already_done} already done, "
            f"{len(pending)} to run with {self.workers} worker(s)"
        )
        if pending:
            counter = 0
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                for record in pool.map(self._work, pending):
                    if record is None:
                        continue
                    counter += 1
                    summary.executed += 1
                    summary.errors += 1 if record["errors"] else 0
                    if counter % max(25, len(pending) // 10) == 0 or counter == len(pending):
                        elapsed = time.perf_counter() - started
                        rate = counter / elapsed if elapsed else 0.0
                        eta = (len(pending) - counter) / rate if rate else 0.0
                        self._progress(
                            f"  {counter}/{len(pending)} done, {summary.errors} with errors, ~{eta:.0f}s left"
                        )
        summary.aborted = self._abort
        summary.elapsed_s = time.perf_counter() - started
        if self._abort:
            self._progress(f"ABORTED: {self._abort}")
        return summary
