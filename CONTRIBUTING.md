# Contributing

Thanks for helping. The aim is a harness whose results can be trusted, so the rules below favour
reproducibility and honesty over speed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
make test lint
```

Everything runs against the bundled mock server: no network, no GPU, no weights. The core
package depends on the standard library and numpy only; do not add runtime dependencies.

## Ground rules

1. **Never commit or paste an empirical result about a real model** unless it comes from a
   `results/*.json` file produced by `refusalscope analyze` on a real run, and then only by
   adding that file. Do not write numbers into the README by hand; `scripts/make_report.py`
   fills the marked block.
2. **Mock output must say MOCK** in its file name, title and tables. `results/MOCK*` is
   git-ignored. Do not remove the labelling code paths.
3. **No harmful-content datasets.** The refusal probe reads a file the user supplies. Do not add
   prompts that request harmful content to the repository, to tests (use obviously fake
   placeholders) or to examples. The benign borderline set must stay benign.
4. **Pre-registration is binding.** `docs/PREREGISTRATION.md` fixes hypotheses, metrics, sample
   sizes, tests and decision rules. A change to any of them after the first real run is a
   deviation: add an entry to section 13 (date, section, what and why, whether real outputs had
   been seen, which results it affects) in the same pull request.
5. **Suites are versioned.** If you change a generator, grader or system prompt, regenerate the
   snapshots with `make suites`, bump the suite version constant, and add a CHANGELOG entry that
   says results from different suite versions must not be pooled.

## Making changes

- Add tests with every change. Statistical code needs a hand-computed or textbook check. Graders
  need a case where they pass and a case where they fail. Aim to keep line coverage above 90
  percent (`make cov`).
- Run `make lint` (`ruff check .`) before pushing. Line length is 100; target Python 3.10.
- Keep error messages actionable: say what is wrong and what to do.

### Adding an injection vector

Add the wrapper and an entry to `VECTORS` in `src/refusalscope/suites/injection.py` (slot is one
of `tail`, `body`, `name`, `meta`). Update the vector table in `docs/METHODOLOGY.md`, the suite
size arithmetic in the pre-registration if the case count changes, regenerate snapshots, and run
the tests (they check coverage of every vector and environment combination).

### Adding a tool environment

Add an `EnvSpec` in `src/refusalscope/suites/injection_envs.py` with its tools, name pools, fact
carrying items, prompts, and attacker goals in `injection.py` (`_goals`) and a legitimate action
call (`_act_call`). Environments must use invented people, `.example` domains and fictional
accounts only.

### Adding a Suite A category or function

Add a function to the library in `toolcall_lib.py` (phrasings must render every expected value) or
a builder in `toolcall.py` plus a `CATEGORIES` entry. The oracle test (a perfect answer must score
correct for every case) and the "expected values appear in the request" test must pass.

## Reporting problems

Open an issue with: the command you ran, the `doctor` output, the harness version
(`refusalscope --version`), the engine and its version, and, for grader disputes, the output of
`refusalscope audit --run runs/<id> --suite <suite> --n 5`.

## Releasing

Update `CHANGELOG.md`, bump `__version__` and `pyproject.toml`, run `make test lint`, regenerate
and commit the suite snapshots and diagrams (`make suites diagrams`), and tag.
