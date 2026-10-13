# cagematch

**Put your agent harnesses in the cage. Evidence-locked results or no result at all.**

Run the same agent task suite across providers, models, and harness policies
(parallel vs serial tool dispatch, prompt variants, budgets), score cost,
latency, tool accuracy, and failure modes — and prove every score by
**replaying the persisted, hash-chained trace** against the original data.

Zero runtime dependencies. Python 3.10+. stdlib only, on purpose: an eval
harness that needs eleven packages is a supply chain, not a tool.

## Why

Coding agents and agent harnesses are proliferating faster than any honest way
to compare them. Leaderboards measure one dimension (task pass rate) with no
cost normalization and no audit trail behind the number. cagematch's contract
is different: **a run's score is only reportable if every tool call in its
trace re-executes to the same result.** A cell cannot score well on numbers it
did not derive, and a edited results database cannot pass `verify`.

## The failure mode this exists for

One command, real output:

```text
$ python tools/tamper_demo.py
run drawdown-spread__scripted: 10 events, chain intact
FORGED: drawdown-spread__scripted event 2: result {"stat": "max_drawdown", "value": -0.25360573419533106}... -> {"stat": "max_drawdown", "value": -0.001}
CAUGHT: trace for run drawdown-spread__scripted diverges at event index 2: chain hash mismatch (expected prev 822be9cfb3c9…, stored 822be9cfb3c9…)
(the attacker could recompute one event's hash but cannot rewrite the forward chain without touching every later row)
```

A single event's payload AND its own hash are rewritten — the realistic
one-row forgery. The chain link to the *next* event is what catches it, and
the error localizes the exact event index.

## Quick start

```bash
pip install -e .            # or: uv sync --extra dev
cagematch run               # hermetic derby with the scripted test double
cagematch report --markdown # results table
cagematch verify            # recompute every trace hash chain
```

Add live cells by exporting credentials (keys are read from the environment
variable *named* in config — never stored in config or code):

```bash
export OPENAI_API_KEY=***          # enables the gpt + gpt-serial cells
export ANTHROPIC_API_KEY=***   # enables the claude cell
export SPARK_BASE_URL=http://<host>:18300/v1   # enables a self-hosted cell
cagematch run
```

Cells whose credentials are absent self-skip, so the same command works on a
laptop, in CI, and on a trading floor with partial credentials.

## What's in the box

| piece | file | what it proves |
|---|---|---|
| agent loop with budgets | `src/cagematch/agent.py` | step + cost budgets checked BEFORE dispatch; a spinning cell dies at the budget, not the clock |
| provider client | `src/cagematch/provider.py` | any OpenAI-compatible endpoint; handles `content: null` from thinking stacks without crashing the scorer |
| sandboxed tools | `src/cagematch/tools.py` | filesystem-confined CSV/stats tools; path escapes and non-arithmetic `calc` payloads refuse with findings |
| hash-chained trace | `src/cagematch/trace.py` | sqlite event log, each event's hash covers the previous; `verify` localizes tampering to an event index |
| replay scorer | `src/cagematch/scoring.py` | ground truth recomputed from raw data, every tool call re-executed; divergence = score unproven |
| cell matrix | `src/cagematch/cells.py` | provider × model × prompt × loop-policy as JSON, not code |
| hermetic eval gate | `scripts/run_eval.py` | CI runs the full loop + scoring + chain with a labelled test double, no network, no keys |
| real market data | `data/` | SPY/QQQ 5y dailies (Yahoo Finance chart API, fetched 2026-10-13), SP500 + DGS10 (FRED, fetched 2026-10-13) |

## Tasks

- **drawdown-spread** — max drawdown of SPY vs QQQ over 5 years of dailies;
  ground truth: SPY −25.36%, QQQ −35.62% (recomputed at score time from the
  CSVs, never trusted from the agent).
- **recent-window** — trailing 252-day SPY mean + last observation in a FRED
  calendar series with missing days; punishes cells that treat blanks as zeros
  or read the wrong column.

## Honest limits

- **The CI gate runs a test double.** `scripted` proves the plumbing — loop,
  budgets, trace chain, replay, scorer — end to end with no network. It does
  not prove model intelligence, and the gate output says so on every run
  (`simulated_test_double=yes`).
- **The task suite is small (2 tasks).** It is an architecture you extend with
  your own tasks JSON, not a leaderboard. Two tasks, honestly measured and
  replay-certified, is a better claim than twenty that aren't.
- **Replay requires deterministic tools.** Replays certify tool *results*, not
  model determinism; live cells run once per report and report variance the
  way sampling models behave, single-sample.
- **The hash chain is tamper-EVIDENCE, not tamper-PROOF.** Anyone with sqlite
  write access can rewrite the whole forward chain; anchoring the chain head
  outside the database (git commit, WORM store) is the next link and is not
  built here.
- **Cost numbers are list-price estimates** from the pricing table in
  `agent.py`; update the table when providers reprice.

## Design decisions worth arguing about

- **Scores must be replayable or they don't print.** Most eval harnesses
  trust their own log. Here the log is a hypothesis until the tools re-run.
- **Budgets are checked before dispatch, never after.** Checking after lets a
  cell take one more unlimited action for free.
- **Tool failures are findings, not crashes.** A TOOL_ERROR goes into the
  transcript and the run continues; the harness reports tool_errors as a
  first-class column. A harness that dies on a bad tool call cannot evaluate
  a model's recovery behavior.
- **Config is data.** Adding a provider is a JSON edit; the loop code never
  changes, which is the whole point when the ecosystem moves every six weeks.

## Layout

```
src/cagematch/   package (agent loop, provider, tools, trace, scoring, cells, CLI)
tasks/tasks.json task suite
cells.demo.json  scripted-only cell matrix used by the README demo
data/            pinned real market data + SOURCES note
scripts/         hermetic CI eval gate
tools/           tamper_demo.py (the README transcript, as a committed contract)
tests/           25 tests: budgets, replay divergence, chain forgery, tool sandbox
```

## License

MIT
