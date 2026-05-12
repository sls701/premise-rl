# Premise Selection RL — Project Plan

## Goal

Train a reinforcement-learning agent that learns to query TheoremSearch for the **logical dependencies** of a target mathematical theorem, not its semantic neighbors. Build the data layer, environment, baselines, and GRPO training pipeline end-to-end on a single Linux GPU host accessed over SSH, and evaluate dependency recall@k against (a) an untrained base model, (b) a prompted base model, and (c) the RL-trained policy.

The plan is intentionally bottom-up: Phases 1–5 are infrastructure and baselines, and Phase 6 is the actual RL training. Do not skip ahead. Most of the bugs that will sink this project — broken ID mapping, miscalibrated thresholds, reward-shaping mistakes, the model spamming the same query — surface in Phases 1–5. They are much cheaper to fix there than during a multi-hour GRPO run.

## Non-goals

- Lean proof checking, formal verification, or anything that requires a Lean kernel.
- Paper-level (lenient) recall scoring during training. Statement-level only until Phase 7's robustness check.
- SLURM, Hyak, or any scheduler integration. Scripts run directly via SSH on a Linux GPU host inside `tmux`.
- Multi-node distributed training. Single-node multi-GPU only, via `accelerate launch`.

## System overview

Two Postgres databases are involved. They live on the same RDS instance and are accessed through the existing `get_rds_connection(db_name)` helper.

- **`postgres`** — the database that backs the TheoremSearch retrieval API. The Qwen3-Embedding-8B slogan embeddings used by the API live here. We never query it directly. We talk to it only through the public REST endpoint at `https://api.theoremsearch.com`.
- **`v2`** — the canonical theorem-statement and dependency-edge database. It contains the ground truth: target statements, true dependency edges, statement bodies, paper IDs, slogans, informal metadata. All queries for training data and labels go here.

The two databases were extracted from overlapping but slightly different snapshots of the same upstream corpus. There is no reliable shared key. The API returns `theorem_id` as an integer scoped to `postgres`; v2's `statement.statement_id` is a UUID. **The only way to bridge them is body-similarity matching over LaTeX statement bodies.** This is a structural fact about the project, not a temporary workaround. Any code that touches retrieval results must go through the matcher (Phase 2.2).

## Existing utilities (use as-is)

- `connect.py`: `get_rds_connection(db_name="v2")` — returns a `psycopg2` connection, pulls credentials from AWS Secrets Manager. Default `db_name="v2"`. Pass `db_name="postgres"` only if directly inspecting the API's backing store, which should be rare.
- `query.py`: `build_query(base_query, base_params, where_clauses, sample)` and `get_query_count(conn, query, params)` — helpers for SQL with optional WHERE clauses and `TABLESAMPLE BERNOULLI(1)` sampling. Reuse for any sampling code in Phase 1.
- `premise_selection.txt`: the system prompt for the prompted baseline. Do not modify it; copy verbatim into `configs/prompts/`.

Do not write new connection or query-building code. If something is missing, extend these files.

## Repo layout

```
premise-rl/
  src/
    db.py                     # one-line re-exports from connect.py + query.py
    data/
      load.py                 # fetch targets, dep edges, dep bodies; pickle cache
      splits.py               # build train/val/test tables in v2
    env/
      search_client.py        # async POST /search wrapper + diskcache
      id_mapping.py           # rapidfuzz body matcher, API int -> v2 UUID
      environment.py          # MDP: reset, step, reward, trajectory log
      prompts.py              # state -> string formatter (single source of truth)
    policies/
      common.py               # shared episode loop: env step + tool-call wiring
      local.py                # Qwen3-4B (untrained) and Qwen3-4B (RL checkpoint)
      api.py                  # frontier baselines, provider chosen via config
    train/
      reward.py               # per-step + terminal reward, imported by env and trainer
      rollout.py              # batched async rollouts for GRPO
      grpo.py                 # GRPO training loop (TRL)
    eval/
      run_eval.py             # entry point; produces summary.json + trajectories.jsonl
      metrics.py              # recall@k, FP rate, mapping diagnostics
  configs/
    baseline_gpt55.yaml       # provider: openai, model: gpt-5.5-<snapshot>
    baseline_gemini.yaml      # provider: google, model: gemini-3.1-pro-<snapshot>
    eval_local.yaml           # for the trained Qwen3-4B; --checkpoint at eval time
    prompts/
      premise_selection.txt   # verbatim copy of the user's file
  scripts/
    calibrate_threshold.py    # one-off: pick match_threshold for id_mapping
    build_splits.py           # one-off: populate v2 with rl_train/val/test tables
  tests/
    test_environment.py
    test_search_client.py
    test_id_mapping.py
    test_reward.py
  pyproject.toml
  README.md
  .env.example                # AWS_REGION, RDS_SECRET_ARN, RDS_HOST, HF_TOKEN,
                              # OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
```

Use `uv` for dependency management. Python 3.11+. CUDA-enabled `torch`. `httpx[http2]` for async HTTP. `rapidfuzz` (NOT `python-Levenshtein`) for body matching. `diskcache` for the search-client cache. `trl`, `transformers`, `peft`, `accelerate`, `bitsandbytes` for policy training.

## Phase 0 — Bootstrap (½ day)

1. Initialize repo. Copy `connect.py`, `query.py`, and `premise_selection.txt` into place. Create `src/db.py` that re-exports `get_rds_connection`, `build_query`, and `get_query_count` so downstream code can `from src.db import get_rds_connection`.
2. `pyproject.toml` with the dependencies above. `uv sync`.
3. `.env.example` documenting all secrets. `README.md` with a short project description and a "how to run" section that walks through Phases 1, 2.2.b, 4, and 6 entry points.
4. Connection smoke test:
   ```bash
   python -c "from src.db import get_rds_connection; c = get_rds_connection('v2'); cur = c.cursor(); cur.execute('SELECT 1'); print(cur.fetchone())"
   ```
   Should print `(1,)`. If it doesn't, do not proceed.

## Phase 1 — Data layer

The 9.2M-statement corpus and dependency edges live in `v2`. The plan uses three datasets:

- **`rl_smoke_100`** (already populated in v2): 100 hand-vetted targets, each with ≥2 statement-level dependencies. Used for plumbing validation, prompt iteration, and trajectory inspection.
- **`rl_train` / `rl_val` / `rl_test`** (Phase 1.3 creates them): main splits for RL training and final eval.

### 1.1 — Loaders

`src/data/load.py` exposes three functions that take a table name (default `rl_smoke_100`) and return lists of dataclasses. All use `get_rds_connection("v2")`. Cache each result to a pickle in `cache_dir` keyed on the table name; subsequent calls read the pickle. Data load is one-time at startup, so synchronous psycopg2 is fine even though the rest of the pipeline is async.

**(a) Target statements:**
```sql
SELECT s.statement_id, s.body, s.proof, s.kind, s.paper_id,
       im.label, im.ref, im.pre_context, im.post_context
FROM <table> t
JOIN statement s ON s.statement_id = t.src_id
LEFT JOIN informal_metadata im ON im.statement_id = s.statement_id;
```

**(b) True dependency edges** (statement-level only — `dep_id IS NOT NULL`):
```sql
SELECT d.src_id, d.dep_id, d.cite_key, d.dep_name, d.dep_key,
       s_src.paper_id AS src_paper_id,
       s_dep.paper_id AS dep_paper_id
FROM informal_dependency d
JOIN <table> t ON t.src_id = d.src_id
JOIN statement s_src ON s_src.statement_id = d.src_id
JOIN statement s_dep ON s_dep.statement_id = d.dep_id
WHERE d.cite_key IS NOT NULL
  AND d.method = 'deterministic'
  AND d.dep_id IS NOT NULL;
```

The two `paper_id` columns let downstream code partition each target's deps into intra-paper (`src_paper_id == dep_paper_id`) and inter-paper. This matters for Phase 6 (training first on intra-paper, then on inter-paper).

**(c) Dep statement bodies** (used for ID mapping in Phase 2.2 and trajectory inspection):
```sql
SELECT DISTINCT s.statement_id, s.body, s.kind, s.paper_id
FROM informal_dependency d
JOIN <table> t ON t.src_id = d.src_id
JOIN statement s ON s.statement_id = d.dep_id
WHERE d.cite_key IS NOT NULL
  AND d.method = 'deterministic'
  AND d.dep_id IS NOT NULL;
```

Final structure: `dict[UUID, Target]` where each Target carries body/proof/kind/metadata, `true_dep_ids: set[UUID]`, and `intra_dep_ids: set[UUID]` ⊆ `true_dep_ids`.

### 1.2 — Stats checkpoint

For each loaded table, print: target count, distribution of `len(true_dep_ids)`, distribution of `len(intra_dep_ids)`, total unique dep IDs across all targets (the "true-dep universe" used by the matcher).

### 1.3 — Train/val/test splits (deferred until baselines validate on smoke)

After the prompted baseline runs cleanly on `rl_smoke_100`, build the main splits in v2 with `scripts/build_splits.py`:

- Pool: all `src_id`s in `informal_dependency` with ≥2 deterministic `dep_id IS NOT NULL` edges.
- **Hold out by paper, not by statement**: pick papers for val/test and take all qualifying statements within those papers. This prevents test contamination via paper-internal lemma reuse, which is the central leakage risk.
- Target sizes: `rl_train` ≈ 50K, `rl_val` ≈ 1K, `rl_test` ≈ 2K. Adjust if the qualifying pool is smaller.
- Persist each split as a v2 table with the same `src_id` schema as `rl_smoke_100`, plus a `split` column. This keeps the splits inspectable from SQL and reproducible across runs.

Until the splits exist, every phase runs on `rl_smoke_100`.

## Phase 2 — Search client and ID mapping

### 2.1 — REST client

`src/env/search_client.py`: async wrapper around `POST https://api.theoremsearch.com/search`. Public endpoint, no auth header.

```python
async def search(query: str, k: int) -> list[SearchResult]
```

`SearchResult` mirrors the API's per-theorem response: `theorem_id (int), slogan_id (int), name, body, slogan, theorem_type, link, similarity, paper`.

Request payload: `{"query": query, "n_results": k}`. Do not expose the rich filter parameters (`sources`, `types`, `tags`, `year_range`, `citation_range`) to the policy. Keeping the action space minimal makes the trained policy comparable to the baselines.

Implementation:
- `httpx.AsyncClient` with HTTP/2 and a connection pool (limit 64).
- `diskcache` backend keyed on `(query, k)` — full-string key, no normalization. Cache dir from config.
- Retries with exponential backoff: 3 attempts, 30 s per-request timeout.
- On persistent failure, log a warning and return `[]`. **Never raise into a rollout** — a thrown exception during GRPO trashes the whole batch.

### 2.2 — ID mapping via body similarity

Both API and v2 ship LaTeX bodies from the same upstream extraction pipeline at slightly different snapshots. True matches have very high character-level overlap; coincidental pairs do not. The matcher uses `rapidfuzz.fuzz.ratio` (0–100, length-normalized) with a calibrated threshold.

#### 2.2.a — Body normalization

Both bodies (API and v2) go through the same normalizer before any comparison.

```python
def normalize(body: str) -> str:
    body = re.sub(r"\\label\{[^}]*\}", "", body)   # strip \label{...}
    body = html.unescape(body)                       # &amp; -> &, etc.
    body = re.sub(r"\s+", " ", body)                 # collapse whitespace
    body = body.strip().rstrip(".")                  # drop trailing period
    return body
```

Lowercase is **not** safe — LaTeX is case-sensitive (`\Theta` ≠ `\theta`).

#### 2.2.b — Threshold calibration (BEFORE running anything else)

`scripts/calibrate_threshold.py` picks `match_threshold` empirically.

1. Sample 50 statements from v2's `informal_dependency.dep_id` set (these have stable identifiers and well-formed bodies).
2. For each, query the API with the statement's slogan (or `body[:100]` if no slogan exists) at `k=10`.
3. For each returned API result, compute `rapidfuzz.fuzz.ratio(normalize(api.body), normalize(pg.body))` against the source statement's v2 body.
4. **True-match distribution:** for each query, the highest-ratio API result is presumed the true match (verify ~5 manually). Collect those ratios.
5. **Cross-pair distribution:** ratios of API results that are NOT the source statement, against the source's v2 body. These are coincidental overlap.
6. Print both distributions to stdout. Pick `match_threshold` between the 5th percentile of true matches and the 95th percentile of cross-pairs. Write to `configs/baseline.yaml`.

**Hard gate:** if 5th-percentile-true-match < 95th-percentile-cross-pair, the distributions overlap and there is no clean threshold. **Stop and report.** Likely causes: parser-version skew between snapshots, or the API was indexed against a substantively different version of the corpus. Do not paper over this with a midpoint threshold — every recall number downstream will be unreliable, and the RL training loop will be optimizing partly against noise.

Expected if all is well: true matches cluster ≥95, cross-pairs cluster ≤50, threshold lands around 85–90.

#### 2.2.c — Matcher

`src/env/id_mapping.py`. At construction time, takes the union of dep bodies for the active dataset (smoke during Phase 4, train+val during Phase 6, test during Phase 7), normalizes each body, and stores them indexed by UUID. Exposes:

```python
def map_int_to_uuid(api_result: SearchResult) -> MatchResult
    # MatchResult = NamedTuple(uuid: UUID | None, score: float, second_best_gap: float)
```

Per-call cost: O(|dep universe|) ratio computations. For smoke (~3K bodies), ~3 ms per call. For full train (likely 100K+), use `rapidfuzz.process.extract` which runs in a tight C loop — should still be sub-50 ms. If profiling shows the matcher dominates GRPO rollout time, prefilter with Qwen3 cosine over normalized bodies (top-100 candidates) and run exact ratio only on the candidates.

**Important — scope:** match against ONLY the active dataset's true-dep universe, not all of v2. False positives don't need a UUID; the default `None` correctly flags them as "not a true dep." Matching against all of v2 would slow things 1000× and create spurious matches on unrelated statements that share boilerplate LaTeX (definitions, standard lemma openers, etc).

**Tiebreak / low-confidence flagging:** if the best match crosses threshold but the second-best is within a configurable gap (default 5 ratio points), log this as a low-confidence match in the trajectory. Track the rate as a metric (`low_confidence_match_rate`).

#### 2.2.d — Tests

`tests/test_id_mapping.py`:
- **Round-trip:** pick 5 statements from the smoke dep set. Query the API for their slogans. Verify the matcher recovers the correct UUIDs above threshold.
- **No-match:** synthetic `SearchResult(body="this is not a real theorem")` returns `uuid=None`.
- **Normalizer idempotence:** `normalize(normalize(x)) == normalize(x)` over a battery of LaTeX inputs.

**Checkpoint:** calibration produces a clean threshold gap, the chosen threshold and both distributions are saved to `<results_dir>/calibration.json`, and the round-trip test passes. Do not move past Phase 2 until all three.

## Phase 3 — MDP environment

### 3.1 — Environment

`src/env/environment.py`: `PremiseSelectionEnv` class.

- `reset(target_id) -> state`: loads target, sets `retrieved_uuids: set[UUID] = set(), query_history: list[str] = [], step_idx = 0, true_deps = target.true_dep_ids`.
- `async step(query) -> (state, reward, done, info)`:
  1. `results = await search_client.search(query, k=top_k)`
  2. `mapped = [matcher.map_int_to_uuid(r) for r in results]`
  3. `new_uuids = {m.uuid for m in mapped if m.uuid is not None} - retrieved_uuids`
  4. `new_tps = new_uuids & true_deps`, `new_fps = new_uuids - true_deps`
  5. Per-step reward: `|new_tps| - alpha * |new_fps|`
  6. Update state, increment `step_idx`. `done = (step_idx == H)`.
  7. On `done`, add terminal bonus: `beta * (|retrieved_uuids ∩ true_deps| / |true_deps|)`.

**Critical detail:** an API result that fails to map (`uuid=None`) is **not** scored as a false positive. We don't know what it is. It's logged as `dropped_no_match` and excluded from reward. This keeps the FP penalty meaningful — a true FP is a confident match to a non-dep.

### 3.2 — Trajectory log

Per step: `{step, query, returned_results: [{int_id, mapped_uuid, match_score, second_best_gap}], new_tps, new_fps, dropped_no_match, step_reward, terminal_reward}`. Log both ID forms — debugging an ID-mapping issue without the raw API IDs preserved is miserable.

### 3.3 — State formatter

`src/env/prompts.py`: `format_state(state) -> str`. Lays out:
- Target slogan, then body. Optionally pre/post context.
- Prior queries (one per line, numbered).
- Prior retrieved slogans (NOT full bodies — token budget). Tag each with which query retrieved it.

One formatter, used by all policies and any debugging. If you want to test a different format, do it via a config flag, not a fork.

### 3.4 — Tests

`tests/test_environment.py` with a fake search client returning canned results:
- 3-step episode with mixed TPs and FPs, asserts cumulative reward matches a hand-computed value.
- **Duplicate query:** same query issued twice. The second issuance yields zero new TPs and zero new FPs (`step_reward = 0`). This is the most common reward-shaping bug; do not skip.
- Terminal bonus fires exactly once, on the final step.
- Mapping failure (`uuid=None`) contributes zero reward and zero penalty; the episode continues normally.

`tests/test_reward.py` covers the reward function in isolation: empty `true_deps`, all-FP step, all-TP step, partial overlap.

**Checkpoint:** all environment and reward tests pass.

## Phase 4 — Baseline policies

Per the proposal, three policies share Phase 5's eval harness:

- **Two prompted frontier baselines:** GPT-5.5 and Gemini-3.1 Pro. Both use the same `premise_selection.txt` system prompt. These measure what good prompting on a strong base model can do — the bar the RL policy should meet, and optimistically clear.
- **The trained policy:** Qwen3-4B fine-tuned with GRPO (Phase 6). Same system prompt as the frontier baselines, served from a local checkpoint.

There is no untrained-Qwen3-4B baseline in the proposal. Do not add one; it would change the comparison story without being asked for.

### 4.1 — Common episode loop

`src/policies/common.py`: provider-agnostic `async run_episode(env, target_id, agent, config) -> trajectory`. Owns:
- The env-step ↔ tool-call plumbing.
- Conversation-history management (full history persists across turns; do not re-prompt with state).
- Termination: model declines to call the tool, OR `env` returns `done=True`, OR safety cap of `H` tool calls.
- Trajectory logging.

The `agent` argument is whatever provider-specific client object knows how to do one round of "given conversation history, return either a tool call or a final message." Providers differ in tool-use APIs; the common loop hides that.

### 4.2 — Frontier baselines (`api.py`)

`src/policies/api.py` dispatches on `config.provider ∈ {openai, google, anthropic}` and constructs the provider-specific agent that `common.run_episode` consumes. One tool: `search_theorems(query: string, k: integer)`. Each tool call is wrapped into `env.step(query)`; the returned slogans (not full bodies) come back as the tool result.

Provider-specific notes:
- **OpenAI (GPT-5.5):** Responses API with native tool use. Pin the dated snapshot (`gpt-5.5-YYYY-MM-DD`) — never use the `gpt-5.5` alias, it drifts.
- **Google (Gemini-3.1 Pro):** Gemini API with function calling. Pin the dated snapshot.

Both at temperature 0 for reproducibility (and so the diskcache hits on re-runs). Each provider's snapshot ID, max output tokens, and any other knobs live in the per-provider config.

Do not introduce a wrapper library (LiteLLM, etc.) just to share code. The two tool-use APIs are different enough that an abstraction will leak; a 50-line per-provider implementation is cleaner and easier to debug. Common.py holds the env logic; api.py holds the two thin provider implementations.

### 4.3 — Trained policy (`local.py`)

`src/policies/local.py`: loads Qwen3-4B with optional LoRA adapters from a checkpoint dir. Uses the model's chat template for native tool use. Same one-tool surface as the API baselines, same temperature 0 at eval. Served via vLLM if available, else `transformers.generate` — pick at config time. During Phase 7 eval, this is the only "local" policy that runs; during Phase 6 training, this is the policy under the GRPO trainer.

### 4.4 — Smoke run and inspection

Run **the prompted GPT 5.5 baseline** on a single target from `rl_smoke_100`. (Pick one; running both on every smoke target during plumbing iteration burns cost for no incremental signal.) Manually inspect the trajectory JSONL:

- **Query variation:** are queries varied across steps, or is the model spamming the same query? Spam means the state formatter isn't surfacing query history clearly. Fix before scaling.
- **Refinement:** is the model using prior retrieved slogans to refine?
- **Predecessors vs neighbors:** is the model querying for restatements of the target (semantic neighbors) instead of likely lemma names (logical predecessors)? **This is the central failure mode** — and the diagnostic that motivates RL training. Expect to see it.
- `dropped_no_match` rate: high is normal. High `low_confidence_match_rate` is worse — investigate matcher or normalizer.

Then run all three frontier baselines on the full `rl_smoke_100` to confirm provider plumbing works end-to-end and to get a first read on which is the strongest baseline.

### 4.5 — Async batching and rate limits

`asyncio.gather` with a per-provider semaphore. Concurrency from config, separate per provider since rate limits differ.

The smoke run is ~600 model calls per provider (100 episodes × ~6 turns). All two providers' tier-5 limits are generous enough that 32-way concurrency should not 429. If 429s happen anyway, halve and retry; do not let retry storms cascade.

Cost: track cumulative API spend per provider in the run summary. The full `rl_test` eval at ~2K targets × 3 providers × ~6 turns is ~36K calls — order of magnitude $50–$200 depending on how the providers price tool-use turns. Report extrapolated spend before launching the test-set eval.

**Checkpoint:** both frontier baselines run end-to-end on `rl_smoke_100`. Two `summary.json` files written. Recall is non-trivially > 0 for both.

## Phase 5 — Evaluation harness

### 5.1 — Metrics

`src/eval/metrics.py`: pure functions over trajectory JSONL.

- `recall_at_k`: per-target `|retrieved_uuids ∩ true_deps| / |true_deps|`. Headline number.
- `mean_queries_per_episode`, `unique_query_rate`, `mean_FP_per_episode`, `mean_terminal_reward`.
- `dropped_no_match_rate`: fraction of API results that didn't cross threshold against any true dep. Expected to be high — informational, not an alarm.
- `low_confidence_match_rate`: fraction of accepted matches where the gap to second-best was small. If >5%, recall numbers carry meaningful uncertainty and should be reported with a caveat.
- Stratified versions of all of the above by `len(true_deps)` bucket: `{2}, {3}, {4–5}, {6+}`. Mixed-dep-count averages are noisy on small sets; the bucket breakdown is where the signal lives.
- Intra-paper recall vs inter-paper recall, computed separately. Phase 6 trains intra first; tracking these separately is essential for spotting when transfer to inter-paper is or isn't working.
- All metrics are computed per-policy from each policy's trajectory JSONL. The eval harness (5.2) treats policies as interchangeable plug-ins; metrics never reach across policies. Cross-policy comparison is done at writeup time from the per-policy summary.jsons, not in code.

### 5.2 — Entry point

`src/eval/run_eval.py --config <path> --policy {api|local} --dataset {smoke|val|test} [--checkpoint <path>]`:

1. Load dataset via Phase 1.
2. Build matcher via Phase 2 against the active dataset's dep universe.
3. Construct the policy:
   - `--policy api`: reads `provider` and provider-specific snapshot from the config (e.g. `baseline_gpt55.yaml` -> openai/gpt-5.5).
   - `--policy local`: loads Qwen3-4B from `--checkpoint` (untrained-Qwen3-4B base if checkpoint omitted, but per Phase 4 we don't ship that as a baseline; this branch exists for the trained policy in Phase 7).
4. Run policy concurrently with the semaphore from config.
5. Write `<results_dir>/<run_name>/trajectories.jsonl` and `<results_dir>/<run_name>/summary.json`.
6. `<run_name>` defaults to `<policy_descriptor>_<dataset>_<UTC timestamp>`, where `<policy_descriptor>` is the provider+snapshot for api, or `qwen3-4b_<checkpoint_name>` for local.

**Checkpoint:** smoke run produces both files. `summary.json` shows:
- recall > 0 (if at floor, suspect calibration first — re-run threshold script and round-trip test).
- `mean_queries_per_episode` close to H (model is using its budget).
- `unique_query_rate` > 0.7 or so (model isn't trivially repeating).
- `low_confidence_match_rate` < 5%.

Do not begin Phase 6 until all four pass on smoke, and the prompted baseline produces sensible numbers on `rl_val`.

## Phase 6 — GRPO training

This is the meat of the project. Do not start until Phases 1–5 are stable.

The training schedule follows the proposal: **train on intra-paper edges first, then continue training on inter-paper edges.** Intra-paper deps are easier (more semantic overlap with the target, often visible in surrounding paper context), so the policy can establish baseline competence before tackling the harder inter-paper transfer task.

### 6.1 — Prerequisites

- `rl_train` / `rl_val` / `rl_test` populated in v2 via Phase 1.3.
- Matcher rebuilt against the union of `rl_train` + `rl_val` dep universes. (Test set's matcher is built separately at Phase 7 to avoid leaking val-time threshold tuning into the test eval.)
- Per-target rollout cost characterized (queries × API latency × top-k matcher cost). Multiply by batch size × group size × steps to confirm a training run fits the time budget you have.

### 6.2 — Rollout infrastructure

`src/train/rollout.py`: batched async rollouts. For each GRPO step:

1. Sample a batch of targets from the active training table.
2. For each target, sample `G` (group size, default 8) trajectories with the current policy at temperature > 0 (default 0.7).
3. Each trajectory is an env episode with the policy issuing tool calls.
4. Return per-trajectory: total reward (per-step + terminal), token-level log-probs from the policy under the current parameters, and the trajectory log.

Correctness items:
- Cache the matcher's normalized body store across the entire training run — it's static.
- Search-client cache should run hot — many rollouts will reuse queries. Allocate ≥10 GB of `diskcache`.
- Per-target rollout group must use independent search-client coroutine sessions; do not share `httpx` clients across asyncio loops.
- Wrap every search-client call in a try/except that returns `[]` on failure. A single thrown exception will trash a batch and waste an hour of rollouts.

### 6.3 — Reward

`src/train/reward.py` is the **only** place the reward function lives. Imported by both `src/env/environment.py` (for baseline/eval rollouts) and `src/train/rollout.py` (for training rollouts). Do not duplicate reward logic between env and trainer; that divergence is a future-Sophie problem you do not want.

GRPO consumes per-trajectory scalar rewards: `sum(per_step_rewards) + terminal_bonus`. Within a group, GRPO normalizes by group mean/std. Watch for degenerate groups where all trajectories tie (zero advantage signal) — log the rate. If >20% of groups are degenerate, the action space isn't producing enough variation: bump rollout temperature, reshape the reward, or both.

### 6.4 — Training loop (intra-paper)

`src/train/grpo.py`: TRL's `GRPOTrainer`, configured by `configs/grpo_intra.yaml`. The reward function passed to GRPO uses `target.intra_dep_ids` rather than `target.true_dep_ids` — only intra-paper hits count for reward at this stage.

Configuration:
- Policy: Qwen3-4B (pin the snapshot in config — never use the moving alias), with LoRA adapters (rank 16, alpha 32, target Q/K/V/O + MLP). Full fine-tune is out of scope.
- Optimizer: AdamW, lr `1e-6` (GRPO is sensitive — keep it small).
- Group size `G = 8`.
- Per-device batch size: as many targets per step as GPU memory allows, each with G trajectories.
- KL penalty against the frozen reference (the same base model). Default `beta = 0.04`. Tune if KL collapses or explodes.
- Gradient checkpointing on. bf16. Flash attention.

Launch:
```bash
accelerate launch -m src.train.grpo --config configs/grpo_intra.yaml
```

Run inside `tmux` so the job survives SSH disconnect.

### 6.5 — Mid-training eval

Every N training steps (default 100), the trainer runs the current policy on a fixed 200-target slice of `rl_val`. Logs: recall@k bucketed by dep count, recall split intra vs inter, mean reward, mean queries/episode, unique query rate, KL to ref. Saves policy LoRA + eval snapshot to `<checkpoint_dir>/step_<N>/`.

Optional `wandb` integration behind a config flag. If not using wandb, append everything to `<results_dir>/training_log.jsonl`.

### 6.6 — Continuation training (inter-paper)

Once intra-paper training has converged (val intra-recall plateaus or starts overfitting), continue training with `configs/grpo_inter.yaml`:

- Starts from the best intra-paper checkpoint.
- Reward uses `target.true_dep_ids` (all deps, not just intra-paper).
- Lower learning rate (`5e-7`) since we're refining, not starting fresh.
- Same ablation surface as intra.

The point of the two-stage schedule is to give the policy a competent starting point before the harder inter-paper transfer task. Track val inter-recall throughout — if intra-recall stays high but inter-recall doesn't budge, the policy may have overfit to intra-paper cues and need a different reward shape.

### 6.7 — Ablations

Once a single training run produces non-degenerate behavior, sweep:

- `alpha` (FP penalty weight): {0.05, 0.1, 0.25, 0.5}.
- `H` (horizon): {3, 5, 8}.
- `top_k` (results per query): {5, 10, 20}.
- Action shaping: free-form vs constrained templates vs retrieve-and-refine. Templates ("what is known about <concept>?") are easier to learn but more limited. Free-form is more expressive but slower to converge. Run all three.

Each ablation lives in `configs/ablations/<name>.yaml`. Do not run a sweep before a single training run produces a sensible result, or you will spend a day waiting for noise.

**Checkpoint:** the best intra+inter checkpoint, evaluated on rl_val, meets (or optimistically wins) both of the prompted frontier baselines on bucket-stratified recall@k by at least ε (decide ε before looking at the numbers — say, 0.05 absolute on the {4–5} bucket). "Best frontier" is computed per-bucket — the RL policy needs to meet whichever frontier model wins each bucket, not an average. If not, do not advance to Phase 7. Debug.

## Phase 7 — Final evaluation

1. Build the matcher against `rl_test`'s dep universe (separately from train/val, to prevent threshold-tuning leakage).
2. Run all three policies on the full rl_test: prompted GPT-5.5, prompted Gemini-3.1 Pro, and the trained Qwen3-4B (best Phase 6 checkpoint). Save trajectories per policy.
3. Headline table: recall@k for each policy, stratified by dep-count bucket and intra/inter, with bootstrap 95% CIs.
4. Two-tier scoring (added here, not earlier):
   - **Strict (statement-level):** the matcher returns a UUID and that UUID is in `true_deps`. As implemented throughout.
   - **Lenient (paper-level):** an API result counts as a hit if its `paper` (paper_id) matches the `paper_id` of any cited statement in v2. Robustness check on the larger paper-level pool. The lenient tier shouldn't change the ranking of policies; if it does, dig in.
5. Qualitative: for each of the three policies, sample 10 trajectories per dep-count bucket. Read them. Write two pages on what the trained Qwen3-4B does differently from each frontier baseline — query strategies, failure modes, surprises. Pay particular attention to whether the RL policy's behavior is "frontier-like prompting that happens to be cheaper" or genuinely different (e.g., shorter, more iterative, more lemma-name-driven).

## Configurability

Everything tunable lives in YAML configs. No magic constants in code.

- **Cache dir, results dir, checkpoint dir:** in config. Defaults: `./cache`, `./results`, `./checkpoints`. Override per host.
- **Concurrency:** in config. Tune to the host's GPU memory.
- **Model snapshot, temperature, max tokens:** in config. Pin model snapshots — never use a moving alias.
- **MDP hyperparameters (`H`, `top_k`, `alpha`, `beta`):** in config.
- **GRPO hyperparameters (lr, group size, KL beta, LoRA rank):** in config.
- **Secrets:** environment only. AWS_REGION, RDS_SECRET_ARN, RDS_HOST, HF_TOKEN, OPENAI_API_KEY, GOOGLE_API_KEY. Document in .env.example. The user sources .env from their shell before running. The two provider keys are required for the corresponding baseline_*.yaml; run_eval.py should fail fast with a clear message if the relevant key is missing.

## Running the pipeline

All commands run on the Linux GPU host over SSH, inside `tmux` for anything long-running.

```bash
# One-time setup
uv sync
cp .env.example .env  # fill in all keys

# Phase 4 smoke (one provider first to validate plumbing, then both)
python -m src.eval.run_eval --config configs/baseline_gpt55.yaml  --policy api --dataset smoke
python -m src.eval.run_eval --config configs/baseline_gemini.yaml --policy api --dataset smoke

# Phase 1.3 (after smoke is clean)
python scripts/build_splits.py

# Frontier baselines on val
python -m src.eval.run_eval --config configs/baseline_gpt55.yaml  --policy api --dataset val
python -m src.eval.run_eval --config configs/baseline_gemini.yaml --policy api --dataset val

# Phase 6 training (inside tmux)
accelerate launch -m src.train.grpo --config configs/grpo_intra.yaml
accelerate launch -m src.train.grpo --config configs/grpo_inter.yaml

# Phase 7 final eval — all three policies on rl_test
python -m src.eval.run_eval --config configs/baseline_gpt55.yaml  --policy api   --dataset test
python -m src.eval.run_eval --config configs/baseline_gemini.yaml --policy api   --dataset test
python -m src.eval.run_eval --config configs/eval_local.yaml      --policy local --dataset test --checkpoint <path>
```

## Sanity checks (post-run, after each phase)

1. **Recall is non-trivial.** If recall ~0, suspect (in order): threshold too strict (re-run calibration), prompt not loaded, temperature non-zero causing cache misses, search API returning empty results.
2. **Cache hits on re-run.** A second identical baseline run should be near-100% search-cache hit. If not, query strings are nondeterministic or the cache key is wrong.
3. **Episode length.** Most baseline episodes should hit H. If many bail at step 1–2, the prompt isn't conveying that multi-step refinement is expected.
4. **Trajectory eyeball.** Pick 5 trajectories per bucket. Read them. Are queries plausible? Is the model refining based on prior retrieved slogans? Is it searching for likely lemmas or for restatements? The last is the central diagnostic.
5. **Low-confidence match audit.** If `low_confidence_match_rate` > 5%, manually inspect a sample. Real matches with formatting drift, or spurious overlap? Adjust normalizer or threshold accordingly.
6. **GRPO group variance.** Healthy training has non-trivial within-group reward variance. If groups consistently tie, the policy isn't exploring; bump temperature or reshape reward.
7. **KL trajectory.** Should rise monotonically but slowly. Sudden spikes = policy collapse; sudden flatlines = no learning. Plot every checkpoint.
8. **Intra vs inter recall during training.** Intra should rise first. If inter rises in lockstep with intra during the intra-only phase, the matcher might be leaking — investigate.

## Handoff prompt for the Claude Code agent

> Implement docs/PLAN.md in phase order. Do not skip ahead. Each phase has a checkpoint; stop at it and report results before continuing. Reuse connect.py and query.py as-is — do not rewrite the connection or query helpers. Use uv for deps, httpx (async, HTTP/2) for the search API, rapidfuzz for body matching, trl for GRPO, Qwen3-4B (pinned snapshot in config) as the policy that gets fine-tuned. The two baselines are prompted frontier models (GPT-5.5, Gemini-3.1 Pro) — each via its provider's native tool-use API, all sharing premise_selection.txt as the system prompt. Do not introduce a multi-provider abstraction library; per-provider implementations in src/policies/api.py are simpler and more debuggable. Phase 2.2.b (threshold calibration) is a hard gate — if calibration shows overlapping distributions with no clean threshold, stop and report. Phase 5 must produce a summary.json with non-trivial recall on at least one frontier baseline before any RL work begins. The reward function lives in exactly one place (src/train/reward.py) and is imported by both env and trainer. The two databases (postgres for the API's backing store and v2 for the dependency ground truth) are bridged only by body-similarity matching (Phase 2.2); this is structural, not a workaround. Provider API keys (OPENAI_API_KEY, GOOGLE_API_KEY) are required for the corresponding baseline configs; fail fast with a clear message if one is missing. If a phase's checkpoint fails, halt and report — do not silently work around it.
