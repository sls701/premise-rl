# Reward Design Implementation Plan

## Current System (Baseline)

```
step_reward(t) = new_tps(t) - α * new_fps(t)   [α = 0.0 currently]
terminal_bonus  = β * |retrieved ∩ deps| / |deps|  [β = 1.0, fires at t=horizon]
total_reward    = Σ step_reward(t) + terminal_bonus
```

SDPO ranks episodes within a group by `total_reward`, imitates the winner's token sequence, and applies DPO contrastive loss between winner and loser. The **group is skipped** if the winner's reward ≤ 0 (`min_winner_reward=0.0`), which causes ~67% of batches to be degenerate — the core training efficiency problem.

---

## Design 1: False-Positive Penalty (α > 0)

**What changes:** Increase `alpha` in the config from 0.0 to 0.1–0.5.

**Effect on reward:**
```
step_reward(t) = new_tps(t) - α * new_fps(t)
```
With `top_k=30` and typical results, a zero-TP search returns 30 FPs, costing `α * 30`. At α=0.1 that's −3 per wasted search, making the total_reward negative for episodes that never find TPs.

**Why this matters for training:**
Currently `total_reward = 0` for all-zero-TP episodes, so the group is always skipped (winner reward = 0). With α > 0, FP-heavy episodes get *negative* total_reward, which means a mixed group (some TPs found) has a non-zero winner, making it trainable. The degenerate rate should drop significantly.

**Implementation:** One-line config change (`alpha: 0.1`). No code changes needed.

**Tradeoff:** Too-high α discourages exploration entirely — the model learns to never search rather than search badly. Start at 0.1–0.2, not higher.

**Risk:** The terminal_bonus still uses the recall formula (which is positive), so a model that finds 1 TP and stops making queries gets `step_reward + terminal_bonus > 0` even with high α. This is correct behavior.

---

## Design 2: Per-Step Recall Shaping (Dense Normalization)

**What changes:** Replace the raw TP count per step with a normalized recall increment:

```python
# src/train/reward.py
def step_reward(new_tps: int, new_fps: int, alpha: float,
                n_deps: int = 1) -> float:
    recall_gain = new_tps / max(n_deps, 1)          # normalize by target size
    fp_cost     = alpha * new_fps / max(top_k, 1)   # normalize by search size
    return recall_gain - fp_cost
```

**Why:** Currently a target with 2 deps gives max step_reward=2, while one with 10 deps gives max=10. This biases the winner selection toward high-dep targets — the model learns to find easy multi-dep papers rather than hard single-dep ones. Normalization makes every target equally valuable.

**Implementation changes:**
- `step_reward` signature gains `n_deps: int`
- `environment.py:step()` passes `len(true_deps)` to `step_reward`
- `terminal_bonus` stays unchanged (already normalized)
- Also pass `top_k` for FP normalization, or divide by a constant

**Tradeoff:** Harder to tune; the reward magnitude changes with target difficulty. Also changes the existing reward scale the model has learned from, which can cause transient regression.

---

## Design 3: Query Novelty Bonus

**What changes:** Add a diversity reward that penalizes repeating semantically similar queries. The search API already returns `similarity` scores; we can use them as proxy for query overlap.

**Reward formula:**
```python
def novelty_bonus(current_query: str, query_history: list[str],
                  gamma: float = 0.1) -> float:
    if not query_history:
        return gamma
    from difflib import SequenceMatcher
    max_sim = max(
        SequenceMatcher(None, current_query.lower(), q.lower()).ratio()
        for q in query_history
    )
    return gamma * (1.0 - max_sim)
```

This fires as a `step_reward` addend: `r_t += novelty_bonus(query, history, gamma)`.

**Why it matters:** The model frequently repeats near-identical queries (evidenced by `unique_query_rate=0.77` in evals). Repeated queries retrieve the same documents and give zero new TPs. A novelty bonus directly trains diversity without requiring different search results.

**Implementation changes:**
- Add `novelty_bonus()` to `src/train/reward.py`
- Add `novelty_gamma: float` config param
- Pass `state.query_history` into `step_reward` call in `environment.py:step()`
- No SDPO changes needed — already uses total_reward sum

**Better alternative:** Call the `/graph/embedding` API with the query text to get the actual embedding vector, then compute true cosine similarity. This is slower (1 extra API call/step) but precise.

**Tradeoff:** Edit-distance novelty can be gamed — the model learns to add unrelated words to each query. True embedding cosine is more robust but adds latency (~0.5s per step, ×6 steps ×200 targets = meaningful slowdown). A cache makes this feasible.

---

## Design 4: Stop Action (Voluntary Episode Termination)

**What changes:** Allow the model to output `<stop>` instead of a search query. The `finish()` method in `PremiseSelectionEnv` already implements this — it fires `terminal_bonus` early and marks the episode done.

**Prompt change:** Add `<stop>` as a valid action in the system prompt:
```
If you have found all the premises you need, output:
<stop>
instead of a search query.
```

**Training change:** In `_run_group_batched()` in `sdpo.py`, after generating each turn's completion, check if it contains `<stop>`. If so, call `env.finish()` and stop that episode's rollout:

```python
# After decoding completion at each turn:
if "<stop>" in decoded_completion.lower():
    r = env.finish()
    rewards[i] += r
    active[i] = False   # remove from next turn's batch
    continue
```

**Why:** Currently the model wastes turns after it has already retrieved all TPs. With stop, an episode that finds all deps in 3 turns gets `terminal_bonus` at turn 3 and 3 "free" turns of no FP cost. This teaches efficiency. It also creates a meaningful behavioral difference between winner and loser — the winner may stop early, teaching the model when to be confident.

**Tradeoff:** The model may learn to stop too early (before finding all TPs) if the terminal_bonus is not high enough. The beta hyperparameter controls this: higher beta rewards recall more strongly and discourages premature stopping. Requires careful prompt engineering — the model must learn `<stop>` is valid syntax.

---

## Design 5: Per-Step DPO (Step-Level Contrastive Loss)

**What changes:** Replace the current *episode-level* winner/loser comparison with *step-level* comparisons. Rather than asking "which full trajectory was better?", ask "at each decision point, which query was better?"

**Algorithm:**
1. Run `group_size=4` rollouts as normal
2. For each turn `t`, collect all 4 queries `{q_1^t, ..., q_4^t}` and their immediate rewards `r_t`
3. Apply DPO between the highest-reward query at step t (winner step) and the lowest (loser step)
4. Loss is averaged across all turns and all groups

**Why this is better for sparse rewards:**
- Currently: if 3/4 episodes have zero TPs, the winner has reward ~0.01 and no contrastive signal exists
- Per-step DPO: even if all 4 episodes get zero TPs at turn 1, we can still compare which query *approached* the answer better (e.g., by using similarity to known deps as a proxy)
- Creates `horizon × group_size` training pairs per batch instead of just `group_size`

**Implementation changes:**
- Major rewrite of `sdpo_update_step()` in `src/train/sdpo.py`
- `EpisodeRecord.steps` already contains per-step `completion_ids` and `step_reward` — the data is there
- Need per-step `log_ratio` computation (already have `_seq_log_ratio()`)
- New loss: `L = Σ_t -log σ(β * (log_ratio_winner_t - log_ratio_loser_t))`

**Tradeoff:** If all episodes make the same query at a given turn (low temperature), there's no per-step contrastive signal. Requires higher temperature or injection of query diversity (e.g., different sampling seeds per rollout). Also harder to debug — the loss now mixes turns of different quality.

---

## Algorithm-Level Tradeoffs

| Algorithm | Reward Signal | Stability | Sample Efficiency | Best For |
|---|---|---|---|---|
| **SDPO (current)** | Episode total_reward ranking | High (skips zero-reward groups) | Low (67% degenerate) | Well-trained model, sparse rewards |
| **GSPO** | Group-normalized advantage | Medium (geometric IS ratio) | Medium | Exploration phase, untrained model |
| **REINFORCE** | Episode reward - baseline | Low (high variance) | Low | Simple reward landscapes |
| **Per-step DPO** | Per-turn query comparison | Medium | High | Dense-reward settings |
| **PPO** | Clipped advantage, value function | High (with tuning) | High | Long horizons, requires value net |

**SDPO vs GSPO:** SDPO outperformed GSPO (10.0% vs 7.1%) because the model is already trained — GSPO's IS ratio stabilization helps during exploration but adds noise when the policy is already close to optimal. SDPO's winner imitation is cleaner signal.

**GSPO's failure mode here:** With a well-trained model, the 4 rollouts in a group are often nearly identical (same queries, same results). The geometric mean IS ratio collapses to near 1, giving zero gradient. SDPO avoids this by skipping the group entirely rather than applying a near-zero gradient.

**Per-step DPO vs Episode-level DPO:** Per-step creates 6× more training pairs but with noisier signal (a single step's reward is less informative than the full trajectory). The tradeoff is data efficiency vs signal quality.

---

## Recommended Implementation Order

**Phase 1 — False-positive penalty (fastest, highest confidence)**
- Change `alpha: 0.1` in `configs/sdpo_intra_v3.yaml`
- Expected effect: reduce degenerate rate from ~67% to ~40%
- Estimated time: 1 day to eval result
- Risk: low (one config line)

**Phase 2 — Stop action (medium effort)**
- Add `<stop>` parsing in `_run_group_batched()` and `env.finish()` dispatch
- Update system prompt in `configs/prompts/premise_selection.txt`
- Estimated time: ~1 day to implement + test
- Risk: medium (prompt engineering; model may not learn stop reliably)

**Phase 3 — Per-step DPO (higher effort)**
- Rewrite `sdpo_update_step()` to operate at step granularity
- Estimated time: 2–3 days
- Risk: medium-high (fundamentally changes training dynamics; needs careful testing)

**Phase 4 — Query novelty via embeddings + per-step normalization (most expensive)**
- API-based embedding similarity in reward loop
- Per-step recall normalization by `n_deps`
- Estimated time: 1–2 days + tuning
- Risk: latency increase; harder to tune γ

**Not recommended (for now):**
- **Process Reward Model (PRM):** requires a separate labeled dataset of query quality
- **Hindsight Experience Replay (HER):** requires goal-conditioned policy; SDPO doesn't support natively
