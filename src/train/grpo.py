"""
REINFORCE training loop — multi-step episode implementation.

Each training step:
  1. Roll out G episodes per target using the current policy (no grad).
     asyncio.run() is called fresh per batch so torch ops never block search callbacks.
  2. Compute within-group advantages (mean/std normalisation over G rewards).
     Falls back to an EMA baseline when group_size == 1.
  3. For each (episode, step): forward pass under current policy → policy gradient.
     No reference-model forward pass — REINFORCE, not GRPO.
  4. Accumulate gradients across grad_accum batches, then optimizer.step().

Launch:
    python -m src.train.grpo --config configs/grpo_intra.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import concurrent.futures

import torch
import torch.nn.functional as F
import yaml
from transformers import LogitsProcessor, LogitsProcessorList

# Single-worker executor so model.generate() runs in a thread, leaving the
# asyncio event loop live to drain search-API responses immediately instead
# of waiting behind the 2-3s blocking GPU call. One worker => CUDA ops are
# serialised, no concurrent-allocator deadlock.
_GENERATE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpcore", "httpx", "asyncio", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class _SafeLogitsProcessor(LogitsProcessor):
    """Guard against bfloat16 overflow producing all-NaN logit rows.

    InfNanRemoveLogitsProcessor replaces NaN→-inf, which is correct for
    individual bad values, but when ALL logits in a row are NaN (full bfloat16
    overflow in some layer) they all become -inf and softmax returns NaN again,
    triggering the multinomial device-side assertion.  This processor adds one
    more step: if an entire row is -inf after cleanup, replace with zeros so
    softmax produces a uniform distribution over the vocabulary.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, nan=float("-inf"), posinf=torch.finfo(scores.dtype).max)
        all_neg_inf = (scores == float("-inf")).all(dim=-1, keepdim=True)
        if all_neg_inf.any():
            logger.warning("logits all -inf for %d token(s); falling back to uniform sampling",
                           int(all_neg_inf.sum().item()))
            scores = scores.masked_fill(all_neg_inf.expand_as(scores), 0.0)
        return scores


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_theorems",
        "description": "Search the theorem corpus by natural language query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    },
}]


# ---------------------------------------------------------------------------
# Rollout data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    prompt_ids: list[int]
    completion_ids: list[int]


@dataclass
class EpisodeRecord:
    target_id: UUID
    total_reward: float
    recall: float = 0.0
    n_tps: int = 0
    n_fps: int = 0
    steps: list[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

def _fill_episode_stats(ep: "EpisodeRecord", env) -> None:
    traj = env.get_trajectory()
    ep.total_reward = traj.total_reward
    n_true = max(len(traj.final_true_dep_ids), 1)
    ep.recall = len(traj.final_retrieved_uuids & traj.final_true_dep_ids) / n_true
    ep.n_tps = sum(len(s.new_tps) for s in traj.steps)
    ep.n_fps = sum(len(s.new_fps) for s in traj.steps)


def _parse_query(text: str) -> str | None:
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            if data.get("name") == "search_theorems":
                args = data.get("arguments", data.get("parameters", {}))
                return args.get("query")
        except (json.JSONDecodeError, KeyError):
            pass
    m2 = re.search(r'\{"query":\s*"([^"]+)"', text)
    if m2:
        return m2.group(1)
    return None


# ---------------------------------------------------------------------------
# Batched group rollout
# ---------------------------------------------------------------------------

async def _run_group_batched(
    target,
    model,
    tokenizer,
    search_client,
    system_prompt: str,
    group_size: int,
    horizon: int,
    top_k: int,
    alpha: float,
    beta: float,
    temperature: float,
    max_new_tokens: int,
    device,
) -> list[EpisodeRecord]:
    """
    Run group_size episodes for one target with a single batched generate call
    per horizon step. env.step() calls are dispatched concurrently via asyncio.
    """
    from src.env.environment import PremiseSelectionEnv
    from src.env.prompts import format_state
    import time as _time

    envs = [
        PremiseSelectionEnv(
            search_client=search_client,
            horizon=horizon, top_k=top_k, alpha=alpha, beta=beta,
        )
        for _ in range(group_size)
    ]
    states = [env.reset(target) for env in envs]
    messages_list: list[list[dict]] = [
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": format_state(s)}]
        for s in states
    ]
    episodes = [
        EpisodeRecord(target_id=target.statement_id, total_reward=0.0)
        for _ in range(group_size)
    ]
    active = list(range(group_size))

    for _turn in range(horizon):
        if not active:
            break

        _t0 = _time.monotonic()
        prompt_texts = [
            tokenizer.apply_chat_template(
                messages_list[i], tools=SEARCH_TOOL, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            for i in active
        ]
        tokenizer.padding_side = "left"
        enc = tokenizer(
            prompt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=1536,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        padded_len = input_ids.shape[1]
        logger.info("turn=%d  n_active=%d  input_shape=%s  tok=%.2fs",
                    _turn, len(active), list(input_ids.shape), _time.monotonic() - _t0)

        _tg = _time.monotonic()
        _loop = asyncio.get_running_loop()
        _ids, _mask = input_ids, attn_mask  # capture for the lambda

        def _do_generate() -> torch.Tensor:
            with torch.no_grad():
                return model.generate(
                    _ids,
                    attention_mask=_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([_SafeLogitsProcessor()]),
                )

        out = await _loop.run_in_executor(_GENERATE_EXECUTOR, _do_generate)
        logger.info("turn=%d  generate done  shape=%s  gen=%.2fs",
                    _turn, list(out.shape), _time.monotonic() - _tg)

        prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        for j in range(len(active)):
            n_real = int(attn_mask[j].sum().item())
            p_ids = input_ids[j, padded_len - n_real:].cpu().tolist()
            c_ids = out[j, padded_len:].cpu().tolist()
            prompt_ids_list.append(p_ids)
            completion_ids_list.append(c_ids)

        del out, input_ids, attn_mask, enc
        # Do NOT call empty_cache() here: the executor can start the next generate() immediately
        # after finishing this one, before asyncio runs this coroutine's continuation.
        # That races on the CUDA allocator mutex and corrupts GPU state. The post-rollout
        # empty_cache() (before model.train()) is the safe point — executor is guaranteed idle.

        completion_texts = [
            tokenizer.decode(c, skip_special_tokens=True)
            for c in completion_ids_list
        ]

        if _turn == 0 and completion_texts:
            logger.info("turn=0 sample completion[0]: %r", completion_texts[0][:200])

        step_tasks: list[tuple[int, str]] = []
        next_active: list[int] = []

        for j, i in enumerate(active):
            query = _parse_query(completion_texts[j])
            if query is None:
                if not envs[i]._state.done:
                    envs[i].finish()
                _fill_episode_stats(episodes[i], envs[i])
            else:
                episodes[i].steps.append(StepRecord(
                    prompt_ids=prompt_ids_list[j],
                    completion_ids=completion_ids_list[j],
                ))
                step_tasks.append((i, query))

        if step_tasks:
            _ta = _time.monotonic()
            results = await asyncio.gather(
                *[envs[i].step(q) for i, q in step_tasks],
                return_exceptions=True,
            )
            logger.info("turn=%d  api done  n=%d  t=%.2fs",
                        _turn, len(step_tasks), _time.monotonic() - _ta)
            for k, (i, query) in enumerate(step_tasks):
                res = results[k]
                if isinstance(res, BaseException):
                    logger.warning("env.step failed: %s", res)
                    if not envs[i]._state.done:
                        envs[i].finish()
                    _fill_episode_stats(episodes[i], envs[i])
                else:
                    state, _, done, _ = res
                    messages_list[i].append({
                        "role": "assistant",
                        "content": f"[search_theorems] query={query!r}",
                    })
                    messages_list[i].append({
                        "role": "user",
                        "content": f"[tool_result]\n{format_state(state)}",
                    })
                    if done:
                        _fill_episode_stats(episodes[i], envs[i])
                    else:
                        next_active.append(i)

        active = next_active

    for i in active:
        if not envs[i]._state.done:
            envs[i].finish()
        _fill_episode_stats(episodes[i], envs[i])

    return episodes


# ---------------------------------------------------------------------------
# REINFORCE update — no reference model, within-group or EMA baseline
# ---------------------------------------------------------------------------

def reinforce_update_step(
    model,
    episode_groups: list[list[EpisodeRecord]],
    baseline: list[float],
    ema_decay: float,
    device: torch.device,
) -> dict:
    """
    Accumulate REINFORCE gradients over episode_groups.

    Advantage per episode:
      - group_size > 1: within-group (R - mean) / (std + eps)  [degenerate groups skipped]
      - group_size == 1: R - EMA_baseline, then update EMA

    One forward pass per (episode, step) at shape [1, seq_len].
    No reference-model forward pass.
    Assumes optimizer.zero_grad() was called before this function.
    Does NOT call optimizer.step().
    """
    all_episodes = [ep for group in episode_groups for ep in group]
    all_rewards = [ep.total_reward for ep in all_episodes]
    if not all_rewards:
        return {"loss": 0.0, "mean_reward": 0.0, "baseline": baseline[0]}

    mean_recall = sum(ep.recall for ep in all_episodes) / len(all_episodes)
    mean_fp = sum(ep.n_fps for ep in all_episodes) / len(all_episodes)
    reward_std = float(torch.tensor(all_rewards, dtype=torch.float32).std().item()) if len(all_rewards) > 1 else 0.0

    total_steps = sum(
        1
        for ep in all_episodes
        for step in ep.steps
        if step.completion_ids
    )
    if total_steps == 0:
        return {
            "loss": 0.0, "mean_reward": sum(all_rewards) / len(all_rewards),
            "mean_recall": mean_recall, "mean_fp": mean_fp, "reward_std": reward_std,
            "baseline": baseline[0],
        }

    total_loss_val = 0.0
    degenerate_groups = 0

    n_groups = sum(1 for g in episode_groups if g)
    for group in episode_groups:
        if not group:
            continue
        rewards = [ep.total_reward for ep in group]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        if len(group) > 1:
            std_r = rewards_t.std().item()
            if std_r < 1e-8:
                degenerate_groups += 1
                continue
            mean_r = rewards_t.mean().item()
            advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]
        else:
            b = baseline[0]
            advantages = [rewards[0] - b]
            baseline[0] = ema_decay * b + (1 - ema_decay) * rewards[0]

        for ep_idx, ep in enumerate(group):
            adv = advantages[ep_idx]
            for step in ep.steps:
                if not step.completion_ids:
                    continue

                n_p = len(step.prompt_ids)
                n_c = len(step.completion_ids)
                seq = torch.tensor(
                    step.prompt_ids + step.completion_ids,
                    dtype=torch.long, device=device,
                ).unsqueeze(0)

                # attention_mask=None lets SDPA choose flash/mem-efficient backend
                # (explicit all-ones mask forces the math backend, materialising
                # the full [1, heads, n, n] attention matrix for all 28 layers).
                logits = model(seq).logits
                logit_slice = logits[0, n_p - 1: n_p + n_c - 1].contiguous().float()
                if logit_slice.isnan().any():
                    logger.warning("NaN logits in backward (n_p=%d, n_c=%d); skipping", n_p, n_c)
                    del seq, logits, comp_ids_t, logit_slice
                    continue
                comp_ids_t = torch.tensor(step.completion_ids, dtype=torch.long, device=device)
                log_probs = F.log_softmax(logit_slice, dim=-1)
                token_lp = log_probs[range(n_c), comp_ids_t].mean()

                loss = (-adv * token_lp) / total_steps
                loss.backward()
                total_loss_val += loss.item()

                del seq, logits, comp_ids_t, log_probs

        # Sync + defrag after each group's backward passes.
        # Sequential backward() calls fragment the CUDA allocator; without
        # synchronize() the GPU may still be executing kernels when empty_cache()
        # runs, so in-flight activations appear "live" and aren't reclaimed.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if degenerate_groups:
        logger.warning("degenerate_groups=%d (all rewards tied — no gradient signal)", degenerate_groups)

    return {
        "loss": total_loss_val,
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "mean_recall": mean_recall,
        "mean_fp": mean_fp,
        "reward_std": reward_std,
        "baseline": baseline[0],
        "degenerate_frac": degenerate_groups / max(n_groups, 1),
        "degenerate_groups": degenerate_groups,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    from src.data.load import load_targets
    from src.env.search_client import SearchClient

    cache_dir = cfg.get("cache_dir", "cache")
    checkpoint_dir = (
        Path(cfg.get("checkpoint_dir", "checkpoints")) / cfg.get("run_name", "reinforce")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = Path("configs/prompts/premise_selection.txt").read_text()

    table = cfg.get("table", "rl_train")
    use_intra = cfg.get("use_intra", False)
    logger.info("Loading targets from %s (use_intra=%s)...", table, use_intra)
    targets_dict = load_targets(table=table, cache_dir=cache_dir)
    targets_list = list(targets_dict.values())
    if use_intra:
        targets_list = [t for t in targets_list if t.intra_dep_ids]
        logger.info("Intra-paper targets: %d", len(targets_list))

    max_targets = cfg.get("max_targets")
    if max_targets is not None and len(targets_list) > max_targets:
        rng_sub = random.Random(cfg.get("seed", 42))
        targets_list = rng_sub.sample(targets_list, max_targets)
        logger.info("Subsampled to %d targets (seed=%d)", len(targets_list), cfg.get("seed", 42))

    model_id = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s on %s...", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="sdpa",
    )
    lora_cfg = LoraConfig(
        r=cfg.get("lora_rank", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("lora_targets", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if cfg.get("resume_from"):
        logger.info("Resuming LoRA adapters from %s", cfg["resume_from"])
        model.load_adapter(cfg["resume_from"], adapter_name="default", is_trainable=True)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 1e-6))

    group_size    = cfg.get("group_size", 4)
    horizon       = cfg.get("horizon", 6)
    top_k         = cfg.get("top_k", 10)
    alpha         = cfg.get("alpha", 0.1)
    beta          = cfg.get("beta", 1.0)
    temperature   = cfg.get("temperature", 0.7)
    max_new_tokens = cfg.get("max_completion_length", 128)
    batch_size    = cfg.get("batch_size", 2)
    grad_accum    = cfg.get("grad_accum", 4)
    epochs        = cfg.get("epochs", 3)
    save_steps    = cfg.get("save_steps", 100)
    logging_steps = cfg.get("logging_steps", 10)
    ema_decay     = cfg.get("ema_decay", 0.99)
    seed          = cfg.get("seed", 42)

    rng = random.Random(seed)
    cache_dir_search = os.path.join(cache_dir, "search")

    log_file = checkpoint_dir / "training_log.jsonl"
    global_step = 0
    accum_count = 0
    baseline = [0.0]   # mutable EMA baseline (used only when group_size == 1)
    optimizer.zero_grad()

    logger.info(
        "Training on %d targets, %d epochs, group_size=%d, batch_size=%d, grad_accum=%d",
        len(targets_list), epochs, group_size, batch_size, grad_accum,
    )

    for epoch in range(epochs):
        epoch_targets = list(targets_list)
        rng.shuffle(epoch_targets)
        logger.info("=== Epoch %d/%d (%d targets) ===", epoch + 1, epochs, len(epoch_targets))

        for batch_start in range(0, len(epoch_targets), batch_size):
            batch = epoch_targets[batch_start: batch_start + batch_size]

            # ---- Rollout phase ----
            # Fresh asyncio.run() per batch: the event loop is torn down before
            # torch backward runs, so search-timeout callbacks are never delayed
            # by blocking GPU operations.
            # Defrag before each rollout: varying KV-cache sizes across turns
            # fragment the pool; left uncleared, this causes illegal-memory-access
            # errors on batch 7-8 when a new allocation hits a fragmented gap.
            torch.cuda.empty_cache()
            model.eval()

            async def _rollout(targets):
                sc = SearchClient(cache_dir=cache_dir_search)
                try:
                    return await asyncio.gather(*[
                        _run_group_batched(
                            t, model, tokenizer, sc,
                            system_prompt, group_size, horizon, top_k,
                            alpha, beta, temperature, max_new_tokens, device,
                        )
                        for t in targets
                    ], return_exceptions=True)
                finally:
                    await sc.close()

            try:
                results = asyncio.run(_rollout(batch))
            except Exception as exc:
                logger.error("Rollout failed: %s", exc)
                continue

            for r in results:
                if isinstance(r, Exception):
                    logger.error("Rollout group failed: %s: %s", type(r).__name__, r)
            episode_groups = [g for g in results if not isinstance(g, Exception) and g]
            if not episode_groups:
                continue

            # ---- Update phase (synchronous — no event loop active) ----
            torch.cuda.empty_cache()
            model.train()
            metrics = reinforce_update_step(
                model, episode_groups, baseline, ema_decay, device,
            )
            accum_count += 1

            if accum_count >= grad_accum:
                _nan_g = sum(
                    1 for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                    and not p.grad.isfinite().all()
                )
                if _nan_g:
                    logger.warning("NaN/inf in %d grad tensors; zeroing before step", _nan_g)
                    for p in model.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.nan_to_num_(0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % logging_steps == 0:
                    log_entry = {"step": global_step, "epoch": epoch + 1, **metrics}
                    logger.info(
                        "step=%d  loss=%.4f  mean_reward=%.4f  recall=%.3f  mean_fp=%.1f  baseline=%.4f  degen=%.2f",
                        global_step, metrics["loss"], metrics["mean_reward"],
                        metrics["mean_recall"], metrics["mean_fp"],
                        metrics["baseline"], metrics["degenerate_frac"],
                    )
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                if global_step % save_steps == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    logger.info("Checkpoint saved: %s", ckpt_path)

        # Flush remaining gradients at epoch end
        if accum_count > 0:
            _nan_g = sum(
                1 for p in model.parameters()
                if p.requires_grad and p.grad is not None
                and not p.grad.isfinite().all()
            )
            if _nan_g:
                logger.warning("NaN/inf in %d grad tensors (epoch flush); zeroing", _nan_g)
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.nan_to_num_(0.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            global_step += 1
            log_entry = {"step": global_step, "epoch": epoch + 1, **metrics, "epoch_end": True}
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

    model.save_pretrained(str(checkpoint_dir / "final"))
    logger.info("Training complete. Final model: %s/final", checkpoint_dir)


if __name__ == "__main__":
    main()
