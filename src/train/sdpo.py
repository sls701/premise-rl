"""
SDPO training loop — Self-Distillation Policy Optimization.

Core idea: for each group of G rollouts from the same target, learn from
the *winner* (highest-reward episode) via supervised imitation, rather than
computing policy-gradient advantages across all rollouts.

Update per group:
  1. winner e+ = argmax_{e_i} R(e_i)
  2. Skip group if R(e+) <= min_winner_reward (nothing worth imitating).
  3. SFT loss on e+'s completions:
         L_sft = -mean_t[ log π_θ(a_t | context_t) ]
  4. KL regularisation toward the base model (optional):
         L_kl  =  mean_t[ log π_θ(a_t) - log π_ref(a_t) ]
     Combined: L = L_sft + beta_kl * L_kl
             = -(1 - beta_kl) * mean_t[log π_θ] + beta_kl * KL penalty
  5. Contrastive (DPO-style, optional): simultaneously push away from the
     loser e- = argmin R(e_i) when winner ≠ loser:
         L_dpo = -log σ( contrastive_beta *
                         (log_ratio(e+) - log_ratio(e-)) )
     where log_ratio(e) = mean_t[log π_θ(a_t) - log π_ref(a_t)] over e.

This avoids the degeneracy problem of GSPO/GRPO: even when all-but-one
rollout finds no TPs, the single high-reward rollout provides a clean
imitation gradient.  Conversely, when no rollout finds TPs (winner reward=0),
the group is skipped entirely — no corrupted gradient from zero-reward demos.

Structure mirrors src/train/gspo.py (rollout, main loop, logging unchanged).

Launch:
    python -m src.train.sdpo --config configs/sdpo_intra.yaml
"""
from __future__ import annotations

import os
import argparse
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpcore", "httpx", "asyncio", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


class _SafeLogitsProcessor(LogitsProcessor):
    """Guard against bfloat16 overflow producing all-NaN logit rows."""
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
# Batched group rollout (identical to gspo.py — kept self-contained)
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
    novelty_gamma: float,
    temperature: float,
    max_new_tokens: int,
    device,
) -> list[EpisodeRecord]:
    from src.env.environment import PremiseSelectionEnv
    from src.env.prompts import format_state
    import time as _time

    envs = [
        PremiseSelectionEnv(
            search_client=search_client,
            horizon=horizon, top_k=top_k, alpha=alpha, beta=beta,
            novelty_gamma=novelty_gamma,
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
            pad_to_multiple_of=64,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        padded_len = input_ids.shape[1]
        logger.info("turn=%d  n_active=%d  input_shape=%s  tok=%.2fs",
                    _turn, len(active), list(input_ids.shape), _time.monotonic() - _t0)

        _tg = _time.monotonic()
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([_SafeLogitsProcessor()]),
                use_cache=True,
                stop_strings=["</tool_call>"],
                tokenizer=tokenizer,
            )
        torch.cuda.synchronize()
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
                next_active.append(i)
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
# SDPO update step
# ---------------------------------------------------------------------------

def _seq_log_ratio(step: StepRecord, model, device) -> torch.Tensor | None:
    """
    Compute mean_t[log π_θ(a_t) - log π_ref(a_t)] for one step's completion.
    Returns a scalar tensor with grad, or None if numerics are bad.
    """
    n_p = len(step.prompt_ids)
    n_c = len(step.completion_ids)
    if n_c == 0:
        return None

    _ids = step.prompt_ids + step.completion_ids
    seq = torch.tensor(_ids, dtype=torch.long, device=device).unsqueeze(0)
    comp_ids_t = torch.tensor(step.completion_ids, dtype=torch.long, device=device)

    logits = model(seq, use_cache=False).logits
    logit_slice = logits[0, n_p - 1: n_p + n_c - 1].contiguous().float()
    del logits
    if logit_slice.shape[0] != n_c or not torch.isfinite(logit_slice).all():
        del seq, comp_ids_t, logit_slice
        return None
    if comp_ids_t.min() < 0 or comp_ids_t.max() >= logit_slice.shape[-1]:
        del seq, comp_ids_t, logit_slice
        return None

    cur_lp = F.log_softmax(logit_slice, dim=-1)
    cur_token_lp = cur_lp.gather(1, comp_ids_t.view(-1, 1)).squeeze(1)
    del logit_slice, cur_lp

    with model.disable_adapter(), torch.no_grad():
        ref_logits = model(seq, use_cache=False).logits
    ref_slice = ref_logits[0, n_p - 1: n_p + n_c - 1].contiguous().float()
    del ref_logits
    if ref_slice.shape[0] != n_c or not torch.isfinite(ref_slice).all():
        del seq, comp_ids_t, ref_slice, cur_token_lp
        return None

    ref_lp = F.log_softmax(ref_slice, dim=-1)
    ref_token_lp = ref_lp.gather(1, comp_ids_t.view(-1, 1)).squeeze(1).detach()
    del ref_slice, ref_lp, seq, comp_ids_t

    if not torch.isfinite(cur_token_lp).all() or not torch.isfinite(ref_token_lp).all():
        return None

    return (cur_token_lp - ref_token_lp).mean()  # scalar, has grad


def sdpo_update_step(
    model,
    episode_groups: list[list[EpisodeRecord]],
    device: torch.device,
    beta_kl: float = 0.1,
    min_winner_reward: float = 0.0,
    use_contrastive: bool = True,
    contrastive_beta: float = 0.1,
    ema_baseline: float | None = None,
    ema_decay: float = 0.95,
    grpo_lambda: float = 0.0,
) -> dict:
    """
    SDPO gradient update over episode_groups.

    For each group:
      - winner = episode with highest total_reward
      - loser  = episode with lowest total_reward (for contrastive only)
      - Skip groups where winner.total_reward <= min_winner_reward

    SFT loss on winner:
      L_sft = -mean over winner's steps of mean_t[log π_θ(a_t | context_t)]

    KL regularisation (always computed since ref logits are needed for
    contrastive anyway):
      L_kl  = mean over winner's steps of mean_t[log π_θ - log π_ref]
    Combined: L = (1 - beta_kl) * L_sft + beta_kl * L_kl
                = -(1 - beta_kl) * mean[log π_θ] + beta_kl * mean[log π_θ - log π_ref]

    Contrastive (DPO-style, optional):
      L_dpo = -log σ( contrastive_beta * (log_ratio_winner - log_ratio_loser) )
    where log_ratio(e) = mean over e's steps of mean_t[log π_θ - log π_ref].

    Assumes optimizer.zero_grad() was called before this function.
    Does NOT call optimizer.step().
    """
    all_episodes = [ep for group in episode_groups for ep in group]
    all_rewards = [ep.total_reward for ep in all_episodes]
    _empty = {
        "loss": 0.0, "mean_reward": 0.0, "mean_recall": 0.0, "mean_fp": 0.0,
        "reward_std": 0.0, "skipped_groups": 0, "active_groups": 0,
        "ema_baseline": ema_baseline,
    }
    if not all_rewards:
        return _empty

    mean_recall = sum(ep.recall for ep in all_episodes) / len(all_episodes)
    mean_fp = sum(ep.n_fps for ep in all_episodes) / len(all_episodes)
    reward_std = (
        float(torch.tensor(all_rewards, dtype=torch.float32).std().item())
        if len(all_rewards) > 1 else 0.0
    )

    total_loss_val = 0.0
    skipped_groups = 0
    active_groups = 0
    n_groups = sum(1 for g in episode_groups if g)

    for group in episode_groups:
        if not group:
            continue

        rewards = [ep.total_reward for ep in group]
        winner_idx = max(range(len(group)), key=lambda i: rewards[i])
        loser_idx  = min(range(len(group)), key=lambda i: rewards[i])
        winner = group[winner_idx]

        # Update EMA with this group's mean reward for external tracking.
        mean_r = sum(rewards) / len(rewards)
        ema_baseline = (
            mean_r if ema_baseline is None
            else ema_decay * ema_baseline + (1 - ema_decay) * mean_r
        )

        # ---- GRPO auxiliary loss (computed first — applies even for skipped groups) ----
        # Group-normalized advantage gives gradient signal when reward variance exists,
        # including groups where the SDPO winner reward is below threshold.
        grpo_aux = torch.zeros(1, device=device)
        has_grpo_signal = False
        if grpo_lambda > 0.0:
            ep_rewards = [ep.total_reward for ep in group]
            rewards_t = torch.tensor(ep_rewards, dtype=torch.float32)
            reward_std_g = rewards_t.std().item()
            if reward_std_g > 1e-6:
                reward_mean_g = rewards_t.mean().item()
                advantages = [(r - reward_mean_g) / reward_std_g for r in ep_rewards]
                grpo_terms: list[torch.Tensor] = []
                for ep, adv in zip(group, advantages):
                    ep_steps_g = [s for s in ep.steps if s.completion_ids]
                    if not ep_steps_g:
                        continue
                    ep_lrs = [_seq_log_ratio(s, model, device) for s in ep_steps_g]
                    ep_lrs = [lr for lr in ep_lrs if lr is not None]
                    if ep_lrs:
                        ep_seq_lr = torch.stack(ep_lrs).mean()
                        grpo_terms.append(-ep_seq_lr * adv)
                if grpo_terms:
                    grpo_aux = torch.stack(grpo_terms).mean()
                    has_grpo_signal = True

        # ---- Check if SDPO winner imitation should apply ----
        sdpo_applicable = winner.total_reward > min_winner_reward
        winner_steps = [s for s in winner.steps if s.completion_ids] if sdpo_applicable else []

        if not sdpo_applicable and not has_grpo_signal:
            skipped_groups += 1
            logger.debug("group skipped: winner_reward=%.3f <= threshold=%.3f, no GRPO signal",
                         winner.total_reward, min_winner_reward)
            continue

        if not winner_steps and not has_grpo_signal:
            skipped_groups += 1
            continue

        sft_kl_loss = torch.zeros(1, device=device)
        contrastive_loss = torch.zeros(1, device=device)
        sdpo_contributed = False

        if sdpo_applicable and winner_steps:
            torch.cuda.synchronize()

            # ---- Per-step log-ratio: mean_t[log π_θ(a_t) - log π_ref(a_t)] ----
            winner_log_ratios: list[torch.Tensor] = []
            for step in winner_steps:
                lr = _seq_log_ratio(step, model, device)
                if lr is not None:
                    winner_log_ratios.append(lr)

            if winner_log_ratios:
                sdpo_contributed = True
                winner_seq_lr = torch.stack(winner_log_ratios).mean()

                # L_sft + beta_kl * L_kl  (combined into a single expression)
                # cur_token_lp = log π_θ, ref_token_lp = log π_ref
                # Final: L_combined = -(1 - 2*beta_kl) * winner_seq_lr
                # (positive beta_kl < 0.5 adds a mild KL penalty that pushes toward ref)
                sft_kl_loss = -(1.0 - 2.0 * beta_kl) * winner_seq_lr

                # ---- Contrastive loss (DPO-style, optional) ----
                if use_contrastive and loser_idx != winner_idx:
                    loser = group[loser_idx]
                    loser_steps = [s for s in loser.steps if s.completion_ids]
                    if loser_steps:
                        loser_log_ratios: list[torch.Tensor] = []
                        for step in loser_steps:
                            lr = _seq_log_ratio(step, model, device)
                            if lr is not None:
                                loser_log_ratios.append(lr)
                        if loser_log_ratios:
                            loser_seq_lr = torch.stack(loser_log_ratios).mean()
                            margin = contrastive_beta * (winner_seq_lr - loser_seq_lr)
                            margin = margin.clamp(-10.0, 10.0)
                            contrastive_loss = -F.logsigmoid(margin)

        # Nothing to learn from: neither SDPO imitation nor GRPO advantage fired.
        if not sdpo_contributed and not has_grpo_signal:
            skipped_groups += 1
            continue

        group_loss = sft_kl_loss + contrastive_loss + grpo_lambda * grpo_aux
        if not torch.isfinite(group_loss):
            logger.warning(
                "Non-finite group loss (winner_reward=%.3f); skipping backward",
                winner.total_reward,
            )
            skipped_groups += 1
            continue

        active_groups += 1

        group_loss = group_loss / max(n_groups - skipped_groups, 1)
        group_loss.backward()
        total_loss_val += group_loss.item()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    if skipped_groups:
        logger.info("sdpo: %d/%d groups skipped (zero-reward winners)",
                    skipped_groups, n_groups)

    return {
        "loss": total_loss_val,
        "mean_reward": sum(all_rewards) / len(all_rewards),
        "mean_recall": mean_recall,
        "mean_fp": mean_fp,
        "reward_std": reward_std,
        "skipped_groups": skipped_groups,
        "active_groups": active_groups,
        "ema_baseline": ema_baseline,
    }


# ---------------------------------------------------------------------------
# Main training loop
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
        Path(cfg.get("checkpoint_dir", "checkpoints")) / cfg.get("run_name", "sdpo")
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

    logger.info(
        "tokenizer_len=%d model_vocab=%d eos_token_id=%s pad_token_id=%s",
        len(tokenizer), model.config.vocab_size,
        tokenizer.eos_token_id, tokenizer.pad_token_id,
    )
    assert tokenizer.eos_token_id is not None, "tokenizer.eos_token_id must be set"
    assert tokenizer.eos_token_id < model.config.vocab_size, "eos_token_id exceeds model vocab"
    if len(tokenizer) > model.config.vocab_size:
        logger.warning(
            "Tokenizer has more tokens than model vocab (%d > %d); resizing embeddings.",
            len(tokenizer), model.config.vocab_size,
        )
        model.resize_token_embeddings(len(tokenizer))

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
    model.config.use_cache = False
    if cfg.get("resume_from"):
        logger.info("Resuming LoRA adapters from %s", cfg["resume_from"])
        model.load_adapter(cfg["resume_from"], adapter_name="default", is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 1e-5))

    group_size       = cfg.get("group_size", 4)
    horizon          = cfg.get("horizon", 6)
    top_k            = cfg.get("top_k", 20)
    alpha            = cfg.get("alpha", 0.0)
    beta             = cfg.get("beta", 1.0)
    novelty_gamma    = cfg.get("novelty_gamma", 0.0)
    grpo_lambda      = cfg.get("grpo_lambda", 0.0)
    temperature      = cfg.get("temperature", 1.0)
    max_new_tokens   = cfg.get("max_completion_length", 128)
    beta_kl          = cfg.get("beta_kl", 0.1)
    use_contrastive  = cfg.get("use_contrastive", True)
    contrastive_beta = cfg.get("contrastive_beta", 0.1)
    min_winner_reward = cfg.get("min_winner_reward", 0.0)
    ema_decay        = cfg.get("ema_decay", 0.95)
    batch_size       = cfg.get("batch_size", 2)
    grad_accum       = cfg.get("grad_accum", 4)
    epochs           = cfg.get("epochs", 3)
    save_steps       = cfg.get("save_steps", 10)
    logging_steps    = cfg.get("logging_steps", 1)
    seed             = cfg.get("seed", 42)

    base_lr = cfg.get("lr", 1e-5)
    lr_scheduler_type = cfg.get("lr_scheduler", None)
    scheduler = None
    if lr_scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        batches_per_epoch = max(1, len(targets_list) // batch_size)
        total_optim_steps = max(1, (batches_per_epoch // grad_accum) * epochs)
        eta_min = base_lr * cfg.get("lr_min_factor", 0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_optim_steps, eta_min=eta_min)
        logger.info(
            "LR scheduler: cosine  total_steps=%d  lr=%.2e -> eta_min=%.2e",
            total_optim_steps, base_lr, eta_min,
        )

    rng = random.Random(seed)
    cache_dir_search = os.path.join(cache_dir, "search")

    log_file = checkpoint_dir / "training_log.jsonl"
    # Offset global_step from resume checkpoint so saves never overwrite prior steps.
    # e.g. resume_from=.../step_20_backup → global_step starts at 20 → next save is step_30.
    _resume_step = 0
    if cfg.get("resume_from"):
        import re as _re
        _m = _re.search(r"step_(\d+)", str(cfg["resume_from"]))
        if _m:
            _resume_step = int(_m.group(1))
    global_step = _resume_step
    accum_count = 0
    ema_baseline: float | None = None
    optimizer.zero_grad()

    logger.info(
        "SDPO training on %d targets, %d epochs, group_size=%d, batch_size=%d, "
        "grad_accum=%d, beta_kl=%.3f, contrastive=%s, contrastive_beta=%.3f",
        len(targets_list), epochs, group_size, batch_size, grad_accum,
        beta_kl, use_contrastive, contrastive_beta,
    )

    for epoch in range(epochs):
        epoch_targets = list(targets_list)
        rng.shuffle(epoch_targets)
        logger.info("=== Epoch %d/%d (%d targets) ===", epoch + 1, epochs, len(epoch_targets))

        for batch_start in range(0, len(epoch_targets), batch_size):
            batch = epoch_targets[batch_start: batch_start + batch_size]

            torch.cuda.empty_cache()
            model.eval()

            async def _rollout(targets):
                sc = SearchClient(cache_dir=cache_dir_search)
                try:
                    return await asyncio.gather(*[
                        _run_group_batched(
                            t, model, tokenizer, sc,
                            system_prompt, group_size, horizon, top_k,
                            alpha, beta, novelty_gamma, temperature, max_new_tokens, device,
                        )
                        for t in targets
                    ])
                finally:
                    await sc.close()

            try:
                results = asyncio.run(_rollout(batch))
            except RuntimeError as exc:
                logger.exception("Rollout failed")
                if "CUDA" in str(exc) or "cuda" in str(exc):
                    raise
                torch.cuda.empty_cache()
                continue
            except Exception:
                logger.exception("Rollout failed")
                torch.cuda.empty_cache()
                continue

            episode_groups = [g for g in results if g]
            if not episode_groups:
                torch.cuda.empty_cache()
                continue

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            model.train()
            metrics = sdpo_update_step(
                model, episode_groups, device,
                beta_kl=beta_kl,
                min_winner_reward=min_winner_reward,
                use_contrastive=use_contrastive,
                contrastive_beta=contrastive_beta,
                ema_baseline=ema_baseline,
                ema_decay=ema_decay,
                grpo_lambda=grpo_lambda,
            )
            ema_baseline = metrics.get("ema_baseline", ema_baseline)
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
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % logging_steps == 0:
                    log_entry = {
                        "step": global_step, "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                        **metrics,
                    }
                    logger.info(
                        "step=%d  loss=%.4f  mean_reward=%.4f  recall=%.3f  "
                        "active_groups=%d  skipped=%d",
                        global_step, metrics["loss"], metrics["mean_reward"],
                        metrics["mean_recall"], metrics["active_groups"],
                        metrics["skipped_groups"],
                    )
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                if global_step % save_steps == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    logger.info("Checkpoint saved: %s", ckpt_path)

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
            if scheduler is not None:
                scheduler.step()
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
