"""
GSPO training loop — Group Sequence Policy Optimization.

Reference: https://arxiv.org/abs/2507.18071 (Qwen/Alibaba)

Key difference from GRPO: importance ratio is sequence-level with length
normalization, not per-token:

    s_i(θ) = exp( (1/|y_i|) * Σ_t log(π_θ(t) / π_old(t)) )

This is the geometric mean of per-token ratios, preventing long sequences
from numerically dominating.  Loss per episode:

    L = -min( s_i * Â_i,  clip(s_i, 1-ε, 1+ε) * Â_i )

Advantages are within-group (mean/std over G rollouts), same as GRPO.

Structure mirrors src/train/grpo.py (REINFORCE):
  - asyncio.run() is called fresh per batch — torch ops never block search
    callbacks.
  - The reference policy is the base model (LoRA adapters disabled via
    model.disable_adapter()).  Only the update step differs from REINFORCE.

Launch:
    python -m src.train.gspo --config configs/gspo_intra.yaml
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

import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpcore", "httpx", "asyncio", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


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
    steps: list[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------

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
# Batched group rollout (identical to grpo.py — kept self-contained)
# ---------------------------------------------------------------------------

async def _run_group_batched(
    target,
    model,
    tokenizer,
    search_client,
    matcher,
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
    from src.env.environment import PremiseSelectionEnv
    from src.env.prompts import format_state
    import time as _time

    envs = [
        PremiseSelectionEnv(
            search_client=search_client, matcher=matcher,
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
            truncation=True, max_length=3072,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        padded_len = input_ids.shape[1]
        logger.info("turn=%d  n_active=%d  input_shape=%s  tok=%.2fs",
                    _turn, len(active), list(input_ids.shape), _time.monotonic() - _t0)

        _tg = _time.monotonic()
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
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
        torch.cuda.empty_cache()

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
                episodes[i].total_reward = envs[i].get_trajectory().total_reward
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
                    episodes[i].total_reward = envs[i].get_trajectory().total_reward
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
                        episodes[i].total_reward = envs[i].get_trajectory().total_reward
                    else:
                        next_active.append(i)

        active = next_active

    for i in active:
        if not envs[i]._state.done:
            envs[i].finish()
        episodes[i].total_reward = envs[i].get_trajectory().total_reward

    return episodes


# ---------------------------------------------------------------------------
# GSPO update step
# ---------------------------------------------------------------------------

def gspo_update_step(
    model,
    episode_groups: list[list[EpisodeRecord]],
    device: torch.device,
    clip_eps: float = 0.2,
) -> dict:
    """
    GSPO gradient update over episode_groups.

    For each episode:
      1. Compute per-token log-ratio:  log π_θ(t) - log π_old(t)
         where π_old = base model (LoRA adapters disabled, no grad).
      2. Length-normalised sequence ratio:
             s_i = exp( mean_t[ log π_θ(t) - log π_old(t) ] )
      3. Within-group advantage Â_i = (r_i - mean_r) / std_r.
      4. Clipped surrogate: -min(s_i * Â_i, clip(s_i, 1-ε, 1+ε) * Â_i).

    Degenerate groups (all rewards identical) are skipped — no gradient signal.
    Assumes optimizer.zero_grad() was already called.
    Does NOT call optimizer.step().
    """
    all_rewards = [ep.total_reward for group in episode_groups for ep in group]
    if not all_rewards:
        return {"loss": 0.0, "mean_reward": 0.0}

    # Count non-empty steps for loss normalisation
    total_steps = sum(
        1
        for group in episode_groups
        for ep in group
        for step in ep.steps
        if step.completion_ids
    )
    if total_steps == 0:
        return {"loss": 0.0, "mean_reward": sum(all_rewards) / len(all_rewards)}

    total_loss_val = 0.0
    degenerate_groups = 0

    for group in episode_groups:
        if not group:
            continue

        rewards = [ep.total_reward for ep in group]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        std_r = rewards_t.std().item()
        if std_r < 1e-8:
            degenerate_groups += 1
            continue
        mean_r = rewards_t.mean().item()
        advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

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
                ).unsqueeze(0)                                      # [1, n_p+n_c]
                seq_attn = torch.ones(1, n_p + n_c, device=device)
                comp_ids_t = torch.tensor(
                    step.completion_ids, dtype=torch.long, device=device,
                )                                                   # [n_c]

                # ---- Current policy (with grad) ----
                logits = model(seq, attention_mask=seq_attn).logits  # [1, seq_len, V]
                cur_lp = F.log_softmax(
                    logits[0, n_p - 1: n_p + n_c - 1].float(), dim=-1
                )                                                   # [n_c, V]
                cur_token_lp = cur_lp[range(n_c), comp_ids_t]      # [n_c]

                # ---- Reference policy (base model, no grad) ----
                with model.disable_adapter(), torch.no_grad():
                    ref_logits = model(seq, attention_mask=seq_attn).logits
                ref_lp = F.log_softmax(
                    ref_logits[0, n_p - 1: n_p + n_c - 1].float(), dim=-1
                )
                ref_token_lp = ref_lp[range(n_c), comp_ids_t]      # [n_c]

                # ---- Length-normalised sequence importance ratio ----
                # s_i = exp( mean_t[ log π_θ(t) - log π_old(t) ] )
                mean_log_ratio = (cur_token_lp - ref_token_lp.detach()).mean()
                mean_log_ratio = mean_log_ratio.clamp(-10.0, 10.0)  # numerical guard
                s_i = mean_log_ratio.exp()                           # scalar, has grad

                # ---- Clipped surrogate (PPO-style) ----
                s_i_clipped = s_i.clamp(1.0 - clip_eps, 1.0 + clip_eps)
                surr1 = s_i * adv
                surr2 = s_i_clipped * adv
                step_loss = -torch.min(surr1, surr2) / total_steps

                step_loss.backward()
                total_loss_val += step_loss.item()

                del seq, seq_attn, logits, ref_logits, comp_ids_t
                del cur_lp, ref_lp, cur_token_lp, ref_token_lp

    if degenerate_groups:
        logger.warning("degenerate_groups=%d (all rewards tied — no gradient signal)", degenerate_groups)

    return {
        "loss": total_loss_val,
        "mean_reward": sum(all_rewards) / len(all_rewards),
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

    from src.data.load import load_targets, load_dep_bodies
    from src.env.id_mapping import IDMapper
    from src.env.search_client import SearchClient

    cache_dir = cfg.get("cache_dir", "cache")
    checkpoint_dir = (
        Path(cfg.get("checkpoint_dir", "checkpoints")) / cfg.get("run_name", "gspo")
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

    baseline_cfg_path = Path("configs/baseline.yaml")
    match_threshold = 88.8
    if baseline_cfg_path.exists():
        match_threshold = yaml.safe_load(
            baseline_cfg_path.read_text()
        ).get("match_threshold", 88.8)
    dep_bodies = load_dep_bodies(table=table, cache_dir=cache_dir)
    matcher = IDMapper(dep_bodies, threshold=match_threshold)

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
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 1e-6))

    group_size     = cfg.get("group_size", 4)
    horizon        = cfg.get("horizon", 6)
    top_k          = cfg.get("top_k", 10)
    alpha          = cfg.get("alpha", 0.1)
    beta           = cfg.get("beta", 1.0)
    temperature    = cfg.get("temperature", 0.7)
    max_new_tokens = cfg.get("max_completion_length", 128)
    clip_eps       = cfg.get("clip_eps", 0.2)
    batch_size     = cfg.get("batch_size", 2)
    grad_accum     = cfg.get("grad_accum", 4)
    epochs         = cfg.get("epochs", 3)
    save_steps     = cfg.get("save_steps", 100)
    logging_steps  = cfg.get("logging_steps", 10)
    seed           = cfg.get("seed", 42)

    rng = random.Random(seed)
    cache_dir_search = os.path.join(cache_dir, "search")

    log_file = checkpoint_dir / "training_log.jsonl"
    global_step = 0
    accum_count = 0
    optimizer.zero_grad()

    logger.info(
        "GSPO training on %d targets, %d epochs, group_size=%d, batch_size=%d, "
        "grad_accum=%d, clip_eps=%.2f",
        len(targets_list), epochs, group_size, batch_size, grad_accum, clip_eps,
    )

    for epoch in range(epochs):
        epoch_targets = list(targets_list)
        rng.shuffle(epoch_targets)
        logger.info("=== Epoch %d/%d (%d targets) ===", epoch + 1, epochs, len(epoch_targets))

        for batch_start in range(0, len(epoch_targets), batch_size):
            batch = epoch_targets[batch_start: batch_start + batch_size]

            # ---- Rollout (fresh event loop per batch) ----
            model.eval()

            async def _rollout(targets):
                sc = SearchClient(cache_dir=cache_dir_search)
                try:
                    return await asyncio.gather(*[
                        _run_group_batched(
                            t, model, tokenizer, sc, matcher,
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

            episode_groups = [g for g in results if not isinstance(g, Exception) and g]
            if not episode_groups:
                continue

            # ---- GSPO update (synchronous — event loop fully torn down) ----
            torch.cuda.empty_cache()
            model.train()
            metrics = gspo_update_step(
                model, episode_groups, device, clip_eps=clip_eps,
            )
            accum_count += 1

            if accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % logging_steps == 0:
                    log_entry = {"step": global_step, "epoch": epoch + 1, **metrics}
                    logger.info(
                        "step=%d  loss=%.4f  mean_reward=%.4f  degenerate=%d",
                        global_step, metrics["loss"], metrics["mean_reward"],
                        metrics.get("degenerate_groups", 0),
                    )
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")

                if global_step % save_steps == 0:
                    ckpt_path = checkpoint_dir / f"step_{global_step}"
                    model.save_pretrained(str(ckpt_path))
                    logger.info("Checkpoint saved: %s", ckpt_path)

        # Flush remaining gradients at epoch end
        if accum_count > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            global_step += 1

    model.save_pretrained(str(checkpoint_dir / "final"))
    logger.info("Training complete. Final model: %s/final", checkpoint_dir)


if __name__ == "__main__":
    main()
