"""
GRPO training loop — custom multi-step episode implementation.

Each training step:
  1. Roll out G episodes per target using the current policy (no grad).
  2. Compute group-relative advantages from total episode rewards.
  3. For each (episode, step): forward-pass under current policy → policy gradient.
     KL penalty uses base model (LoRA adapters disabled) as frozen reference.
  4. Accumulate gradients across grad_accum batches, then optimizer.step().

Launch:
    accelerate launch -m src.train.grpo --config configs/grpo_intra.yaml
    accelerate launch -m src.train.grpo --config configs/grpo_inter.yaml
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
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# Silence noisy third-party loggers
for _noisy in ("httpcore", "httpx", "asyncio", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# Tool definition passed to Qwen3's chat template so it emits <tool_call> JSON
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
    """Token IDs for one model generation step within an episode."""
    prompt_ids: list[int]
    completion_ids: list[int]


@dataclass
class EpisodeRecord:
    target_id: UUID
    total_reward: float
    steps: list[StepRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Query parser (shared with LocalAgent)
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
# Batched group rollout — one model.generate([G, seq]) call per horizon step
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
    """
    Run all group_size episodes in parallel with a single batched generate call
    per horizon step. env.step() calls are dispatched concurrently via asyncio.

    vs. the old sequential approach (G × H separate generate calls), this issues
    only H generate calls total, each with batch_size=G — much better GPU
    utilisation on large-VRAM cards.
    """
    from src.env.environment import PremiseSelectionEnv
    from src.env.prompts import format_state

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
    active = list(range(group_size))   # indices of still-running episodes

    for _turn in range(horizon):
        if not active:
            break

        import time as _time
        _t0 = _time.monotonic()

        # ---- Build & batch-tokenize prompts for all active episodes ----
        prompt_texts = [
            tokenizer.apply_chat_template(
                messages_list[i], tools=SEARCH_TOOL, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            for i in active
        ]
        tokenizer.padding_side = "left"   # required for batch decoder-only gen
        enc = tokenizer(
            prompt_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=3072,
        )
        input_ids = enc["input_ids"].to(device)    # [n_active, padded_len]
        attn_mask = enc["attention_mask"].to(device)
        padded_len = input_ids.shape[1]
        logger.info("turn=%d  n_active=%d  input_shape=%s  tok=%.2fs",
                    _turn, len(active), list(input_ids.shape), _time.monotonic()-_t0)

        # ---- Single batched generate ----
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
                    _turn, list(out.shape), _time.monotonic()-_tg)
        # out: [n_active, padded_len + n_new]

        # ---- Recover unpadded prompt_ids and completion_ids per episode ----
        prompt_ids_list: list[list[int]] = []
        completion_ids_list: list[list[int]] = []
        for j in range(len(active)):
            n_real = int(attn_mask[j].sum().item())
            # left-padding: real tokens occupy the LAST n_real columns of input
            p_ids = input_ids[j, padded_len - n_real:].cpu().tolist()
            c_ids = out[j, padded_len:].cpu().tolist()
            prompt_ids_list.append(p_ids)
            completion_ids_list.append(c_ids)

        # Free GPU tensors before async API phase
        del out, input_ids, attn_mask, enc
        torch.cuda.empty_cache()

        completion_texts = [
            tokenizer.decode(c, skip_special_tokens=True)
            for c in completion_ids_list
        ]

        # Log first completion so we can see what model is generating
        if _turn == 0 and completion_texts:
            logger.info("turn=0 sample completion[0]: %r", completion_texts[0][:200])

        # ---- Parse queries; fan-out env.step() calls concurrently ----
        step_tasks: list[tuple[int, str]] = []   # (episode_idx, query)
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

        logger.info("turn=%d  step_tasks=%d  next_active_before_step=%d",
                    _turn, len(step_tasks), len(step_tasks))

        if step_tasks:
            _ta = _time.monotonic()
            results = await asyncio.gather(
                *[envs[i].step(q) for i, q in step_tasks],
                return_exceptions=True,
            )
            logger.info("turn=%d  api done  n=%d  t=%.2fs",
                        _turn, len(step_tasks), _time.monotonic()-_ta)
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
# GRPO gradient update — batched forward pass per horizon step
# ---------------------------------------------------------------------------

def grpo_update_step(
    model,
    optimizer,
    episode_groups: list[list[EpisodeRecord]],
    kl_coef: float,
    device: torch.device,
) -> dict:
    """
    Accumulate GRPO gradients over episode_groups.

    Each episode is forward-passed individually to bound peak GPU memory:
    one [1, seq_len] call at a time rather than one [G, max_seq_len] call.
    Gradients accumulate across episodes; backward() is called per episode
    so PyTorch can free each computation graph immediately.

    Assumes optimizer.zero_grad() was already called.
    Does NOT call optimizer.step() — caller handles accumulation scheduling.
    """
    total_steps = sum(
        1
        for group in episode_groups
        for ep in group
        for step in ep.steps
        if step.completion_ids
    )
    all_rewards = [ep.total_reward for group in episode_groups for ep in group]

    if total_steps == 0:
        return {"loss": 0.0, "mean_reward": sum(all_rewards) / max(len(all_rewards), 1)}

    total_loss_val = 0.0

    for group in episode_groups:
        if not group:
            continue
        rewards = [ep.total_reward for ep in group]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        std_r = rewards_t.std().item()
        if std_r < 1e-8:
            advantages = [0.0] * len(rewards)
        else:
            mean_r = rewards_t.mean().item()
            advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards]

        max_horizon = max((len(ep.steps) for ep in group), default=0)

        for t in range(max_horizon):
            for ep_idx, ep in enumerate(group):
                if t >= len(ep.steps) or not ep.steps[t].completion_ids:
                    continue

                step = ep.steps[t]
                adv = advantages[ep_idx]
                n_p = len(step.prompt_ids)
                n_c = len(step.completion_ids)

                seq = torch.tensor(
                    step.prompt_ids + step.completion_ids,
                    dtype=torch.long, device=device,
                ).unsqueeze(0)                               # [1, seq_len]
                seq_attn = torch.ones(1, n_p + n_c, device=device)

                # Current policy forward pass (with grad)
                logits_k = model(seq, attention_mask=seq_attn).logits   # [1, seq_len, V]

                # Reference policy (adapters off, no grad)
                with model.disable_adapter(), torch.no_grad():
                    ref_logits_k = model(seq, attention_mask=seq_attn).logits

                comp_ids_t = torch.tensor(step.completion_ids, dtype=torch.long, device=device)
                cur_lp = F.log_softmax(logits_k[0, n_p - 1: n_p + n_c - 1].float(), dim=-1)
                token_lp = cur_lp[range(n_c), comp_ids_t].mean()

                ref_lp = F.log_softmax(ref_logits_k[0, n_p - 1: n_p + n_c - 1].float(), dim=-1)
                ref_token_lp = ref_lp[range(n_c), comp_ids_t].mean()

                kl = (token_lp - ref_token_lp).clamp(-10.0, 10.0)
                ep_loss = (-adv * token_lp + kl_coef * kl) / total_steps

                # backward immediately so PyTorch can free the computation graph
                ep_loss.backward()
                total_loss_val += ep_loss.item()

                del seq, seq_attn, logits_k, ref_logits_k, comp_ids_t, cur_lp, ref_lp

    return {
        "loss": total_loss_val,
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load .env so HF_TOKEN, RDS_SECRET_ARN, etc. are available
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

    # Paths
    cache_dir = cfg.get("cache_dir", "cache")
    checkpoint_dir = (
        Path(cfg.get("checkpoint_dir", "checkpoints")) / cfg.get("run_name", "grpo")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = Path("configs/prompts/premise_selection.txt").read_text()

    # Data
    table = cfg.get("table", "rl_train")
    use_intra = cfg.get("use_intra", False)
    logger.info("Loading targets from %s (use_intra=%s)...", table, use_intra)
    targets_dict = load_targets(table=table, cache_dir=cache_dir)
    targets_list = list(targets_dict.values())
    if use_intra:
        targets_list = [t for t in targets_list if t.intra_dep_ids]
        logger.info("Intra-paper targets: %d", len(targets_list))

    baseline_cfg_path = Path("configs/baseline.yaml")
    match_threshold = 88.8
    if baseline_cfg_path.exists():
        match_threshold = (
            yaml.safe_load(baseline_cfg_path.read_text()).get("match_threshold", 88.8)
        )
    dep_bodies = load_dep_bodies(table=table, cache_dir=cache_dir)
    matcher = IDMapper(dep_bodies, threshold=match_threshold)

    # Model
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

    # Hyperparameters
    group_size    = cfg.get("group_size", 8)
    horizon       = cfg.get("horizon", 6)
    top_k         = cfg.get("top_k", 10)
    alpha         = cfg.get("alpha", 0.1)
    beta          = cfg.get("beta", 1.0)
    temperature   = cfg.get("temperature", 0.7)
    max_new_tokens = cfg.get("max_completion_length", 512)
    kl_coef       = cfg.get("kl_coef", 0.04)
    batch_size    = cfg.get("batch_size", 1)
    grad_accum    = cfg.get("grad_accum", 4)
    epochs        = cfg.get("epochs", 3)
    save_steps    = cfg.get("save_steps", 100)
    logging_steps = cfg.get("logging_steps", 10)
    seed          = cfg.get("seed", 42)

    rng = random.Random(seed)
    search_client = SearchClient(cache_dir=os.path.join(cache_dir, "search"))

    log_file = checkpoint_dir / "training_log.jsonl"
    global_step = 0
    accum_count = 0
    optimizer.zero_grad()

    logger.info(
        "Training on %d targets, %d epochs, group_size=%d, batch_size=%d, grad_accum=%d",
        len(targets_list), epochs, group_size, batch_size, grad_accum,
    )

    async def _rollout_batch(batch):
        return await asyncio.gather(*[
            _run_group_batched(
                t, model, tokenizer, search_client, matcher,
                system_prompt, group_size, horizon, top_k,
                alpha, beta, temperature, max_new_tokens, device,
            )
            for t in batch
        ], return_exceptions=True)

    async def _train_loop():
        nonlocal global_step, accum_count

        for epoch in range(epochs):
            epoch_targets = list(targets_list)
            rng.shuffle(epoch_targets)
            logger.info("=== Epoch %d/%d (%d targets) ===", epoch + 1, epochs, len(epoch_targets))

            for batch_start in range(0, len(epoch_targets), batch_size):
                batch = epoch_targets[batch_start: batch_start + batch_size]

                # --- Rollout phase (inference, no grad) ---
                model.eval()
                try:
                    results = await _rollout_batch(batch)
                except Exception as exc:
                    logger.error("Rollout batch failed: %s", exc)
                    continue
                episode_groups = [
                    g for g in results
                    if not isinstance(g, Exception) and g
                ]
                if not episode_groups:
                    continue

                # --- Training phase (with grad, synchronous GPU ops) ---
                torch.cuda.empty_cache()
                model.train()
                metrics = grpo_update_step(
                    model, optimizer, episode_groups, kl_coef, device,
                )
                accum_count += 1

                if accum_count >= grad_accum:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1

                    if global_step % logging_steps == 0:
                        log_entry = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            **metrics,
                        }
                        logger.info(
                            "step=%d  loss=%.4f  mean_reward=%.4f",
                            global_step, metrics["loss"], metrics["mean_reward"],
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
        await search_client.close()

    # Single asyncio.run() for entire training — keeps httpx client and
    # semaphore alive across batches, avoiding per-batch event-loop teardown.
    asyncio.run(_train_loop())


if __name__ == "__main__":
    main()
