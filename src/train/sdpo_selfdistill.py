"""
SDPO — Self-Distillation Policy Optimization, faithful to Hübotter et al.,
"Reinforcement Learning via Self-Distillation" (arXiv 2601.20802).

This is the PAPER's SDPO, distinct from src/train/sdpo.py (which is a
winner-imitation + DPO-contrastive hybrid that merely shares the acronym).

Core idea
---------
For each prompt x we sample G rollouts with the (feedback-free) student.
We then build feedback f for each rollout from a *successful sibling* rollout
in the same group (the highest-recall one), and distil the feedback-conditioned
self-teacher into the feedback-free student:

    L_SDPO(theta) = sum_t  JS( pi_theta(.|x, y_<t)  ||  stopgrad pi_teacher(.|x, f, y_<t) )

  - teacher = an EMA-regularised copy of the student's LoRA weights (alpha=0.01),
    conditioned on the feedback f; stopgrad blocks the teacher from regressing
    toward the student and ignoring f.
  - student = the live LoRA weights, conditioned WITHOUT feedback.
  - symmetric Jensen-Shannon divergence, approximated over the teacher's top-K
    tokens (+ a tail bucket) for memory efficiency (paper uses K=100).
  - loss is masked to the completion (generated query) tokens.

Premise-selection feedback
--------------------------
The feedback is the best sibling rollout's queries — a self-generated successful
search strategy, NOT ground-truth leakage. Groups whose best rollout finds zero
dependencies carry no positive teacher signal and are skipped.

Launch:
    python -m src.train.sdpo_selfdistill --config configs/sdpo_sd_8b_v1.yaml
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import torch
import torch.nn.functional as F
from transformers import LogitsProcessorList

from src.train.sdpo import (
    SEARCH_TOOL,
    _SafeLogitsProcessor,
    _parse_query,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for _noisy in ("httpcore", "httpx", "asyncio", "urllib3", "filelock", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout data structures (carry messages_prefix + query for teacher rebuild)
# ---------------------------------------------------------------------------

@dataclass
class SDStep:
    messages_prefix: list[dict]   # conversation BEFORE this turn's assistant query
    completion_ids: list[int]     # generated tool-call tokens
    query: str


@dataclass
class SDEpisode:
    target_id: UUID
    recall: float = 0.0
    total_reward: float = 0.0
    queries: list[str] = field(default_factory=list)
    steps: list[SDStep] = field(default_factory=list)


def _fill_stats(ep: SDEpisode, env) -> None:
    traj = env.get_trajectory()
    ep.total_reward = traj.total_reward
    n_true = max(len(traj.final_true_dep_ids), 1)
    ep.recall = len(traj.final_retrieved_uuids & traj.final_true_dep_ids) / n_true


# ---------------------------------------------------------------------------
# Group rollout (feedback-free student) — adapted from sdpo._run_group_batched,
# additionally capturing each step's messages prefix and query text.
# ---------------------------------------------------------------------------

async def _rollout_group(
    target, model, tokenizer, search_client, system_prompt: str,
    group_size: int, horizon: int, top_k: int, alpha: float, beta: float,
    novelty_gamma: float, temperature: float, max_new_tokens: int, device,
) -> list[SDEpisode]:
    from src.env.environment import PremiseSelectionEnv
    from src.env.prompts import format_state

    envs = [
        PremiseSelectionEnv(
            search_client=search_client, horizon=horizon, top_k=top_k,
            alpha=alpha, beta=beta, novelty_gamma=novelty_gamma,
        )
        for _ in range(group_size)
    ]
    states = [env.reset(target) for env in envs]
    messages_list = [
        [{"role": "system", "content": system_prompt},
         {"role": "user", "content": format_state(s)}]
        for s in states
    ]
    episodes = [SDEpisode(target_id=target.statement_id) for _ in range(group_size)]
    active = list(range(group_size))

    for _turn in range(horizon):
        if not active:
            break
        prompt_texts = [
            tokenizer.apply_chat_template(
                messages_list[i], tools=SEARCH_TOOL, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            for i in active
        ]
        tokenizer.padding_side = "left"
        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=1536, pad_to_multiple_of=64)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        padded_len = input_ids.shape[1]

        torch.cuda.empty_cache()
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attn_mask, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                logits_processor=LogitsProcessorList([_SafeLogitsProcessor()]),
                use_cache=True, stop_strings=["</tool_call>"], tokenizer=tokenizer,
            )
        torch.cuda.synchronize()

        completion_ids_list = [out[j, padded_len:].cpu().tolist() for j in range(len(active))]
        del out, input_ids, attn_mask, enc
        completion_texts = [tokenizer.decode(c, skip_special_tokens=True) for c in completion_ids_list]

        step_tasks: list[tuple[int, str]] = []
        next_active: list[int] = []
        for j, i in enumerate(active):
            query = _parse_query(completion_texts[j])
            if query is None:
                next_active.append(i)
                continue
            episodes[i].steps.append(SDStep(
                messages_prefix=copy.deepcopy(messages_list[i]),
                completion_ids=completion_ids_list[j],
                query=query,
            ))
            episodes[i].queries.append(query)
            step_tasks.append((i, query))

        if step_tasks:
            results = await asyncio.gather(
                *[envs[i].step(q) for i, q in step_tasks], return_exceptions=True,
            )
            for k, (i, query) in enumerate(step_tasks):
                res = results[k]
                if isinstance(res, BaseException):
                    logger.warning("env.step failed: %s", res)
                    if not envs[i]._state.done:
                        envs[i].finish()
                    _fill_stats(episodes[i], envs[i])
                    continue
                state, _, done, _ = res
                messages_list[i].append({"role": "assistant", "content": f"[search_theorems] query={query!r}"})
                messages_list[i].append({"role": "user", "content": f"[tool_result]\n{format_state(state)}"})
                if done:
                    _fill_stats(episodes[i], envs[i])
                else:
                    next_active.append(i)
        active = next_active

    for i in active:
        if not envs[i]._state.done:
            envs[i].finish()
        _fill_stats(episodes[i], envs[i])
    return episodes


# ---------------------------------------------------------------------------
# Feedback construction
# ---------------------------------------------------------------------------

def build_feedback(group: list[SDEpisode]) -> tuple[str, int] | None:
    """Return (feedback_text, best_idx) from the highest-recall sibling, or None
    if no rollout in the group found any dependency (no positive teacher signal)."""
    best_idx = max(range(len(group)), key=lambda i: group[i].recall)
    best = group[best_idx]
    if best.recall <= 0.0 or not best.queries:
        return None
    qs = "; ".join(dict.fromkeys(best.queries))  # dedup, preserve order
    feedback = (
        "[SEARCH HINT] To find this theorem's logical dependencies (the lemmas and "
        "theorems its proof cites), effective search queries include: " + qs + "."
    )
    return feedback, best_idx


def _inject_feedback(messages_prefix: list[dict], feedback: str) -> list[dict]:
    """Return a copy of the prefix with the feedback appended to the system msg."""
    msgs = copy.deepcopy(messages_prefix)
    for m in msgs:
        if m["role"] == "system":
            m["content"] = m["content"] + "\n\n" + feedback
            return msgs
    msgs.insert(0, {"role": "system", "content": feedback})
    return msgs


# ---------------------------------------------------------------------------
# EMA teacher over LoRA trainable weights
# ---------------------------------------------------------------------------

def _trainable_named_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def init_ema(model) -> dict:
    return {n: p.detach().clone() for n, p in _trainable_named_params(model)}


@torch.no_grad()
def ema_update(ema: dict, model, alpha: float) -> None:
    for n, p in _trainable_named_params(model):
        ema[n].mul_(1.0 - alpha).add_(p.detach(), alpha=alpha)


@torch.no_grad()
def _load_weights(model, state: dict) -> None:
    for n, p in _trainable_named_params(model):
        p.copy_(state[n])


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _prompt_ids(tokenizer, messages: list[dict], device) -> torch.Tensor:
    text = tokenizer.apply_chat_template(
        messages, tools=SEARCH_TOOL, tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1536)
    return enc["input_ids"].to(device)


@torch.no_grad()
def _teacher_topk(model, tokenizer, step: SDStep, feedback: str, device, top_k: int):
    """Teacher (EMA weights already loaded) top-K next-token probs over the
    completion positions, conditioned on feedback. Returns (ids[T,K], probs[T,K],
    tail[T]) detached, or None if degenerate."""
    msgs = _inject_feedback(step.messages_prefix, feedback)
    p_ids = _prompt_ids(tokenizer, msgs, device)
    c_ids = torch.tensor(step.completion_ids, dtype=torch.long, device=device)
    n_c = c_ids.numel()
    if n_c == 0:
        return None
    seq = torch.cat([p_ids[0], c_ids]).unsqueeze(0)
    n_p = p_ids.shape[1]
    logits = model(seq, use_cache=False).logits[0, n_p - 1: n_p + n_c - 1].float()
    if logits.shape[0] != n_c or not torch.isfinite(logits).all():
        return None
    probs = F.softmax(logits, dim=-1)
    top_p, top_i = probs.topk(min(top_k, probs.shape[-1]), dim=-1)
    tail = (1.0 - top_p.sum(dim=-1)).clamp_min(0.0)
    return top_i.detach(), top_p.detach(), tail.detach()


def _student_js(model, tokenizer, step: SDStep, teacher, device) -> torch.Tensor | None:
    """Student (live weights, no feedback) symmetric-JS to the cached teacher
    top-K distribution, summed over completion tokens. Has grad."""
    top_i, top_p, t_tail = teacher
    p_ids = _prompt_ids(tokenizer, step.messages_prefix, device)
    c_ids = torch.tensor(step.completion_ids, dtype=torch.long, device=device)
    n_c = c_ids.numel()
    seq = torch.cat([p_ids[0], c_ids]).unsqueeze(0)
    n_p = p_ids.shape[1]
    logits = model(seq, use_cache=False).logits[0, n_p - 1: n_p + n_c - 1].float()
    if logits.shape[0] != n_c or not torch.isfinite(logits).all():
        return None
    s_logprobs = F.log_softmax(logits, dim=-1)             # [T, V]
    s_top = s_logprobs.gather(1, top_i).exp()              # [T, K] student prob at teacher's top-K ids
    s_tail = (1.0 - s_top.sum(dim=-1)).clamp_min(1e-8)     # [T]
    # distributions over K+1 buckets (top-K tokens + tail)
    eps = 1e-8
    P = torch.cat([top_p, t_tail.unsqueeze(-1)], dim=-1).clamp_min(eps)   # teacher
    Q = torch.cat([s_top, s_tail.unsqueeze(-1)], dim=-1).clamp_min(eps)   # student
    P = P / P.sum(-1, keepdim=True)
    Q = Q / Q.sum(-1, keepdim=True)
    M = 0.5 * (P + Q)
    js = 0.5 * (P * (P / M).log()).sum(-1) + 0.5 * (Q * (Q / M).log()).sum(-1)  # [T]
    return js.sum()


def selfdistill_update_step(model, tokenizer, groups, ema, device, top_k, cur_state):
    """One gradient accumulation over the batch's groups. Assumes
    optimizer.zero_grad() already called; does NOT call optimizer.step()."""
    # 1) gather (step, feedback) work items from groups that have positive feedback
    work: list[tuple[SDStep, str]] = []
    n_groups_used = 0
    for group in groups:
        fb = build_feedback(group)
        if fb is None:
            continue
        feedback, _best = fb
        n_groups_used += 1
        for ep in group:
            for st in ep.steps:
                if st.completion_ids:
                    work.append((st, feedback))
    if not work:
        return {"loss": 0.0, "groups_used": 0, "n_steps": 0, "skipped_groups": len(groups)}

    # 2) teacher pass with EMA weights (no grad) -> cache top-K targets
    _load_weights(model, ema)
    model.eval()
    teachers = []
    with torch.no_grad():
        for st, feedback in work:
            teachers.append(_teacher_topk(model, tokenizer, st, feedback, device, top_k))

    # 3) restore live weights, student pass with grad -> JS loss.
    # Backward per work-item (scaled) so only ONE forward graph is alive at a time
    # — summing 32+ 8B forward graphs before a single backward would OOM.
    _load_weights(model, cur_state)
    model.train()
    n_valid = sum(1 for t in teachers if t is not None)
    if n_valid == 0:
        return {"loss": 0.0, "groups_used": n_groups_used, "n_steps": 0,
                "skipped_groups": len(groups) - n_groups_used}
    loss_sum = 0.0
    n_used = 0
    for (st, _fb), teacher in zip(work, teachers):
        if teacher is None:
            continue
        js = _student_js(model, tokenizer, st, teacher, device)
        if js is None or not torch.isfinite(js):
            continue
        (js / n_valid).backward()
        loss_sum += float(js.item()) / n_valid
        n_used += 1
    return {"loss": loss_sum, "groups_used": n_groups_used,
            "n_steps": n_used, "skipped_groups": len(groups) - n_groups_used}


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
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / cfg.get("run_name", "sdpo_sd")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = Path("configs/prompts/premise_selection.txt").read_text()

    table = cfg.get("table", "rl_train")
    targets_dict = load_targets(table=table, cache_dir=cache_dir)
    targets_list = list(targets_dict.values())
    if cfg.get("use_intra", False):
        targets_list = [t for t in targets_list if t.intra_dep_ids]
    max_targets = cfg.get("max_targets")
    if max_targets and len(targets_list) > max_targets:
        targets_list = random.Random(cfg.get("seed", 42)).sample(targets_list, max_targets)
        logger.info("Subsampled to %d targets", len(targets_list))

    model_id = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s ...", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
    )
    lora_cfg = LoraConfig(
        r=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
        target_modules=cfg.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj",
                                                "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.config.use_cache = False
    if cfg.get("resume_from"):
        logger.info("Resuming LoRA from %s", cfg["resume_from"])
        model.load_adapter(cfg["resume_from"], adapter_name="default", is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=cfg.get("lr", 1e-6))

    group_size = cfg.get("group_size", 4)
    horizon = cfg.get("horizon", 6)
    top_k = cfg.get("top_k", 30)
    alpha = cfg.get("alpha", 0.1)
    beta = cfg.get("beta", 1.0)
    novelty_gamma = cfg.get("novelty_gamma", 0.05)
    temperature = cfg.get("temperature", 1.0)
    max_new_tokens = cfg.get("max_completion_length", 128)
    distill_top_k = cfg.get("distill_top_k", 100)
    ema_alpha = cfg.get("ema_alpha", 0.01)
    batch_size = cfg.get("batch_size", 2)
    grad_accum = cfg.get("grad_accum", 4)
    epochs = cfg.get("epochs", 3)
    save_steps = cfg.get("save_steps", 10)
    seed = cfg.get("seed", 42)

    rng = random.Random(seed)
    cache_dir_search = os.path.join(cache_dir, "search")
    log_file = checkpoint_dir / "training_log.jsonl"

    ema = init_ema(model)
    global_step = 0
    accum = 0
    optimizer.zero_grad()
    logger.info("SDPO(self-distill) on %d targets, %d epochs, G=%d, distill_top_k=%d, ema_alpha=%.3f",
                len(targets_list), epochs, group_size, distill_top_k, ema_alpha)

    for epoch in range(epochs):
        ep_targets = list(targets_list)
        rng.shuffle(ep_targets)
        logger.info("=== Epoch %d/%d ===", epoch + 1, epochs)
        for bstart in range(0, len(ep_targets), batch_size):
            batch = ep_targets[bstart: bstart + batch_size]
            torch.cuda.empty_cache()
            model.eval()

            async def _roll(targets):
                sc = SearchClient(cache_dir=cache_dir_search)
                try:
                    return await asyncio.gather(*[
                        _rollout_group(t, model, tokenizer, sc, system_prompt, group_size,
                                       horizon, top_k, alpha, beta, novelty_gamma,
                                       temperature, max_new_tokens, device)
                        for t in targets
                    ])
                finally:
                    await sc.close()

            try:
                groups = asyncio.run(_roll(batch))
            except Exception:
                logger.exception("Rollout failed")
                torch.cuda.empty_cache()
                continue

            cur_state = {n: p.detach().clone() for n, p in _trainable_named_params(model)}
            metrics = selfdistill_update_step(model, tokenizer, groups, ema, device,
                                              distill_top_k, cur_state)
            recalls = [ep.recall for g in groups for ep in g]
            mean_recall = sum(recalls) / max(len(recalls), 1)
            accum += 1

            if accum >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                ema_update(ema, model, ema_alpha)   # EMA tracks the student after each update
                accum = 0
                global_step += 1
                entry = {"step": global_step, "epoch": epoch + 1, "loss": metrics["loss"],
                         "mean_recall": round(mean_recall, 4), "groups_used": metrics["groups_used"],
                         "n_distill_steps": metrics["n_steps"], "skipped_groups": metrics["skipped_groups"]}
                logger.info("step=%d loss=%.4f recall=%.3f groups_used=%d distill_steps=%d skipped=%d",
                            global_step, metrics["loss"], mean_recall, metrics["groups_used"],
                            metrics["n_steps"], metrics["skipped_groups"])
                with open(log_file, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                if global_step % save_steps == 0:
                    model.save_pretrained(str(checkpoint_dir / f"step_{global_step}"))
                    logger.info("saved %s", checkpoint_dir / f"step_{global_step}")

    model.save_pretrained(str(checkpoint_dir / "final"))
    logger.info("Done. Final: %s/final", checkpoint_dir)


if __name__ == "__main__":
    main()
