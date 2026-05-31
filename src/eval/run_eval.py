"""
Evaluation entry point.

Usage:
    python -m src.eval.run_eval --config configs/baseline_gpt55.yaml --policy api --dataset smoke
    python -m src.eval.run_eval --config configs/eval_local.yaml --policy local --dataset smoke --checkpoint checkpoints/step_100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.data.load import load_targets
from src.env.environment import PremiseSelectionEnv, Trajectory
from src.env.search_client import SearchClient
from src.eval.metrics import compute_metrics
from src.policies.common import run_episode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_TABLE = {
    "smoke": "rl_test_100",
    "val": "rl_val",
    "test": "rl_test",
}

SYSTEM_PROMPT_PATH = Path("configs/prompts/premise_selection.txt")


def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text()


def _run_name(policy: str, config: dict, dataset: str, checkpoint: str | None) -> str:
    if policy == "api":
        descriptor = f"{config['provider']}_{config['model'].replace('/', '_')}"
    else:
        ckpt_name = Path(checkpoint).name if checkpoint else "base"
        descriptor = f"qwen3-4b_{ckpt_name}"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{descriptor}_{dataset}_{ts}"


async def _run_all(
    targets_dict,
    system_prompt: str,
    agent,
    config: dict,
    concurrency: int,
) -> list[Trajectory]:
    top_k = config.get("top_k", 10)
    horizon = config.get("horizon", 6)
    alpha = config.get("alpha", 0.1)
    beta = config.get("beta", 1.0)
    novelty_gamma = config.get("novelty_gamma", 0.0)
    cache_dir = config.get("cache_dir", "cache/search")

    trajectories: list[Trajectory] = []
    sem = asyncio.Semaphore(concurrency)
    search_client = SearchClient(cache_dir=cache_dir)

    async def run_one(target):
        async with sem:
            env = PremiseSelectionEnv(
                search_client=search_client,
                horizon=horizon,
                top_k=top_k,
                alpha=alpha,
                beta=beta,
                novelty_gamma=novelty_gamma,
            )
            return await run_episode(env, target, agent, system_prompt, top_k=top_k)

    tasks = [run_one(target) for target in targets_dict.values()]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error("Episode failed: %s", r)
        else:
            trajectories.append(r)

    await search_client.close()
    return trajectories


def _traj_to_dict(traj: Trajectory) -> dict:
    return {
        "target_id": str(traj.target_id),
        "total_reward": traj.total_reward,
        "final_retrieved_uuids": [str(u) for u in traj.final_retrieved_uuids],
        "final_true_dep_ids": [str(u) for u in traj.final_true_dep_ids],
        "steps": [
            {
                "step": s.step,
                "query": s.query,
                "returned_results": s.returned_results,
                "new_tps": [str(u) for u in s.new_tps],
                "new_fps": [str(u) for u in s.new_fps],
                "step_reward": s.step_reward,
                "terminal_reward": s.terminal_reward,
            }
            for s in traj.steps
        ],
    }


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--policy", choices=["api", "local"], required=True)
    parser.add_argument("--dataset", choices=["smoke", "val", "test"], default="smoke")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n", type=int, default=None, help="Limit to N targets (for quick smoke)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    table = DATASET_TABLE[args.dataset]
    cache_dir = config.get("cache_dir", "cache")
    data_cache_dir = config.get("data_cache_dir", "cache")

    logger.info("Loading targets from table: %s", table)
    targets = load_targets(table=table, cache_dir=data_cache_dir)
    if args.n:
        targets = dict(list(targets.items())[: args.n])

    system_prompt = _load_system_prompt()

    if args.policy == "api":
        from src.policies.api import make_agent
        agent = make_agent(config)
    else:
        from src.policies.local import make_local_agent
        agent = make_local_agent(config, checkpoint_dir=args.checkpoint)

    concurrency = config.get("concurrency", 4)
    run_name = _run_name(args.policy, config, args.dataset, args.checkpoint)
    results_dir = Path(config.get("results_dir", "results")) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting eval: %s  n_targets=%d  concurrency=%d", run_name, len(targets), concurrency)

    trajectories = asyncio.run(
        _run_all(targets, system_prompt, agent, config, concurrency)
    )

    traj_path = results_dir / "trajectories.jsonl"
    with open(traj_path, "w") as f:
        for traj in trajectories:
            f.write(json.dumps(_traj_to_dict(traj)) + "\n")

    metrics = compute_metrics(trajectories)
    metrics["run_name"] = run_name
    metrics["config"] = config
    metrics["table"] = table

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Done. Results in %s", results_dir)
    logger.info("recall=%.3f  unique_query_rate=%.3f  mean_queries=%.1f",
        metrics.get("recall", 0),
        metrics.get("unique_query_rate", 0),
        metrics.get("mean_queries_per_episode", 0),
    )

    recall = metrics.get("recall", 0)
    if recall == 0:
        logger.warning("CHECKPOINT FAIL: recall=0. Check API connectivity and prompt loading.")
    uqr = metrics.get("unique_query_rate", 0)
    if uqr < 0.7:
        logger.warning("CHECKPOINT WARN: unique_query_rate=%.2f < 0.7 — model may be repeating queries.", uqr)


if __name__ == "__main__":
    main()
