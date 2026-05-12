"""
Batched async rollouts for GRPO.

For each training step:
  1. Sample a batch of targets from the active table.
  2. For each target, sample G trajectories at temperature > 0.
  3. Return per-trajectory: total reward, token log-probs, trajectory log.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from uuid import UUID

from src.data.load import Target
from src.env.environment import PremiseSelectionEnv, Trajectory
from src.env.id_mapping import IDMapper
from src.env.search_client import SearchClient
from src.policies.common import run_episode
from src.train.reward import step_reward, terminal_bonus

logger = logging.getLogger(__name__)


@dataclass
class RolloutResult:
    target_id: UUID
    trajectory: Trajectory
    total_reward: float
    # token log-probs populated by the trainer via model forward pass on the trajectory tokens
    log_probs: list[float] | None = None


async def _single_rollout(
    target: Target,
    agent,
    search_client: SearchClient,
    matcher: IDMapper,
    system_prompt: str,
    horizon: int,
    top_k: int,
    alpha: float,
    beta: float,
) -> RolloutResult:
    env = PremiseSelectionEnv(
        search_client=search_client,
        matcher=matcher,
        horizon=horizon,
        top_k=top_k,
        alpha=alpha,
        beta=beta,
    )
    try:
        traj = await run_episode(env, target, agent, system_prompt, top_k=top_k)
    except Exception as exc:
        logger.warning("rollout failed for target %s: %s", target.statement_id, exc)
        env.finish()
        traj = env.get_trajectory()

    return RolloutResult(
        target_id=target.statement_id,
        trajectory=traj,
        total_reward=traj.total_reward,
    )


async def run_rollout_batch(
    targets: list[Target],
    agent,
    search_client: SearchClient,
    matcher: IDMapper,
    system_prompt: str,
    group_size: int,
    horizon: int,
    top_k: int,
    alpha: float,
    beta: float,
    concurrency: int = 32,
) -> list[list[RolloutResult]]:
    """
    For each target, sample group_size trajectories.
    Returns list[group_size results] per target.
    """
    sem = asyncio.Semaphore(concurrency)

    async def run_group(target: Target) -> list[RolloutResult]:
        results = []
        for _ in range(group_size):
            async with sem:
                r = await _single_rollout(
                    target, agent, search_client, matcher,
                    system_prompt, horizon, top_k, alpha, beta,
                )
                results.append(r)
        return results

    groups = await asyncio.gather(*[run_group(t) for t in targets], return_exceptions=True)

    out = []
    for g in groups:
        if isinstance(g, Exception):
            logger.error("group rollout failed: %s", g)
        else:
            out.append(g)
    return out
