from __future__ import annotations

import logging
from typing import Protocol

from src.data.load import Target
from src.env.environment import PremiseSelectionEnv, Trajectory
from src.env.prompts import format_state

logger = logging.getLogger(__name__)


class Agent(Protocol):
    """Provider-specific agent: given a conversation history, return either a
    tool call (query string, k) or None to signal no further queries."""

    async def chat(
        self,
        messages: list[dict],
        tool_result: str | None = None,
    ) -> tuple[str | None, int | None]:
        """Returns (query, k) if a tool call was issued, or (None, None) to stop."""
        ...


async def run_episode(
    env: PremiseSelectionEnv,
    target: Target,
    agent: Agent,
    system_prompt: str,
    top_k: int = 10,
) -> Trajectory:
    state = env.reset(target)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_state(state)},
    ]

    for _turn in range(env.horizon):
        try:
            query, k = await agent.chat(messages)
        except Exception as exc:
            logger.warning("agent.chat raised on target %s: %s", target.statement_id, exc)
            break

        if query is None:
            logger.debug("no tool call parsed on turn %d; skipping", _turn)
            continue

        k = k or top_k
        state, reward, done, info = await env.step(query)

        tool_output = format_state(state)
        messages.append({"role": "assistant", "content": f"[search_theorems] query={query!r} k={k}"})
        messages.append({"role": "user", "content": f"[tool_result]\n{tool_output}"})

        if done:
            break

    if not state.done:
        env.finish()

    return env.get_trajectory()
