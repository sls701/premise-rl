from __future__ import annotations

import logging
from typing import Any, Protocol

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
            logger.debug("agent declined to query, ending episode")
            break

        k = k or top_k
        state, reward, done, info = await env.step(query)

        # Build tool result from slogans of the most-recent step's results
        latest_step = env.get_trajectory().steps[-1]
        retrieved_this_step = [
            r for r in latest_step.returned_results if r["mapped_uuid"] is not None
        ]
        dropped = latest_step.dropped_no_match
        slogan_lines = []
        for r in latest_step.returned_results:
            # Include slogan if we have it; otherwise note the body was unmatched
            pass

        # We surface slogans via the state formatter on the next turn
        # rather than constructing raw tool output here
        tool_output = format_state(state)

        # Append assistant tool call + tool result to history
        messages.append({"role": "assistant", "content": f"[search_theorems] query={query!r} k={k}"})
        messages.append({"role": "user", "content": f"[tool_result]\n{tool_output}"})

        if done:
            break

    # If the agent stopped early (before horizon), fire the terminal bonus
    # without issuing further search queries.
    if not state.done:
        env.finish()

    return env.get_trajectory()
