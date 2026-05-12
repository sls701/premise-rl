from __future__ import annotations
from dataclasses import dataclass, field
from uuid import UUID


@dataclass
class State:
    target_statement_id: UUID
    target_body: str
    target_slogan: str | None
    target_pre_context: str | None
    target_post_context: str | None
    query_history: list[str] = field(default_factory=list)
    # list of (query_index, slogan) tuples
    retrieved_slogans: list[tuple[int, str]] = field(default_factory=list)
    retrieved_uuids: set[UUID] = field(default_factory=set)
    step_idx: int = 0
    done: bool = False


def format_state(state: State, include_context: bool = False) -> str:
    parts: list[str] = []

    if state.target_slogan:
        parts.append(f"Target slogan: {state.target_slogan}")
    parts.append(f"Target body:\n{state.target_body}")

    if include_context:
        if state.target_pre_context:
            parts.append(f"Pre-context:\n{state.target_pre_context}")
        if state.target_post_context:
            parts.append(f"Post-context:\n{state.target_post_context}")

    if state.query_history:
        parts.append("Prior queries:")
        for i, q in enumerate(state.query_history, 1):
            parts.append(f"  {i}. {q}")

    if state.retrieved_slogans:
        parts.append("Retrieved results (slogans only):")
        for query_idx, slogan in state.retrieved_slogans:
            parts.append(f"  [from query {query_idx}] {slogan}")

    return "\n\n".join(parts)
