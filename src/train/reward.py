from difflib import SequenceMatcher
from uuid import UUID


def step_reward(new_tps: int, new_fps: int, alpha: float, top_k: int = 0) -> float:
    """Per-step reward: TP count minus an FP penalty.

    When top_k > 0 the FP penalty is normalized by top_k so that a fully-wasted
    search (all results FP) costs exactly `alpha` regardless of how many results
    the query returned. Without normalization (top_k=0, legacy behavior), a high
    top_k makes the FP penalty dominate the +1-per-TP signal and every episode
    goes negative — which makes SDPO skip every group. See docs/reward_design_plan.md
    Design 2 (FP normalization).
    """
    if top_k > 0:
        fp_cost = alpha * float(new_fps) / float(top_k)
    else:
        fp_cost = alpha * float(new_fps)
    return float(new_tps) - fp_cost


def terminal_bonus(retrieved: set[UUID], true_deps: set[UUID], beta: float) -> float:
    if not true_deps:
        return 0.0
    return beta * (len(retrieved & true_deps) / len(true_deps))


def novelty_bonus(query: str, query_history: list[str], gamma: float) -> float:
    """Reward diversity: bonus proportional to how different query is from prior queries.

    Uses character-level SequenceMatcher ratio as a cheap proxy for semantic similarity.
    Returns gamma when history is empty (first query always novel).
    """
    if gamma == 0.0:
        return 0.0
    if not query_history:
        return gamma
    max_sim = max(
        SequenceMatcher(None, query.lower(), q.lower()).ratio()
        for q in query_history
    )
    return gamma * (1.0 - max_sim)
