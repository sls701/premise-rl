import asyncio
from uuid import UUID, uuid4

import pytest

from src.data.load import Target
from src.env.environment import PremiseSelectionEnv
from src.env.search_client import SearchResult


# --- Fake search client ---

class FakeSearchClient:
    def __init__(self, results_map: dict[str, list[SearchResult]]):
        self._map = results_map

    async def search(self, query: str, k: int) -> list[SearchResult]:
        return self._map.get(query, [])


def _sr(uid: UUID, slogan: str = "") -> SearchResult:
    return SearchResult(
        statement_id=uid,
        paper_id=uuid4(),
        name="",
        body="",
        slogan=slogan,
        source="",
        similarity=1.0,
        score=1.0,
    )


def _target(true_dep_ids: set[UUID], intra_dep_ids: set[UUID] | None = None) -> Target:
    return Target(
        statement_id=uuid4(),
        body="target body",
        proof=None,
        kind="theorem",
        paper_id="p1",
        label=None,
        ref=None,
        pre_context=None,
        post_context=None,
        true_dep_ids=true_dep_ids,
        intra_dep_ids=intra_dep_ids or set(),
    )


# --- Tests ---

@pytest.mark.asyncio
async def test_three_step_episode_reward():
    tp1, tp2, fp1 = uuid4(), uuid4(), uuid4()
    true_deps = {tp1, tp2}

    client = FakeSearchClient({
        "query1": [_sr(tp1), _sr(fp1)],
        "query2": [_sr(tp2)],
        "query3": [],
    })
    env = PremiseSelectionEnv(client, horizon=3, alpha=0.1, beta=1.0)
    env.reset(_target(true_deps))

    # Step 1: 1 TP, 1 FP → 1 - 0.1 = 0.9
    state, r1, done, info = await env.step("query1")
    assert not done
    assert abs(r1 - 0.9) < 1e-9

    # Step 2: 1 TP, 0 FP → 1.0
    state, r2, done, info = await env.step("query2")
    assert not done
    assert abs(r2 - 1.0) < 1e-9

    # Step 3 (final): 0 new + terminal bonus (2/2 recalled) * beta=1.0 → 1.0
    state, r3, done, info = await env.step("query3")
    assert done
    assert abs(info["terminal_reward"] - 1.0) < 1e-9
    assert abs(r3 - 1.0) < 1e-9

    traj = env.get_trajectory()
    assert abs(traj.total_reward - (0.9 + 1.0 + 1.0)) < 1e-9


@pytest.mark.asyncio
async def test_duplicate_query_zero_reward():
    tp1 = uuid4()
    true_deps = {tp1}
    client = FakeSearchClient({
        "query1": [_sr(tp1)],
    })
    env = PremiseSelectionEnv(client, horizon=2, alpha=0.1, beta=0.0)
    env.reset(_target(true_deps))

    _, r1, _, _ = await env.step("query1")
    assert r1 > 0  # first time: TP

    _, r2, done, info = await env.step("query1")
    assert done
    # Same query → no new UUIDs → step_reward = 0
    assert abs(info["step_reward"]) < 1e-9


@pytest.mark.asyncio
async def test_terminal_bonus_fires_once():
    tp1 = uuid4()
    true_deps = {tp1}
    client = FakeSearchClient({
        "q1": [_sr(tp1)],
        "q2": [],
        "q3": [],
    })
    env = PremiseSelectionEnv(client, horizon=3, alpha=0.0, beta=2.0)
    env.reset(_target(true_deps))

    _, _, done, i1 = await env.step("q1")
    assert not done
    assert abs(i1["terminal_reward"]) < 1e-9

    _, _, done, i2 = await env.step("q2")
    assert not done
    assert abs(i2["terminal_reward"]) < 1e-9

    _, _, done, i3 = await env.step("q3")
    assert done
    assert abs(i3["terminal_reward"] - 2.0) < 1e-9  # beta * (1/1) = 2.0

    traj = env.get_trajectory()
    terminal_count = sum(1 for s in traj.steps if s.terminal_reward != 0.0)
    assert terminal_count == 1


@pytest.mark.asyncio
async def test_horizon_terminates():
    client = FakeSearchClient({"q": []})
    env = PremiseSelectionEnv(client, horizon=3, alpha=0.1, beta=0.0)
    env.reset(_target(set()))

    for i in range(2):
        _, _, done, _ = await env.step("q")
        assert not done

    _, _, done, _ = await env.step("q")
    assert done


@pytest.mark.asyncio
async def test_early_stop_fires_terminal():
    tp1 = uuid4()
    true_deps = {tp1}
    client = FakeSearchClient({"q1": [_sr(tp1)]})
    env = PremiseSelectionEnv(client, horizon=6, alpha=0.0, beta=1.0)
    env.reset(_target(true_deps))

    await env.step("q1")
    bonus = env.finish()
    assert abs(bonus - 1.0) < 1e-9  # 1/1 deps found, beta=1.0

    traj = env.get_trajectory()
    assert traj.total_reward >= 1.0
