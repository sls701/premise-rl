from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from uuid import UUID

from src.data.load import Target
from src.env.id_mapping import IDMapper, MatchResult
from src.env.prompts import State
from src.env.search_client import SearchClient
from src.train.reward import step_reward, terminal_bonus

logger = logging.getLogger(__name__)


@dataclass
class StepLog:
    step: int
    query: str
    returned_results: list[dict]
    new_tps: list[UUID]
    new_fps: list[UUID]
    dropped_no_match: int
    low_confidence_matches: int
    step_reward: float
    terminal_reward: float


@dataclass
class Trajectory:
    target_id: UUID
    steps: list[StepLog] = field(default_factory=list)
    total_reward: float = 0.0
    final_retrieved_uuids: set[UUID] = field(default_factory=set)
    final_true_dep_ids: set[UUID] = field(default_factory=set)


class PremiseSelectionEnv:
    def __init__(
        self,
        search_client: SearchClient,
        matcher: IDMapper,
        horizon: int = 6,
        top_k: int = 10,
        alpha: float = 0.1,
        beta: float = 1.0,
        include_context: bool = False,
    ):
        self._client = search_client
        self._matcher = matcher
        self.horizon = horizon
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.include_context = include_context

        self._target: Target | None = None
        self._state: State | None = None
        self._trajectory: Trajectory | None = None

    def reset(self, target: Target) -> State:
        self._target = target
        self._state = State(
            target_statement_id=target.statement_id,
            target_body=target.body,
            target_slogan=None,
            target_pre_context=target.pre_context,
            target_post_context=target.post_context,
        )
        self._trajectory = Trajectory(
            target_id=target.statement_id,
            final_true_dep_ids=set(target.true_dep_ids),
        )
        return self._state

    async def step(self, query: str) -> tuple[State, float, bool, dict]:
        assert self._state is not None and self._target is not None

        results = await self._client.search(query, self.top_k)

        # rapidfuzz.process.cdist() is synchronous C code that blocks the
        # event loop. Run a single batched call (10 results × ~10K choices)
        # in the executor so other concurrent step() coroutines can proceed.
        _loop = asyncio.get_running_loop()
        _matcher = self._matcher
        mapped: list[MatchResult] = await _loop.run_in_executor(
            None, _matcher.map_batch, results
        )

        new_uuids: set[UUID] = set()
        dropped_no_match = 0
        low_confidence = 0
        result_logs: list[dict] = []

        for r, m in zip(results, mapped):
            if m.uuid is None:
                dropped_no_match += 1
                result_logs.append({
                    "int_id": r.theorem_id,
                    "mapped_uuid": None,
                    "match_score": m.score,
                    "second_best_gap": m.second_best_gap,
                })
            else:
                if m.second_best_gap < self._matcher.low_confidence_gap:
                    low_confidence += 1
                if m.uuid not in self._state.retrieved_uuids:
                    new_uuids.add(m.uuid)
                result_logs.append({
                    "int_id": r.theorem_id,
                    "mapped_uuid": str(m.uuid),
                    "match_score": m.score,
                    "second_best_gap": m.second_best_gap,
                })

        true_deps = self._target.true_dep_ids
        new_tps = list(new_uuids & true_deps)
        new_fps = list(new_uuids - true_deps)

        r_step = step_reward(len(new_tps), len(new_fps), self.alpha)

        self._state.retrieved_uuids.update(new_uuids)
        self._state.query_history.append(query)
        for r, m in zip(results, mapped):
            if m.uuid is not None and r.slogan:
                self._state.retrieved_slogans.append(
                    (len(self._state.query_history), r.slogan)
                )
        self._state.step_idx += 1
        done = self._state.step_idx >= self.horizon

        r_terminal = 0.0
        if done:
            r_terminal = terminal_bonus(self._state.retrieved_uuids, true_deps, self.beta)

        log = StepLog(
            step=self._state.step_idx,
            query=query,
            returned_results=result_logs,
            new_tps=new_tps,
            new_fps=new_fps,
            dropped_no_match=dropped_no_match,
            low_confidence_matches=low_confidence,
            step_reward=r_step,
            terminal_reward=r_terminal,
        )
        assert self._trajectory is not None
        self._trajectory.steps.append(log)
        self._trajectory.total_reward += r_step + r_terminal
        if done:
            self._trajectory.final_retrieved_uuids = set(self._state.retrieved_uuids)

        self._state.done = done

        info = {
            "new_tps": new_tps,
            "new_fps": new_fps,
            "dropped_no_match": dropped_no_match,
            "low_confidence_matches": low_confidence,
            "step_reward": r_step,
            "terminal_reward": r_terminal,
        }
        return self._state, r_step + r_terminal, done, info

    def finish(self) -> float:
        """Finalize the episode early (agent stopped before horizon).
        Fires the terminal bonus once without issuing any search query.
        Returns the terminal bonus value."""
        assert self._state is not None and self._target is not None
        if self._state.done:
            return 0.0
        r_terminal = terminal_bonus(
            self._state.retrieved_uuids, self._target.true_dep_ids, self.beta
        )
        assert self._trajectory is not None
        self._trajectory.total_reward += r_terminal
        self._trajectory.final_retrieved_uuids = set(self._state.retrieved_uuids)
        self._state.done = True
        # Append a synthetic final log entry
        self._trajectory.steps.append(StepLog(
            step=self._state.step_idx + 1,
            query="<STOP>",
            returned_results=[],
            new_tps=[],
            new_fps=[],
            dropped_no_match=0,
            low_confidence_matches=0,
            step_reward=0.0,
            terminal_reward=r_terminal,
        ))
        return r_terminal

    def get_trajectory(self) -> Trajectory:
        assert self._trajectory is not None
        return self._trajectory
