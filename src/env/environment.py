from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from src.data.load import Target
from src.env.prompts import State
from src.env.search_client import SearchClient
from src.train.reward import novelty_bonus, step_reward, terminal_bonus

logger = logging.getLogger(__name__)


@dataclass
class StepLog:
    step: int
    query: str
    returned_results: list[dict]
    new_tps: list[UUID]
    new_fps: list[UUID]
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
        horizon: int = 6,
        top_k: int = 10,
        alpha: float = 0.1,
        beta: float = 1.0,
        novelty_gamma: float = 0.0,
        include_context: bool = False,
    ):
        self._client = search_client
        self.horizon = horizon
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.novelty_gamma = novelty_gamma
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

        new_uuids: set[UUID] = set()
        result_logs: list[dict] = []

        for r in results:
            uid = r.statement_id
            if uid not in self._state.retrieved_uuids:
                new_uuids.add(uid)
            result_logs.append({
                "statement_id": str(uid),
                "similarity": r.similarity,
            })

        true_deps = self._target.true_dep_ids
        new_tps = list(new_uuids & true_deps)
        new_fps = list(new_uuids - true_deps)

        r_step = step_reward(len(new_tps), len(new_fps), self.alpha, self.top_k)
        r_step += novelty_bonus(query, self._state.query_history, self.novelty_gamma)

        self._state.retrieved_uuids.update(new_uuids)
        self._state.query_history.append(query)
        for r in results:
            if r.slogan:
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
        self._trajectory.steps.append(StepLog(
            step=self._state.step_idx + 1,
            query="<STOP>",
            returned_results=[],
            new_tps=[],
            new_fps=[],
            step_reward=0.0,
            terminal_reward=r_terminal,
        ))
        return r_terminal

    def get_trajectory(self) -> Trajectory:
        assert self._trajectory is not None
        return self._trajectory
