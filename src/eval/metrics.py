from __future__ import annotations

from collections import defaultdict

from src.env.environment import Trajectory


def _dep_count_bucket(n: int) -> str:
    if n <= 2:
        return "2"
    elif n == 3:
        return "3"
    elif n <= 5:
        return "4-5"
    else:
        return "6+"


def compute_metrics(trajectories: list[Trajectory]) -> dict:
    if not trajectories:
        return {}

    recall_list: list[float] = []
    queries_per_episode: list[int] = []
    unique_query_rates: list[float] = []
    fp_per_episode: list[float] = []
    terminal_rewards: list[float] = []

    by_bucket: dict[str, list[float]] = defaultdict(list)

    for traj in trajectories:
        true_deps = traj.final_true_dep_ids
        retrieved = traj.final_retrieved_uuids
        n_true = len(true_deps)

        recall = len(retrieved & true_deps) / max(n_true, 1)
        recall_list.append(recall)

        bucket = _dep_count_bucket(n_true)
        by_bucket[bucket].append(recall)

        n_queries = len(traj.steps)
        queries_per_episode.append(n_queries)

        all_queries = [s.query for s in traj.steps]
        unique_q = len(set(all_queries))
        unique_query_rates.append(unique_q / max(n_queries, 1))

        total_fp = sum(len(s.new_fps) for s in traj.steps)
        fp_per_episode.append(float(total_fp))

        last_terminal = next(
            (s.terminal_reward for s in reversed(traj.steps) if s.terminal_reward != 0.0),
            0.0,
        )
        terminal_rewards.append(last_terminal)

    n = len(trajectories)

    def mean(xs: list) -> float:
        return sum(xs) / max(len(xs), 1)

    return {
        "n_episodes": n,
        "recall": mean(recall_list),
        "mean_queries_per_episode": mean(queries_per_episode),
        "unique_query_rate": mean(unique_query_rates),
        "mean_fp_per_episode": mean(fp_per_episode),
        "mean_terminal_reward": mean(terminal_rewards),
        "recall_by_bucket": {k: mean(v) for k, v in by_bucket.items()},
        "recall_by_bucket_counts": {k: len(v) for k, v in by_bucket.items()},
        "total_reward": sum(traj.total_reward for traj in trajectories),
    }
