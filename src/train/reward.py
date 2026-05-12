from uuid import UUID


def step_reward(new_tps: int, new_fps: int, alpha: float) -> float:
    return float(new_tps) - alpha * float(new_fps)


def terminal_bonus(retrieved: set[UUID], true_deps: set[UUID], beta: float) -> float:
    if not true_deps:
        return 0.0
    return beta * (len(retrieved & true_deps) / len(true_deps))
