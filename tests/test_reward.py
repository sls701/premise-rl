from uuid import uuid4
from src.train.reward import step_reward, terminal_bonus


def test_all_tp():
    assert step_reward(3, 0, alpha=0.1) == 3.0


def test_all_fp():
    assert abs(step_reward(0, 3, alpha=0.1) - (-0.3)) < 1e-9


def test_partial_overlap():
    assert abs(step_reward(2, 1, alpha=0.1) - 1.9) < 1e-9


def test_empty_step():
    assert step_reward(0, 0, alpha=0.1) == 0.0


def test_fp_normalized_by_top_k():
    # Fully-wasted search (all top_k results are FPs) costs exactly alpha.
    assert abs(step_reward(0, 30, alpha=0.1, top_k=30) - (-0.1)) < 1e-9


def test_tp_dominates_normalized_fp():
    # Finding 1 TP among 29 FPs is clearly positive with normalization.
    assert step_reward(1, 29, alpha=0.1, top_k=30) > 0.9


def test_terminal_full_recall():
    uids = {uuid4() for _ in range(4)}
    assert terminal_bonus(uids, uids, beta=1.0) == 1.0


def test_terminal_partial_recall():
    uids = [uuid4() for _ in range(4)]
    retrieved = set(uids[:2])
    true_deps = set(uids)
    assert abs(terminal_bonus(retrieved, true_deps, beta=1.0) - 0.5) < 1e-9


def test_terminal_zero_recall():
    true_deps = {uuid4() for _ in range(3)}
    assert terminal_bonus(set(), true_deps, beta=1.0) == 0.0


def test_terminal_empty_true_deps():
    # No division by zero; returns 0
    assert terminal_bonus({uuid4()}, set(), beta=1.0) == 0.0


def test_terminal_beta_scaling():
    uids = {uuid4() for _ in range(2)}
    assert abs(terminal_bonus(uids, uids, beta=2.5) - 2.5) < 1e-9
