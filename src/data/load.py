import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from src.db import get_rds_connection


@dataclass
class Target:
    statement_id: UUID
    body: str
    proof: str | None
    kind: str
    paper_id: str
    label: str | None
    ref: str | None
    pre_context: str | None
    post_context: str | None
    true_dep_ids: set[UUID] = field(default_factory=set)
    intra_dep_ids: set[UUID] = field(default_factory=set)


@dataclass
class DepEdge:
    src_id: UUID
    dep_id: UUID
    cite_key: str | None
    dep_name: str | None
    dep_key: str | None
    src_paper_id: str
    dep_paper_id: str


@dataclass
class DepBody:
    statement_id: UUID
    body: str
    kind: str
    paper_id: str


def _cache_path(cache_dir: Path, table: str, kind: str) -> Path:
    return cache_dir / f"{table}_{kind}.pkl"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_targets(
    table: str = "rl_test_100",
    cache_dir: str | Path = "cache",
) -> dict[UUID, Target]:
    cache_dir = Path(cache_dir)
    cache = _cache_path(cache_dir, table, "targets")
    if cache.exists():
        return _load_pickle(cache)

    conn = get_rds_connection("v2")
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT s.statement_id, s.body, s.proof, s.kind, s.paper_id,
                       im.label, im.ref, im.pre_context, im.post_context
                FROM {table} t
                JOIN "statement" s ON s.statement_id = t.src_id
                LEFT JOIN informal_metadata im ON im.statement_id = s.statement_id
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    targets: dict[UUID, Target] = {}
    for row in rows:
        sid = UUID(str(row[0]))
        targets[sid] = Target(
            statement_id=sid,
            body=row[1] or "",
            proof=row[2],
            kind=row[3] or "",
            paper_id=str(row[4]) if row[4] is not None else "",
            label=row[5],
            ref=row[6],
            pre_context=row[7],
            post_context=row[8],
        )

    # attach dep edges
    edges = load_dep_edges(table, cache_dir)
    for edge in edges:
        if edge.src_id in targets:
            targets[edge.src_id].true_dep_ids.add(edge.dep_id)
            if edge.src_paper_id == edge.dep_paper_id:
                targets[edge.src_id].intra_dep_ids.add(edge.dep_id)

    _save_pickle(cache, targets)
    return targets


def load_dep_edges(
    table: str = "rl_test_100",
    cache_dir: str | Path = "cache",
) -> list[DepEdge]:
    cache_dir = Path(cache_dir)
    cache = _cache_path(cache_dir, table, "dep_edges")
    if cache.exists():
        return _load_pickle(cache)

    conn = get_rds_connection("v2")
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT d.src_id, d.dep_id, d.cite_key, d.dep_name, d.dep_key,
                       s_src.paper_id AS src_paper_id,
                       s_dep.paper_id AS dep_paper_id
                FROM informal_dependency d
                JOIN {table} t ON t.src_id = d.src_id
                JOIN "statement" s_src ON s_src.statement_id = d.src_id
                JOIN "statement" s_dep ON s_dep.statement_id = d.dep_id
                WHERE d.cite_key IS NOT NULL
                  AND 'deterministic' = ANY(d.methods)
                  AND d.dep_id IS NOT NULL
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    edges = [
        DepEdge(
            src_id=UUID(str(row[0])),
            dep_id=UUID(str(row[1])),
            cite_key=row[2],
            dep_name=row[3],
            dep_key=row[4],
            src_paper_id=str(row[5]) if row[5] is not None else "",
            dep_paper_id=str(row[6]) if row[6] is not None else "",
        )
        for row in rows
    ]

    _save_pickle(cache, edges)
    return edges


def load_dep_bodies(
    table: str = "rl_test_100",
    cache_dir: str | Path = "cache",
    scope: str = "deps_only",
) -> dict[UUID, DepBody]:
    """Load bodies for the matcher's UUID universe.

    scope:
      - "deps_only": only statements that ARE a true dep of some target in `table`.
        ~10K bodies; agent's retrievals mostly drop as uuid=None.
      - "dep_papers": all statements in papers that contain at least one true dep
        of some target in `table`. Broader pool so most API retrievals map to a UUID,
        giving honest TP/FP reward signal instead of silently dropping unrelated hits.
    """
    if scope not in {"deps_only", "dep_papers"}:
        raise ValueError(f"unknown scope: {scope}")

    cache_dir = Path(cache_dir)
    suffix = "dep_bodies" if scope == "deps_only" else f"dep_bodies_{scope}"
    cache = _cache_path(cache_dir, table, suffix)
    if cache.exists():
        return _load_pickle(cache)

    conn = get_rds_connection("v2")
    try:
        with conn.cursor() as cur:
            if scope == "deps_only":
                cur.execute(f"""
                    SELECT DISTINCT s.statement_id, s.body, s.kind, s.paper_id
                    FROM informal_dependency d
                    JOIN {table} t ON t.src_id = d.src_id
                    JOIN "statement" s ON s.statement_id = d.dep_id
                    WHERE d.cite_key IS NOT NULL
                      AND 'deterministic' = ANY(d.methods)
                      AND d.dep_id IS NOT NULL
                """)
            else:  # dep_papers
                cur.execute(f"""
                    SELECT DISTINCT s.statement_id, s.body, s.kind, s.paper_id
                    FROM "statement" s
                    WHERE s.body IS NOT NULL
                      AND s.paper_id IN (
                        SELECT DISTINCT s_dep.paper_id
                        FROM informal_dependency d
                        JOIN {table} t ON t.src_id = d.src_id
                        JOIN "statement" s_dep ON s_dep.statement_id = d.dep_id
                        WHERE d.cite_key IS NOT NULL
                          AND 'deterministic' = ANY(d.methods)
                          AND d.dep_id IS NOT NULL
                      )
                """)
            rows = cur.fetchall()
    finally:
        conn.close()

    bodies = {
        UUID(str(row[0])): DepBody(
            statement_id=UUID(str(row[0])),
            body=row[1] or "",
            kind=row[2] or "",
            paper_id=str(row[3]) if row[3] is not None else "",
        )
        for row in rows
    }

    _save_pickle(cache, bodies)
    return bodies


def print_stats(table: str = "rl_test_100", cache_dir: str | Path = "cache") -> None:
    targets = load_targets(table, cache_dir)
    dep_bodies = load_dep_bodies(table, cache_dir)

    true_counts = [len(t.true_dep_ids) for t in targets.values()]
    intra_counts = [len(t.intra_dep_ids) for t in targets.values()]
    all_dep_ids = set()
    for t in targets.values():
        all_dep_ids |= t.true_dep_ids

    def bucket_dist(counts: list[int]) -> dict[str, int]:
        buckets = {"0": 0, "1": 0, "2": 0, "3": 0, "4-5": 0, "6+": 0}
        for c in counts:
            if c == 0:
                buckets["0"] += 1
            elif c == 1:
                buckets["1"] += 1
            elif c == 2:
                buckets["2"] += 1
            elif c == 3:
                buckets["3"] += 1
            elif c <= 5:
                buckets["4-5"] += 1
            else:
                buckets["6+"] += 1
        return {k: v for k, v in buckets.items() if v > 0}

    print(f"\n=== Stats for table: {table} ===")
    print(f"  targets:              {len(targets)}")
    print(f"  true_dep_ids dist:    {bucket_dist(true_counts)}")
    print(f"  intra_dep_ids dist:   {bucket_dist(intra_counts)}")
    print(f"  unique dep IDs:       {len(all_dep_ids)}")
    print(f"  dep bodies loaded:    {len(dep_bodies)}")


if __name__ == "__main__":
    table = sys.argv[1] if len(sys.argv) > 1 else "rl_test_100"
    print_stats(table)
