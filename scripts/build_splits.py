"""
Phase 1.3 — Build rl_train / rl_val / rl_test splits in v2.

Pool: all src_id values in informal_dependency with ≥2 deterministic dep_id IS NOT NULL edges.
Split by paper, not by statement, to prevent test contamination via paper-internal lemma reuse.

Target sizes: rl_train ~50K, rl_val ~1K, rl_test ~2K.
Each table has schema: (src_id UUID, n_inter_deps BIGINT).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db import get_rds_connection

TRAIN_FRAC = 0.94
VAL_FRAC   = 0.02
TEST_FRAC  = 0.04   # ~2× val so test CIs are tighter

SEED = 42


def main():
    conn = get_rds_connection("v2")
    cur = conn.cursor()

    print("Counting qualifying statements...")
    cur.execute("""
        SELECT COUNT(DISTINCT src_id)
        FROM informal_dependency
        WHERE cite_key IS NOT NULL
          AND 'deterministic' = ANY(methods)
          AND dep_id IS NOT NULL
    """)
    total_stmts = cur.fetchone()[0]
    print(f"  Total qualifying statements: {total_stmts:,}")

    # Pool: src_id with ≥2 deterministic dep_id edges, grouped by paper
    print("Building paper-level pool...")
    cur.execute("""
        SELECT s.paper_id, s.statement_id AS src_id,
               COUNT(DISTINCT d.dep_id) AS n_deps,
               SUM(CASE WHEN s2.paper_id != s.paper_id THEN 1 ELSE 0 END) AS n_inter_deps
        FROM informal_dependency d
        JOIN statement s ON s.statement_id = d.src_id
        JOIN statement s2 ON s2.statement_id = d.dep_id
        WHERE d.cite_key IS NOT NULL
          AND 'deterministic' = ANY(d.methods)
          AND d.dep_id IS NOT NULL
        GROUP BY s.paper_id, s.statement_id
        HAVING COUNT(DISTINCT d.dep_id) >= 2
        ORDER BY s.paper_id, s.statement_id
    """)
    rows = cur.fetchall()
    print(f"  Qualifying (paper_id, src_id) pairs: {len(rows):,}")

    # Group by paper
    from collections import defaultdict
    paper_to_stmts: dict[str, list[tuple]] = defaultdict(list)
    for paper_id, src_id, n_deps, n_inter_deps in rows:
        paper_to_stmts[str(paper_id)].append((str(src_id), int(n_inter_deps or 0)))

    papers = list(paper_to_stmts.keys())
    print(f"  Unique papers: {len(papers):,}")

    # Shuffle papers with fixed seed
    import random
    rng = random.Random(SEED)
    rng.shuffle(papers)

    total_stmts_pool = len(rows)
    val_target  = max(1000, int(total_stmts_pool * VAL_FRAC))
    test_target = max(2000, int(total_stmts_pool * TEST_FRAC))

    val_stmts:   list[tuple[str, int]] = []
    test_stmts:  list[tuple[str, int]] = []
    train_stmts: list[tuple[str, int]] = []

    # Assign papers to val/test first (to hit size targets), rest to train
    val_done = test_done = False
    for paper in papers:
        stmts = paper_to_stmts[paper]
        if not val_done:
            val_stmts.extend(stmts)
            if len(val_stmts) >= val_target:
                val_done = True
        elif not test_done:
            test_stmts.extend(stmts)
            if len(test_stmts) >= test_target:
                test_done = True
        else:
            train_stmts.extend(stmts)

    print(f"\nSplit sizes:")
    print(f"  rl_train: {len(train_stmts):,} statements")
    print(f"  rl_val:   {len(val_stmts):,} statements")
    print(f"  rl_test:  {len(test_stmts):,} statements")

    def create_split_table(name: str, stmts: list[tuple[str, int]]) -> None:
        print(f"\nCreating table {name}...")
        cur.execute(f"""
            CREATE TABLE {name} (
                src_id UUID NOT NULL,
                n_inter_deps BIGINT NOT NULL DEFAULT 0
            )
        """)
        from psycopg2.extras import execute_values
        execute_values(
            cur,
            f"INSERT INTO {name} (src_id, n_inter_deps) VALUES %s",
            stmts,
        )
        conn.commit()
        cur.execute(f"SELECT COUNT(*) FROM {name}")
        n = cur.fetchone()[0]
        print(f"  {name}: {n:,} rows committed")

    create_split_table("rl_train", train_stmts)
    create_split_table("rl_val",   val_stmts)
    create_split_table("rl_test",  test_stmts)

    conn.close()
    print("\nSplits built successfully.")


if __name__ == "__main__":
    main()
