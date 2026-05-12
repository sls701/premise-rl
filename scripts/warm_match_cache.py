"""Pre-warm the body→UUID match cache from already-cached search results.

rapidfuzz.process.extract over the 10K dep-body corpus takes ~10s per call,
so 10 results × N unique theorem_ids would dominate training. This script
walks every entry already in cache/search/, runs the rapidfuzz match once,
and stores the result in cache/match/. After running, env.step() is fast
for any query that was previously cached.

Use rapidfuzz workers=-1 here (this is the only thing using CPU during the
warm-up, so we can use all cores).
"""
import argparse
import logging
import time
from pathlib import Path

import diskcache
from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import partial_ratio

from src.data.load import load_dep_bodies
from src.env.id_mapping import MatchResult, normalize
from src.env.search_client import SearchResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="rl_train")
    ap.add_argument("--threshold", type=float, default=85.0)
    ap.add_argument("--low-confidence-gap", type=float, default=5.0)
    ap.add_argument("--search-cache", default="cache/search")
    ap.add_argument("--match-cache", default="cache/match")
    args = ap.parse_args()

    logger.info("Loading dep bodies for %s...", args.table)
    dep_bodies = load_dep_bodies(args.table)
    uuids = list(dep_bodies.keys())
    normalized = [normalize(dep_bodies[u].body) for u in uuids]
    logger.info("corpus size=%d", len(uuids))

    search_cache = diskcache.Cache(args.search_cache, size_limit=10 * 2**30)
    match_cache = diskcache.Cache(args.match_cache, size_limit=2 * 2**30)
    logger.info("search cache entries=%d  match cache entries=%d",
                len(search_cache), len(match_cache))

    # Collect every unique SearchResult across every cached query.
    unique_results: dict[int, SearchResult] = {}
    for key in search_cache:
        results = search_cache[key]
        for r in results:
            if r.theorem_id not in unique_results:
                unique_results[r.theorem_id] = r
    logger.info("unique theorem_ids to match: %d", len(unique_results))

    to_match = [r for tid, r in unique_results.items() if tid not in match_cache]
    logger.info("already cached: %d  to compute: %d",
                len(unique_results) - len(to_match), len(to_match))

    if not to_match:
        logger.info("nothing to do")
        return

    # Batch all queries through process.cdist with workers=-1 to parallelise
    # across CPU cores. Output is a (n_queries, n_choices) score matrix.
    query_bodies = [normalize(r.body) for r in to_match]
    empty_idx = {i for i, b in enumerate(query_bodies) if not b}
    logger.info("running batched cdist (%d queries × %d choices, workers=-1)...",
                len(query_bodies), len(normalized))

    t_start = time.monotonic()
    score_matrix = rf_process.cdist(
        query_bodies, normalized,
        scorer=partial_ratio,
        workers=-1,
        dtype=None,  # default float32
    )
    cdist_time = time.monotonic() - t_start
    logger.info("cdist done in %.1fs (%.3fs/query)", cdist_time, cdist_time / len(to_match))

    # For each query row, extract top-2 to compute the second_best_gap.
    import numpy as np
    n_q = score_matrix.shape[0]
    # argpartition is O(n) and gets the indices of the top-2 (unordered)
    top2_idx = np.argpartition(-score_matrix, kth=1, axis=1)[:, :2]
    rows = np.arange(n_q)[:, None]
    top2_scores = score_matrix[rows, top2_idx]
    # Sort the 2 picks per row so [0]=best, [1]=second
    order = np.argsort(-top2_scores, axis=1)
    top2_idx_sorted = top2_idx[rows, order]
    top2_scores_sorted = top2_scores[rows, order]

    for i, r in enumerate(to_match):
        if i in empty_idx:
            match_cache[r.theorem_id] = MatchResult(uuid=None, score=0.0, second_best_gap=0.0)
            continue
        best_score = float(top2_scores_sorted[i, 0])
        second_score = float(top2_scores_sorted[i, 1])
        gap = best_score - second_score
        best_idx = int(top2_idx_sorted[i, 0])
        if best_score < args.threshold:
            match_cache[r.theorem_id] = MatchResult(uuid=None, score=best_score, second_best_gap=gap)
        else:
            match_cache[r.theorem_id] = MatchResult(uuid=uuids[best_idx], score=best_score, second_best_gap=gap)

    total = time.monotonic() - t_start
    logger.info("Done. matched %d in %.1fs (cdist+write)", len(to_match), total)


if __name__ == "__main__":
    main()
