import html
import logging
import re
import threading
from typing import NamedTuple
from uuid import UUID

import diskcache
import numpy as np
from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import partial_ratio

from src.data.load import DepBody
from src.env.search_client import SearchResult

logger = logging.getLogger(__name__)

# Serialise cdist calls process-wide. Concurrent cdists with workers=-1
# spawn N*48 threads contending for 48 cores; the lock ensures one cdist
# at a time gets full parallelism instead.
_CDIST_LOCK = threading.Lock()


def normalize(body: str) -> str:
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = html.unescape(body)
    body = re.sub(r"\s+", " ", body)
    body = body.strip().rstrip(".")
    return body


class MatchResult(NamedTuple):
    uuid: UUID | None
    score: float
    second_best_gap: float


class IDMapper:
    def __init__(
        self,
        dep_bodies: dict[UUID, DepBody],
        threshold: float = 85.0,
        low_confidence_gap: float = 5.0,
        cache_dir: str | None = "cache/match",
    ):
        self.threshold = threshold
        self.low_confidence_gap = low_confidence_gap

        self._uuids: list[UUID] = []
        self._normalized_bodies: list[str] = []

        for uid, dep in dep_bodies.items():
            self._uuids.append(uid)
            self._normalized_bodies.append(normalize(dep.body))

        # Persistent cache of theorem_id → MatchResult. rapidfuzz over ~10K
        # bodies takes ~10s per call; with 10 results per search this would
        # add ~100s per env.step(). The cache makes repeat lookups instant
        # and persists across training runs.
        self._match_cache: diskcache.Cache | dict[int, MatchResult]
        if cache_dir is not None:
            self._match_cache = diskcache.Cache(cache_dir, size_limit=2 * 2**30)
        else:
            self._match_cache = {}

    def map_int_to_uuid(self, api_result: SearchResult) -> MatchResult:
        return self.map_batch([api_result])[0]

    def map_batch(self, api_results: list[SearchResult]) -> list[MatchResult]:
        """Map a batch of SearchResults via a single cdist call.

        Single-query cdist has high thread-spawn overhead (~4s per call);
        batching all N queries amortises that cost across N. With 10 results
        per env.step this drops per-step matching from ~40s to ~4s.
        """
        out: list[MatchResult | None] = [None] * len(api_results)
        to_compute: list[tuple[int, str]] = []  # (index, normalized body)

        for i, r in enumerate(api_results):
            cached = self._match_cache.get(r.theorem_id)
            if cached is not None:
                out[i] = cached
                continue
            body = normalize(r.body)
            if not body or not self._normalized_bodies:
                result = MatchResult(uuid=None, score=0.0, second_best_gap=0.0)
                self._match_cache[r.theorem_id] = result
                out[i] = result
                continue
            to_compute.append((i, body))

        if to_compute:
            queries = [b for _, b in to_compute]
            # Single batched cdist: N queries × M choices. Lock so concurrent
            # callers don't oversubscribe the 48 cores with 48 workers each.
            with _CDIST_LOCK:
                scores = rf_process.cdist(
                    queries,
                    self._normalized_bodies,
                    scorer=partial_ratio,
                    workers=-1,
                )
            # Top-2 per row via argpartition (O(M) vs O(M log M) for argsort)
            top2_idx = np.argpartition(-scores, kth=1, axis=1)[:, :2]
            rows = np.arange(len(queries))[:, None]
            top2_scores = scores[rows, top2_idx]
            order = np.argsort(-top2_scores, axis=1)
            best_idx_arr = top2_idx[rows, order][:, 0]
            best_scores = top2_scores[rows, order][:, 0]
            second_scores = top2_scores[rows, order][:, 1]

            for k, (i, _) in enumerate(to_compute):
                best_score = float(best_scores[k])
                second_score = float(second_scores[k])
                gap = best_score - second_score
                if best_score < self.threshold:
                    result = MatchResult(uuid=None, score=best_score, second_best_gap=gap)
                else:
                    uuid = self._uuids[int(best_idx_arr[k])]
                    if gap < self.low_confidence_gap:
                        logger.debug(
                            "low-confidence match: uuid=%s score=%.1f gap=%.1f",
                            uuid, best_score, gap,
                        )
                    result = MatchResult(uuid=uuid, score=best_score, second_best_gap=gap)
                self._match_cache[api_results[i].theorem_id] = result
                out[i] = result

        return out  # type: ignore[return-value]
