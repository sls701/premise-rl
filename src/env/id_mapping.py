import html
import logging
import re
from typing import NamedTuple
from uuid import UUID

from rapidfuzz import process as rf_process
from rapidfuzz.fuzz import partial_ratio

from src.data.load import DepBody
from src.env.search_client import SearchResult

logger = logging.getLogger(__name__)


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
    ):
        self.threshold = threshold
        self.low_confidence_gap = low_confidence_gap

        self._uuids: list[UUID] = []
        self._normalized_bodies: list[str] = []

        for uid, dep in dep_bodies.items():
            self._uuids.append(uid)
            self._normalized_bodies.append(normalize(dep.body))

    def map_int_to_uuid(self, api_result: SearchResult) -> MatchResult:
        if not self._normalized_bodies:
            return MatchResult(uuid=None, score=0.0, second_best_gap=0.0)

        query_body = normalize(api_result.body)
        if not query_body:
            return MatchResult(uuid=None, score=0.0, second_best_gap=0.0)

        # rapidfuzz.process.extract runs in a tight C loop
        hits = rf_process.extract(
            query_body,
            self._normalized_bodies,
            scorer=partial_ratio,
            limit=2,
            score_cutoff=0,
        )

        if not hits:
            return MatchResult(uuid=None, score=0.0, second_best_gap=0.0)

        best_body, best_score, best_idx = hits[0]
        second_score = hits[1][1] if len(hits) > 1 else 0.0
        gap = best_score - second_score

        if best_score < self.threshold:
            return MatchResult(uuid=None, score=best_score, second_best_gap=gap)

        uuid = self._uuids[best_idx]

        if gap < self.low_confidence_gap:
            logger.debug(
                "low-confidence match: uuid=%s score=%.1f gap=%.1f", uuid, best_score, gap
            )

        return MatchResult(uuid=uuid, score=best_score, second_best_gap=gap)
