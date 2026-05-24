import asyncio
import logging
import time as _time
from dataclasses import dataclass
from uuid import UUID

import diskcache
import httpx

logger = logging.getLogger(__name__)

GRAPH_EMBEDDING_URL = "https://api.theoremsearch.com/graph/embedding"


@dataclass
class SearchResult:
    statement_id: UUID
    paper_id: UUID
    name: str
    body: str
    slogan: str
    source: str      # external_id (arXiv ID, repo slug, etc.)
    similarity: float
    score: float


def _make_result(item: dict) -> SearchResult | None:
    try:
        return SearchResult(
            statement_id=UUID(item["statement_id"]),
            paper_id=UUID(item["paper_id"]),
            name=item.get("name", ""),
            body=item.get("body", ""),
            slogan=item.get("slogan", ""),
            source=item.get("external_id", "") or item.get("source", ""),
            similarity=item.get("similarity", 0.0),
            score=item.get("score", 0.0),
        )
    except (KeyError, ValueError):
        return None


class SearchClient:
    """Async search client with httpx and asyncio-native timeouts.

    asyncio.timeout() is used instead of threading.Thread.join() — it
    participates in the event loop's cancellation machinery and is reliable
    even when model.generate() briefly blocks the loop between await points.
    Slow or trickle-response servers are cut off after request_timeout seconds.

    Cache key includes a version suffix ("v2") so stale int-based entries from
    the old POST /search endpoint are ignored on the first run.
    """

    def __init__(
        self,
        cache_dir: str = "cache/search",
        concurrency: int = 16,
        request_timeout: float = 30.0,
    ):
        self._cache = diskcache.Cache(cache_dir, size_limit=10 * 2**30)
        self._concurrency = concurrency
        self._request_timeout = request_timeout
        self._semaphore: asyncio.Semaphore | None = None
        self._http: httpx.AsyncClient | None = None

    def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            # HTTP/1.1 with a generous connection pool. HTTP/2 serializes
            # multiplexed requests on the single TCP connection server-side,
            # so 8 concurrent requests took 10s in benchmark vs 5.87s on
            # HTTP/1.1 across multiple keep-alive connections.
            limits = httpx.Limits(max_connections=32, max_keepalive_connections=32)
            self._http = httpx.AsyncClient(
                timeout=self._request_timeout,
                limits=limits,
            )
        return self._http

    async def search(self, query: str, k: int) -> list[SearchResult]:
        cache_key = (query, k, "v2")
        loop = asyncio.get_running_loop()

        cached = await loop.run_in_executor(None, self._cache.get, cache_key)
        if cached is not None:
            logger.debug("search cache hit  query=%r", query[:120])
            return cached

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency)

        params = {"query": query, "n_results": k}

        async with self._semaphore:
            for attempt in range(3):
                _t0 = _time.monotonic()
                try:
                    async with asyncio.timeout(self._request_timeout):
                        resp = await self._get_http().get(GRAPH_EMBEDDING_URL, params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    items = data.get("results", [])
                    results = [r for item in items if (r := _make_result(item)) is not None]
                    elapsed = _time.monotonic() - _t0
                    logger.info(
                        "search ok  t=%.2fs  n=%d  query=%r",
                        elapsed, len(results), query[:120],
                    )
                    await loop.run_in_executor(None, self._cache.set, cache_key, results)
                    return results
                except Exception as exc:
                    elapsed = _time.monotonic() - _t0
                    if attempt == 2:
                        logger.warning(
                            "search failed after 3 attempts (last=%.2fs) for query=%r: %s",
                            elapsed, query[:120], exc,
                        )
                        return []
                    wait = 2 ** attempt
                    logger.debug(
                        "search attempt=%d failed (%.2fs): %s  retry in %ds",
                        attempt, elapsed, exc, wait,
                    )
                    if self._http and not self._http.is_closed:
                        await self._http.aclose()
                    self._http = None
                    await asyncio.sleep(wait)

        return []

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
        self._cache.close()

    async def __aenter__(self) -> "SearchClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
