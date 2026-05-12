import asyncio
import logging
import time as _time
from dataclasses import dataclass

import diskcache
import httpx

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.theoremsearch.com/search"


@dataclass
class SearchResult:
    theorem_id: int
    slogan_id: int
    name: str
    body: str
    slogan: str
    theorem_type: str
    link: str
    similarity: float
    paper: str


def _make_result(item: dict) -> SearchResult:
    paper = item.get("paper") or {}
    if isinstance(paper, dict):
        # API returns paper.source (arXiv ID); paper_id kept as fallback
        paper_id = paper.get("source", "") or paper.get("paper_id", "")
    else:
        paper_id = str(paper)
    return SearchResult(
        theorem_id=item.get("theorem_id", 0),
        slogan_id=item.get("slogan_id", 0),
        name=item.get("name", ""),
        body=item.get("body", ""),
        slogan=item.get("slogan", ""),
        theorem_type=item.get("theorem_type", ""),
        link=item.get("link") or "",
        similarity=item.get("similarity", 0.0),
        paper=paper_id,
    )


class SearchClient:
    """Async search client with httpx and asyncio-native timeouts.

    asyncio.timeout() is used instead of threading.Thread.join() — it
    participates in the event loop's cancellation machinery and is reliable
    even when model.generate() briefly blocks the loop between await points.
    Slow or trickle-response servers are cut off after request_timeout seconds.
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
        cache_key = (query, k)
        loop = asyncio.get_running_loop()

        cached = await loop.run_in_executor(None, self._cache.get, cache_key)
        if cached is not None:
            logger.debug("search cache hit  query=%r", query[:120])
            return cached

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency)

        payload = {"query": query, "n_results": k}

        async with self._semaphore:
            for attempt in range(3):
                _t0 = _time.monotonic()
                try:
                    async with asyncio.timeout(self._request_timeout):
                        resp = await self._get_http().post(SEARCH_URL, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    items = (
                        data.get("theorems")
                        or data.get("results")
                        or (data if isinstance(data, list) else [])
                    )
                    results = [_make_result(item) for item in items]
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
                    # Recreate client after a failed/cancelled request
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
