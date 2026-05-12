import asyncio
import logging
import threading
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import diskcache
import requests as _requests

logger = logging.getLogger(__name__)

SEARCH_URL = "https://api.theoremsearch.com/search"

# Executor for blocking HTTP calls. max_workers >= concurrency * 2 to absorb
# retries without queuing: 4 concurrent slots × 2 (retry overlap) = 8 minimum.
_executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="search")


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


def _sync_post(payload: dict, total_timeout: float) -> list:
    """HTTP POST with a hard wall-clock deadline via thread.join().

    requests.post(timeout=N) is a per-read socket timeout: if the server
    trickles the response body one chunk every N-1 seconds, the request runs
    forever. This wrapper spawns a daemon thread and joins it with a hard
    wall-clock limit — thread.join(timeout) always fires on wall-clock time,
    independent of the asyncio event loop or socket-level keepalives.
    """
    result: list = [[]]
    exc_holder: list = [None]

    def _run() -> None:
        try:
            per_read = max(5.0, total_timeout / 3)
            resp = _requests.post(SEARCH_URL, json=payload, timeout=per_read)
            resp.raise_for_status()
            data = resp.json()
            result[0] = (
                data.get("theorems")
                or data.get("results")
                or (data if isinstance(data, list) else [])
            )
        except Exception as exc:
            exc_holder[0] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=total_timeout)

    if t.is_alive():
        raise TimeoutError(f"search request exceeded {total_timeout}s wall-clock limit")
    if exc_holder[0] is not None:
        raise exc_holder[0]
    return result[0]


class SearchClient:
    def __init__(
        self,
        cache_dir: str = "cache/search",
        concurrency: int = 4,
        request_timeout: float = 30.0,
    ):
        self._cache = diskcache.Cache(cache_dir, size_limit=10 * 2**30)
        self._concurrency = concurrency
        self._request_timeout = request_timeout
        self._semaphore: asyncio.Semaphore | None = None

    async def search(self, query: str, k: int) -> list[SearchResult]:
        cache_key = (query, k)
        loop = asyncio.get_running_loop()

        # Cache lookup in executor — diskcache.get() holds a SQLite lock that
        # would block the event loop if called directly.
        cached = await loop.run_in_executor(None, self._cache.get, cache_key)
        if cached is not None:
            logger.debug("search cache hit  query=%r", query[:120])
            return cached

        # Lazy semaphore — must be created inside a running event loop.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency)

        payload = {"query": query, "n_results": k}

        async with self._semaphore:
            for attempt in range(3):
                _t0 = _time.monotonic()
                try:
                    # _sync_post enforces a wall-clock hard deadline internally
                    # via thread.join(); no asyncio.timeout needed here.
                    items = await loop.run_in_executor(
                        _executor, _sync_post, payload, self._request_timeout
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
                    await asyncio.sleep(wait)

        return []

    async def close(self) -> None:
        self._cache.close()

    async def __aenter__(self) -> "SearchClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
