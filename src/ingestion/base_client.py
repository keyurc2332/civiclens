"""Shared HTTP client behavior for all ingestion sources: retries,
timeouts, and structured logging. Source-specific clients (data.gov.in,
CPCB, IMD) subclass or wrap this rather than each rolling their own
requests logic.
"""

import logging
import time
import uuid
from typing import Any

import requests

logger = logging.getLogger("civiclens.ingestion")
logging.basicConfig(level=logging.INFO)


class IngestionError(Exception):
    pass


class BaseAPIClient:
    def __init__(self, base_url: str, max_retries: int = 3, backoff_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    def new_batch_id(self) -> uuid.UUID:
        """Every ingestion run gets a batch id so raw rows can be traced
        back to the run that landed them, and a failed run can be
        identified/replayed without guessing."""
        return uuid.uuid4()

    # data.gov.in (and some other Indian gov APIs) return 502 for
    # Python's default "python-requests/x.x" user-agent but accept
    # browser-like ones. Set a default here so every source client
    # gets this for free instead of rediscovering it independently.
    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
    }

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("GET %s attempt=%d params=%s", url, attempt, params)
                resp = requests.get(url, params=params, headers=self.DEFAULT_HEADERS, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Request failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * attempt)  # simple linear backoff

        raise IngestionError(f"Failed to GET {url} after {self.max_retries} attempts") from last_exc
