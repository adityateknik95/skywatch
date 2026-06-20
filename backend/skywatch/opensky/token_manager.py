"""OAuth2 client-credentials token manager for OpenSky.

OpenSky removed basic auth in March 2026; access now requires exchanging a
``client_id`` / ``client_secret`` for a short-lived bearer token (spec §2). Tokens
expire after ~30 minutes, so this manager refreshes **proactively** — a few seconds
before expiry — rather than waiting for a 401.

Concurrency: refreshes are serialized with an ``asyncio.Lock`` and guarded by a
double-checked expiry test, so concurrent callers share a single in-flight refresh.
This is async-safe for a single event loop (which is how the collector uses it);
it is not designed for use across OS threads.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import httpx

log = logging.getLogger(__name__)


class TokenError(RuntimeError):
    """Raised when a token cannot be obtained from the OpenSky auth server."""


class TokenManager:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        token_url: str,
        http_client: httpx.AsyncClient,
        *,
        refresh_skew_seconds: float = 60.0,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        """
        Args:
            http_client: shared async client (the collector passes the same one it
                uses for ``/states/all``).
            refresh_skew_seconds: refresh this many seconds *before* the real
                expiry, so a token is never used in its final moments.
            now: monotonic clock, injectable for tests.
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._token_url = token_url
        self._http = http_client
        self._refresh_skew = refresh_skew_seconds
        self._now = now

        self._access_token: str | None = None
        self._expires_at: float = 0.0  # in `now()` units
        self._lock = asyncio.Lock()

    def _is_valid(self) -> bool:
        return (
            self._access_token is not None
            and self._now() < self._expires_at - self._refresh_skew
        )

    async def get_token(self, *, force_refresh: bool = False) -> str:
        """Return a valid bearer token, refreshing proactively if needed."""
        if not force_refresh and self._is_valid():
            return self._access_token  # type: ignore[return-value]

        async with self._lock:
            # Re-check inside the lock: another coroutine may have just refreshed.
            if not force_refresh and self._is_valid():
                return self._access_token  # type: ignore[return-value]
            await self._refresh()
            return self._access_token  # type: ignore[return-value]

    async def _refresh(self) -> None:
        log.debug("Requesting new OpenSky access token")
        try:
            resp = await self._http.post(
                self._token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.HTTPError as exc:  # network/timeout
            raise TokenError(f"Token request failed: {exc}") from exc

        if resp.status_code != 200:
            raise TokenError(
                f"Token endpoint returned {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise TokenError(f"Token response missing access_token: {data}")

        # OpenSky tokens are ~1800s; trust the server's expires_in, default to 30m.
        expires_in = float(data.get("expires_in", 1800))
        self._access_token = token
        self._expires_at = self._now() + expires_in
        log.info(
            "Obtained OpenSky token (expires in %.0fs, refresh ~%.0fs before)",
            expires_in,
            self._refresh_skew,
        )

    def invalidate(self) -> None:
        """Force the next :meth:`get_token` to refresh (e.g. after a 401)."""
        self._access_token = None
        self._expires_at = 0.0
