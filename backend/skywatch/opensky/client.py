"""Async client for the OpenSky ``/states/all`` endpoint.

Only fetches raw JSON for the configured bounding box; decoding is delegated to
:mod:`skywatch.opensky.parser`. Keeps the bbox small and polls infrequently
because ``/states/all`` is billed against a daily credit budget that scales with
bbox area (spec §2/§3).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from skywatch.config import Settings
from skywatch.opensky.token_manager import TokenManager

log = logging.getLogger(__name__)


class RateLimitedError(RuntimeError):
    """Raised on HTTP 429 — the daily credit budget / rate limit was hit."""


class OpenSkyClient:
    def __init__(
        self,
        settings: Settings,
        token_manager: TokenManager,
        http_client: httpx.AsyncClient,
    ) -> None:
        self._settings = settings
        self._tokens = token_manager
        self._http = http_client

    async def fetch_states(self) -> dict[str, Any]:
        """Fetch the current state vectors for the configured bbox.

        Retries once on a 401 by force-refreshing the token (it may have expired
        between the proactive check and the request). Raises
        :class:`RateLimitedError` on 429 so the caller can back off.
        """
        payload = await self._request_states()
        return payload

    async def _request_states(self, _retried: bool = False) -> dict[str, Any]:
        token = await self._tokens.get_token()
        resp = await self._http.get(
            self._settings.opensky_states_url,
            params=self._settings.bbox_params,
            headers={"Authorization": f"Bearer {token}"},
        )

        if resp.status_code == 401 and not _retried:
            log.warning("OpenSky returned 401; forcing token refresh and retrying")
            self._tokens.invalidate()
            await self._tokens.get_token(force_refresh=True)
            return await self._request_states(_retried=True)

        if resp.status_code == 429:
            raise RateLimitedError(
                "OpenSky rate limit / daily credit budget hit (HTTP 429)"
            )

        resp.raise_for_status()
        return resp.json()
