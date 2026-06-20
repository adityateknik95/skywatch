"""OpenSkyClient tests: bbox params, 401 -> force-refresh-and-retry, 429 backoff."""

from __future__ import annotations

import httpx
import pytest
import respx

from skywatch.config import Settings
from skywatch.opensky.client import OpenSkyClient, RateLimitedError
from skywatch.opensky.token_manager import TokenManager

TOKEN_URL = "https://auth.test/token"
STATES_URL = "https://api.test/states/all"


def _settings() -> Settings:
    return Settings(
        opensky_client_id="id",
        opensky_client_secret="secret",
        opensky_token_url=TOKEN_URL,
        opensky_states_url=STATES_URL,
    )


def _token_response(tok: str = "tok") -> httpx.Response:
    return httpx.Response(200, json={"access_token": tok, "expires_in": 1800})


@respx.mock
async def test_fetch_states_sends_bbox_and_bearer():
    respx.post(TOKEN_URL).mock(return_value=_token_response("tok1"))
    states_route = respx.get(STATES_URL).mock(
        return_value=httpx.Response(200, json={"time": 1, "states": []})
    )

    settings = _settings()
    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http)
        client = OpenSkyClient(settings, tm, http)
        payload = await client.fetch_states()

    assert payload == {"time": 1, "states": []}
    request = states_route.calls.last.request
    assert request.headers["Authorization"] == "Bearer tok1"
    assert request.url.params["lamin"] == str(settings.bbox_lamin)
    assert request.url.params["lomax"] == str(settings.bbox_lomax)


@respx.mock
async def test_401_triggers_refresh_and_retry():
    respx.post(TOKEN_URL).mock(
        side_effect=[_token_response("stale"), _token_response("fresh")]
    )
    states_route = respx.get(STATES_URL).mock(
        side_effect=[
            httpx.Response(401, text="expired"),
            httpx.Response(200, json={"time": 2, "states": []}),
        ]
    )

    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http)
        client = OpenSkyClient(_settings(), tm, http)
        payload = await client.fetch_states()

    assert payload["time"] == 2
    assert states_route.call_count == 2
    # The retry used the refreshed token.
    assert states_route.calls.last.request.headers["Authorization"] == "Bearer fresh"


@respx.mock
async def test_429_raises_rate_limited():
    respx.post(TOKEN_URL).mock(return_value=_token_response())
    respx.get(STATES_URL).mock(return_value=httpx.Response(429, text="too many"))

    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http)
        client = OpenSkyClient(_settings(), tm, http)
        with pytest.raises(RateLimitedError):
            await client.fetch_states()
