"""Token manager tests: caching, proactive refresh before expiry, force refresh,
invalidation, and error handling — all with a fake clock and mocked HTTP."""

from __future__ import annotations

import httpx
import pytest
import respx

from skywatch.opensky.token_manager import TokenError, TokenManager

TOKEN_URL = "https://auth.test/token"


class Clock:
    """Deterministic monotonic clock."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _token_response(access_token: str, expires_in: int = 1800) -> httpx.Response:
    return httpx.Response(
        200, json={"access_token": access_token, "expires_in": expires_in, "token_type": "bearer"}
    )


@respx.mock
async def test_caches_token_until_near_expiry():
    clock = Clock()
    route = respx.post(TOKEN_URL).mock(
        side_effect=[_token_response("tok1"), _token_response("tok2")]
    )

    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http, refresh_skew_seconds=60, now=clock)

        assert await tm.get_token() == "tok1"
        assert route.call_count == 1

        # Still well within validity: cached, no new request.
        clock.advance(100)
        assert await tm.get_token() == "tok1"
        assert route.call_count == 1


@respx.mock
async def test_refreshes_proactively_before_expiry():
    clock = Clock()
    route = respx.post(TOKEN_URL).mock(
        side_effect=[_token_response("tok1", expires_in=1800), _token_response("tok2")]
    )

    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http, refresh_skew_seconds=60, now=clock)
        assert await tm.get_token() == "tok1"

        # Move to inside the 60s skew window (expiry at 1800, skew 60 -> refresh at 1740).
        clock.advance(1750)
        assert await tm.get_token() == "tok2"
        assert route.call_count == 2


@respx.mock
async def test_force_refresh_and_invalidate():
    clock = Clock()
    route = respx.post(TOKEN_URL).mock(
        side_effect=[_token_response("tok1"), _token_response("tok2"), _token_response("tok3")]
    )

    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http, now=clock)
        assert await tm.get_token() == "tok1"

        assert await tm.get_token(force_refresh=True) == "tok2"
        assert route.call_count == 2

        tm.invalidate()
        assert await tm.get_token() == "tok3"
        assert route.call_count == 3


@respx.mock
async def test_error_on_non_200():
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(401, text="invalid_client"))
    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "bad", TOKEN_URL, http)
        with pytest.raises(TokenError):
            await tm.get_token()


@respx.mock
async def test_error_on_missing_access_token():
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"expires_in": 1800}))
    async with httpx.AsyncClient() as http:
        tm = TokenManager("id", "secret", TOKEN_URL, http)
        with pytest.raises(TokenError):
            await tm.get_token()
