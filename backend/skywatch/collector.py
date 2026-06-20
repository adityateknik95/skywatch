"""Poll loop: OpenSky ``/states/all`` -> parse -> bulk insert into ``raw_states``.

Run it for a few hours to build a normal-traffic dataset before Phase 4 (spec §13).

CLI::

    python -m skywatch.collector            # poll forever, every POLL_INTERVAL_SECONDS
    python -m skywatch.collector --once      # a single poll, then exit
    python -m skywatch.collector --from-file sample.json   # offline: parse+insert a
                                                           # saved response (no creds)

The ``--from-file`` mode exercises the full parse + DB write path without calling
the API, which is handy for verifying the pipeline before you have OpenSky
credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import insert

from skywatch.config import Settings, get_settings
from skywatch.db.models import RawState
from skywatch.db.session import dispose_engine, session_scope
from skywatch.opensky.client import OpenSkyClient, RateLimitedError
from skywatch.opensky.parser import parse_states_response
from skywatch.opensky.token_manager import TokenManager, TokenError

log = logging.getLogger("skywatch.collector")

# Back off this long after hitting the rate limit / credit budget.
RATE_LIMIT_BACKOFF_SECONDS = 60.0
HTTP_TIMEOUT_SECONDS = 30.0


async def insert_rows(rows: list[dict[str, Any]]) -> int:
    """Bulk-insert parsed state rows into ``raw_states``. Returns the row count."""
    if not rows:
        return 0
    async with session_scope() as session:
        await session.execute(insert(RawState), rows)
    return len(rows)


async def poll_once(client: OpenSkyClient) -> int:
    """Fetch one snapshot, parse it, and write the rows. Returns rows written."""
    payload = await client.fetch_states()
    request_time, rows = parse_states_response(payload)
    written = await insert_rows(rows)
    log.info("Snapshot t=%s: parsed %d states, wrote %d rows", request_time, len(rows), written)
    return written


async def run_collector(
    settings: Settings | None = None,
    *,
    run_once: bool = False,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Main poll loop. Stops when ``stop_event`` is set or after one poll if ``run_once``."""
    settings = settings or get_settings()
    if not settings.has_credentials:
        raise SystemExit(
            "OpenSky credentials are not configured. Set OPENSKY_CLIENT_ID and "
            "OPENSKY_CLIENT_SECRET in .env (see .env.example), or use "
            "`--from-file` to exercise the pipeline offline."
        )

    stop_event = stop_event or asyncio.Event()
    interval = settings.poll_interval_seconds

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as http:
        tokens = TokenManager(
            settings.opensky_client_id,
            settings.opensky_client_secret,
            settings.opensky_token_url,
            http,
        )
        client = OpenSkyClient(settings, tokens, http)
        log.info(
            "Collector started: bbox=%s interval=%ss%s",
            settings.bbox_params,
            interval,
            " (once)" if run_once else "",
        )

        while not stop_event.is_set():
            delay: float = interval
            try:
                await poll_once(client)
            except RateLimitedError as exc:
                log.warning("%s — backing off %.0fs", exc, RATE_LIMIT_BACKOFF_SECONDS)
                delay = RATE_LIMIT_BACKOFF_SECONDS
            except TokenError as exc:
                log.error("Auth failure: %s — backing off %.0fs", exc, interval)
            except httpx.HTTPError as exc:
                log.error("HTTP error during poll: %s", exc)
            except Exception:  # never let one bad cycle kill the loop
                log.exception("Unexpected error during poll")

            if run_once:
                break

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass  # normal: interval elapsed, poll again

    await dispose_engine()
    log.info("Collector stopped.")


async def run_from_file(path: Path) -> int:
    """Offline path: parse a saved ``/states/all`` response and insert it."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    request_time, rows = parse_states_response(payload)
    written = await insert_rows(rows)
    log.info("From %s (t=%s): parsed %d states, wrote %d rows", path, request_time, len(rows), written)
    await dispose_engine()
    return written


def _install_signal_handlers(loop: asyncio.AbstractEventLoop, stop_event: asyncio.Event) -> None:
    def _request_stop() -> None:
        log.info("Shutdown signal received; finishing current cycle...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, ValueError):
            # Windows event loops don't support add_signal_handler for SIGTERM;
            # KeyboardInterrupt still propagates out of asyncio.run for SIGINT.
            pass


async def _amain(args: argparse.Namespace) -> None:
    if args.from_file:
        await run_from_file(Path(args.from_file))
        return

    stop_event = asyncio.Event()
    _install_signal_handlers(asyncio.get_running_loop(), stop_event)
    await run_collector(run_once=args.once, stop_event=stop_event)


def main() -> None:
    parser = argparse.ArgumentParser(description="SkyWatch OpenSky collector")
    parser.add_argument("--once", action="store_true", help="poll a single time, then exit")
    parser.add_argument(
        "--from-file",
        metavar="PATH",
        help="parse and insert a saved /states/all JSON response (offline, no creds)",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="logging level (default: INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    # asyncpg + Windows: the selector loop avoids Proactor edge cases.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(_amain(args))
    except KeyboardInterrupt:
        log.info("Interrupted.")


if __name__ == "__main__":
    main()
