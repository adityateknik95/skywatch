"""Parse OpenSky ``/states/all`` responses.

OpenSky state vectors are **arrays-of-arrays**, not JSON objects: each aircraft is
a flat list whose meaning is purely positional (spec §6). Fields are frequently
``null`` (receiver gaps, free-tier omissions) and the trailing ``category`` field
(index 17) is sometimes absent entirely, so every access is bounds- and None-safe.

``parse_states_response`` turns one response into ``(request_time, rows)`` where
each row is a ``dict`` keyed by ``raw_states`` column name, ready for a bulk insert.
"""

from __future__ import annotations

from typing import Any

# Positional field indices for an OpenSky state vector (spec §6).
ICAO24 = 0
CALLSIGN = 1
ORIGIN_COUNTRY = 2
TIME_POSITION = 3
LAST_CONTACT = 4
LONGITUDE = 5
LATITUDE = 6
BARO_ALTITUDE = 7
ON_GROUND = 8
VELOCITY = 9
TRUE_TRACK = 10
VERTICAL_RATE = 11
SENSORS = 12
GEO_ALTITUDE = 13
SQUAWK = 14
SPI = 15
POSITION_SOURCE = 16
CATEGORY = 17


def _at(vec: list[Any], idx: int) -> Any:
    """Return ``vec[idx]`` or ``None`` if the field is absent (short vector)."""
    return vec[idx] if idx < len(vec) else None


def _str(value: Any) -> str | None:
    """Normalize a string field: strip padding, treat blank as ``None``."""
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def parse_state_vector(vec: list[Any]) -> dict[str, Any] | None:
    """Convert one positional state vector into a column dict.

    Returns ``None`` for unusable rows (missing ``icao24``), which the caller
    skips — ``icao24`` is the only non-null state-vector field we hard-require.
    """
    if not vec:
        return None

    icao24 = _str(_at(vec, ICAO24))
    if icao24 is None:
        return None
    icao24 = icao24.lower()

    return {
        "icao24": icao24,
        "callsign": _str(_at(vec, CALLSIGN)),
        "origin_country": _str(_at(vec, ORIGIN_COUNTRY)),
        "time_position": _at(vec, TIME_POSITION),
        "last_contact": _at(vec, LAST_CONTACT),
        "longitude": _at(vec, LONGITUDE),
        "latitude": _at(vec, LATITUDE),
        "baro_altitude": _at(vec, BARO_ALTITUDE),
        "on_ground": _at(vec, ON_GROUND),
        "velocity": _at(vec, VELOCITY),
        "true_track": _at(vec, TRUE_TRACK),
        "vertical_rate": _at(vec, VERTICAL_RATE),
        "sensors": _at(vec, SENSORS),
        "geo_altitude": _at(vec, GEO_ALTITUDE),
        "squawk": _str(_at(vec, SQUAWK)),
        "spi": _at(vec, SPI),
        "position_source": _at(vec, POSITION_SOURCE),
        "category": _at(vec, CATEGORY),
    }


def parse_states_response(payload: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
    """Parse a full ``/states/all`` response.

    Returns ``(request_time, rows)``. ``request_time`` is the response's top-level
    ``time`` (epoch seconds for the snapshot); ``rows`` are ready-to-insert dicts
    with ``request_time`` attached. ``states`` may be ``null`` when no aircraft are
    in the bbox.
    """
    request_time = int(payload.get("time"))
    states = payload.get("states") or []

    rows: list[dict[str, Any]] = []
    for vec in states:
        row = parse_state_vector(vec)
        if row is None:
            continue
        row["request_time"] = request_time
        rows.append(row)
    return request_time, rows
