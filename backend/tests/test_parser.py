"""Parser tests against a captured /states/all sample including null fields,
a padded callsign, a short (category-absent) vector, and a null-icao24 row."""

from __future__ import annotations

import json

from skywatch.opensky.parser import parse_state_vector, parse_states_response


def test_parse_states_response_from_sample(sample_states_path):
    payload = json.loads(sample_states_path.read_text(encoding="utf-8"))
    request_time, rows = parse_states_response(payload)

    assert request_time == 1718600000
    # 4 vectors in, but the null-icao24 row is dropped -> 3 rows out.
    assert len(rows) == 3
    assert all(row["request_time"] == 1718600000 for row in rows)
    assert all(row["icao24"] is not None for row in rows)


def test_full_vector_fields(sample_states_path):
    payload = json.loads(sample_states_path.read_text(encoding="utf-8"))
    _, rows = parse_states_response(payload)
    row = rows[0]

    assert row["icao24"] == "3c6444"           # lower-cased
    assert row["callsign"] == "DLH9LH"          # padding stripped
    assert row["origin_country"] == "Germany"
    assert row["time_position"] == 1718599998
    assert row["last_contact"] == 1718599999
    assert row["longitude"] == 8.5432
    assert row["latitude"] == 50.1109
    assert row["baro_altitude"] == 11277.6
    assert row["on_ground"] is False
    assert row["velocity"] == 231.5
    assert row["true_track"] == 182.3
    assert row["vertical_rate"] == 0.0
    assert row["sensors"] is None
    assert row["geo_altitude"] == 11582.4
    assert row["squawk"] == "1000"
    assert row["spi"] is False
    assert row["position_source"] == 0
    assert row["category"] == 0


def test_null_fields_and_short_vector(sample_states_path):
    """Second row: lots of nulls, on_ground True, and category absent (17 fields)."""
    payload = json.loads(sample_states_path.read_text(encoding="utf-8"))
    _, rows = parse_states_response(payload)
    row = rows[1]

    assert row["icao24"] == "4b1815"
    assert row["callsign"] is None       # null callsign stays None
    assert row["time_position"] is None
    assert row["longitude"] is None
    assert row["latitude"] is None
    assert row["on_ground"] is True
    assert row["velocity"] is None
    assert row["sensors"] is None
    assert row["position_source"] == 3
    assert row["category"] is None       # trailing field absent -> None, no IndexError


def test_sensors_array_and_category(sample_states_path):
    payload = json.loads(sample_states_path.read_text(encoding="utf-8"))
    _, rows = parse_states_response(payload)
    row = rows[2]

    assert row["icao24"] == "406a3e"
    assert row["callsign"] == "BAW283"
    assert row["sensors"] == [12, 34, 56]
    assert row["squawk"] is None
    assert row["category"] == 1


def test_parse_state_vector_skips_missing_icao24():
    assert parse_state_vector([None, "X", "Y"]) is None
    assert parse_state_vector([]) is None
    assert parse_state_vector(["   "]) is None  # blank icao24 -> None


def test_parse_state_vector_blank_callsign_is_none():
    row = parse_state_vector(["abc123", "        ", "Germany"])
    assert row is not None
    assert row["callsign"] is None


def test_empty_states_list():
    request_time, rows = parse_states_response({"time": 123, "states": None})
    assert request_time == 123
    assert rows == []
