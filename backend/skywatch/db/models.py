"""ORM models.

Phase 1 only needs ``raw_states`` — one row per (icao24, snapshot) parsed from
the OpenSky ``/states/all`` array-of-arrays response (spec §6). The columns mirror
the 18 positional state-vector fields exactly, in order, plus bookkeeping
(``request_time`` = the response's top-level snapshot time, ``ingested_at`` = when
we wrote the row).

Hypertable note: TimescaleDB requires the partitioning column to be part of every
unique constraint, so the primary key is composite ``(id, ingested_at)`` and the
hypertable (created in the migration when the extension is present) partitions on
the non-null ``ingested_at``. ``time_position`` can be null, so it isn't usable as
the partition column even though the per-aircraft index is on it (spec §6).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Identity,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from skywatch.db.base import Base


class RawState(Base):
    __tablename__ = "raw_states"

    # Bookkeeping
    id: Mapped[int] = mapped_column(BigInteger, Identity(), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    # Top-level "time" from the response: the epoch second this snapshot is valid for.
    request_time: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)

    # --- 18 positional state-vector fields, in spec order (parse by index) ---
    icao24: Mapped[str] = mapped_column(String(6), nullable=False)             # 0
    callsign: Mapped[str | None] = mapped_column(String(16), nullable=True)    # 1
    origin_country: Mapped[str | None] = mapped_column(String(64), nullable=True)  # 2
    time_position: Mapped[int | None] = mapped_column(BigInteger, nullable=True)    # 3
    last_contact: Mapped[int | None] = mapped_column(BigInteger, nullable=True)     # 4
    longitude: Mapped[float | None] = mapped_column(Double, nullable=True)     # 5
    latitude: Mapped[float | None] = mapped_column(Double, nullable=True)      # 6
    baro_altitude: Mapped[float | None] = mapped_column(Double, nullable=True)  # 7
    on_ground: Mapped[bool | None] = mapped_column(Boolean, nullable=True)     # 8
    velocity: Mapped[float | None] = mapped_column(Double, nullable=True)      # 9
    true_track: Mapped[float | None] = mapped_column(Double, nullable=True)    # 10
    vertical_rate: Mapped[float | None] = mapped_column(Double, nullable=True)  # 11
    sensors: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)  # 12
    geo_altitude: Mapped[float | None] = mapped_column(Double, nullable=True)  # 13
    squawk: Mapped[str | None] = mapped_column(String(8), nullable=True)       # 14
    spi: Mapped[bool | None] = mapped_column(Boolean, nullable=True)           # 15
    position_source: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 16
    category: Mapped[int | None] = mapped_column(Integer, nullable=True)       # 17

    __table_args__ = (
        PrimaryKeyConstraint("id", "ingested_at", name="pk_raw_states"),
        # Per spec §6: index on (icao24, time_position) for per-aircraft lookups.
        Index("ix_raw_states_icao24_time_position", "icao24", "time_position"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<RawState icao24={self.icao24!r} t={self.time_position} "
            f"lat={self.latitude} lon={self.longitude}>"
        )


class Trajectory(Base):
    """A reconstructed flight segment for one aircraft (spec §6, Phase 2).

    Produced by :mod:`skywatch.reconstruct` from ``raw_states``: a continuous,
    airborne stretch of one ``icao24`` with no gap longer than the segmentation
    threshold, resampled onto a fixed ``dt`` grid. ``start_time``/``end_time`` are
    epoch seconds (matching the raw time fields).
    """

    __tablename__ = "trajectories"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    icao24: Mapped[str] = mapped_column(String(6), nullable=False)
    start_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    end_time: Mapped[int] = mapped_column(BigInteger, nullable=False)
    point_count: Mapped[int] = mapped_column(Integer, nullable=False)
    dt_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    points: Mapped[list["TrajectoryPoint"]] = relationship(
        back_populates="trajectory",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("ix_trajectories_icao24_start", "icao24", "start_time"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Trajectory id={self.id} icao24={self.icao24!r} "
            f"points={self.point_count}>"
        )


class TrajectoryPoint(Base):
    """One resampled point of a trajectory (spec §6). Columns match the spec's
    trajectory_points list: airborne position/altitude/velocity/heading on the
    fixed ``dt`` grid. ``lat``/``lon`` are required (points without a position are
    dropped during reconstruction); the rest may be null when not reported."""

    __tablename__ = "trajectory_points"

    trajectory_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("trajectories.id", ondelete="CASCADE"),
        nullable=False,
    )
    t: Mapped[int] = mapped_column(BigInteger, nullable=False)  # epoch seconds (grid)
    lat: Mapped[float] = mapped_column(Double, nullable=False)
    lon: Mapped[float] = mapped_column(Double, nullable=False)
    baro_altitude: Mapped[float | None] = mapped_column(Double, nullable=True)
    geo_altitude: Mapped[float | None] = mapped_column(Double, nullable=True)
    velocity: Mapped[float | None] = mapped_column(Double, nullable=True)
    true_track: Mapped[float | None] = mapped_column(Double, nullable=True)
    vertical_rate: Mapped[float | None] = mapped_column(Double, nullable=True)

    trajectory: Mapped["Trajectory"] = relationship(back_populates="points")

    __table_args__ = (
        PrimaryKeyConstraint("trajectory_id", "t", name="pk_trajectory_points"),
    )


class AnomalyScore(Base):
    """Live anomaly score for one aircraft at one cycle (spec §6).

    Written by the scoring service each poll/replay cycle. ``reason`` is ``ml`` (model
    residual over threshold), ``physics:<rule>``, or null when not anomalous.
    """

    __tablename__ = "anomaly_scores"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    icao24: Mapped[str] = mapped_column(String(6), nullable=False)
    t: Mapped[int] = mapped_column(BigInteger, nullable=False)  # cycle epoch second
    score: Mapped[float] = mapped_column(Double, nullable=False)
    threshold: Mapped[float] = mapped_column(Double, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_anomaly_scores_icao24_t", "icao24", "t"),
        Index("ix_anomaly_scores_t", "t"),
    )
