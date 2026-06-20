"""create raw_states (+ TimescaleDB hypertable when available)

Revision ID: 0001
Revises:
Create Date: 2026-06-17

Mirrors the 18 positional OpenSky state-vector fields (spec §6) plus bookkeeping.
When the TimescaleDB extension is present, ``raw_states`` becomes a hypertable
partitioned on ``ingested_at``; on plain Postgres it stays a regular table.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "raw_states",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column(
            "ingested_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("request_time", sa.BigInteger(), nullable=False),
        # 18 positional state-vector fields, in order (spec §6)
        sa.Column("icao24", sa.String(length=6), nullable=False),          # 0
        sa.Column("callsign", sa.String(length=16), nullable=True),        # 1
        sa.Column("origin_country", sa.String(length=64), nullable=True),  # 2
        sa.Column("time_position", sa.BigInteger(), nullable=True),        # 3
        sa.Column("last_contact", sa.BigInteger(), nullable=True),         # 4
        sa.Column("longitude", sa.Double(), nullable=True),                # 5
        sa.Column("latitude", sa.Double(), nullable=True),                 # 6
        sa.Column("baro_altitude", sa.Double(), nullable=True),            # 7
        sa.Column("on_ground", sa.Boolean(), nullable=True),               # 8
        sa.Column("velocity", sa.Double(), nullable=True),                 # 9
        sa.Column("true_track", sa.Double(), nullable=True),               # 10
        sa.Column("vertical_rate", sa.Double(), nullable=True),            # 11
        sa.Column("sensors", sa.ARRAY(sa.Integer()), nullable=True),       # 12
        sa.Column("geo_altitude", sa.Double(), nullable=True),             # 13
        sa.Column("squawk", sa.String(length=8), nullable=True),           # 14
        sa.Column("spi", sa.Boolean(), nullable=True),                     # 15
        sa.Column("position_source", sa.Integer(), nullable=True),         # 16
        sa.Column("category", sa.Integer(), nullable=True),                # 17
        sa.PrimaryKeyConstraint("id", "ingested_at", name="pk_raw_states"),
    )
    op.create_index(
        "ix_raw_states_icao24_time_position",
        "raw_states",
        ["icao24", "time_position"],
    )
    op.create_index("ix_raw_states_request_time", "raw_states", ["request_time"])

    # Promote to a TimescaleDB hypertable only if the extension is available.
    conn = op.get_bind()
    has_ts = conn.execute(
        sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
    ).scalar()
    if has_ts:
        conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
        conn.execute(
            sa.text(
                "SELECT create_hypertable('raw_states', 'ingested_at', "
                "if_not_exists => TRUE)"
            )
        )


def downgrade() -> None:
    op.drop_index("ix_raw_states_request_time", table_name="raw_states")
    op.drop_index("ix_raw_states_icao24_time_position", table_name="raw_states")
    op.drop_table("raw_states")
