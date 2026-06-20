"""create trajectories + trajectory_points (Phase 2 reconstruction)

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-18

Reconstructed flight segments (spec §6). Derived from raw_states by
skywatch.reconstruct; safe to drop and rebuild.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "trajectories",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("icao24", sa.String(length=6), nullable=False),
        sa.Column("start_time", sa.BigInteger(), nullable=False),
        sa.Column("end_time", sa.BigInteger(), nullable=False),
        sa.Column("point_count", sa.Integer(), nullable=False),
        sa.Column("dt_seconds", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="pk_trajectories"),
    )
    op.create_index(
        "ix_trajectories_icao24_start", "trajectories", ["icao24", "start_time"]
    )

    op.create_table(
        "trajectory_points",
        sa.Column("trajectory_id", sa.BigInteger(), nullable=False),
        sa.Column("t", sa.BigInteger(), nullable=False),
        sa.Column("lat", sa.Double(), nullable=False),
        sa.Column("lon", sa.Double(), nullable=False),
        sa.Column("baro_altitude", sa.Double(), nullable=True),
        sa.Column("geo_altitude", sa.Double(), nullable=True),
        sa.Column("velocity", sa.Double(), nullable=True),
        sa.Column("true_track", sa.Double(), nullable=True),
        sa.Column("vertical_rate", sa.Double(), nullable=True),
        sa.ForeignKeyConstraint(
            ["trajectory_id"],
            ["trajectories.id"],
            name="fk_trajectory_points_trajectory",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("trajectory_id", "t", name="pk_trajectory_points"),
    )


def downgrade() -> None:
    op.drop_table("trajectory_points")
    op.drop_index("ix_trajectories_icao24_start", table_name="trajectories")
    op.drop_table("trajectories")
