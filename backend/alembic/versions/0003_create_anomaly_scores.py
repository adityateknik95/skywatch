"""create anomaly_scores (Phase 6 scoring service)

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-18

Live anomaly scores written by the scoring service (spec §6).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "anomaly_scores",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("icao24", sa.String(length=6), nullable=False),
        sa.Column("t", sa.BigInteger(), nullable=False),
        sa.Column("score", sa.Double(), nullable=False),
        sa.Column("threshold", sa.Double(), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_anomaly_scores"),
    )
    op.create_index("ix_anomaly_scores_icao24_t", "anomaly_scores", ["icao24", "t"])
    op.create_index("ix_anomaly_scores_t", "anomaly_scores", ["t"])


def downgrade() -> None:
    op.drop_index("ix_anomaly_scores_t", table_name="anomaly_scores")
    op.drop_index("ix_anomaly_scores_icao24_t", table_name="anomaly_scores")
    op.drop_table("anomaly_scores")
