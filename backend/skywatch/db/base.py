"""Declarative base shared by all ORM models and by Alembic's autogenerate."""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
