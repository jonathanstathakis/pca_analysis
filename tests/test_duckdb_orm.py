"""
testing duckdb orm
"""

import pytest
import duckdb
import sqlalchemy
from tests.test_definitions import TEST_DB_PATH
from sqlalchemy import Column, Integer, Sequence, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import mapped_column, Mapped
from sqlalchemy import select

def test_orm():
    Base = declarative_base()

    class Solvents(Base):
        __tablename__ = "solvents"

        runid: Mapped[str] = mapped_column(primary_key=True)
        a: Mapped[str] = mapped_column()
        b: Mapped[str] = mapped_column()

    eng = create_engine(f"duckdb:///{TEST_DB_PATH}")

    session = Session(bind=eng)

    result = select(Solvents).limit(5)

