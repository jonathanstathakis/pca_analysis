from enum import StrEnum
import duckdb as db
import numpy as np
import polars as pl
import logging
from sqlalchemy import create_engine, Integer, Sequence, text, Engine
from sqlalchemy.orm import mapped_column, Mapped, Session
from .orm import Base
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from typing import List

logger = logging.getLogger(__name__)


class CoreTbls(StrEnum):
    EXEC_ID = "exec_id"
    RESULT_ID = "result_id"
    SAMPLES = "samples"


sample_sequence = Sequence(
    name="sample_id",
    start=1,
    increment=1,
)


class RunIDs(Base):
    """
    holds the run ids. To be merged into Samples at a later point
    """

    __tablename__ = "runids"

    runid: Mapped[str] = mapped_column(primary_key=True)
    runid_idx: Mapped[int] = mapped_column()


class Samples(Base):
    __tablename__ = "samples"

    # sample_id: Mapped[int] = mapped_column(sample_sequence, primary_key=True)

    exec_id: Mapped[str] = mapped_column(ForeignKey("exec_id.exec_id"))
    sample: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    runid: Mapped[str] = mapped_column(ForeignKey("runids.runid"))

    # children: Mapped["Samples"]


class ExecID(Base):
    __tablename__ = "exec_id"

    exec_id: Mapped[str] = mapped_column(primary_key=True)

    # many results for one execution
    # result_id: Mapped[List["ResultNames"]] = relationship()

    # many executions for one sample
    # parent: Mapped["Samples"] = relationship(back_populates="children")


class ResultNames(Base):
    __tablename__ = "result_id"

    # many results for one execution
    result_id: Mapped[str] = mapped_column(primary_key=True)
    exec_id: Mapped[str] = mapped_column(ForeignKey("exec_id.exec_id"))


class CoreTablesDB:
    def __init__(self, engine: Engine):
        """wrapper around the core tables of the results db, providing IO for those tables"""

        self._engine = engine

    def get_loader(self, exec_id, runids):
        return CoreTableLoader(exec_id=exec_id, runids=runids, engine=self._engine)

    # def clear_tables(self):
    #     """delete all rows of core tables 'exec_id', 'results_id' and 'samples'."""

    #     for table in CoreTbls:
    #         self._conn.execute(f"truncate {table}")


class CoreTableLoader:
    def __init__(
        self,
        exec_id: str,
        runids: pl.DataFrame,
        engine: Engine,
    ):
        """loads the core tables that other loaders depend on"""
        self._exec_id = exec_id
        self._engine = engine

        self._runids = runids

    def load_core_tables(self):
        # need to modify this to either only create the core tables OR called somewhere else

        self._create_exec_id_tbl()
        self._create_runid_tbl()
        self._create_sample_table()
        self.create_result_id_tbl()

        logger.debug("core tables loaded.")

    def _create_runid_tbl(self):
        """create the runid table"""

        Base.metadata.create_all(self._engine, tables=[RunIDs.__table__])

        self._runids.write_database(
            table_name=RunIDs.__tablename__,
            connection=self._engine,
            if_table_exists="append",
        )

    def _create_exec_id_tbl(self):
        """creates a exec_id table containing the exec_ids"""

        Base.metadata.create_all(self._engine, tables=[ExecID.__table__])

        with Session(self._engine) as session:
            entered = ExecID(exec_id=self._exec_id)
            session.add(entered)
            session.commit()
        logger.debug("entered exec_id..")

    def _create_sample_table(self):
        """write a table containing the unique sample ids"""

        logger.debug("writing sample table..")

        Base.metadata.create_all(self._engine, tables=[Samples.__table__])

        self._runids.rename({"runid_idx": "sample"}).with_columns(
            pl.lit(self._exec_id).alias("exec_id")
        ).write_database(
            table_name=Samples.__tablename__,
            connection=self._engine,
            if_table_exists="append",
        )

        logger.debug("inserted into sample table..")

    def create_result_id_tbl(self):
        """create an empty result_id table containing the identifier for each result type"""

        Base.metadata.create_all(self._engine, tables=[ResultNames.__table__])
