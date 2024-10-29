from enum import StrEnum
import duckdb as db
import numpy as np
import polars as pl
import logging
from sqlalchemy import create_engine, Integer, Sequence, text, Engine
from sqlalchemy.orm import mapped_column, Mapped, Session
from .orm import ParafacResultsBase
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

runid_sequence = Sequence(
    name="runid_idx",
    start=1,
    increment=1,
)


class RunIDs(ParafacResultsBase):
    """
    holds the run ids. To be merged into Samples at a later point
    """

    __tablename__ = "runids"

    runid_idx: Mapped[int] = mapped_column(runid_sequence)
    runid: Mapped[str] = mapped_column(primary_key=True)


class ExecID(ParafacResultsBase):
    __tablename__ = "exec_id"

    exec_id: Mapped[str] = mapped_column(primary_key=True)


class ResultNames(ParafacResultsBase):
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
        self.create_result_id_tbl()

        logger.debug("core tables loaded.")

    def _create_runid_tbl(self):
        """create the runid table"""

        ParafacResultsBase.metadata.create_all(self._engine, tables=[RunIDs.__table__])

        with Session(self._engine) as session:
            for runid in self._runids:
                session.add(RunIDs(runid=runid))
            session.commit()

    def _create_exec_id_tbl(self):
        """creates a exec_id table containing the exec_ids"""

        ParafacResultsBase.metadata.create_all(self._engine, tables=[ExecID.__table__])

        with Session(self._engine) as session:
            entered = ExecID(exec_id=self._exec_id)
            session.add(entered)
            session.commit()
        logger.debug("entered exec_id..")

    def create_result_id_tbl(self):
        """create an empty result_id table containing the identifier for each result type"""

        ParafacResultsBase.metadata.create_all(
            self._engine, tables=[ResultNames.__table__]
        )
