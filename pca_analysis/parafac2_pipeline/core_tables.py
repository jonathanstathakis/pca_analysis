from enum import StrEnum
import logging
from sqlalchemy import Sequence
from sqlalchemy.orm import mapped_column, Mapped, Session
from .orm import ParafacResultsBase
from sqlalchemy import ForeignKey

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


def load_core_tables(engine, exec_id, runids):
    # need to modify this to either only create the core tables OR called somewhere else

    _create_exec_id_tbl(engine, exec_id)
    _create_runid_tbl(engine, runids)
    _create_result_id_tbl(engine)

    logger.debug("core tables loaded.")


def _create_runid_tbl(engine, runids):
    """create the runid table"""

    ParafacResultsBase.metadata.create_all(engine, tables=[RunIDs.__table__])

    with Session(engine) as session:
        for runid in runids:
            session.merge(RunIDs(runid=runid))
        session.commit()


def _create_exec_id_tbl(engine, exec_id):
    """creates a exec_id table containing the exec_ids"""

    ParafacResultsBase.metadata.create_all(engine, tables=[ExecID.__table__])

    with Session(engine) as session:
        try:
            entered = ExecID(exec_id=exec_id)
            session.merge(entered)
        except Exception:
            session.rollback()
            session.close()
            raise
        else:
            session.commit()

    logger.debug("entered exec_id..")


def _create_result_id_tbl(engine):
    """create an empty result_id table containing the identifier for each result type"""

    ParafacResultsBase.metadata.create_all(engine, tables=[ResultNames.__table__])
