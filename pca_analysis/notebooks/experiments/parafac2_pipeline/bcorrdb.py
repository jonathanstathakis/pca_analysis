from pca_analysis.notebooks.experiments.parafac2_pipeline.estimators import (
    logger,
)

from .core_tables import CoreTbls
from enum import StrEnum
import duckdb as db
import polars as pl
from numpy.typing import NDArray
from .core_tables import ResultNames
from sqlalchemy.orm import Session
from sqlalchemy import Engine
from .orm import ParafacResultsBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import ForeignKey
from sqlalchemy import Index


class BcorrTbls(StrEnum):
    BCORR = "baseline_corrected"


class BcorrIdx(StrEnum):
    BCORR_IDX = "baseline_corrected_idx"


class SignalNames(StrEnum):
    CORR = "corrected"
    BLINE = "baseline"
    INPUT = "input"


class BCRCols(StrEnum):
    """BaselineCorrectionResultsColumns"""

    EXEC_ID = "exec_id"
    RESULT_ID = "result_id"
    SAMPLE = "sample"
    SAMPLE_IDX = "sample_idx"
    TIME_IDX = "time_idx"
    WAVELENGTH = "wavelength"
    WAVELENGTH_IDX = "wavelength_idx"
    ABS = "abs"
    SIGNAL = "signal"


class BCorr(ParafacResultsBase):
    __tablename__ = "baseline_correction"
    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"), primary_key=True
    )
    result_id: Mapped[str] = mapped_column(
        ForeignKey("result_id.result_id"), primary_key=True
    )
    runid: Mapped[str] = mapped_column(ForeignKey("runids.runid"), primary_key=True)
    signal: Mapped[str] = mapped_column(nullable=False, primary_key=True)
    wavelength: Mapped[int] = mapped_column(nullable=False, primary_key=True)
    time_idx: Mapped[int] = mapped_column(nullable=False, primary_key=True)
    abs: Mapped[float] = mapped_column(nullable=False)


bcorr_index = Index(
    "bcorr_index",
    BCorr.exec_id,
    BCorr.result_id,
    BCorr.result_id,
    BCorr.runid,
    BCorr.signal,
    BCorr.wavelength,
    BCorr.time_idx,
)


def to_dataframe(arrs: list[NDArray], signal_name=None):
    df = pl.concat(
        [
            pl.DataFrame(x)
            .with_columns(pl.lit(idx).alias(BCRCols.SAMPLE_IDX).cast(pl.UInt32))
            .with_row_index(BCRCols.TIME_IDX)
            for idx, x in enumerate(arrs)
        ]
    )

    df = df.unpivot(
        index=[BCRCols.SAMPLE_IDX, BCRCols.TIME_IDX],
        variable_name="wavelength_idx",
        value_name=BCRCols.ABS,
    )

    df = df.with_columns(
        pl.col(BCRCols.WAVELENGTH_IDX).str.replace("column_", "").cast(pl.UInt32)
    )

    if signal_name:
        df = df.with_columns(pl.lit(signal_name).alias(BCRCols.SIGNAL))
    return df


class BCorrLoader:
    def __init__(
        self,
        engine: Engine,
    ):
        self._engine = engine

    def _insert_into_result_id(self):
        """add the result_id into the result_id table"""

        logger.debug(f"inserting {self._result_id} into result_id table..")

        with Session(self._engine) as session:
            result_name = ResultNames(exec_id=self._exec_id, result_id=self._result_id)

            session.merge(result_name)
            session.commit()

        logger.debug("inserted result name into result_ids")

    def load_results(
        self,
        runids: list[str],
        wavelength_labels: list[int],
        exec_id: str,
        corrected: list[NDArray],
        baselines: list[NDArray],
        result_name: str = "bcorr",
    ):
        """write the baselines and corrected to a database"""

        logger.debug("loading bcorr results..")

        self.corrected = corrected
        self.baselines = baselines
        self._exec_id = exec_id
        self._result_id = result_name

        ParafacResultsBase.metadata.create_all(self._engine, tables=[BCorr.__table__])

        self._insert_into_result_id()

        with Session(self._engine) as session:
            for ss, sample in enumerate(self.corrected):
                for tt, time_point in enumerate(sample):
                    for nm_idx, abs in enumerate(time_point):
                        bcorr = BCorr(
                            exec_id=self._exec_id,
                            result_id=self._result_id,
                            runid=runids[ss],
                            signal=SignalNames.CORR,
                            wavelength=wavelength_labels[nm_idx],
                            time_idx=tt,
                            abs=abs,
                        )

                        session.merge(bcorr)

            for ss, sample in enumerate(self.baselines):
                for tt, time_point in enumerate(sample):
                    for nm_idx, abs in enumerate(time_point):
                        bcorr = BCorr(
                            exec_id=self._exec_id,
                            result_id=self._result_id,
                            runid=runids[ss],
                            signal=SignalNames.BLINE,
                            wavelength=wavelength_labels[nm_idx],
                            time_idx=tt,
                            abs=abs,
                        )

                        session.merge(bcorr)
            session.commit()

        logger.debug(("baseline correction table written."))


class BCorrDB:
    def __init__(self, engine: Engine):
        """wrapper around the bcorr results"""
        self._engine = engine

    def get_loader(self) -> BCorrLoader:
        return BCorrLoader(
            engine=self._engine,
        )
