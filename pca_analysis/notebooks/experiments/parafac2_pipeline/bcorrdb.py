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
from .orm import Base
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


class BCorr(Base):
    __tablename__ = "baseline_correction"
    exec_id: Mapped[str] = mapped_column(
        ForeignKey("exec_id.exec_id"), primary_key=True
    )
    result_id: Mapped[str] = mapped_column(ForeignKey("result_id.result_id"))
    runid: Mapped[str] = mapped_column(ForeignKey("samples.runid"))
    signal: Mapped[str] = mapped_column(nullable=False)
    wavelength: Mapped[int] = mapped_column(nullable=False)
    time_idx: Mapped[int] = mapped_column(nullable=False)
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

            session.add(result_name)
            session.commit()

        logger.debug("inserted result name into result_ids")

    def load_results(
        self,
        runids: pl.DataFrame,
        wavelength_labels: pl.DataFrame,
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

        corr_df = to_dataframe(self.corrected, signal_name=SignalNames.CORR)
        bline_df = to_dataframe(self.baselines, signal_name=SignalNames.BLINE)

        self.df = (
            pl.concat([corr_df, bline_df])
            .sort(
                BCRCols.SAMPLE_IDX,
                BCRCols.SIGNAL,
                BCRCols.WAVELENGTH_IDX,
                BCRCols.TIME_IDX,
            )
            .with_columns(
                pl.lit(self._exec_id).alias("exec_id"),
                pl.lit(self._result_id).alias("result_id"),
            )
        )

        # label the sample runids based on their order of apparance in the BCorr
        # Estimator
        # note that this hasnt been verified..

        runids_ = runids.rename({"runid_idx": "sample_idx"})

        self.df = self.df.join(
            runids_,
            on=[BCRCols.SAMPLE_IDX],
        )

        self.df = self.df.join(wavelength_labels, on=BCRCols.WAVELENGTH_IDX)

        self._insert_into_result_id()

        self.df.write_database(
            BCorr.__tablename__, self._engine, if_table_exists="append"
        )

        logger.debug(("baseline correction table written."))


class BCorrDB:
    def __init__(self, engine: Engine):
        """wrapper around the bcorr results"""
        self._engine = engine

    def get_loader(self) -> BCorrLoader:
        return BCorrLoader(
            engine=self._engine,
        )
