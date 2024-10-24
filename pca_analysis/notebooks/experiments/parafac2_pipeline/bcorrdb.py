from pca_analysis.notebooks.experiments.parafac2_pipeline.estimators import (
    BCRCols,
    SignalNames,
    logger,
    to_dataframe,
)

from .core_tables import CoreTbls
from enum import StrEnum
import duckdb as db
import polars as pl
from numpy.typing import NDArray


class BcorrTbls(StrEnum):
    BCORR = "baseline_corrected"


class BcorrIdx(StrEnum):
    BCORR_IDX = "baseline_corrected_idx"


class BCorrLoader:
    def __init__(
        self,
        exec_id: str,
        corrected: list[NDArray],
        baselines: list[NDArray],
        conn=db.connect(),
        result_name: str = "bcorr",
    ):
        self._conn = conn
        self.corrected = corrected
        self.baselines = baselines
        self._exec_id = exec_id
        self._result_id = result_name

        corr_df = to_dataframe(self.corrected, signal_name=SignalNames.CORR)

        bline_df = to_dataframe(self.baselines, signal_name=SignalNames.BLINE)

        self.df = (
            pl.concat([corr_df, bline_df])
            .sort(BCRCols.SAMPLE, BCRCols.SIGNAL, BCRCols.WAVELENGTH, BCRCols.IDX)
            .with_columns(
                pl.lit(self._exec_id).alias("exec_id"),
                pl.lit(self._result_id).alias("result_id"),
            )
        )

    def _insert_into_result_id(self):
        """add the result_id into the result_id table"""

        query = """--sql
        insert into result_id
            values (?, ?)
        """
        logger.debug(f"inserting {self._result_id} into result_id table..")
        self._conn.execute(query, parameters=[self._exec_id, self._result_id])

    def load_results(self):
        """write the baselines and corrected to a database"""

        logger.debug("loading bcorr results..")
        self._insert_into_result_id()

        create_query = f"""--sql
        create table if not exists {BcorrTbls.BCORR} (
        exec_id varchar not null references {CoreTbls.EXEC_ID}(exec_id),
        result_id varchar not null references {CoreTbls.RESULT_ID}(result_id),
        sample int not null references {CoreTbls.SAMPLES}(sample),
        signal varchar not null,
        wavelength int not null,
        idx int not null,
        abs float not null,
        primary key (exec_id, result_id, sample, signal, wavelength, idx)
        );

        create unique index if not exists {BcorrIdx.BCORR_IDX} on baseline_corrected(exec_id, result_id, sample, signal, wavelength, idx);
        """

        df = self.df
        load_query = f"""--sql
        insert into {BcorrTbls.BCORR}
            select
                exec_id,
                result_id,
                sample,
                signal,
                wavelength,
                idx,
                abs
            from
                df
            on conflict (exec_id, result_id, sample, signal, wavelength, idx) do update set abs = EXCLUDED.abs
                ;
        """
        self._conn.execute(create_query)
        self._conn.execute(load_query)


class BCorrDB:
    def __init__(self, output_conn):
        """wrapper around the bcorr results"""
        self._conn = output_conn

    def get_loader(
        self, exec_id, corrected, baselines, result_name="bcorr"
    ) -> BCorrLoader:
        return BCorrLoader(
            exec_id=exec_id,
            corrected=corrected,
            baselines=baselines,
            conn=self._conn,
            result_name=result_name,
        )

    def clear_tables(self):
        for tbl in BcorrIdx:
            self._conn.execute(f"truncate {tbl}")
