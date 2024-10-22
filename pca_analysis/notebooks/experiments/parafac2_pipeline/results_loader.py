import duckdb as db
from sklearn.pipeline import Pipeline
import logging
from .estimators import BCorr_ASLS, BCorrLoader, PARAFAC2
from .parafac2results import Pfac2Loader
import polars as pl
from .pipeline import PipeSteps
import numpy as np

logger = logging.getLogger(__name__)


class ResultsLoader:
    def __init__(self, conn: db.DuckDBPyConnection, exec_id: str, runids: list[str]):
        """
        ETL from a sklearn Pipeline to the database connected via `conn`.

        exec_id: a unique execution id to identify consecutive pipeline runs in the database.
        """
        self._exec_id = exec_id
        self._conn = conn
        self._loadfuncs = {}
        self._runids = runids

    def load_results(
        self,
        pipeline: Pipeline,
        steps: list[str] | str = "all",
        overwrite: bool = False,
    ):
        logger.debug("loading results..")

        logger.debug("loading core tables..")

        cl = CoreTableLoader(
            exec_id=self._exec_id, runids=self._runids, conn=self._conn
        )

        cl.load_core_tables()

        if steps == "all":
            steps = [PipeSteps.BCORR, PipeSteps.PARAFAC2]

        if str(PipeSteps.BCORR) in steps:
            bcorr_est = pipeline.named_steps[PipeSteps.BCORR]
            self.load_baseline_corr(
                estimator=bcorr_est,
                conn=self._conn,
                exec_id=self._exec_id,
                overwrite=overwrite,
            )

        if str(PipeSteps.PARAFAC2) in steps:
            parafac2_est = pipeline.named_steps[PipeSteps.PARAFAC2]
            self.load_parafac2(
                estimator=parafac2_est,
                conn=self._conn,
                exec_id=self._exec_id,
                overwrite=overwrite,
            )

        logger.debug("results loading complete.")

    def load_baseline_corr(
        self,
        estimator: BCorr_ASLS,
        conn: db.DuckDBPyConnection,
        exec_id: str,
        result_id: str = "bcorr",
        overwrite: bool = False,
    ):
        """extract the results of baseline correction"""

        loader = BCorrLoader(
            exec_id=exec_id,
            result_id=result_id,
            baselines=estimator.bline_slices_,
            corrected=estimator.Xt,
            conn=conn,
        )

        loader.load_results()

        logger.debug("bcorr results ETL complete.")

    def load_parafac2(
        self,
        estimator: PARAFAC2,
        conn: db.DuckDBPyConnection,
        exec_id: str,
        overwrite: bool = False,
    ):
        decomp = estimator.decomp
        loader = Pfac2Loader(
            exec_id=exec_id,
            decomp=decomp,
            conn=conn,
        )
        loader.create_datamart()

        # need to avoid referring to the input data thus recreating 'create_datamart``


def write_to_db(
    signal_frame: pl.DataFrame,
    table_name: str,
    conn: db.DuckDBPyConnection,
    overwrite: bool = False,
) -> None:
    if overwrite:
        create_clause = "create or replace table"
    else:
        create_clause = "create table if not exists"
    query = f"""--sql
        {create_clause} {table_name} (
        exec_id varchar not null,
        sample int not null,
        signal varchar not null,
        wavelength int not null,
        idx int not null,
        abs float not null,
        primary key (exec_id, sample, signal, wavelength, idx)
        );

        insert or replace into {table_name}
            select
                exec_id,
                sample,
                signal,
                wavelength,
                idx,
                abs
            from
                signal_frame
        """

    conn.execute(query)


class CoreTableLoader:
    def __init__(
        self,
        exec_id: str,
        runids: list[str],
        conn: db.DuckDBPyConnection = db.connect(),
    ):
        """loads the core tables that other loaders depend on"""
        self._exec_id = exec_id
        self._conn = conn
        self._runids = runids

    def load_core_tables(self):
        self._create_exec_id_tbl()
        self.create_result_id_tbl()
        self._create_sample_table(runids=self._runids)

    def _create_exec_id_tbl(self):
        """creates a exec_id table containing the exec_ids"""

        query = """--sql
        create table if not exists exec_id (
        exec_id varchar primary key,
        );
        insert or replace into exec_id values (?)
        """

        self._conn.execute(query, parameters=[self._exec_id])

    def _create_sample_table(self, runids: list[str]):
        """write a table containing the unique sample ids
        :param runids: the labels of each sample in the dataset
        :type runids: list[str]
        """

        logger.debug("writing sample table..")
        sample_df = pl.DataFrame(
            {"sample": np.arange(0, len(runids), 1), "runid": runids}
        ).with_columns(pl.lit(self._exec_id).alias("exec_id"))
        sample_df.shape

        self._conn.execute("""--sql
            create table if not exists samples (
            exec_id varchar references exec_id(exec_id),
            sample int not null unique,
            runid varchar not null unique,
            primary key (exec_id, sample, runid)
            );
            """)

        self._conn.execute("""--sql
            insert into samples
            select
                exec_id,
                sample,
                runid
            from
                sample_df
            """)

    def create_result_id_tbl(self):
        """create an empty result_id table containing the identifier for each result type"""

        query = """--sql
        create table if not exists result_id (
        exec_id varchar references exec_id(exec_id),
        result_id varchar unique not null,
        primary key (exec_id, result_id)
        );
        """

        self._conn.execute(query)
