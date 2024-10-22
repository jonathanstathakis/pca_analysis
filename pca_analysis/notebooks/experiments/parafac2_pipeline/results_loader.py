import duckdb as db
from sklearn.pipeline import Pipeline
from typing import Any
import logging
from .estimators import BCorr_ASLS, to_dataframe, PARAFAC2
from .parafac2results import Parafac2Results

logger = logging.getLogger(__name__)
import polars as pl


class ResultsLoader:
    def __init__(self, conn: db.DuckDBPyConnection, exec_id: str):
        """
        ETL from a sklearn Pipeline to the database connected via `conn`.

        exec_id: a unique execution id to identify consecutive pipeline runs in the database.
        """
        self.exec_id = exec_id
        self._conn = conn
        self._extractors = {}

    def load_extractors(self, extractors: dict[str, Any]):
        """
        add specialised extractors whose key matches the name of the step in the pipeline. These functions will be used for ETL. Expect the extractor to take and estimator object and database connection.
        """
        if not isinstance(extractors, dict):
            raise TypeError("expecting dict")

        logger.debug("adding extractors..")
        self._extractors = {**self._extractors, **extractors}
        logger.debug("extractors added")

    def load_results(
        self,
        pipeline: Pipeline,
        steps: list[str] = ["value"],
        exec_id: str = "",
        overwrite: bool = False,
    ):
        """
        extract the results from each step in `steps` from `pipeline` in the order of appearance in `steps`.
        """

        logger.debug("loading results..")
        for step in steps:
            logger.debug(f"extracting {step}..")
            estimator = pipeline.named_steps[step]
            extractor = self._extractors[step]

            logger.debug("execiting extractor..")
            extractor(estimator, self._conn, exec_id, overwrite)
            logger.debug(f"{step} extractor complete.")

        logger.debug("results loading complete.")


def baseline_corr_extractor(
    estimator: BCorr_ASLS,
    conn: db.DuckDBPyConnection,
    exec_id: str,
    overwrite: bool = False,
):
    """extract the results of baseline correction"""

    logger.debug("getting bcorr results..")
    results = estimator.get_bcorr_results(conn=conn)

    logger.debug("creating dataframes..")
    corrected = to_dataframe(results.corrected, "corrected").with_columns(
        pl.lit(exec_id).alias("exec_id")
    )
    baselines = to_dataframe(results.baselines, "corrected").with_columns(
        pl.lit(exec_id).alias("exec_id")
    )

    logger.debug("writing to db..")
    write_to_db(
        signal_frame=corrected, table_name="corrected", conn=conn, overwrite=overwrite
    )
    write_to_db(
        signal_frame=baselines, table_name="baselines", conn=conn, overwrite=overwrite
    )

    logger.debug("bcorr results ETL complete.")


def parafac2_results_estractor(
    estimator: PARAFAC2,
    conn: db.DuckDBPyConnection,
    exec_id: str,
    overwrite: bool = False,
):
    decomp = estimator.decomp
    results = Parafac2Results(
        decomp=decomp,
        conn=conn,
    )

    # need to avoid referring to the input data thus recreating 'create_datamart``
    results._create_component_table()
    results._create_sample_table(
        runids=[str(x) for x in range(0, results._A.shape[0], 1)]
    )
    results._create_table_A()
    results._create_table_B()
    results._create_table_C()
    results._create_component_slices_table()
    results._create_recon_slices()


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
