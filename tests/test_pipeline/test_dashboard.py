import pytest
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    Orchestrator,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.results_db import ResultsDB
import polars as pl
import duckdb as db
from pathlib import Path
import logging
from sqlalchemy import create_engine
from sqlalchemy import Engine

logger = logging.getLogger(__name__)


@pytest.fixture
def pipeline_results(
    testdata_filter_expr: pl.Expr,
    pers_results_db_path: str,
    database_etl_db_engine: Engine,
    test_sample_ids: list[str],
    results_db_path: str,
    exec_id="test_dashboard",
):
    """
    establish logic that will run the pipeline below if the db doesnt exist..
    """

    if results_db_path:
        logger.debug("result db found, connecting..")
        return ResultsDB(engine=database_etl_db_engine)
    else:
        orc = Orchestrator(exec_id=exec_id)

        results_db = (
            orc.load_data(
                filter_expr=testdata_filter_expr,
                input_db_path=pers_results_db_path,
                runids=test_sample_ids,
            )
            .get_pipeline()
            .set_params(dict(parafac2__rank=9))
            .run_pipeline()
            .load_results(output_db_path=results_db_path)
        )

    return results_db


@pytest.mark.xfail
def test_pipeline_results(pipeline_results):
    assert pipeline_results
