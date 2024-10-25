import duckdb as db
import pytest

from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    Orchestrator,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.results_db import ResultsDB


def test_orc_load_data(orc_loaded: Orchestrator):
    assert orc_loaded


def test_orc_run_pipeline(orc_run_pipeline):
    assert orc_run_pipeline


def test_load_results_tables_content(results_db_loaded: ResultsDB):
    # test that the expected tables are in the database

    assert results_db_loaded


def test_get_database_report(results_db_loaded: ResultsDB):
    """generate database report, ensure count is >1 for all tables"""
    report = results_db_loaded._get_database_report()
    assert not report.is_empty()
    import polars as pl

    # all tables should have >0 rows.
    assert report.filter(pl.col("count").eq(0)).is_empty()


def test_loading_duplicate_results(
    orc_run_pipeline: Orchestrator, results_db_path: str
):
    """load the pipeline results twice to see what happens if duplicate values entered"""

    orc_run_pipeline.load_results(output_db_path=":memory:")

    try:
        orc_run_pipeline.load_results(output_db_path=results_db_path)
    except db.ConstraintException:
        pass
