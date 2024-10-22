import pytest
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    Orchestrator,
)
import duckdb as db

from pca_analysis.notebooks.experiments.parafac2_pipeline.results_loader import (
    ResultsLoader,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Parafac2Tables,
)


def test_orc_load_data(orc: Orchestrator):
    assert orc


@pytest.fixture()
def param_grid():
    return dict(parafac2__rank=9)


@pytest.fixture()
def orc_get_pipeline_set_params(orc: Orchestrator, param_grid) -> Orchestrator:
    return orc.get_pipeline().set_params(param_grid)


@pytest.fixture(scope="module")
def pipeline_output_con():
    from pathlib import Path

    return db.connect(Path(__file__).parent / "pipeline_output_con.db")


@pytest.fixture()
def orc_run_pipeline(
    orc_get_pipeline_set_params: Orchestrator,
    pipeline_output_con: db.DuckDBPyConnection,
):
    """
    execute the pipeline in full.
    """
    return orc_get_pipeline_set_params.run_pipeline()


@pytest.mark.skip
def test_display_orc_results(orc_run_pipeline: Orchestrator):
    app = orc_run_pipeline.results.results_dashboard()
    import webbrowser

    # host = "127.0.0.1"
    port = "8050"
    from threading import Timer

    def open_browser():
        webbrowser.open_new(url="http://127.0.0.1:8050")

    Timer(2, open_browser).start()
    app.run(port=port, debug=False)


@pytest.fixture(scope="module")
def results_loader(test_sample_ids: list[str], exec_id="test"):
    results_loader = ResultsLoader(
        conn=db.connect(), exec_id=exec_id, runids=test_sample_ids
    )

    return results_loader


def test_results_loader(
    orc_run_pipeline: Orchestrator,
    results_loader: ResultsLoader,
    overwrite: bool = True,
):
    """test whether the result loader works.."""

    results_loader.load_results(
        pipeline=orc_run_pipeline._pipeline,
        steps=["bcorr", "parafac2"],
        overwrite=overwrite,
    )


def test_orc_run_pipeline(orc_run_pipeline):
    assert orc_run_pipeline


@pytest.fixture(scope="module")
def results_conn():
    return db.connect()


def test_load_results(
    orc_run_pipeline: Orchestrator,
    results_conn: db.DuckDBPyConnection,
):
    loader = orc_run_pipeline.load_results(
        output_con=results_conn, exec_id="test_load_results"
    )

    assert loader

    # test that the expected tables are in the database

    table_names = results_conn.execute("select name from (show)").fetchnumpy()["name"]

    for table in list(Parafac2Tables):
        assert table in table_names
