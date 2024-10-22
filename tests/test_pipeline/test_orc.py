import pytest
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    Orchestrator,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Parafac2Results,
)
import duckdb as db

from pca_analysis.notebooks.experiments.parafac2_pipeline.results_loader import (
    ResultsLoader,
)

from pca_analysis.notebooks.experiments.parafac2_pipeline.results_loader import (
    ResultsLoader,
    baseline_corr_extractor,
    parafac2_results_estractor,
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
    return orc_get_pipeline_set_params.run_pipeline(output_con=pipeline_output_con)


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
def results_loader(testcon, exec_id="test"):
    results_loader = ResultsLoader(conn=testcon, exec_id=exec_id)

    results_loader.load_extractors(
        {"bcorr": baseline_corr_extractor, "parafac2": parafac2_results_estractor}
    )
    return results_loader


def test_results_loader(
    orc_run_pipeline: Orchestrator,
    results_loader: ResultsLoader,
    exec_id="test",
    overwrite: bool = True,
):
    """test whether the result loader works.."""

    results_loader.load_results(
        pipeline=orc_run_pipeline._pipeline,
        steps=["bcorr", "parafac2"],
        exec_id=exec_id,
        overwrite=overwrite,
    )


def test_orc_run_pipeline(orc_run_pipeline):
    assert orc_run_pipeline
