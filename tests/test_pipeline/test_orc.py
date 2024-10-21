import pytest
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    Orchestrator,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Parafac2Results,
)


def test_orc_load_data(orc: Orchestrator):
    assert orc


@pytest.fixture()
def param_grid():
    return dict(parafac2__rank=9)


@pytest.fixture()
def orc_get_pipeline_set_params(orc: Orchestrator, param_grid) -> Orchestrator:
    return orc.get_pipeline().set_params(param_grid)


@pytest.fixture()
def orc_run_pipeline(orc_get_pipeline_set_params: Orchestrator):
    """
    execute the pipeline in full.
    """
    return orc_get_pipeline_set_params.run_pipeline()


@pytest.fixture()
def orc_results(orc_run_pipeline: Orchestrator):
    """results of the orchestrator run pipeline"""

    return orc_run_pipeline.results


@pytest.fixture()
def test_display_orc_results(orc_results: Parafac2Results):
    orc_results.results_dashboard()


def test_orc_run_pipeline(orc_run_pipeline):
    assert orc_run_pipeline
