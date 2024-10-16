import pytest
import duckdb as db
from pca_analysis.definitions import DB_PATH_UV
from pca_analysis.code.get_sample_data import get_ids_by_varietal
from database_etl.etl.etl_pipeline_raw import get_data
import polars as pl
from pca_analysis.notebooks.experiments.parafac2_pipeline.data import Data, XX
from pca_analysis.notebooks.experiments.parafac2_pipeline.input_data import InputData
from pca_analysis.notebooks.experiments.parafac2_pipeline.estimators import PARAFAC2
from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Parafac2Results,
)
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    DCols,
    Orchestrator,
)
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def datacon():
    return db.connect(DB_PATH_UV, read_only=True)


@pytest.fixture(scope="module")
def test_sample_ids(datacon, varietal="shiraz"):
    """get test sample ids"""
    return get_ids_by_varietal(varietal=varietal, con=datacon)


@pytest.fixture(scope="module")
def input_data(
    datacon,
    test_sample_ids,
) -> InputData:
    """the test sample data collection object"""
    return InputData(con=datacon, ids=test_sample_ids)


@pytest.fixture(scope="module")
def testdata(input_data: InputData) -> Data:
    """a Data object wrapping the test data"""
    return Data(
        time_col=DCols.TIME,
        runid_col=DCols.RUNID,
        nm_col=DCols.NM,
        abs_col=DCols.ABS,
        scalar_cols=[DCols.PATH, DCols.ID],
    ).load_data(input_data)


def test_load_data(testdata: Data):
    assert testdata


@pytest.fixture(scope="module")
def XXX(testdata) -> XX:
    """the input data for the pipeline"""
    return testdata.to_X()


@pytest.fixture(scope="module")
def parafac2_estimator(XXX, rank=9):
    """the fit_transformed parafac2 estimator object"""
    pfac2 = PARAFAC2(rank=9, n_iter_max=100, nn_modes="all", linesearch=False)
    pfac2.fit_transform(XXX)
    return pfac2


@pytest.fixture(scope="module")
def decomp(parafac2_estimator):
    """the decomp output of the PARAFAC2 estimator"""
    return parafac2_estimator.decomp


def test_parafac2_exc(decomp):
    """test if the decomp object can be created"""
    assert decomp


@pytest.fixture(scope="module")
def resultscon():
    """the conn to the results db"""
    return db.connect()


@pytest.fixture(scope="module")
def pfac2results(resultscon, decomp, XXX: XXX):
    """the parafac2 results object"""
    return Parafac2Results(con=resultscon, decomp=decomp).create_datamart(
        input_imgs=XXX
    )


def test_pfac2results(pfac2results):
    assert pfac2results


def test_viz_recon_3d(pfac2results):
    assert pfac2results.viz_recon_3d()


def test_show_tables(pfac2results):
    assert not pfac2results._show_tables().is_empty()


def test_viz_overlay_components(pfac2results):
    assert pfac2results._viz_overlay_components_sample_wavelength(0, 0)


def test_viz_input_img_curve(pfac2results):
    assert pfac2results._viz_input_img_curve_sample_wavelength(0, 0)


def test_viz_overlay_curve_components(pfac2results):
    assert pfac2results.viz_overlay_curve_components(0, 0)


def test_viz_recon_input_overlay(pfac2results):
    assert pfac2results.viz_recon_input_overlay(0, 0)


def test_viz_recon_input_overlay_facet(pfac2results):
    assert pfac2results.viz_recon_input_overlay_facet(0, 0, "sample")


def test_computation_matches_tly(pfac2results):
    """assert that any variation between my implementation and tensorlys is insiginficant. Variation arises from calculations performed by different libraries."""
    assert pfac2results._check_computations_match_tly()


@pytest.fixture(scope="module")
def testdata_filter_expr():
    return pl.col("mins").is_between(0.7, 1.39) & pl.col("nm").is_between(240, 270)


@pytest.fixture(scope="module")
def orc(
    test_sample_ids: list[str],
    testcon: db.DuckDBPyConnection,
    testdata_filter_expr: pl.Expr,
):
    orc = Orchestrator()
    orc.load_data(con=testcon, runids=test_sample_ids, filter_expr=testdata_filter_expr)

    return orc


def test_orc_load_data(orc):
    assert orc


@pytest.fixture(scope="module")
def orc_run_pipeline(orc):
    return orc.run_pipeline(rank=9)


def test_orc_run_pipeline(orc_run_pipeline):
    assert orc_run_pipeline
