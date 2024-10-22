import pytest
import duckdb as db
from tests.test_definitions import TEST_DB_PATH
import polars as pl
from pca_analysis.notebooks.experiments.parafac2_pipeline.data import Data, XX
from pca_analysis.notebooks.experiments.parafac2_pipeline.input_data import InputData
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    DCols,
    Orchestrator,
)
from pathlib import Path
from pca_analysis.notebooks.experiments.parafac2_pipeline.estimators import PARAFAC2

from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Parafac2Results,
)

import logging

from tensorly.parafac2_tensor import Parafac2Tensor
from ..pickle_cache import PickleCache
from . import CACHE_PATH

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def datacon():
    return db.connect(TEST_DB_PATH, read_only=True)


@pytest.fixture(scope="module")
def test_sample_ids(datacon):
    """get test sample ids"""
    ids = [x[0] for x in datacon.execute("select runid from inc_chm").fetchall()]

    return ids


@pytest.fixture(scope="module")
def input_data(
    datacon,
    test_sample_ids,
) -> InputData:
    """the test sample data collection object"""
    return InputData(conn=datacon, ids=test_sample_ids)


@pytest.fixture(scope="module")
def testdata(input_data: InputData) -> Data:
    """a Data object wrapping the test data"""
    testdata = (
        Data(
            time_col=DCols.TIME,
            runid_col=DCols.RUNID,
            nm_col=DCols.NM,
            abs_col=DCols.ABS,
            scalar_cols=[DCols.PATH, DCols.ID],
        )
        .load_data(input_data)
        .filter_nm_tbl(
            pl.col(DCols.TIME).is_between(0, 0.5)
            & pl.col(DCols.NM).is_between(250, 256)
        )
    )
    return testdata


@pytest.fixture(scope="module")
def XXX(testdata) -> XX:
    """the input data for the pipeline"""
    return testdata.to_X()


@pytest.fixture(scope="module")
def pickle_cache():
    """a collection of object name, cache paths for objects requiring pickling. To be used in conjunction with pytest caching"""

    pickle_cache = PickleCache(
        cache_parent=str(Path(CACHE_PATH).parent), cache_name=str(Path(CACHE_PATH.name))
    )
    return pickle_cache


@pytest.fixture(scope="module")
def parafac2_ft(
    XXX,
    pickle_cache: PickleCache,
    rank=2,
):
    """the fit_transformed parafac2 estimator object"""

    pfac2_ft_key = "pfac2_ft"
    # see <https://docs.pytest.org/en/stable/how-to/cache.html#the-new-config-cache-object>

    # use cache merely to store state, pickle_cache contains the paths to the actual cached pickles.
    logger.debug("attempting to fetch pfac2 transformer from cache..")
    result = pickle_cache.fetch_from_cache(pfac2_ft_key)

    if result is None:
        logger.debug("no cached transformer found.")
        logger.debug("initializing PARAFAC2 transformer..")
        pfac2 = PARAFAC2(rank=rank, n_iter_max=100, nn_modes="all", linesearch=False)
        logger.debug("executing fit_transform..")
        pfac2.fit_transform(XXX)
        logger.debug("transformation complete.")

        logger.debug("writing pfac2 transformer to cache..")
        pickle_cache.write_to_cache(pfac2_ft_key, pfac2)
    else:
        logger.debug("using cached pfac2 transformer..")
        pfac2 = result

    return pfac2


@pytest.fixture(scope="module")
def decomp(parafac2_ft: PARAFAC2) -> Parafac2Tensor:
    """the decomp output of the PARAFAC2 estimator"""

    return parafac2_ft.decomp


@pytest.fixture(scope="module")
def testdata_filter_expr():
    return pl.col("mins").is_between(0.7, 1.39) & pl.col("nm").is_between(240, 270)


@pytest.fixture(scope="module")
def orc(
    test_sample_ids: list[str],
    testcon: db.DuckDBPyConnection,
    testdata_filter_expr: pl.Expr,
    exec_id: str = "test",
):
    orc = Orchestrator(exec_id=exec_id)
    orc.load_data(con=testcon, runids=test_sample_ids, filter_expr=testdata_filter_expr)

    return orc


@pytest.fixture(scope="module")
def resultscon():
    """the conn to the results db"""
    return db.connect()


@pytest.fixture(scope="module")
def pfac2results(resultscon, decomp, XXX: XX):
    """the parafac2 results object"""
    return Parafac2Results(conn=resultscon, decomp=decomp).create_datamart(
        input_imgs=XXX
    )
