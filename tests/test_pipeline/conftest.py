import pytest
import duckdb as db
from pca_analysis.definitions import DB_PATH_UV
from pca_analysis.code.get_sample_data import get_ids_by_varietal
from database_etl.etl.etl_pipeline_raw import get_data
import polars as pl
from pca_analysis.notebooks.experiments.parafac2_pipeline.data import Data, XX
from pca_analysis.notebooks.experiments.parafac2_pipeline.input_data import InputData
from pca_analysis.notebooks.experiments.parafac2_pipeline.orchestrator import (
    DCols,
    Orchestrator,
)
from pathlib import Path
import pickle
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
    rank=9,
):
    """the fit_transformed parafac2 estimator object"""

    pfac2_ft_key = "pfac2_ft"
    # see <https://docs.pytest.org/en/stable/how-to/cache.html#the-new-config-cache-object>

    # use cache merely to store state, pickle_cache contains the paths to the actual cached pickles.

    pfac2 = PARAFAC2(rank=rank, n_iter_max=100, nn_modes="all", linesearch=False)
    pfac2.fit_transform(XXX)

    result = pickle_cache.fetch_from_cache(pfac2_ft_key)

    if result is None:
        pickle_cache.write_to_cache(pfac2_ft_key, pfac2)

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
):
    orc = Orchestrator()
    orc.load_data(con=testcon, runids=test_sample_ids, filter_expr=testdata_filter_expr)

    return orc


@pytest.fixture(scope="module")
def resultscon():
    """the conn to the results db"""
    return db.connect()


@pytest.fixture(scope="module")
def pfac2results(resultscon, decomp, XXX: XX):
    """the parafac2 results object"""
    return Parafac2Results(con=resultscon, decomp=decomp).create_datamart(
        input_imgs=XXX
    )