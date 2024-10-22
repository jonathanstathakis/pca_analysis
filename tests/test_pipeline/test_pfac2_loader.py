from pca_analysis.notebooks.experiments.parafac2_pipeline.parafac2results import (
    Pfac2Loader,
)
from tensorly.parafac2_tensor import Parafac2Tensor
import pytest
import duckdb as db
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def pfac2loader_conn():
    return db.connect()


@pytest.fixture(scope="module")
def runids(decomp: Parafac2Tensor):
    return [str(x) for x in range(0, decomp[1][0].shape[0], 1)]


@pytest.fixture(scope="module")
def pfac2loader(
    pfac2loader_conn: db.DuckDBPyConnection,
    decomp: Parafac2Tensor,
    exec_id: str = "pfac2loader_test",
):
    ...

    assert pfac2loader_conn
    assert decomp

    loader = Pfac2Loader(exec_id=exec_id, decomp=decomp, conn=pfac2loader_conn)

    return loader


@pytest.fixture(scope="module")
def exc_loader(pfac2loader: Pfac2Loader, runids: list[str]):
    return pfac2loader.create_datamart(runids=runids)


def test_pfac2loader(exc_loader: Pfac2Loader):
    assert exc_loader
