from . import test_definitions as test_defs
from pca_analysis import definitions as defs
from pathlib import Path
import pytest
import duckdb as db


@pytest.fixture(scope="module")
def test_data_dir():
    return Path(test_defs.TEST_DATA_DIR)


@pytest.fixture(scope="module")
def test_db_path() -> Path:
    return Path(test_defs.TEST_OUTPUT_PATH / "test_db.db")


@pytest.fixture(scope="module")
def ct_un():
    return defs.CT_UN


@pytest.fixture(scope="module")
def ct_pw():
    return defs.CT_PW


@pytest.fixture(scope="module")
def dirty_st_path() -> Path:
    return defs.DIRTY_ST


@pytest.fixture(scope="module")
def testcon(test_db_path: Path):
    return db.connect(test_db_path)
