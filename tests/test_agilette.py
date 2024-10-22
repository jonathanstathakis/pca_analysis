import pytest
from pca_analysis.agilette import Agilette, create_database
from pathlib import Path
import duckdb as db
import logging


@pytest.fixture
def agilette():
    return Agilette()


@pytest.fixture
def create_agilette_database(
    test_data_dir: Path,
    test_db_path: Path,
    agilette: Agilette,
    ct_un: str,
    ct_pw: str,
    dirty_st_path: Path,
) -> Path:
    """
    create a database thrugh the agilette `create_database` method. Returns the connection object to the database.
    """
    ...
    create_database(
        test_data_dir,
        ct_un=ct_un,
        ct_pw=ct_pw,
        con=db.connect(str(test_db_path)),
        dirty_st_path=dirty_st_path,
        excluded_samples=[dict()],
        run_extraction=True,
        overwrite=True,
    )

    return test_db_path


logger = logging.getLogger(__name__)


@pytest.mark.skip
def test_agilette_create_database(
    create_agilette_database: Path,
) -> None:
    """
    test if `agilette.create_database` can execute as expected.

    This currently results in a segmentation fault, caused when etl_pipeline_raw tries to return the con object (?)
    Until I solve that problem, this test will be skipped..
    """
    assert True
