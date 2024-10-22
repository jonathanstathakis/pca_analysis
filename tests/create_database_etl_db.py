from database_etl import etl_pipeline_raw
import duckdb as db
from pca_analysis.definitions import (
    CT_PW,
    CT_UN,
    DIRTY_ST,
)
from tests.test_definitions import TEST_DATA_DIR, TEST_DB_PATH

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level="DEBUG")


def create_database():
    logger.info("establishing database..")
    conn = db.connect(TEST_DB_PATH)
    etl_pipeline_raw(
        data_dir=TEST_DATA_DIR,
        dirty_st_path=DIRTY_ST,
        ct_pw=CT_PW,
        ct_un=CT_UN,
        con=conn,
        overwrite=True,
        run_extraction=False,
        excluded_samples=[
            {
                "runid": "2021-debortoli-cabernet-merlot_avantor",
                "reason": "aborted run",
            }
        ],
    )

    logger.info("database creation complete.")


if __name__ == "__main__":
    create_database()
