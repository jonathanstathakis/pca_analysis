"""
While I would normally contain this within a notebook, it seems to causing th kernel to crash
"""

import logging
import polars as pl
import duckdb as db
from database_etl import etl_pipeline_raw

from pca_analysis.definitions import (
    CT_PW,
    CT_UN,
    DB_PATH_UV,
    DIRTY_ST,
    RAW_LIB_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(encoding="utf-8", level=logging.INFO)
con = db.connect(DB_PATH_UV)

pl.Config.set_tbl_rows(99)
overwrite = True

if overwrite:
    etl_pipeline_raw(
        data_dir=RAW_LIB_DIR,
        dirty_st_path=DIRTY_ST,
        ct_pw=CT_PW,
        ct_un=CT_UN,
        con=con,
        overwrite=True,
        run_extraction=False,
        excluded_samples=[
            {
                "runid": "2021-debortoli-cabernet-merlot_avantor",
                "reason": "aborted run",
            }
        ],
    )

con.close()
