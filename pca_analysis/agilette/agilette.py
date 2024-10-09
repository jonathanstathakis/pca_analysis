import duckdb as db
import polars as pl
import altair as alt
from database_etl import etl_pipeline_raw
from pathlib import Path
from .runset import RunSet
import logging

logger = logging.getLogger(__name__)


def create_database(
    data_dir: Path,
    dirty_st_path: Path,
    ct_un: str,
    ct_pw: str,
    excluded_samples: list[dict[str, str]],
    con: db.DuckDBPyConnection = db.connect(),
    run_extraction: bool = True,
    overwrite: bool = True,
):
    """
    given a directory of .D dirs, extract the data and output to a number of formats. Refer to
    """
    logger.info("running `etl_pipeline_raw`..")
    etl_pipeline_raw(
        data_dir=data_dir,
        dirty_st_path=dirty_st_path,
        ct_un=ct_un,
        ct_pw=ct_pw,
        excluded_samples=excluded_samples,
        con=con,
        run_extraction=run_extraction,
        overwrite=overwrite,
    )


class Agilette:
    def __init__(self, con: db.DuckDBPyConnection):
        """
        Connect to Database
        """
        self.con = con

    def get_run_set(
        self,
        labels: dict[str, str],
        wavelengths: list[int],
        mins: tuple[float, float],
    ) -> RunSet:
        return RunSet(con=self.con, labels=labels, wavelengths=wavelengths, mins=mins)
