import duckdb as db
from pathlib import Path
from sqlalchemy import create_engine, select
import polars as pl
from .parafac2db import A, Bs, C, SampleRecons, ComponentSlices

class DashboardExtractor:
    def __init__(self, db_path: str | Path):
        """provide data for the dashboard app from the results database"""
        ...

        self._db_path = db_path
        self._url = f"duckdb:///{self._db_path}"
        self._engine = create_engine(self._url)

    def get_parafac2_results(self, limit=-1):
        """return the parafac2 results object
        simply want to provide a means of accessing the data as dataframes..
        """
        
        # recon =

        # pl.read_database("select * from recon")

    def get_bcorr_results(self):
        """return the bcorr results object"""
