import duckdb as db
from pathlib import Path


class DashboardExtractor:
    def __init__(self, db_path: str | Path):
        """provide data for the dashboard app from the results database"""
        ...

        self._conn = db.connect(db_path, read_only=True)

    def get_parafac2_results(self):
        """return the parafac2 results object"""
        ...

    def get_bcorr_results(self):
        """return the bcorr results object"""
