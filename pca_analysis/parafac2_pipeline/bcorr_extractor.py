from pathlib import Path
import duckdb as db


class BCorrExtractor:
    def __init__(self, db_path: str | Path):
        """extract the bcorr data from the db"""

        self._conn = db.connect(db_path, read_only=True)

    def extract_tables(self):
        """
        got
        """
