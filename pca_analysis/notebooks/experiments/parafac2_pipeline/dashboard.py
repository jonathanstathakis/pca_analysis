from .dashboard_extractor import DashboardExtractor


class Dashboard:
    def __init__(self, db_path: str):
        """a dashboard providing insights into the parafac2 decomp pipeline results"""
        self._extractor = DashboardExtractor(db_path=db_path)
