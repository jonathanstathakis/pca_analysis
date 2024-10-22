import duckdb as db


class RunSet:
    def __init__(
        self,
        con: db.DuckDBPyConnection,
        labels: dict[str, str],
        wavelengths: list[int],
        mins=tuple[float, float],
    ):
        """
        A collection of Run objects.
        """
        ...
