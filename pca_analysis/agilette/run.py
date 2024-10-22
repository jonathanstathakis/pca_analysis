import plotly.express as px
import duckdb as db
import altair as alt
import polars as pl
from pathlib import Path


class Run:
    def __init__(self, con: db.DuckDBPyConnection, runid: str):
        """
        A representation of a single experimental run.
        """

        self.con = db.cursor(connection=con)
        self.runid = runid
        self.data_path = get_data_path(con=self.con, runid=self.runid)
        self.img_path = self.data_path / "data.parquet"

    def get_img(self, wavelengths: list[int] = [256]):
        """
        TODO: add select columns, select rows to options to speed up read times.
        """
        ["mins"] + wavelengths
        return get_img(
            path=str(self.img_path),
            runid=self.runid,
            # read_parquet_kwargs=dict(columns=columns),
        ).drop("id")

    def plot_chromatogram(self, wavelength: int = 256) -> alt.Chart:
        img = self.get_img()
        img_ = img.select(pl.col("runid"), pl.col("mins"), pl.col("256"))
        return px.line(img_, x="mins", y=str(wavelength))
        # return img_.plot.line(x="mins", y=str(wavelength))

    def plot_line_3d(self):
        img = self.get_img()
        img_ = img.select(pl.exclude("runid")).unpivot(
            index="mins", variable_name="nm", value_name="abs"
        )

        return px.line_3d(img_, x="mins", y="nm", z="abs")

    # def plot_chromatogram(self)->alt.Chart:


def get_img(runid: str, path: str, read_parquet_kwargs: dict = {}) -> pl.DataFrame:
    return pl.read_parquet(path, **read_parquet_kwargs).with_columns(
        pl.lit(runid).alias("runid")
    )


def get_id(con: db.DuckDBPyConnection, runid: str) -> str:
    """
    return the id of a given run
    """
    result = con.execute(
        "select id from chm where runid = ?", parameters=[runid]
    ).fetchone()

    if not result:
        raise ValueError("no result")
    elif not isinstance(result[0], str):
        raise TypeError("expected str")
    else:
        return result[0]


def get_data_path(con: db.DuckDBPyConnection, runid: str) -> Path:
    result = con.execute(
        "select path from chm where runid = ?", parameters=[runid]
    ).fetchone()

    if not result:
        raise ValueError("no path found")
    elif not isinstance(result[0], str):
        raise TypeError("expected str")
    else:
        return Path(result[0])
