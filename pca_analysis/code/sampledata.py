import polars as pl

# from collections import UserDict
from IPython.display import display
import altair as alt
from typing import Self
from copy import copy
# build a class to store the data output

# class Img:
#     def __init__(self, img: pl.DataFrame):
#         """
#         class containing the image data and relevant stats:
#             - dimensions
#             - argmax
#             - argmin
#             - wavelength range
#             - time range
#             - peak count
#         """
#         self.df: pl.DataFrame = img
#         self.dims: tuple[int, int] = img.shape
#         self.argmax =


def move_id_path_to_mta(sample):
    """
    move id and path fields to the metadata table
    """

    img = sample[0]
    mta = sample[1]
    path = img["path"][0]
    img["id"][0]
    mta = mta.with_columns(pl.lit(path).alias("path"))
    img = img.drop(["path", "id"])
    # sample[0] = sample[0].

    return img, mta


class Run(dict):
    def __init__(self, run: tuple):
        """
        class wrapping a run, with an image and metadata, analogous to a pandas Series
        """

        sample_ = move_id_path_to_mta(sample=run)

        img = sample_[0]
        mta = sample_[1]
        self = {}
        self["img"] = (
            img.unpivot(
                index=["runid", "mins"],
                variable_name="wavelength",
                value_name="abs",
            )
            .select(
                "runid",
                pl.col("wavelength").cast(int),
                "mins",
                "abs",
            )
            .sort("runid", "wavelength", "mins")
        )

        self["mta"] = mta


class SampleData(dict):
    """
    use `long_img` as the central data catalog - figure out strategies for large data later..

    ergo ditch dict and go for two big tables. functions can return information from them..

    This design will allow for migration to db later..
    """

    def load_data(
        self, input_data: list[tuple[pl.DataFrame, pl.DataFrame]], name: str = "data"
    ) -> None:
        """
        populate the class instance with `data`
        """
        self.name = name

        # self.con = db.connect(f":memory:data{name}")

        if input_data:
            self.num_samples = len(input_data)

            # load the sample img, metadata into a dict, moving superf img cols to
            # metadata
            self.sample_order = []
            for sample in input_data:
                runid = sample[0]["runid"][0]
                self.sample_order.append(runid)
                self[runid] = Run(sample)

            print(self.keys())

    def isel(self, i):
        sd_ = copy(self)
        print(sd_.values())
        return list(sd_.values())[i]

    def get_run(self, runid: str) -> Self:
        self_ = copy(self)

        self_["img"] = self_["img"].filter(runid=runid)
        self_["mta"] = self_["mta"].filter(runid=runid)

        return self_

    # def load_db(self):
    #     """
    #     try loading the data into a db instance.

    #     Very possible, promising for use of keys etc, but I dont know how to update the db
    #     based on python operations..
    #     """

    #     self.con = db.connect()
    #     img_: pl.DataFrame = self["img"]

    #     for run in img_.partition_by("runid"):
    #         self.con.sql(
    #             """--sql
    #         create table if not exists img (
    #         runnum int not null,
    #         runid varchar not null,
    #         wavelength integer not null,
    #         mins float not null,
    #         abs float not null,
    #         primary key (runnum, runid, wavelength, mins)
    #         )
    #         """
    #         )
    #         self.con.sql(
    #             """--sql
    #         insert into img
    #             select
    #                 *
    #             from
    #                 run
    #         """
    #         )

    #     self.con.sql(
    #         """--sql
    #     summarize img
    #     """
    #     ).pl().pipe(display)

    def describe_images(
        self,
        statistics: list[str] = ["count", "null_count", "min", "max", "mean", "50%"],
    ) -> pl.DataFrame:
        """
        get image descriptions for included sample

        possible statistics: 'count','null_count','mean','std','min','25%','50%','75%','max'
        """

        # display(self["img"].group_by("runid").agg(statistics))

        stats = []
        for img in self["img"].partition_by(["runid", "wavelength"]):
            runnum = img["runnum"][0]
            wavelength = img["wavelength"][0]
            runid = img["runid"][0]

            img_stats = (
                img.drop("runnum", "runid", "wavelength")
                .describe()
                .filter(pl.col("statistic").is_in(statistics))
                .select(
                    pl.lit(runnum).alias("runnum"),
                    pl.lit(runid).alias("runid"),
                    pl.lit(wavelength).alias("wavelength"),
                    pl.all(),
                )
                # .select(pl.lit(sample).alias("runid"), pl.all())
            )
            stats.append(img_stats)

        return pl.concat(stats, how="vertical").sort(
            "runnum", "runid", "wavelength", "statistic"
        )

    def __repr__(self):
        return str(tuple(self.keys()))

    # def describe_samples(self):
    #     display(self.isel(1)["img"].head())
    #     display(self.isel(1)["mta"].head())

    # def sample_features(self, sample: str):

    def img_max_histograms(self):
        """
        return histogram of the image maxes over the wavelengths for each sample
        """
        img_stats = self.describe_images(statistics=["max"])
        # cols = img_stats.columns
        display(img_stats.head())
        maxes: pl.DataFrame = img_stats.drop("statistic", "mins").unpivot(
            index="runid", variable_name="wavelength", value_name="abs"
        )
        display(maxes.head())

        display(
            maxes.plot.bar(x="wavelength", y="abs").facet(column="runid", columns=3)
        )

    def img_chromatograms(self, wavelengths: int | list[int]):
        """
        plot each sample chromatogram at the given wavelength
        """
        if isinstance(wavelengths, list):
            wavelengths_ = wavelengths
        else:
            wavelengths_ = [wavelengths]

        display(
            self["img"]
            .filter(pl.col("wavelength").is_in(wavelengths_))
            .pipe(alt.Chart)
            .mark_line()
            .encode(
                x="mins:Q",
                y="abs:Q",
                color="wavelength:N",
            )
            .facet("runid:N", columns=4)
        )

    def peak_tables(self):
        """
        generate peak tables of each sample over the wavelengths
        """

        # the result will be a mapping of sample : wavelength ; peak idx : peak height

        display(self["img"])
