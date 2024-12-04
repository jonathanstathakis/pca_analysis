import polars as pl
from typing import Self
from .utility import plot_imgs
from .input_data import (
    InputDataGetter,
)
import plotly.graph_objects as go
from collections import UserList
from .pipeline_defs import DCols
from copy import deepcopy


class XX(UserList):
    def __init__(self, nm_tbl: pl.DataFrame):
        X_dict = nm_tbl.drop(DCols.TIME).partition_by(DCols.RUNID, as_dict=True)

        # this assumes that all samples have the same wavelengths. I believe this is
        # precluded by the formation of nm_tbl in Data..

        self.wavelength_labels = pl.DataFrame(
            {"wavelength": [int(x) for x in nm_tbl.drop("runid", "mins").columns]}
        ).with_row_index("wavelength_idx")

        # used for reconstruction
        self.X_schemas = {
            runid[0]: df.drop(DCols.RUNID).schema for runid, df in X_dict.items()
        }

        self.runids = pl.DataFrame(
            {DCols.RUNID: [str(x) for x in self.X_schemas.keys()]}
        ).with_row_index("runid_idx")

        # time labels by sample
        self.time_labels = nm_tbl.select("runid", "mins").with_columns(
            pl.col("mins").rank("dense").over("runid").alias("time_label_idx")
        )

        self.data = [
            X.drop(DCols.RUNID).to_numpy(writable=True) for X in X_dict.values()
        ]

    def as_list_numpy_arrays(self):
        """return the data as a list of numpy arrays"""

        return self.data

    def as_3_mode_tensor(self):
        """return the data as a 3 way tensor with shape samples x elution x spectrum"""
        import numpy as np

        tensor = np.stack(self.data)

        return tensor

    def as_unfolded_samplewise(self):
        """returns X as a 2 mode tensor unfolded samplewise"""
        tensor = self.as_3_mode_tensor()

        num_samples = tensor.shape[0]
        num_timepoints = tensor.shape[1]
        unfolded_row_count = num_samples * num_timepoints

        unfolded = tensor.reshape(unfolded_row_count, 106)

        return unfolded

    def to_mat(self, path) -> None:
        """output the tensor as a matlab dataset at `path`"""

        from pathlib import Path

        path_ = Path(path)
        from scipy.io import savemat

        savemat(path, {"X": self.data})

        print(f"X saved to {path_.resolve()}")


class Data:
    def __init__(
        self,
        time_col: str,
        runid_col: str,
        nm_col: str,
        abs_col: str,
        scalar_cols: list[str],
    ):
        """
        A wrapper for the input test data, providing a method of subsetting,
        visualisation and transformation into a list of numpy arrays suitable for input
        into the PARAFAC2 function.

        the initialisation will decompose the imgs table into a number of tables - a
        scalar table of sample-spcific values, a time label table, and a wavelength
        table. Each is labelled with at least the `runid` acting as the primary key. The
        wavelength table will be the core table, used for visualisation and later
        transformation into the PARAFAC2 input X.

        imgs: pl.DataFrame
            A long, augmented table with columns: `runid`, `mins`, `nm` and `abs`.
            TestData expects that individual samples are stacked vertically with a
            unique `runid` labeling each sample and  `mins`, `nm` cols labeling each
            row. As such, the combination of `runid`, `nm`, and `mins` forms a primary
            key.
        scalar_cols: list[str]
            any label columns in the `imgs` DataFrame that are unique labels such as
            samplewise runids. These will be organised into a normalised table.
        nm_col: str
            the wavelength label column.
        abs_col: str
            the absorbance value column.
        time_col: str
            the time mode label column, expecting `mins`. This will be used to form a
            long time label column.
        runid_col: str
            The sample id column key.
        """
        self._scalar_cols: list[str] = scalar_cols
        self._time_col: str = time_col
        self._runid_col: str = runid_col
        self._nm_col: str = nm_col
        self._abs_col: str = abs_col
        self._idx_col: str = "idx"
        self.mta = pl.DataFrame()
        self._loaded: bool = False

        self._time_tbl: pl.DataFrame
        self._scalar_tbl: pl.DataFrame
        self._nm_tbl: pl.DataFrame
        # get some stats about the imgs table for process validation

        # mins range

        # wavelength range

    def load_data(
        self,
        raw_data_extractor: InputDataGetter,
        drop_samples: list[str] = ["82", "0151"],
    ) -> Self:
        """
        load the imgs and metadata tables from the data object

        performs preprocessing prior to populating the internals

        :param data: the list-like data object containing the sample data
        :type data: InputData
        :raises TypeError: if data is not an InputData object
        :return: a loaded copy of this object
        :rtype: Self
        """
        if not isinstance(raw_data_extractor, InputDataGetter):
            raise TypeError(
                f"Expecting an InputData object, got {type(raw_data_extractor)}"
            )

        self_ = deepcopy(self)
        raw_data_extractor.get_data_as_list_of_tuples()
        imgs, mta = raw_data_extractor.as_long_tables()
        imgs = imgs.pipe(_preprocess_imgs, drop_samples)

        self_._load_imgs(imgs)
        self_.mta = mta

        self_._loaded = True
        return self_

    def __repr__(self):
        repr_str = ""
        if hasattr(self, "_loaded"):
            repr_str = f"""
            Data
            ----

            Metadata
            --------
            num. samples: {self.mta.shape[0]}
            num. metadata fields: {self.mta.shape[1]}
            metadata fields:
            {self.mta.columns}

            NM TBL
            ------

            time_points: {self._nm_tbl.shape[0]}
            wavelengths: {self._nm_tbl.shape[1]}

            columns: {self._nm_tbl.columns}

            ranges:
                idx: {self._nm_tbl[DCols.IDX].min(), self._nm_tbl[DCols.IDX].max()}
                wavelength: {self._nm_tbl[DCols.NM].min(), self._nm_tbl[DCols.NM].max()}
            """

        else:
            repr_str = "no data loaded, nothing to display."

        return repr_str

    def _load_imgs(self, imgs: pl.DataFrame) -> None:
        """
        load the preprocessed sample images.

        validates the input, adds a time-based index column and sorts prior to
        normalising. Once complete sets the flag `_img_loaded` to True in order to track
        state.

        :param imgs: a long sample-wise unfolded table of sample images.
        :type imgs: pl.DataFrame
        """
        imgs.pipe(self._validate_input_imgs)

        imgs = imgs.with_columns(
            pl.col("mins").rank("dense").over("runid").sub(1).cast(int).alias("idx")
        ).sort(self._runid_col, *self._scalar_cols, self._nm_col, self._idx_col)

        imgs.pipe(self._normalise_imgs)

        self._img_loaded = True

    def _validate_input_imgs(self, imgs: pl.DataFrame) -> None:
        """
        check if the `imgs` schema matches the expected names datatypes set in the
        object initialisation.

        :param imgs: a sample-wise unfolded square-ish table of stacked sample images.
        :type imgs: pl.DataFrame
        :raises ValueError: if column names dont match expectation
        :raises TypeError: if column datatypes dont match expectation
        """

        # check named input columns match table exactly.
        diff = set(
            [
                *self._scalar_cols,
                self._time_col,
                self._runid_col,
                self._nm_col,
                self._abs_col,
            ]
        ).difference(imgs.columns)

        if diff:
            raise ValueError(
                f"input column names dont match input table. difference: {diff}"
            )

        # validate the type schema
        expected_schema = pl.Schema(
            {
                **dict(zip(self._scalar_cols, [str])),
                self._time_col: float,
                self._runid_col: str,
                self._nm_col: int,
                self._abs_col: float,
            }
        )

        # the pairs in the input table that differ from the expectation
        schema_diff = {
            k: imgs.schema[k] for k, v in expected_schema.items() if v != imgs.schema[k]
        }

        if schema_diff:
            diff_pairs_from_exp = {k: expected_schema[k] for k in schema_diff.keys()}
            raise TypeError(
                f"schema mismatch: {schema_diff}, expected: {diff_pairs_from_exp}"
            )

    def _normalise_imgs(self, imgs: pl.DataFrame) -> None:
        """
        Deconstruct `imgs` into a series of normalised tables ala SQL normalisation.

        take the input `imgs` table and decompose it into a scalar, wavelength, and time
        tables, deleting the input imgs table. Attributes added include:

        - `_time_tbl` contains the unique time labels for the time-wise index.
        - `_scalar_tbl` contains sample-specific information such as 'id'. One row per sample
        - `_nm_tbl` contains the image data, including time labels.

        :param imgs: a sample-wise unfolded square-ish table of stacked sample images.
        :type imgs: pl.DataFrame
        """

        # a table containing the runwise time mode labels
        self._time_tbl = imgs.select(
            self._runid_col, self._idx_col, self._time_col
        ).unique()

        # a table containing the scalar labels of each sample
        self._scalar_tbl = imgs.select(self._runid_col, *self._scalar_cols).unique()

        self._nm_tbl = imgs.select(
            self._runid_col, self._nm_col, self._idx_col, self._time_col, self._abs_col
        )

    def plot_3d(self) -> go.Figure:
        """
        Produce a 3d line plot of all the samples overlaid.

        :raises RuntimeError: if `_nm_tbl` attribute is not present
        :return: 3d line plot
        :rtype: go.Figure
        """
        if not hasattr(self, "_nm_tbl"):
            raise RuntimeError("run 'load_imgs' first")

        return plot_imgs(
            imgs=self._nm_tbl,
            nm_col=self._nm_col,
            abs_col=self._abs_col,
            time_col=self._time_col,
            runid_col=self._runid_col,
        )

    def to_X(self) -> XX:
        """
        Retuns an XX object, the samples images as a list of numpy arrays sliced
        samplewise.

        :return: Image data sliced samplewise
        :rtype: XX
        """

        return XX(nm_tbl=self.nm_tbl_as_wide())

    def nm_tbl_as_wide(self) -> pl.DataFrame:
        """
        return the internal wavelength table as a wide form, with wavelengths as columns.

        :return: samplewise-unfolded image tensor table.
        :rtype: pl.DataFrame
        """

        return self._nm_tbl.pivot(
            index=[self._runid_col, self._time_col],
            on=self._nm_col,
            values=self._abs_col,
        ).sort(self._runid_col, self._time_col)

    def filter_nm_tbl(self, expr: pl.Expr) -> Self:
        """
        filter `nm_tbl` based on the input Polars expression. Does no validation of
        column names so check before inputting.

        TODO: validate expression column names?

        :param expr: A valid Polars expression able to be input into a `DataFrame.filter`
          method.
        :type expr: pl.Expr
        :raises AttributeError: if this is called before `load_imgs`
        :return: a copy of this object with the nm table filtered.
        :rtype: Self
        """
        if not hasattr(self, "_nm_tbl"):
            raise AttributeError("no nm_tbl found. Run `load_imgs` first")

        _data = deepcopy(self)
        _data._nm_tbl = self._nm_tbl.filter(expr).sort(
            self._runid_col, self._nm_col, self._time_col
        )

        current_runids = self._nm_tbl.get_column("runid").unique().sort()
        remaining_runids = _data._nm_tbl.get_column("runid").unique().sort()

        if not set(current_runids) == set(remaining_runids):
            # remove samples from other tables based on whether
            _data.mta = _data.mta.filter(pl.col("runid").is_in(remaining_runids))
            _data._time_tbl = _data._time_tbl.filter(
                pl.col("runid").is_in(remaining_runids)
            )
            _data._scalar_tbl = _data._scalar_tbl.filter(
                pl.col("runid").is_in(remaining_runids)
            )

        return _data

    def plot_2d_facet(self, wavelength, facet_col_wrap, title="", template=None):
        """plot a samplewise faceting of the data at the given `wavelength` with optional `title` and number of columns `facet_col_wrap`."""
        import plotly.express as px

        return self._nm_tbl.filter(pl.col("nm") == wavelength).pipe(
            px.line,
            x="mins",
            y="abs",
            facet_col="runid",
            facet_col_wrap=facet_col_wrap,
            title=title,
            template=template,
        )


def _normalise_imgs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalise a sample-wise unfolded image tensor table such that each column bar 'abs' is an identifier, and each row is unique.

    Prepares the sample images for further normalisation into seperate tables, sql-like

    :param df: samplewise unfolded image tensor table.
    :type df: pl.DataFrame
    :return: normalised image table
    :rtype: pl.DataFrame
    """
    return df.unpivot(
        index=["runid", "id", "path", "mins"], variable_name="nm", value_name="abs"
    ).with_columns(pl.col("nm").cast(int))


def _remove_samples(df: pl.DataFrame, samples: list[str]) -> pl.DataFrame:
    """
    filter out the input samples from the image df.

    :param df: image df
    :type df: pl.DataFrame
    :param samples: sample runids to be removed
    :type samples: list[str]
    :return: image df without the specified samples
    :rtype: pl.DataFrame
    """
    return df.filter(~pl.col("runid").is_in(samples))


def _preprocess_imgs(imgs: pl.DataFrame, drop_samples: list[str]) -> pl.DataFrame:
    """
    normnalise and remove specified samples.

    :param imgs: image df
    :type imgs: pl.DataFrame
    :param drop_samples: runids of the samples to be removed
    :type drop_samples: list[str]
    :return: normalised image df w/o the specified samples
    :rtype: pl.DataFrame
    """
    return imgs.pipe(_normalise_imgs).pipe(_remove_samples, drop_samples)
