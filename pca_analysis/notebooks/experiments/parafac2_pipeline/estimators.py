from sklearn.base import BaseEstimator, TransformerMixin
from tensorly.decomposition import parafac2 as tl_parafac2
from pybaselines import Baseline
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import duckdb as db
from numpy.typing import ArrayLike


class PARAFAC2(TransformerMixin, BaseEstimator):
    """
    See <https://github.com/scikit-learn-contrib/project-template/blob/main/skltemplate/_template.py#L227>


    TODO: docstring
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(
        self,
        rank: int = 5,
        n_iter_max: int = 2000,
        init: str = "random",
        svd: str = "truncated_svd",
        normalize_factors: bool = False,
        tol: float = 1e-8,
        absolute_tol: float = 1e-13,
        nn_modes=None,
        random_state=None,
        verbose: bool = False,
        # return_errors: bool = False,
        n_iter_parafac: int = 5,
        linesearch: bool = True,
    ):
        """
        TODO: docstring
        """
        self.rank = rank
        self.n_iter_max = n_iter_max
        self.init = init
        self.svd = svd
        self.normalize_factors = normalize_factors
        self.tol = tol
        self.absolute_tol = absolute_tol
        self.nn_modes = nn_modes
        self.random_state = random_state
        self.verbose = verbose
        # self.return_errors = return_errors
        self.n_iter_parafac = n_iter_parafac
        self.linesearch = linesearch

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        # X = self._validate_data(X, accept_sparse=True)

        # Return the transformer
        return self

    def transform(self, X):
        """
        Execute the PARAFAC2 decomposition. iteration reconstruction errors are stored in `self.errors`, the
        decomposition tensors are stored in `self.decomp`.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        # X = self._validate_data(X, accept_sparse=True, reset=False)

        self.decomp, self.errors = tl_parafac2(
            tensor_slices=X,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            normalize_factors=self.normalize_factors,
            tol=self.tol,
            absolute_tol=self.absolute_tol,
            nn_modes=self.nn_modes,
            random_state=self.random_state,
            verbose=self.verbose,
            return_errors=True,
            n_iter_parafac=self.n_iter_parafac,
            linesearch=self.linesearch,
        )

        return self.decomp

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}


class BCorr_ASLS(TransformerMixin, BaseEstimator):
    """
    See <https://github.com/scikit-learn-contrib/project-template/blob/main/skltemplate/_template.py#L227>


    TODO: docstring

    Notes
    -----

    Implementing with iteration over each wavelength but should explore 2D implementation.
    the [Docs](https://pybaselines.readthedocs.io/en/latest/algorithms_2d/index.html)
    say that 1D iteration is faster than 2D.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(
        self,
        lam: float = 1e6,
        p: float = 1e-2,
        diff_order: int = 2,
        max_iter: int = 50,
        tol: float = 1e-3,
        weights=None,
    ):
        """
        Baseline correction via ASLS.

        Parameters
        ----------
        lam : float, optional
            The smoothing parameter. Larger values will create smoother baselines.
            Default is 1e6.
        p : float, optional
            The penalizing weighting factor. Must be between 0 and 1. Values greater
            than the baseline will be given `p` weight, and values less than the baseline
            will be given `p - 1` weight. Default is 1e-2.
        diff_order : int, optional
            The order of the differential matrix. Must be greater than 0. Default is 2
            (second order differential matrix). Typical values are 2 or 1.
        max_iter : int, optional
            The max number of fit iterations. Default is 50.
        tol : float, optional
            The exit criteria. Default is 1e-3.
        weights : array-like, shape (N,), optional
            The weighting array. If None (default), then the initial weights
            will be an array with size equal to N and all values set to 1.



        Other
        -----

        see docs <https://pybaselines.readthedocs.io/en/latest/api/pybaselines/whittaker/index.html#pybaselines.whittaker.asls>
        """

        self.lam = lam
        self.p = p
        self.diff_order = diff_order
        self.max_iter = max_iter
        self.tol = tol
        self.weights = weights

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        # X = self._validate_data(X, accept_sparse=True)

        # Return the transformer

        return self

    def transform(self, X):
        """ """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        # X = self._validate_data(X, accept_sparse=True, reset=False)

        # compute the baseliens
        self._input = X
        self.slice_wavelength_params = []
        X_blines = []
        for slice in X:
            wavelength_blines = []
            wavelength_params = []
            for wavelength in slice.T:
                bline, params = Baseline().arpls(
                    wavelength,
                    lam=self.lam,
                    # p=self.p,
                    diff_order=self.diff_order,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    weights=self.weights,
                )
                wavelength_blines.append(bline)
                wavelength_params.append(params)

            self.slice_wavelength_params.append(wavelength_params)
            X_blines.append(wavelength_blines)

        self._X_bline_slices = [np.stack(X_bline).T for X_bline in X_blines]

        # compute the correctd signals
        self.Xt = []
        for idx, slice in enumerate(X):
            self.Xt.append(slice - self._X_bline_slices[idx])

        return self.Xt

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}

    def get_bcorr_results(self, conn=db.connect()):
        return BCorrResults(
            conn=conn,
            input=self._input,
            corrected=self.Xt,
            baselines=self._X_bline_slices,
        )


class BCorr_SNIP(TransformerMixin, BaseEstimator):
    """
    See <https://github.com/scikit-learn-contrib/project-template/blob/main/skltemplate/_template.py#L227>


    TODO: docstring

    Notes
    -----

    Implementing with iteration over each wavelength but should explore 2D implementation.
    the [Docs](https://pybaselines.readthedocs.io/en/latest/algorithms_2d/index.html)
    say that 1D iteration is faster than 2D.
    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(
        self,
        max_half_window=None,
        decreasing: bool = False,
        smooth_half_window=None,
        filter_order: int = 2,
        pad_kwargs={},
    ):
        """ """

        self.max_half_window = max_half_window
        self.decreasing = decreasing
        self.smooth_half_window = smooth_half_window
        self.filter_order = filter_order
        self.pad_kwargs = pad_kwargs

    # @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        # X = self._validate_data(X, accept_sparse=True)

        # Return the transformer

        return self

    def transform(self, X):
        """ """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        # X = self._validate_data(X, accept_sparse=True, reset=False)

        # compute the baseliens
        self._input = X
        self.slice_wavelength_params = []
        X_blines = []
        for slice in X:
            wavelength_blines = []
            wavelength_params = []
            for wavelength in slice.T:
                bline, params = Baseline().snip(
                    wavelength,
                    max_half_window=self.max_half_window,
                    decreasing=self.decreasing,
                    smooth_half_window=self.smooth_half_window,
                    filter_order=self.filter_order,
                    pad_kwargs=self.pad_kwargs,
                )
                wavelength_blines.append(bline)
                wavelength_params.append(params)

            self.slice_wavelength_params.append(wavelength_params)
            X_blines.append(wavelength_blines)

        self._X_bline_slices = [np.stack(X_bline).T for X_bline in X_blines]

        # compute the correctd signals
        self.Xt = []
        for idx, slice in enumerate(X):
            self.Xt.append(slice - self._X_bline_slices[idx])

        return self.Xt

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}

    def get_bcorr_results(self, conn=db.connect()):
        return BCorrResults(
            conn=conn,
            input=self._input,
            corrected=self.Xt,
            baselines=self._X_bline_slices,
        )


import duckdb as db
from enum import StrEnum
import polars as pl


class BCRCols(StrEnum):
    """BaselineCorrectionResultsColumns"""

    SAMPLE = "sample"
    IDX = "idx"
    WAVELENGTH = "wavelength"
    ABS = "abs"
    SIGNAL = "signal"


class SignalNames(StrEnum):
    CORR = "corrected"
    BLINE = "baseline"
    INPUT = "input"


def to_dataframe(arrs: list, signal_name=None):
    df = (
        pl.concat(
            [
                pl.DataFrame(x)
                .with_columns(pl.lit(idx).alias(BCRCols.SAMPLE))
                .with_row_index(BCRCols.IDX)
                for idx, x in enumerate(arrs)
            ]
        )
        .unpivot(
            index=[BCRCols.SAMPLE, BCRCols.IDX],
            variable_name=BCRCols.WAVELENGTH,
            value_name=BCRCols.ABS,
        )
        .with_columns(pl.col(BCRCols.WAVELENGTH).str.replace("column_", "").cast(int))
    )

    if signal_name:
        df = df.with_columns(pl.lit(signal_name).alias(BCRCols.SIGNAL))
    return df


class BCorrResults:
    def __init__(self, input, corrected, baselines, conn=db.connect()):
        self._conn = conn
        self.input = input
        self.corrected = corrected
        self.baselines = baselines

        corr = to_dataframe(self.corrected, signal_name=SignalNames.CORR)
        baselines = to_dataframe(self.baselines, signal_name=SignalNames.BLINE)
        input = to_dataframe(self.input, signal_name=SignalNames.INPUT)
        self.df = pl.concat([corr, baselines, input]).sort(
            BCRCols.SAMPLE, BCRCols.SIGNAL, BCRCols.WAVELENGTH, BCRCols.IDX
        )

    def _to_db(self):
        """write the baselines and corrected to a database"""

        query = """--sql
        create table baseline_corrected (
        sample int not null,
        signal varchar not null,
        wavelength int not null,
        idx int not null,
        abs float not null,
        primary key (sample, signal, wavelength, idx)
        );

        create unique index baseline_corrected_idx on baseline_corrected(sample, signal, wavelength, idx);

        insert into baseline_corrected
            select
                sample,
                signal,
                wavelength,
                idx,
                abs
            from
                df;
        """
        self._conn.execute(query)

    def viz_3d_line_plots_by_sample(self, cols=2, title=None) -> go.Figure:
        """
        TODO: fix plot so no *underlying* lines, add input(?)
        """
        sample_slices = self.df.partition_by(
            BCRCols.SAMPLE, include_key=False, as_dict=True
        )
        num_samples = len(sample_slices)

        rows = num_samples // cols

        flat_specs = [{"type": "scatter3d"}] * rows * cols

        reshaped_specs = np.asarray(flat_specs).reshape(-1, cols)
        reshaped_specs_list = [list(x) for x in reshaped_specs]

        titles = [f"{BCRCols.SAMPLE} = {str(x[0])}" for x in sample_slices.keys()]

        # see <https://plotly.com/python/subplots/>,
        # API: <https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html>
        fig = make_subplots(
            rows=rows,
            cols=cols,
            specs=reshaped_specs_list,
            horizontal_spacing=0.0001,
            vertical_spacing=0.05,
            subplot_titles=titles,
        )

        coords = gen_row_col_coords(rows, cols)

        for idx, sample in enumerate(sample_slices.values()):
            # margin = dict(b=1, t=1, l=1, r=1)
            traces = px.line_3d(
                data_frame=sample,
                x=BCRCols.IDX,
                y=BCRCols.WAVELENGTH,
                z=BCRCols.ABS,
                line_group=BCRCols.WAVELENGTH,
            ).data

            # got to write each wavelength individually to each subplot
            for trace in traces:
                fig.add_trace(trace, row=coords[idx][0], col=coords[idx][1])

        # camera <https://plot.ly/python/3d-camera-controls/>
        # camera in subplots <https://community.plotly.com/t/make-subplots-with-3d-surfaces-how-to-set-camera-scene/15137/4>
        eye_mult = 1.4
        default_camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            # eye=dict(x=1.25 * eye_mult, y=1.25 * eye_mult, z=1.25 * eye_mult),
        )

        fig.update_layout(
            height=500 * rows,
            template="plotly_dark",
            #   margin=margin
            title=title,
        )

        fig.update_scenes(camera=default_camera)

        return fig

    def viz_compare_signals(self, wavelength: int) -> go.Figure:
        """return a 2d plot at a given wavelength for each sample, corrected, baseline
        and original"""

        wavelength_vals = self.df.get_column(BCRCols.WAVELENGTH).unique(
            maintain_order=True
        )

        if wavelength not in wavelength_vals:
            raise ValueError(f"{wavelength} not in {wavelength_vals}")

        filtered_df = self.df.filter(pl.col(BCRCols.WAVELENGTH) == wavelength)

        fig = px.line(
            data_frame=filtered_df,
            x=BCRCols.IDX,
            y=BCRCols.ABS,
            color=BCRCols.SIGNAL,
            facet_col=BCRCols.SAMPLE,
            facet_col_wrap=wavelength_vals.len() // 4,
            title=f"corrected and fitted baseline @ wavelength = {wavelength}",
        )

        return fig


def gen_row_col_coords(rows: int, cols: int) -> list[tuple[int, int]]:
    """generate a list of tuples where every list element is the row, containing the row and column position."""
    indexes = []
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            indexes.append((row, col))

    return indexes


def prepare_scatter_3d_from_df(df, x_col, y_col, z_col):
    # scatter3d API: https://plotly.com/python/reference/scatter3d/#scatter3d-line

    x = df.select(x_col).to_numpy().flatten()
    y = df.select(y_col).to_numpy().flatten()
    z = df.select(z_col).to_numpy().flatten()

    return go.Scatter3d(x=x, y=y, z=z, mode="lines")
