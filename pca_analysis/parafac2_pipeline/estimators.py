import logging

import numpy as np
from pybaselines import Baseline
from sklearn.base import BaseEstimator, TransformerMixin
from tensorly.decomposition import parafac2 as tl_parafac2
import xarray as xr

logger = logging.getLogger(__name__)


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
        implement tensorly's PARAFAC2 via sklearn transformer with convenience functions
        for postprocessing.
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

        self.decomp_, self.errors = tl_parafac2(
            tensor_slices=X,
            rank=self.rank,
            n_iter_max=self.n_iter_max,
            init=self.init,
            svd=self.svd,
            normalize_factors=self.normalize_factors,
            tol=self.tol,
            nn_modes=self.nn_modes,
            random_state=self.random_state,
            verbose=self.verbose,
            return_errors=True,
            n_iter_parafac=self.n_iter_parafac,
            linesearch=self.linesearch,
        )

        self.weights_ = self.decomp_[0]
        self.A_ = self.decomp_[1][0]
        self.B_pure_ = self.decomp_[1][1]
        self.C_ = self.decomp_[1][2]
        self.projections_ = self.decomp_[2]

        from tensorly.parafac2_tensor import apply_parafac2_projections

        # the function requires A, C and the weights and returns them without mutation
        # thus only select the list of projected B (or 'evolving factors')
        self.B_proj_ = apply_parafac2_projections(self.decomp_)[1][1]

        return self.decomp_

    def parafac2_to_tensor(self):
        """
        return the model as a tensor
        """
        from tensorly.parafac2_tensor import parafac2_to_tensor

        return parafac2_to_tensor(self.decomp_)

    def parafac2_to_slices(self):
        """
        return the model as a list of slices
        """

        from tensorly.parafac2_tensor import parafac2_to_slices

        return parafac2_to_slices(self.decomp_)

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}

    def get_sample_component_tensors(self, as_xarr: bool = False, input_coords=None):
        """Get each component image of the model in a variety of data formats.

        By default it returns a list of 4 mode tensors in order i, r, j, k where r is
        the model rank and also the number of components. The first mode i is a list
        and the remaining three modes make up the axes of each list element's numpy
        array.

        You also have the option of returning the tensor as an xarray DataArray with
        each mode represented in the order given above. Furthermore you can optionally
        provide the input datasets coordinate values to label each mode. The component
        mode labels are generated from the rank argument given in the PARAFAC2 init.
        """
        from .parafac2postprocessing import _construct_sample_component_tensors

        components = _construct_sample_component_tensors(
            A=self.A_, Bs=self.B_proj_, C=self.C_
        )

        # dim 1 is samples, dim 2 is rank, dim3 is time, dim 4 is spectra.

        if as_xarr:
            full_tensor = np.stack(components)
            if input_coords:
                rank_labels = [int(x) for x in range(self.rank)]
                new_order = list(input_coords.dims)
                new_order.insert(1, "component")

                new_coords = input_coords.assign({"component": rank_labels})[new_order]

                xr_components = xr.DataArray(data=full_tensor, coords=new_coords)
            else:
                xr_components = xr.DataArray(data=full_tensor)

            return xr_components

        return components


class BCorr_ARPLS(TransformerMixin, BaseEstimator):
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
        """subtracts the baseline of a sample x wavelength x time tensor by iterating
        fisrt over the samples then wavelengths before subtracting the baseline of each
        selected 1D array then pasting them back together.
        """
        # Since this is a stateless transformer, we should not call `check_is_fitted`.
        # Common test will check for this particularly.

        # Input validation
        # We need to set reset=False because we don't want to overwrite `n_features_in_`
        # `feature_names_in_` but only check that the shape is consistent.
        # X = self._validate_data(X, accept_sparse=True, reset=False)

        # compute the baseline
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

        self.bline_slices_ = [np.stack(X_bline).T for X_bline in X_blines]

        # compute the correctd signals
        self.Xt = []
        for idx, slice in enumerate(X):
            self.Xt.append(slice - self.bline_slices_[idx])

        return self.Xt

    def _more_tags(self):
        # This is a quick example to show the tags API:\
        # https://scikit-learn.org/dev/developers/develop.html#estimator-tags
        # Here, our transformer does not do any operation in `fit` and only validate
        # the parameters. Thus, it is stateless.
        return {"stateless": True}


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
