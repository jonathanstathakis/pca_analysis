from sklearn.base import BaseEstimator, TransformerMixin
from tensorly.decomposition import parafac2 as tl_parafac2


class PARAFAC2(TransformerMixin, BaseEstimator):
    """
    See <https://github.com/scikit-learn-contrib/project-template/blob/main/skltemplate/_template.py#L227>

    Parameters
    ----------

    Attributes
    ----------

    """

    # This is a dictionary allowing to define the type of parameters.
    # It used to validate parameter within the `_fit_context` decorator.
    _parameter_constraints = {
        "demo_param": [str],
    }

    def __init__(
        self,
        rank: int,
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
