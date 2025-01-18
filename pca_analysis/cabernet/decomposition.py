"""
Cabernet tensor decomposition namespace.
"""

from xarray import DataTree, DataArray
from .cabernet import StrOrPath, Cabernet
from pca_analysis.parafac2_pipeline.estimators import PARAFAC2
from pathlib import PurePath


class Decomposition:
    def __init__(self, dt: DataTree):
        self._dt = dt

    def parafac2(
        self,
        path: StrOrPath,
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
        n_iter_parafac: int = 5,
        linesearch: bool = False,
    ):
        parafac2 = PARAFAC2(
            rank=rank,
            n_iter_max=n_iter_max,
            init=init,
            svd=svd,
            normalize_factors=normalize_factors,
            tol=tol,
            absolute_tol=absolute_tol,
            nn_modes=nn_modes,
            random_state=random_state,
            verbose=verbose,
            n_iter_parafac=n_iter_parafac,
            linesearch=linesearch,
        )
        get_result = self._dt.get(str(path))

        if not isinstance(get_result, DataArray):
            raise TypeError

        X = get_result

        parafac2.fit_transform(X=X)

        decomposition = parafac2.results_as_xr()

        decomp_path = PurePath(path).parent / "parafac2"

        dt = self._dt.copy()

        dt[str(decomp_path)] = decomposition

        cab = Cabernet.from_tree(dt=dt)

        return cab
