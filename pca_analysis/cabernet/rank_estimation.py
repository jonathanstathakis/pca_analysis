from ..decomposers import CORCONDIA
from xarray import DataTree
from dataclasses import dataclass
from plotly.graph_objects import Figure
from pandas import DataFrame
from xarray import DataArray


@dataclass
class CorcondiaResults:
    diagnostic_over_rank: Figure
    diagnostic_table: DataFrame


class RankEstimation:
    def __init__(self, dt: DataTree):
        self._dt = dt

    def corcondia(self, path, rank_range: tuple[int, int] = (1, 2)):
        get_result = self._dt.get(path)

        if not isinstance(get_result, DataArray):
            raise TypeError

        X = get_result.to_numpy()

        corcon = CORCONDIA(X=X)
        corcon.compute_range(rank_range=rank_range)
        fig = corcon.plot_diagonstic()

        diagnostic_table = corcon.diagnostic_table

        return CorcondiaResults(
            diagnostic_over_rank=fig, diagnostic_table=diagnostic_table
        )
