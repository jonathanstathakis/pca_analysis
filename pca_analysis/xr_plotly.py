"""
Adds a plotly accessor for xarray Datasets and DataArrays
"""

import xarray as xr
import plotly.express as px
from pca_analysis.xr_signal import facet_plot_multiple_traces

colormap = px.colors.qualitative.Plotly


@xr.register_dataset_accessor("plotly")
class PlotlyAccessorDS:
    def __init__(self, ds):
        self._ds = ds

    def facet_plot_overlay(
        self,
        grouper=None,
        var_keys=None,
        x_key=None,
        col_wrap: int = 1,
        fig_kwargs={},
        trace_kwargs: dict = {},
    ):
        """
        overlay multiple vars, assuming that after grouping by `grouper`, each var
        is 1D.
        """

        return facet_plot_multiple_traces(
            ds=self._ds,
            grouper=grouper,
            var_keys=var_keys,
            x_key=x_key,
            col_wrap=col_wrap,
            fig_kwargs=fig_kwargs,
            trace_kwargs=trace_kwargs,
        )


@xr.register_dataarray_accessor("plotly")
class PlotlyAccessorDA:
    def __init__(self, da):
        self._da = da

    def line(self, x, y, **kwargs):
        """
        Use `plotly_express` line plot api.

        Example
        -------

        ```
        fig = (da
            .plotly
            .line(x="mins", y="raw_data", color="id_rank")
            )
        ```

        """
        import plotly.express as px

        df = self._da.to_dataframe().reset_index()
        return px.line(df, x=x, y=y, **kwargs)
    
    def smooth(self, **kwargs):
        from pca_analysis.preprocessing import smooth

        smooth.savgol_smooth(self._da)
