import pytest
from pca_analysis.peak_picking_viz import plot_peaks
from pca_analysis.cabernet.cabernet import Cabernet
import xarray as xr
from plotly.graph_objects import Figure

xr.set_options(display_expand_data=False, display_expand_coords=False)


@pytest.fixture
def subset_samples(cab: Cabernet):
    subset = (
        cab.sel(wavelength=256, mins=slice(0, 30))
        .isel(sample=slice(0, 5))
        .pick_peaks(
            "input_data",
            # find_peaks_kwargs=dict(prominence=0.25)
        )
    )
    return subset


@pytest.fixture
def peak_map_figure_faceting(subset_samples: Cabernet):
    fig = plot_peaks(
        ds=subset_samples._dt.to_dataset(),
        x="mins",
        group_dim="sample",
        col_wrap=2,
        input_signal_key="input_data",
        peak_table_key="peaks",
    )

    return fig


def test_peak_map_figure_faceting(peak_map_figure_faceting: Figure):
    peak_map_figure_faceting.show()
