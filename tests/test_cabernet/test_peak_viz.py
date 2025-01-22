import pytest
from pca_analysis.peak_picking_viz import plot_peaks
from pca_analysis.cabernet.cabernet import Cabernet
import xarray as xr
from plotly.graph_objects import Figure

xr.set_options(display_expand_data=False, display_expand_coords=False)


@pytest.fixture
def subset_samples(cab: Cabernet):
    subset = (
        cab.sel(wavelength=256, mins=slice(0, 15))
        .isel(sample=slice(0, 5))
        .pick_peaks(
            "input_data",
        )
    )
    return subset


@pytest.mark.parametrize(
    "peak_outlines, peak_width_calc", [(True, True), (True, False), (False, False)]
)
def test_peak_map_w_faceting(subset_samples: Cabernet, peak_outlines, peak_width_calc):
    """
    test whether we can viz the peaks and their markers. Parametrized to first
    test all markers, then just outlines, then no markers bar maxima.
    """
    fig = plot_peaks(
        ds=subset_samples._dt.to_dataset(),
        x="mins",
        group_dim="sample",
        col_wrap=2,
        input_signal_key="input_data",
        peak_table_key="peaks",
        peak_outlines=peak_outlines,
        peak_width_calc=peak_width_calc,
    )

    fig.show()


@pytest.mark.parametrize(
    "peak_outlines, peak_width_calc", [(True, True), (True, False), (False, False)]
)
def test_peak_map_no_faceting(subset_samples: Cabernet, peak_outlines, peak_width_calc):
    """
    test whether we can viz the peaks and their markers. Parametrized to first
    test all markers, then just outlines, then no markers bar maxima.
    """
    fig = plot_peaks(
        ds=subset_samples.isel(sample=1)._dt.to_dataset(),
        x="mins",
        # col_wrap=2,
        input_signal_key="input_data",
        peak_table_key="peaks",
        peak_outlines=peak_outlines,
        peak_width_calc=peak_width_calc,
    )

    fig.show()
