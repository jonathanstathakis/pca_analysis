from pca_analysis.cabernet.shiraz.viz import VizShiraz
from xarray import DataArray
import plotly.graph_objects as go

from pca_analysis.cabernet.shiraz.shiraz import Shiraz

SHOW_VIZ = True

config = dict(display_logo=False)


def test_shiraz_init():
    """
    test whether a `Shiraz` object can be initialised
    """
    assert Shiraz(da=DataArray()) is not None


def test_shiraz_sel(shz_input_data: Shiraz):
    """
    test whether Shiraz can be subset with sel and return a Shiraz obj.
    """

    shz = shz_input_data.sel(wavelength=250)
    assert isinstance(shz, Shiraz), f"{type(shz)}"


def test_shiraz_isel_does_subset(shz_input_data: Shiraz):
    """
    Test whether isel does subset internal data correctly
    """

    shz = shz_input_data.isel(sample=0)

    assert shz
    assert len(shz.shape) == 2


def test_shiraz_isel_returns_shz(shz_input_data: Shiraz):
    """
    test whether Shiraz can be subset with isel and return a Shiraz obj.
    """

    shz = shz_input_data.isel(sample=0)
    assert isinstance(shz, Shiraz), f"{type(shz)}"


def test_shiraz_getitem_any(shz_input_data: Shiraz):
    """
    `__getitem__` returns a Coordinate object. Test if it returns anything.
    """
    sample = shz_input_data["sample"]
    assert sample is not None


def test_get_viz_namespace_shz(shz_input_data: Shiraz):
    """
    Test whether accessing the viz attribute returns a VizShiraz object
    """
    namespace_obj = shz_input_data.viz
    assert isinstance(namespace_obj, VizShiraz), f"{type(namespace_obj)}"


def test_shz_heatmap_single(shz_input_data: Shiraz):
    """
    test generation of a heatmap for a single sample, which depends on there being only 1 sample in the SAMPLE dimension.
    """

    heatmap = shz_input_data.isel(sample=0).viz.heatmap()

    assert isinstance(heatmap, go.Figure)

    if SHOW_VIZ:
        heatmap.show(config=config)


def test_shz_heatmap_facet(shz_input_data: Shiraz):
    """
    test generation of a heatmap as a facet, which depends on there being more than 1
    sample in the DataArray.
    """

    heatmap = shz_input_data.viz.heatmap(n_cols=3)

    assert isinstance(heatmap, go.Figure)

    if SHOW_VIZ:
        heatmap.show(config=config)


def test_shz_line_single_chromgram(shz_input_data: Shiraz):
    """
    generate a line plot for a single sample at a single wavelength
    """
    line_plot = shz_input_data.isel(sample=0, wavelength=5).viz.line(x="mins")

    assert isinstance(line_plot, go.Figure)

    if SHOW_VIZ:
        line_plot.show()


def test_shz_line_single_specgram(shz_input_data: Shiraz):
    """
    generate a line plot for a single sample at a single wavelength
    """
    line_plot = shz_input_data.isel(sample=0, mins=5).viz.line(x="wavelength")

    assert isinstance(line_plot, go.Figure)


def test_shz_line_samplewise_overlay_chromatogram(shz_input_data: Shiraz):
    """
    generate a line plot for a single sample and multiple wavelengths.
    """
    line_plot = (
        shz_input_data.isel(sample=0)
        # .sel(wavelength=[220, 240, 260, 280])
        .viz.line(x="mins", overlay_dim="wavelength")
    )

    assert isinstance(line_plot, go.Figure)
    line_plot.show(config=config)


def test_shz_line_samplewise_overlay_spectrogram(shz_input_data: Shiraz):
    """
    generate a line plot for a single sample and multiple wavelengths.
    """
    line_plot = (
        shz_input_data.isel(sample=0)
        .sel(method="nearest", mins=[5, 10, 15, 20])
        .viz.line(x="wavelength", overlay_dim="mins")
    )

    assert isinstance(line_plot, go.Figure)
    line_plot.show(config=config)

    # TODO we've got 1d, 2d overlay and now need 3D facet/overlay.
    # TODO 3D line plot


def test_shz_line_facet_overlay_every_dim_multiple(shz_input_data: Shiraz):
    """
    test facet line plotting where every dim has multiple values. Should be possible
    to have 1 value in a given dim while still using this API.
    """
    fig = (
        shz_input_data.isel(sample=[0, 3, 5])
        .sel(mins=slice(0, 30))
        .viz.line(x="mins", facet_dim="sample", overlay_dim="wavelength", n_cols=2)
    )

    assert isinstance(fig, go.Figure)

    if SHOW_VIZ:
        fig.show()
