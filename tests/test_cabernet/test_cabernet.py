import pytest
import xarray as xr
from pca_analysis.exp_manager.cabernet import Cabernet, Shiraz


def test_Cabernet_init():
    """
    test whether a `Cabernet` object can be initialised
    """
    assert Cabernet(da=xr.DataArray()) is not None


def test_Cabernet_assign():
    """
    test whether a `Cabernet` object can be assigned to, returning an instance of Cabernet.
    """

    cab = Cabernet(da=xr.DataArray()).assign(child=xr.DataTree())

    assert isinstance(cab, Cabernet)


def test_load_from_dataset(cab: Cabernet):
    assert cab
    assert cab._dt


def test_cabernet_getitem_any(cab: Cabernet):
    """
    `__getitem__` returns a `DataArray` or DataTree
    depending on whether the key refers to a
    child node or internal `var`. This test checks
    if the call returns anything.
    """
    input_data = cab["input_data"]
    assert input_data is not None


def test_cabernet_getitem_cab(cab: Cabernet):
    """
    tests if `__getitem__` returns a instance of
    `Cabernet` rather than `DataTree`, as we're
    wrapping `DataTree`.
    """

    assert isinstance(cab["child"], Cabernet), f"{type(cab['child'])}"


def test_cabernet_getitem_shz(cab: Cabernet):
    """
    tests if `__getitem__` returns a instance of `Shiraz` rather than `DataArray` when
     accessing a `DataArray`
    """

    shz = cab["input_data"]

    assert isinstance(shz, Shiraz)


def test_setitem(cab: Cabernet):
    """
    `__setitem__` takes arrays or child nodes. As `Cabernet`
    simply passes it through to the `DataTree`, we can use
    that for validation.
    """

    node = xr.DataTree()

    cab["new_item"] = node

    assert cab.get("new_item") is not None


# TODO establish viz protocols. The basic ones need to be instilled at the Shiraz level. Cabernet can have viz methods but they will be very high level. The idea is that you should directly access the object you want to viz without respect to the higher level structure. Cabernet can have multi-var viz methods such as Comparing input data and baseline corrected data.

# TODO establish preprocessing methods. This will be both a Cabernet and Pinot Noir level method. Cabernet will provide the user UI, but Pinot Noir will provide the means of executing the process, returning the corrected object. We can provide a flag in the `attrs` to tell the user what has happened to the data. A receipt. This will include baseline correction, smoothing, sharpening and clustering. Clustering is the difficult one. Pinot Noir will have methods of viewing the result - for example a smoothed Pinot Noir will be able to viz the smoothed signal, same for sharpened, baseline corrected will be able to plot the corrected and the baseline, the clustered will be able to portray the clustering on the input signal, but only Cabernet will be able to viz the input signal against the mutations. Thats the difference - mutations such as signal processing methods cannot carry the input signal as well, but learning tasks such as clustering can.

# # TODO establish PARAFAC2 methods. Cabernet will be able to do this on a target var/child with an option to respect any clustering labels present. The result is a different kind of node: parafac2 and will consist of the factors. It will have children nodes: component slices and reconstruction. Thus the path will be something like /parafac2/comp_slices or /parafac2/recon.

# TODO2 establish validation of 'Cabernet', 'Pinot Noir' and 'Shiraz' xarray data objects. That means that the internal xarray object has at least (possibly exclusively) dims SAMPLE, TIME, SPECTRA, as defined by the class level constants. As we define chromatograms to consist of the signal and the peak mapping, we should enforce the DataArray to consist only of these dims, and if a peak map is present, it will have its own constrictions.
