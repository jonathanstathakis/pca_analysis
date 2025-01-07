"""
Contains the experiment manager
"""

import xarray as xr
from pca_analysis.get_dataset import get_shiraz_dataset
from pca_analysis import xr_plotly
from xarray import Dataset, DataTree, DataArray
from dataclasses import dataclass


@dataclass
class ChromDims:
    SAMPLE = "sample"
    TIME = "mins"
    SPECTRA = "wavelength"

    def __len__(self):
        return len(self.__dict__.keys())


chrom_dims = ChromDims()


class AbstChrom:
    """
    provides a common *chromatogram* validation via the dims - a
    chromatogram should have `SAMPLE`, `TIME`, `SPECTRA` dims only.
    """

    SAMPLE = chrom_dims.SAMPLE
    TIME = chrom_dims.TIME
    SPECTRA = chrom_dims.SPECTRA


class Cabernet(AbstChrom):
    @classmethod
    def from_tree(cls, dt: xr.DataTree):
        """
        Initilise Cabernet from a DataTree.
        """

        cab = Cabernet(da=xr.DataArray())
        cab._dt = dt

        return cab

    @classmethod
    def load_from_dataset(cls, name=None):
        if name == "shiraz":
            da = get_shiraz_dataset().input_data
        else:
            raise NotImplementedError

        return Cabernet(da=da)

    def __init__(self, da: xr.DataArray):
        """
        Manager of data processes
        """

        assert isinstance(da, DataArray), f"{type(da)}"

        self._dt = xr.DataTree(dataset=xr.Dataset({da.name: da}))
        self.viz = Viz(dt=self._dt)

    def sel(self, **kwargs):
        cab = self.copy()
        cab = cab._dt.sel(**kwargs)
        return cab

    def isel(self, **kwargs):
        cab = self.copy()

        cab._dt = cab._dt.isel(**kwargs)
        return cab

    def __getitem__(self, key):
        result = self._dt[key]

        if isinstance(result, DataTree):
            return Cabernet.from_tree(dt=result)
        elif isinstance(result, DataArray):
            return Shiraz(da=result)
        else:
            return result

    def copy(self):
        """
        make a copy of the internal DataTree, returning a new Cabernet object.
        """

        dt = self._dt.copy()

        cab = Cabernet(xr.DataArray())

        cab._dt = dt

        return cab

    def __setitem__(self, key, data) -> None:
        self._dt[key] = data

    def get(self, key, default: DataTree | DataArray | None = None):
        result = self._dt.get(key=key, default=default)
        return result

    def assign(self, items=None, **items_kwargs):
        """
        Assign `DataTree` or `DataArray` to internal `DataTree`.
        """
        cab = self.copy()
        cab._dt = cab._dt.copy().assign(items=items, **items_kwargs)

        return cab

    def __repr__(self):
        return repr(self._dt)

    def _repr_html_(self):
        return self._dt._repr_html_()


class Viz:
    def __init__(self, dt: xr.DataTree):
        self._dt = dt


class PinotNoir(AbstChrom):
    def __init__(self, ds: Dataset):
        """
        Manager for Datasets.
        """

        assert isinstance(ds, Dataset)

        self._ds = ds

    def __getitem__(self, key):
        result = self._ds[key]

        if isinstance(result, DataTree):
            return
        elif isinstance(result, DataArray):
            return Shiraz(da=result)
        else:
            return result

    def __setitem__(self, key, data):
        self._ds[key] = data

    def get(self, key, default=None):
        result = self._ds.default(name=key, default=default)

        if isinstance(result, xr.Dataset):
            return PinotNoir(ds=result)
        elif isinstance(result, xr.DataArray):
            return Shiraz(da=result)
        else:
            return result


class Shiraz(AbstChrom):
    def __init__(self, da: xr.DataArray):
        """
        Manager for DataArrays
        """

        assert isinstance(da, DataArray)

        self._da = da

    def __getitem__(self, key):
        result = self._da[key]

        if isinstance(result, Dataset):
            return Shiraz(da=result)
        else:
            return result

    def __setitem__(self, key, data):
        self._da[key] = data

        return Shiraz(da=self._da)

    def get(self, key, default=None):
        result = self._da.__get__(name=key, default=default)
        return result
