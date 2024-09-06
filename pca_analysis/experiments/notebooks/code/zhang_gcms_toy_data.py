import xarray as xr
import numpy as np
from pymatreader import read_mat

path = "/Users/jonathan/mres_thesis/pca_analysis/Wine_v7.mat"

# from my labeling to the dataset labeling
_label_mapping = {
    "sample" : "Label_Wine_samples",
    "time" : "Label_Elution_time",
    "mz": "Label_Mass_channels",
}

def get_zhang_data():
    full_data = _get_full_zhang_data()

    idx_start, idx_end = _calculate_zhang_slice(full_data['time'])
    
    sliced_data = full_data.isel(time=slice(idx_start,idx_end))
    sliced_data.isel(sample=0).plot.line(x="time", add_legend=False);

    return sliced_data
    

def _get_full_zhang_data():
    return _prepare_data(key_mapping=_label_mapping)

def _prepare_data(key_mapping):
    
    data = read_mat(filename=path, variable_names=["Data_GC"] + list(key_mapping.values()))
    
    raw_data = xr.DataArray(
        data["Data_GC"],
        coords=[data[k] for k in key_mapping.values()],
        dims=list(key_mapping.keys()),
    )

    return raw_data
    
# adjusting the indexes to center the peaks

def _calculate_zhang_slice(times, time_start=16.52, time_end=16.76, left_offset=-6, right_offset=6):
    
    idx_start = np.nonzero(np.isclose(times, time_start, atol=1e-2))[0][0] + left_offset
    idx_end = np.nonzero(np.isclose(times, time_end, atol=1e-2))[0][0] + right_offset
    
    return idx_start, idx_end
