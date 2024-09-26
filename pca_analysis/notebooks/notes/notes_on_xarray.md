---
cdt: 2024-09-04T20:51:23
title: Notes on XArray
description: An outline of notes sourced from the XArray docs
todo: 
    - summarise
    - fix citekeys
---

# Introduction to XArray

- the following labels; dimensions, coordinates and attributes [@xarray_whyxarray, para. 1]
- works with multidimensional arrays (tensors) [@xarray_whyxarray, sec. "what labels enable"]
- xarray can be used in all scientific domains, but especially: physics, astronomy, geoscience, bioinformatics, engineering, finance, and deep learning [@xarray_whyxarray, sec. "what labels enable"].
- xarray provides human-readable labeling for tensors, for use in both seletion and operations. This can be compared to numpy in which the user needs to track what axis index is what [@xarray_whyxarray, sec. "what labels enable"].
- provides groupby operations [@xarray_whyxarray, sec. "what labels enable"].
- alignment [@xarray_whyxarray, sec. "what labels enable"].
- metadata in dictionary form [@xarray_whyxarray, sec. "what labels enable"].
- There are two data structures, DataArray and Dataset [@xarray_whyxarray, sec. "Core data structures"]
- DataArrays are N-dimensional arrays (tensors), an analog of a pandas Series [@xarray_whyxarray, sec. "what labels enable"]
- Dataset is a mapping of DataArrays to shared dimensions, an analog of a pandas DataFrame [@xarray_whyxarray, sec. "what labels enable"].
- The Dataset enables selection and operation across the constituant DataArrays [@xarray_whyxarray, sec. "what labels enable"].
- It is based on netCDF [@xarray_whyxarray, sec. "what labels enable"].

# Dataset

- a Dataset is intended as a "multi-dimensional, in memory, array database" [@xarray_dataset, para. 1].
- A Dataset corresponds to a NetCDF file [@xarray_dataset, para. 2].
- A dataset conists of variables, corrodinates, and attributes [@xarray_dataset, para. 2].
- A Dataset (and NetCDF file) are intended to be self-describing datasets [@xarray_dataset, para. 2].
- Each DataArray is mapped to a variable name [@xarray_dataset, para. 3].
- One dimensional variables are represented as pandas indexes with name equal to dimension [@xarray_dataset, para. 4].

@xarray_whyxarray: https://docs.xarray.dev/en/latest/getting-started-guide/why-xarray.html
@xarray_dataset: https://docs.xarray.dev/en/latest/generated/xarray.Dataset.html#xarray.Dataset