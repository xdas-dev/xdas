---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
os.chdir("_data")
```

# Getting Started   

## Welcome to xdas!

*xdas* is an open source Python library that's used to work with huge labeled
N-dimensional arrays as it is the case in Distributed Acoustic Sensing (DAS). *xdas* API
is highly inspired by the *xarray* project. It provides *xarray* like objects with custom
functionalities that enable to deal with big multi-file netCDF4/HDF5 dataset with
generally one very long dimension (usually time). It provides the classical signal
processing tools to treat time-series that do not fit in memory. It also enables I/O
capabilities with some DAS formats.

## Installing xdas

*xdas* is a pure python package. It can easily be installed with *pip*:

`````{tab-set}
````{tab-item} Stable
```bash
pip install xdas
```
````
````{tab-item} Latest
```bash
pip install "git+https://github.com/xdas-dev/xdas.git@dev"
```

````
`````

## How to use xdas

*xdas* must first be imported:

```{code-cell}
import xdas as xd
```

Data can be fetched from a file:

```{code-cell} 
da = xd.open_dataarray("sample.nc")
da
```

Label-based selection can be done using the [*xarray* API][xarray API].

```{code-cell}
da = da.sel(
    time=slice("2023-01-01T00:00:01", "2023-01-01T00:00:05"),
    distance=slice(1000, 9000),
)
da
```

Once the selection is small enough data can be loaded into memory:

```{code-cell}
da = da.load()
da
```

It can be converted to a [`DataArray`][DataArray] object to enables the full use of the 
*xarray* API (e.g., for plotting):

```{code-cell}
da = da.to_xarray()  # Data will be loaded automatically if not already done.
da.plot(yincrease=False, vmin=-0.5, vmax=0.5);
```

[xarray API]: <https://docs.xarray.dev/en/stable/user-guide/indexing.html>
[DataArray]: <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html#xarray.DataArray>