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

## Welcome to Xdas!

 *Xdas* is an open source Python library that's used to work with huge labeled N-dimensional arrays as it is the case in Distributed Acoustic Sensing (DAS). *Xdas* API
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
pip install "git+https://github.com/xdas-dev xs.git@dev"
```

````
`````

*xdas* must first be imported:

```{code-cell}
import xdas 
```

### Create a DataArray 
Link your DAS data to a DataArray:

```{code-cell} 
da = xdas.open_mfdataarray("00*.h5", engine=None)
````

If you do not have GPS synchronization during your DAS acquisition, you may have gaps or overlaps between files. With Xdas, you can define a tolerance to what extent you accept to reinterpolate the time to avoid overlaps in the time axis. In the case you have overlaps in time you may have errors when slicing the DataArray. 

```{code-cell} 
import numpy as np
da["time"]=da["time"].simplify(np.timedelta64(30,"ms"))
da.to_netcdf("da.nc", virtual = True)

```

Reading ASN, Febus, Optasense and Sintela data is already implemented and must be specified in engine. You also have the option to develop your own customized [engine](user-guide/engine.md). 
If you want to create a single DataArray for multiple acquisitions (i.e. different fibers, changing acquisition parameters), you can use the [DataCollection](user-guide/DataCollection.md) object.  


### Load DataArray

Data can be fetched from a file:

```{code-cell} 
da = xdas.open_dataarray("da.nc")
da
```

DataArray can be sliced using a Label-based selection:

```{code-cell}
da = da.sel(
    time=slice("2023-01-01T00:00:01", "2023-01-01T00:00:05"),
    distance=slice(1000, 9000),
)
da
```

Once the selection is small enough data can be loaded into memory or directly used for further processing:

```{code-cell}
da = da.load()
da
```

### Processing

DataArray can be processed without need to convert it to numpyarray. The methods of DataArray are listed **here Need Link** . Xdas uses the following conventions : (i) instead of providing the axis number, the dimension label must be provided, (ii) a parallel keyword argument may be passed to require multithreading processing. For instance, decimating the DataArray in space and time can we done as : 


```{code-cell}
import xdas.signal as xs
da = xs.decimate(da,2,ftype="fir", dim="distance")
da = xs.decimate(da,2,ftype="iir", dim="time")
```

### Plotting

Visualizing your DataArray is convenient:

```{code-cell}
da.plot(yincrease=False, vmin=-1, vmax=1)
```

Xdas makes it easy to display data in the spectral domain by renaming the axis as shown in the FK plot: 

```{code-cell}
import xdas.fft as xfft
fk = xs.taper(da, dim="distance")
fk = xs.taper(fk, dim="time")
fk = xfft.rfft(fk, dim={"time": "frequency"})
fk = xfft.fft(fk, dim={"distance": "wavenumber"})
np.abs(fk).plot(robust=True, interpolation="antialiased")
```

### Saving
Processed data can be saved as DataArray:
```{code-cell}
np.abs(fk).to_netcdf("fk.nc")
```



[xarray API]: <https://docs.xarray.dev/en/stable/user-guide/indexing.html>
[DataArray]: <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html#xarray.DataArray>
