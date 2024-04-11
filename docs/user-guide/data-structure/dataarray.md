---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
os.chdir("../../_data")
```

## DataArray

{py:class}`~xdas.DataArray` is the base class to load and manipulate big datasets to in *xdas*. It is mainly composed of two attributes: 

- `data`: any N-dimensional array-like object. Compared to *xarray* `xdas.DataArray` are more permissive to the kinds of array-like objects that can be used. In particular, [virtual arrays](virtual-datasets) can be used.
- `coords`: a dict-like container of coordinates. As opposed to *xarray*, which uses dense arrays to label each point, *xdas* also implements [interpolated coordinates](interpolated-coordinates) that provides an efficient representation of evenly spaced data (gracefully handling gaps and small sampling variations).

![](/_static/dataarray.svg)

Other important attributes are:

- `dims`: a tuple that assign to each axis position a dimension name that is defined in the `coords` attribute. Note that having a coordinate per dimension is not mandatory and that the order of the `coords` does not necessary follow the order of the `dims` attribute. 
- `name`: the name of the array to specify the quantity stored (e.g., `"velocity"`).
- `attribute`: a dictionary containing metadata. Note that *xdas* does not use those metadata. It tries to keep as much as possible the information stored there as the `DataArray` is manipulated but it is up to the user to update information there if needed.

In the following examples, we use only one `DataArray`, if you have several `DataArray`s, please use the multi-file version of the opening function: {py:func}`~xdas.open_mfdataarray`. You will just have to adapt the paths argument.

### Creating a DataArray

The user can wrap together an n-dimensional array and some related coordinates. See the related description of how to create coordinates [here](interpolated-coordinates.md). For example:


```{code-cell}
import numpy as np
import xdas

data = np.zeros((6000, 1000))
starttime = np.datetime64("2023-01-01T00:00:00")
endtime = starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1)
distance = 5.0 * np.arange(data.shape[1])

da = xdas.DataArray(
    data=data,
    coords={
        "time": {
            "tie_indices": [0, data.shape[0] - 1],
            "tie_values": [starttime, endtime],
        },
        "distance": distance,
    },
)
da
```

### Writing a DataArray to disk

*xdas* uses the CF conventions to write {py:class}`xdas.DataArray` to disk as netCDF4 files. If the DataArray was generated from a netCDF4/HDF5 file and only slicing was performed, the DataArray can be written as a pointer to the original data using the `virtual` argument. See the part on [](virtual-datasets.md).

```{code-cell}
da.to_netcdf("dataarray.nc", virtual=None)  # try to write virtual, here it's impossible
```

### Reading a DataArray from disk.

Xdas can read several DAS file format with {py:func}`~xdas.open_dataarray` along with its own format. Xdas uses the netCDF4 format with CF conventions. By default Xdas assumes that files are Xdas NetCDF format. If not the case the `engine` argument must be passed.

To learn how to read your custom DAS data format with *xdas*, please see the chapter on [](data-formats.md).

```{code-cell}
da = xdas.open_dataarray("dataarray.nc", engine=None)  # by default Xdas NetCDF
```

### Assign new coordinates to your DataArray

You can either replace the existing coordinates by new ones or assign new coordinates to a {py:class}`xdas.DataArray` and link it them an existing dimension. 

#### Replace existing coordinates

In the example below, we replace the "distance" coordinate with new ones.

```{code-cell}
new_distances = np.linspace(30.8, 40.9, da.shape[1])
assigned = da.assign_coords(distance=new_distances)
assigned
```

#### Add new coordinates and link them to an existing dimension

In the example below, we will add the new coordinate "latitude" linked with the "distance" dimension.

```{code-cell}
latitudes = np.linspace(-33.90, -35.90, da.shape[1])
assigned = da.assign_coords(latitude=("distance", latitudes))
assigned
```

You can also swap a dimension to one of the new coordinates.

```{code-cell}
swapped = da.swap_dims({"distance": "latitude"})
swapped
```

### Plot your DataArray

{py:class}`xdas.DataArray` includes the function {py:func}`xdas.DataArray.plot`. It uses the *xarray* way of plotting data depending on the number of dimensions your data array has. You'll have to adapt the arguments and keyword arguments in {py:func}`xdas.DataArray.plot` depending on the dimensionality of your data:

- If your {py:class}`xdas.DataArray` has one dimension, please refer to the arguments and kwargs from the 'xarray.plot.line' function.
- For 2 dimensions or more, please refer to the 'xarray.plot.imshow' function.
- For other, please refer to 'xarray.plot.hist' function.