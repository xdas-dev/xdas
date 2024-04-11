---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
os.chdir("../_data")
```

# Data Structures

```{warning}
The whole section is rather technical, and these details are not very relevant
for a casual user. I propose to split the User Guide in a *Beginner Guide* and
an *Expert Guide*.
```

```{warning}
Does `xarray.DataArray` need to be discussed here? It doesn't seem very relevant...
```

*xdas* leverages two kinds of data structures. The workhorse data structures are the 
{py:class}`xarray.DataArray` and {py:class}`xdas.DataArray` objects, which are 
N-dimensional labeled array object. The second, more abstract data structures are the 
{py:class}`xarray.Dataset` and {py:class}`xdas.DataCollection` objects that are 
mappings of DataArray and DataArray objects respectively. `DataArray`s are intended for
single contiguous datasets, whereas `DataCollection`s combine multiple `DataArray`s (or
nested `DataCollection`s) to facilitate operations across contiguous blocks.

The *xdas* objects extend their equivalent *xarray* counterpart in term of huge multi-file datasets handling, but 
at the cost of a limited subset of the *xarray* API. When manipulating data that are 
small enough to be stored in-memory, the full capabilities of *xarray* objects are available, and is 
generally preferred. In a near future, we hope that the extra functionalities provided 
by the *xdas* objects will be part of the *xarray* library to enjoy the best of both 
words.

The *xarray* objects structure description can be found in the 
[xarray documentation](https://docs.xarray.dev/en/stable/user-guide/data-structures.html). 
Here, we focus on the *xdas* objects. They follow the same philosophy as
*xarray* objects, so reading the *xarray* documentation is a good start to use *xdas*.

## DataArray

{py:class}`xdas.DataArray` is the base class to load and manipulate big datasets to in 
*xdas*. It is mainly composed of two attributes: 

- `data`: any N-dimensional array-like object. Compared to *xarray* `xdas.DataArray` are
more permissive to the kinds of array-like objects that can be used. In particular, 
[](virtual-datasets.md) can be used.
- `coords`: a dict-like container of coordinates. As opposed to *xarray*, which uses dense
arrays to label each point, *xdas* uses [](interpolated-coordinates.md) that provides
an efficient representation of evenly spaced data (gracefully handling gaps and small
sampling variations).

![](/_static/dataarray.svg)

Additional optional attributes can be provided:

```{warning}
The description of `dims` and `attribute` needs more clarification
```

- `dims`: for now in *xdas* you must provide a `Coordinate` per dimension. If given, 
this attribute must match the keys of the `coords` dict-like container.
- `name`: the name of the array to specify the quantity stored (e.g., `"velocity"`).
- `attribute`: a dictionary containing metadata. Note that as with *xarray*, *xdas* does not
use those metadata. It tries to keep as much as possible the information stored there 
as the `DataArray` is manipulated but it is up to the user to update information there 
if needed.

In the following examples, we use only one `DataArray`, if you have several `DataArray`s, please use the multi-file version of the opening function: {py:func}`xdas.open_mfdataarray`. You will just have to adapt the paths argument.

### Creating a DataArray

The user must provide at least an n-dimensional array with its related coordinates. See 
the related description of how to create coordinates 
[here](interpolated-coordinates.md). For example:

```{code-cell}
import numpy as np
import xdas

shape = (6000, 1000)
resolution = (np.timedelta64(10, "ms"), 5.0)
starttime = np.datetime64("2023-01-01T00:00:00")

data = np.random.randn(*shape).astype("float32")

coords={
    "time": {
        "tie_indices" :[0, shape[0] - 1],
        "tie_values" :[starttime, starttime + resolution[0] * (shape[0] - 1)],
    },
    "distance": {
        "tie_indices": [0, shape[1] - 1],
        "tie_values": [0.0, resolution[1] * (shape[1] - 1)],
    },
}

da = xdas.DataArray(data, coords)
da
```

### Reading a DataArray from a DAS file.

```{warning}
From this description, it is not clear what the default behaviour is, and why ASN/Febus are different.
What about other manufacturers?

This section needs an example.
```

*xdas* can read some DAS file format with {py:func}`xdas.open_dataarray`. *xdas* uses 
the netCDF4 format with CF conventions. ASN and Febus file can also be read. In that 
case the `engine` argument must be passed.

To learn how to read your custom DAS data format with *xdas*, please see the chapter on [](data-formats.md).

### Writing a DataArray to disk

*xdas* uses the CF conventions to write {py:class}`xdas.DataArray` to disk as netCDF4 
files. If the DataArray was generated from a netCDF4/HDF5 file and only slicing was 
performed, the DataArray can be written as a pointer to the original data using the 
`virtual` argument. See the part on [](virtual-datasets.md).

```{code-cell}
da.to_netcdf("dataarray.nc", virtual=False)
```

### Assign new coordinates to your DataArray

You can either replace the existing coordinates by new ones or assign new coordinates to a {py:class}`xdas.DataArray` and link it them an existing dimension. 

#### Replace existing coordinates

In the example below, we replace the "distance" coordinate with new ones.

```{code-cell}
da = xdas.open_dataarray("dataarray.nc")
new_distances = np.linspace(30.8, 40.9, da.shape[1])
da = da.assign_coords(distance=new_distances)
da
```

#### Add new coordinates and link them to an existing dimension

The new coordinates types can be a list of strings or numbers. In the example below, we will add the new coordinate "latitude" linked with the "distance" dimension.

```{code-cell}
da = xdas.open_dataarray("dataarray.nc")
da
```
```{code-cell}
latitudes = np.linspace(-33.90, -35.90, da.shape[1])
da = da.assign_coords(latitude=("distance", list(latitudes)))
da
```

You can also swap a dimension to one of the new coordinates.

```{code-cell}
da = da.swap_dims({"distance": "latitude"})
da
```

### Plot your DataArray

{py:class}`xdas.DataArray` includes the function {py:func}`xdas.DataArray.plot`. It uses the *xarray* way of plotting data depending on the number of dimensions your data has. You'll have to adapt the arguments and kwargs in {py:func}`xdas.DataArray.plot`.

If your {py:class}`xdas.DataArray` has one dimension, please refer to the arguments and kwargs from the 'xarray.plot.line' function.

For 2 dimensions or more, please refer to the 'xarray.plot.imshow' function.

For other, please refer to 'xarray.plot' function.