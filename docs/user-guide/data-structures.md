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

*xdas* mainly uses two kind of data structures. The base kind of structure are the 
{py:class}`xarray.DataArray` and {py:class}`xdas.DataArray` object. Those are 
N-dimensional labeled array object. The second kind of structure are the 
{py:class}`xarray.Dataset` and {py:class}`xdas.DataCollection` objects that are 
mappings of DataArray and DataArray objects respectively. The *xdas* objects extend 
their equivalent *xarray* counterpart in term of huge multi-file datasets handling but 
at the price of a limited subset of the *xarray* API. When manipulating data that are 
small enough to be stored in-memory, the full capabilities of *xarray* objects are 
generally preferred. In a near future, we hope that the extra functionalities provided 
by the *xdas* objects will be part of the *xarray* library to enjoy the best of both 
words.

The *xarray* objects structure description can be found in the 
[xarray documentation](https://docs.xarray.dev/en/stable/user-guide/data-structures.html). 
Here a focus on the *xdas* objects are presented. They follow the same philosophy than
*xarray* objects so reading the *xarray* documentation is a good start to use *xdas*.

## DataArray

{py:class}`xdas.DataArray` is the base class to load and manipulate big datasets to in 
*xdas*. It is mainly composed of two attributes: 

- `data`: any N-dimensional array-like object. Compared to *xarray* `xdas.DataArray` are
more flexible on the kind of array-like object that can be used. In particular, 
[](virtual-datasets.md) can be used.
- `coords`: a dict-like container of coordinates. Instead of *xarray* that uses dense
arrays to label each point, *xdas* uses [](interpolated-coordinates.md) that provides
an efficient representation of evenly spaced data (with eventual gaps and small
sampling variations). 

![](/_static/dataarray.svg)

Additional optional attributes can be provided:

- `dims`: For now in *xdas* you must provide a `Coordinate` per dimension. If given, 
this attribute must match the keys of the `coords` dict-like container.
- `name`: The name of the array. Can explicit the quantity stored (e.g., `"velocity"`).
- `attribute`: a dictionary containing metadata. Note that as *xarray*, *xdas* does not
use those metadata. It tries to keep as much as possible the information stored there 
as the `DataArray` is manipulated but it is up to the user to update information there 
if needed.



### Creating a DataArray

The user must at least provide a n-dimensional array with its related coordinates. See 
the related description of how to create coordinates 
[here](interpolated-coordinates.md). Bellow an example :

```{code-cell}
import numpy as np
import xdas as xd

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

da = xd.DataArray(data, coords)
da
```

### Reading a DataArray from a DAS file.

*xdas* can read some DAS file format with {py:func}`xdas.open_dataarray`. *xdas* use 
netCDF4 format with CF conventions. ASN and Febus file can also be read. In that 
case the `engine` argument must be passed.

To know how to read your DAS data format with *xdas*, please see the part on [](data-formats.md).

### Writing a DataArray to disk

*xdas* uses the CF conventions to write {py:class}`xdas.DataArray` to disk as netCDF4 
files. If the dataarray was generated from a netCDF4/HDF5 file and only slicing was 
perform, the dataarray can be written as a pointer to the original data using the 
`virtual` argument. See the part on [](virtual-datasets.md).

```{code-cell}
da.to_netcdf("dataarray.nc", virtual=False)
```

## DataCollection

Use {py:class}`xdas.DataCollection` when your experiment is composed of different acquisitions on a single or several cables/fibers. In this section, you will see how to use this functionality.

{py:class}`xdas.DataCollection` is a dict-like container of {py:class}`xdas.DataArray`. 
It is mainly used to save several {py:class}`xdas.DataArray` in a unique file. Unlike 
{py:class}`xarray.Dataset`, the different {py:class}`xdas.DataArray` do not need to 
share coordinates. {py:class}`xdas.DataCollection` can for example useful to save a 
list of zone of interest. 

A {py:class}`xdas.DataCollection` can be view as a flexible way of oragnizing data. It has different levels corresponding to your file tree. In this example below, we can see different levels for one global experiment that have several cables, with several acquisitions:
- 1rst (mapping) level: can be the "Newtork" code, relative to the experience name.
- 2nd (mapping) level: contains the "Node" codes relative to the places where the fibers/cables are.
- 3rd (mapping) level: concerns the fibers/cables names.
- 4th (sequence) level: is related to the number of acquisitions with changing parameters.

![](/_static/datacollection.svg)

### Case A: DataCollection as a set of DataArray

In the example below, our DataCollection will be a sequence of 2 {py:class}`xdas.DataArray`.

```{code-cell}
# Reopen dataarray as virtual source
da = xd.open_dataarray("dataarray.nc") 

# Create a DataCollection
dc = xd.DataCollection(
    {
        "event_1": da.sel(time=slice("2023-01-01T00:00:10", "2023-01-01T00:00:20")), 
        "event_2": da.sel(time=slice("2023-01-01T00:00:40", "2023-01-01T00:00:50")),
    }
)
dc
```

If the dataarrays are opened from files (having as data a 
{py:class}`xdas.virtual.VirtualSource`) then the data collection can be saved virtually 
to minimize redundant data writing. 

```{code-cell}
# Write the DataCollection
dc.to_netcdf("datacollection.nc", virtual=True)
```

```{code-cell}
# Read a DataCollection
dc = xd.open_datacollection("datacollection.nc")
dc
```

### Case B: DataCollection as handelling a complex network of acquisitions

In the next example, we have several acquisitions for the same fiber so we will use {py:fn}`xdas.open_mfdatatree` to create our DataCollection.

```{code-cell}
# Open your DataArrays with open_mfdatatree as virtual source
# Write them as a DataCollection
# Read the DataCollection
```

If you have several {py:class}`xdas.DataCollection`, you can gather them in one file following this example:

```{code-cell}
# Open all your DataCollections with open_mfdatacollection
# Write it as your global DataCollection in .nc
# Read your global DataCollection with open_datacollection
```

```python
import xdas
# Open all your DataCollections with open_mfdatacollection
# Write it as your global DataCollection in .nc
# Read your global DataCollection with open_datacollection
```
```text
import xdas
# Open all your DataCollections with open_mfdatacollection
# Write it as your global DataCollection in .nc
# Read your global DataCollection with open_datacollection
```