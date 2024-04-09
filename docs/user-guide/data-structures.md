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

## DataCollection

Use {py:class}`xdas.DataCollection` when your experiment is composed of several acquisitions on a single or on multiple cables/fibers. In the following section, we will discuss how to use this functionality.

{py:class}`xdas.DataCollection` is a dict-like container of {py:class}`xdas.DataArray`. 
It is mainly used to save several `DataArray`s in a single file. Unlike 
{py:class}`xarray.Dataset`, the different `DataArray`s do not need to 
share coordinates. `DataCollection` can, for example, be useful to save a 
list of Regions of Interest (ROIs).

A {py:class}`xdas.DataCollection` can be viewed as a flexible way of organizing data. It has different hierarchical levels corresponding to your file tree. In this example below, we can see different levels for one global experiment that included several cables, with several acquisitions:
- 1rst (mapping) level: can be the "Network" code, relative to the experience name.
- 2nd (mapping) level: contains the "Node" codes relative to the places where the fibers/cables are.
- 3rd (mapping) level: concerns the fibers/cables names.
- 4th (sequence) level: is related to the number of acquisitions with changing parameters.

```{warning}
The alignment in the figure is a bit messed up
```

![](/_static/datacollection.svg)

### Case A: DataCollection as a set of DataArrays

In the example, our `DataCollection` will be a sequence of {py:class}`xdas.DataArray`.

```{code-cell}
# Reopen dataarray as virtual source
da = xdas.open_dataarray("dataarray.nc") 

# Create a DataCollection
dc = xdas.DataCollection(
    {
        "event_1": da.sel(time=slice("2023-01-01T00:00:10", "2023-01-01T00:00:20")), 
        "event_2": da.sel(time=slice("2023-01-01T00:00:40", "2023-01-01T00:00:50")),
    }
)
dc
```

If the DataArrays are opened from files (having as data a 
{py:class}`xdas.virtual.VirtualSource`) then the data collection can be saved virtually 
to minimize redundant data writing. 

```{code-cell}
# Write the DataCollection
dc.to_netcdf("datacollection.nc", virtual=True)
```

```{code-cell}
# Read a DataCollection
dc = xdas.open_datacollection("datacollection.nc")
dc
```

### Case B: DataCollection comprising a set of acquisitions

```{warning}
I think the difference between `open_mfdatatree`, `open_mfdataarray`, and `open_mfdatacollection` needs
to be made more explicit.
```

You can also create a DataCollection to gather different acquisitions on a same fiber with {py:func}`xdas.open_mfdatatree`. This function opens a directory tree structure as a data collection. 
The tree structure is described by a path descriptor provided as a string
containing placeholders. Two flavors of placeholder can be provided:

- `{field}`: this level of the tree will behave as a dict. It will use the
directory/file names as keys.
- `[field]`: this level of the tree will behave as a list. The directory/file
names are not considered (as if the placeholder was replaced by a `*`) and
files are gathered and combined as if using {py:func}`xdas.open_mfdataarray`.

Several dict placeholders with different names can be provided. They must be
followed by one or more list placeholders that must share a unique name. The
resulting data collection will be a nesting of dicts down to the lower level
which will be a list of dataarrays.

In this example, for the 19th of November 2023, our network REKA featured 2 cables (RK1 and RK2), with the RK1 cable having 3 different acquisitions and RK2 one acquisition. 

```python
# Open all your DataCollections with open_mfdatatree
paths = "/data/{network}/{cable}/20231119/proc/[acquisition].hdf5"
dc = xdas.open_mfdatatree(paths, engine='asn')
dc
```
```text
Network:
  REKA:
    Cable:
        RK1: 
        Acquisition:
            0: <xdas.DataArray (time: 54000, distance: 10000)>
            1: <xdas.DataArray (time: 10000, distance: 5000)>
            2: <xdas.DataArray (time: 9000, distance: 10000)>
        RK2: 
        Acquisition:
            0: <xdas.DataArray (time: 54000, distance: 10000)>
```
```python
# Write it as your global DataCollection in .nc with the virtual argument True
dc.to_netcdf("datacollection.nc", virtual=True)
```
```python
# Read your global DataCollection with open_datacollection
dc = open_datacollection("datacollection.nc")
dc
```
```text
Network:
  REKA:
    Cable:
        RK1: 
        Acquisition:
            0: <xdas.DataArray (time: 54000, distance: 10000)>
            1: <xdas.DataArray (time: 10000, distance: 5000)>
            2: <xdas.DataArray (time: 9000, distance: 10000)>
        RK2: 
        Acquisition:
            0: <xdas.DataArray (time: 54000, distance: 10000)>
```

If you have several DataCollections, you can gather them in one file using {py:func}`xdas.open_mfdatacollection` and write it to one single DataCollection.