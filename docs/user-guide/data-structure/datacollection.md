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

# DataCollection

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

## Case A: DataCollection as a set of DataArrays

In the example, our `DataCollection` will be a sequence of {py:class}`xdas.DataArray`.

```{code-cell}
import numpy as np
import xdas

# Reopen dataarray from previous section as a virtual source
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
dc.to_netcdf("datacollection.nc", virtual=False)
```

```{code-cell}
# Read a DataCollection
dc = xdas.open_datacollection("datacollection.nc")
dc
```

## Case B: DataCollection comprising a set of acquisitions

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

### Gather all your DataCollections

In this example, for the 19th of November 2023, our network REKA featured 2 cables (RK1 and RK2), with the RK1 cable having 3 different acquisitions and RK2 one acquisition. 

If your data paths are something like: "/data/REKA/RK1/20231119/proc/*.hdf5" and "/data/REKA/RK2/20231119/proc/*.hdf5", you can define your data path as "/data/{network}/{cable}/20231119/proc/[acquisition].hdf5". You free to choose other words to define "network" and "cable", you juste have to replace them.

```python
path = "/data/{network}/{cable}/20231119/proc/[acquisition].hdf5"
dc = xdas.open_mfdatatree(path, engine='asn')
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
dc = xdas.open_datacollection("datacollection.nc")
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

### Extend your DataCollection

You can extend your {py:class}`xdas.DataCollection` by inserting new {py:class}`xdas.DataArray` to the acquisitons list.

```python
# Read your global DataCollection with open_datacollection
dc = xdas.open_datacollection("datacollection.nc")

# Read the dataarray you want to add
da = xdas.open_dataarray("dataarray.nc")
da
```
```text
<xdas.DataArray (time: 68577, distance: 50000)>
VirtualSource: 72.5TB (float32)
Coordinates:
  * time (time): 2021-10-27T15:44:10.722 to 2021-12-03T15:45:18.419
  * distance (distance): 0.000 to 204255.953
```

```python
# Add the dataarray to the datacollection at the acquisition number 0
dc['REKA']['RK2'].insert(0, da)
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
            0: <xdas.DataArray (time: 68577, distance: 50000)>
            1: <xdas.DataArray (time: 54000, distance: 10000)>
```

You now have 2 acquisitions in your acquisitions list.