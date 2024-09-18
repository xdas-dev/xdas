---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
import xdas as xd
os.chdir("../_data")
```

# Virtual Datasets

To deal with large multi-file dataset, *Xdas* uses the concept of virtual datasets. A virtual dataset is a file that contains pointers towards an arbitrary number of files that can then be accessed seamlessly as a single, contiguous dataset. 

*Xdas* uses two types of virtualization:

- For HDF5 based format, it leverages the performance offered by the [virtual datasets](https://docs.h5py.org/en/stable/vds.html) native capabilities of netCDF4/HDF5 which comes with almost no overhead (C compiled).
- For other type of files, it leverage the flexibility of [Dask arrays](https://docs.Dask.org/en/stable/array.html).

## HDF5 Virtualization

```{note}
Because netCDF4 are valid HDF5 files, the virtual dataset feature of HDF5 can be used with netCDF4 files.
```

In *xdas*, a {py:class}`VirtualSource` is a pointer towards a file, while a {py:class}`VirtualLayout` is table linking multiple {py:class}`VirtualSource`s. Below is an example of a virtual dataset linking three files:

![](/_static/virtual-datasets.svg)

In most cases, users do not need to deal with this object directly. 

```{note}
When opening a virtual dataset, this later will appear as a {py:class}`VirtualSource`. This is because HDF5 treats virtual dataset as regular files.
```

## Use cases

To handle individual files, multiple files, and virtual datasets, *xdas* offers the following routines:

| Function                             | Output                           | Description                                                                 |
|--------------------------------------|----------------------------------|-----------------------------------------------------------------------------|
| {py:func}`xdas.open_dataarray`       | {py:class}`~xdas.DataArray`      | Open a (virtual) file.                                               |
| {py:func}`xdas.open_mfdataarray`     | {py:class}`~xdas.DataArray`      | Open multiple (virtual) files and concatenate them.                         |
| {py:func}`xdas.open_mfdatacollection`| {py:class}`~xdas.DataCollection` | Open multiple (virtual) files, grouping and concatenating compatible files. |
| {py:func}`xdas.open_mfdatatree`      | {py:class}`~xdas.DataCollection` | Open a directory tree of files, organizing data in a data collection.       |
| {py:func}`xdas.open_datacollection`  | {py:class}`~xdas.DataCollection` | Open a (virtual) collection.                                         |

Please refer to the [](data-structure/datacollection.md) section for the functions that return a data collection.

## Linking multi-file datasets

Multiple physical data files can be opened simultaneously with the {py:func}`xdas.open_mfdataarray`:

```{code-cell}
:tags: [remove-stdout,remove-stderr]

da = xd.open_mfdataarray("00*.nc")
da
```

Here, `*` is a wildcard operator. `open_mfdataarray` only creates file handles and loads basic metadata, but does not directly load the underlying DAS data in memory. Hence this method can open an arbitrary number
of files with no concern over memory allocation. Next, the DataArray can be written to disk as a single dataset. The `virtual` argument ensures that only the pointers to the original data files are written to disk (otherwise the whole data set will be written to disk):

```{code-cell}
da.to_netcdf("vds.nc", virtual=True)
```

It can then be read again as a single file using {py:func}`xdas.open_dataarray`:

```{code-cell}
xd.open_dataarray("vds.nc")
```

```{hint}
A virtual dataset can point to another virtual dataset. This can be beneficial for huge real time dataset where new data can be linked regularly by batches. Those batches can then be linked in a master virtual dataset. This avoids relinking all the files. 
```

```{warning}
When loading large part of a virtual dataset, you might end up with nan values. This normally happens when linked files are missing. But due to a [known limitation](https://forum.hdfgroup.org/t/virtual-datasets-and-open-file-limit/6757) of the HDF5 C library it can be due to the opening of too many files. Try increasing the number of possible file to open with the `ulimit` command. Or load smaller chunk of data. 
```

## Dask Virtualization

Other type of formats will be loaded as Dask arrays. Those latter are a N-dimensional stack of chunks. At each chunk is associated a task to complete to get the values of that chunk. It results in a computation graph that Xdas is capable to serialize and store within its native NetCDF format. To be able to serialize the graph, it must only contain xdas or Dask functions. 

From an user point of view the use of this type of virtualization is very similar to HDF5 one. 

The main difference is that when opening a dataset with Dask virtualization, the entire graph of pointers to the files is loaded, can be modified and saved again. In the HDF5 case, opening a virtual dataset is handled the same way as if it is a regular file meaning that the underlying mapping of pointers is hidden and cannot be modified. Dask graph can be slow when they start to become very big (more than one million tasks).
