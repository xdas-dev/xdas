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

To deal with large multi-file dataset, *xdas* uses the flexibility offered by the 
[virtual dataset](https://docs.h5py.org/en/stable/vds.html) capabilities of 
netCDF4/HDF5. A virtual dataset is a file that contains pointers towards an arbitrary number of files that 
can then be accessed seamlessly as a single, contiguous dataset. Since this is
handled by HDF5 under the hood (which is C compiled) it comes with almost no overhead. 

```{note}
Because netCDF4 are valid HDF5 files, the virtual dataset feature of HDF5 can be used 
with netCDF4 files.
```

In *xdas*, a {py:class}`VirtualSource` is a pointer towards a file, while a 
{py:class}`VirtualLayout` is table linking multiple {py:class}`VirtualSource`s. Below is an
example of a virtual dataset linking three files:

![](/_static/virtual-datasets.svg)

In most cases, users do not need to deal with this object directly. To handle individual files, multiple files, and virtual datasets, *xdas* offers the following routines:

- {py:func}`xdas.open_dataarray` is used to open a single (virtual) data file, and create a {py:class}`xdas.DataArray` object out of it.
- {py:func}`xdas.open_mfdataarray` is used to open multiple (virtual) data files at once, creating a single {py:class}`xdas.DataArray` object that can be written to disk as a single virtual data file.

```{note}
When opening a virtual dataset, this later will appear as a {py:class}`VirtualSource`. 
This is because HDF5 treats virtual dataset as regular files.
```

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
A virtual dataset can point to another virtual dataset. This can be beneficial for huge
real time dataset where new data can be linked regularly by batches. Those batches can 
then be linked in a master virtual dataset. This avoids relinking all the files. 
```

```{warning}
When loading large part of a virtual dataset, you might end up with nan values. This
normally happens when linked files are missing. But due to a 
[known limitation](https://forum.hdfgroup.org/t/virtual-datasets-and-open-file-limit/6757) 
of the HDF5 C library it can be due to the opening of too many files. Try increasing 
the number of possible file to open with the `ulimit` command. Or load smaller chunk of 
data. 
```