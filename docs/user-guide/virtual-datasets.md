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
netCDF4/HDF5. A virtual dataset is a file that contains pointers towards files that 
can then be accessed seamlessly as a unique dataset. Because behind the scene this is 
handled by HDF5 (which is C compiled) this can be done with almost no overhead. 

```{note}
Because netCDF4 are valid HDF5 files, the virtual dataset feature of HDF5 can be used 
with netCDF4 files.
```

In *xdas*, a {py:class}`VirtualSource` is a pointer toward a file while a 
{py:class}`VirtualLayout` is linking table of multiple {py:class}`VirtualLayout`. Below an
example of virtual dataset linking three files:

![](/_static/virtual-datasets.svg)

The user normally do not need to deal directly with this objects. Below an example of 
to link a multi-file dataset.

```{note}
When opening a virtual dataset, this later will appear as a {py:class}`VirtualSource`. 
This is because HDF5 treats virtual dataset as regular files.
```

## Linking multi-file datasets

The files can all be opened with the {py:func}`xdas.open_mfdataarray`:

```{code-cell}
:tags: [remove-stdout,remove-stderr]

da = xd.open_mfdataarray("00*.nc")
da
```

Then the dataarray can be written as a virtual dataset using the `virtual` argument
(otherwise the whole data will be written to disk):

```{code-cell}
da.to_netcdf("vds.nc", virtual=True)
```

It can then be read again as a usual file:

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