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

# Data Formats

*xdas* implements some of the more commonly used DAS data formats, but it can be extended to work with other specific formats. In this part we will cover:

- How to use *xdas* with an already implemented file format.
- How to use *xdas* with your specific data format.

## Implemented file formats

The formats that are currently implemented are: ASN, FEBUS, OPTASENSE, SINTELA and TERRA15. To read them you have to specifiy which one you want in the `engine` argument in {py:func}`xdas.open_dataarray` for a single file or {py:func}`xdas.open_mfdataarray` for multiple files:

| DAS constructor   | `engine` argument |
|:-----------------:|:-----------------:|
| ASN               | `"asn"`           |
| FEBUS             | `"febus"`         |
| OPTASENSE         | `"optasense"`     |
| SINTELA           | `"sintela"`       |
| TERRA15           | `"terra15"`       |

## Extending *xdas* with your file format

*xdas* insists on its extensibility, the power is in the hands of the users. Extending *xdas* usually consists of writing few-line-of-code-long functions. The process consists in dealing with the two main aspects of a {py:class}`xarray.DataArray`: unpacking the data and coordinates objects, eventually processing them and packing them back into a Database object. 

To add a new file format the user can specify a function that read one file and outputs a {py:class}`xarray.DataArray`. This function can then be passed as an engine keyword argument to the {py:func}`xdas.open_dataarray` or {py:func}`xdas.open_mfdataarray` functions. The reading function must fetch and parse the data and coordinates information. 

Adding the support for a new file format generally consists in providing the path to the data array and parsing the start time and spatial and temporal spacing as in the example below.

```{code-cell}
import h5py
import numpy as np
import xdas
from xdas import DataArray
from xdas.virtual import VirtualSource

def read(fname):
  with h5py.File(fname, "r") as file:
    t0 = np.datetime64(file["dataset"].attrs['t0']).astype('datetime64[ms]')
    dt = np.timedelta64(int(file["dataset"].attrs['dt']*1e3), 'ms')
    dx = file["dataset"].attrs['dx'][()]
    data = VirtualSource(file["dataset"])
  nt, nd = data.shape
  t = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
  d = {"tie_indices": [0, nd - 1], "tie_values": [0.0, (nd - 1) * dx]}
  return DataArray(data, {"time": t, "distance": d})

# Replace "other_format.hdf5" by the path of your file
da = xdas.open_dataarray("other_format.hdf5", engine=read)
da
```

This example is for one file, please use {py:func}`xdas.open_mfdataarray`. Indicate the path of your files with a '*' before the file format if all your files are in the same folder or pass a list of paths.