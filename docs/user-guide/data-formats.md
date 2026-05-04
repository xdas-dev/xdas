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

Here below the list of formats that are currently implemented. All HDF5 based formats support native virtualization. Other formats support Dask virtualization. Please refer to the [](virtual-datasets) section. To read a them you have to specify which one you want in the `engine` argument in {py:func}`xdas.open`.

Xdas support the following DAS formats:

| Constructor       | Instrument        | `engine` argument | Virtualization    |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| AP Sensing        | DAS N5*           | `"apsensing"`     | HDF5              |
| ASN               | OptoDAS           | `"asn"`           | HDF5              |
| FEBUS             | A1                | `"febus"`         | HDF5              |
| OptaSense         | OLA, ODH*, ...    | `"optasense"`     | HDF5              |
| Silixa            | iDAS              | `"silixa"`        | Dask              |
| SINTELA           | ONYX              | `"sintela"`       | HDF5              |
| Terra15           | Treble            | `"terra15"`       | HDF5              |

It also implements its own format and support ProdML and miniSEED:

| Format            | `engine` argument | Virtualization    |   
|:-----------------:|:-----------------:|:-----------------:|           
| Xdas              | `None`            | HDF5              |
| ProdML            | `"prodml"`        | HDF5              |
| miniSEED          | `"miniseed"`      | Dask              |

```{warning}
Due to poor documentation of the various version of the Febus format, it is recommended to manually provide the required trimming and the position of the timestamps within each block. For example to trim 100 samples on both side of each block and to set the timestamp location at the center of the block for a block of 2000 samples:
`xdas.open("path.h5", engine="febus", overlaps=(100, 100), offset=1000)`
```

## Extending *xdas* with your file format

*xdas* insists on its extensibility, the power is in the hands of the users. Extending *xdas* usually consists of writing few-line-of-code-long functions. The process consists in dealing with the two main aspects of a {py:class}`xarray.DataArray`: unpacking the data and coordinates objects, eventually processing them and packing them back into a Database object. 

### Function-based solution

To add a new file format the user can specify a function that read one file and outputs a {py:class}`xarray.DataArray`. This function can then be passed as an engine keyword argument to the {py:func}`xdas.open` function. The reading function must fetch and parse the data and coordinates information. 

Adding the support for a new file format generally consists in providing the path to the data array and parsing the start time and spatial and temporal spacing as in the example below.

```{code-cell}
import h5py
import numpy as np
import xdas as xd
from xdas import DataArray
from xdas.virtual import VirtualSource

def open_dataarray(fname):
    with h5py.File(fname, "r") as file:
        t0 = np.datetime64(file["dataset"].attrs["t0"]).astype("datetime64[ms]")
        dt = np.timedelta64(int(file["dataset"].attrs["dt"]*1e3), "ms")
        dx = file["dataset"].attrs["dx"][()]
        data = VirtualSource(file["dataset"])
    nt, nx = data.shape
    t = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    x = {"tie_indices": [0, nx - 1], "tie_values": [0.0, (nx - 1) * dx]}
    return DataArray(data, {"time": t, "distance": x})

# Replace "other_format.hdf5" by the path of your file
da = xd.open("other_format.hdf5", engine=open_dataarray)
da
```

This example is for one file. For multi-file datasets please indicate the path of your files with a '*' before the file format if all your files are in the same folder or pass a list of paths.

### Class-based solution

To add support in a more complete way, you can also create your own engine by inheriting from the `xdas.io.Engine` abstract class. Note that when the class is defined, the `name` keyword argument allows to register the new engine along with the `aliases` one that is useful when several instruments share the same data format. This allows to add your engine to the `Engine._registry` and to retrieve it by doing `Engine[name]`. The `_supported_vtypes` and `_supported_ctypes` class attributes allow to determine which kind of virtualization backend and type of coordinates can be used with this file format. When you open any file, you can additionally provide the `vtype` and `ctype` keyword arguments to specify which backends to use. The `Engine` class defines the `__init__` method that checks those passed kwargs and stores in `self.vtype` and `self.ctype` the chosen backends.

```{code-cell}
from xdas.io import Engine
from xdas.coordinates import Coordinate

class MyEngine(Engine, name="my_engine", aliases=["other_engine"]):
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "distance": ["interpolated", "sampled", "dense"],
        "time": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname):
        with h5py.File(fname, "r") as file:
            t0 = np.datetime64(file["dataset"].attrs["t0"]).astype("datetime64[ms]")
            dt = np.timedelta64(int(file["dataset"].attrs["dt"]*1e3), "ms")
            x0 = file["dataset"].attrs["x0"][()]
            dx = file["dataset"].attrs["dx"][()]
            data = VirtualSource(file["dataset"])
        nt, nx = data.shape
        t = Coordinate[self.ctype["time"]].from_block(t0, nt, dt, dim="time")
        x = Coordinate[self.ctype["distance"]].from_block(x0, nx, dx, dim="distance")
        return DataArray(data, {"time": t, "distance": x})
```

Once the class is created and instanciated you can then use it :

```{code-cell}
# Replace "other_format.hdf5" by the path of your file
da = xd.open("other_format.hdf5", engine="my_engine", ctype="sampled")
da
```