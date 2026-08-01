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

# Data Formats

*xdas* implements some of the more commonly used DAS data formats, but it can be extended to work with other specific formats. In this part we will cover:

- How to use *xdas* with an already implemented file format.
- How to use *xdas* with your specific data format.

## Implemented file formats

Here below the list of formats that are currently implemented. Every format supports tile virtualization; HDF5 based formats also support native HDF5 virtualization, and the tables give the backing each engine uses when you do not ask for one. Pass `vtype` to {py:func}`xdas.open` to choose the other, and see [](virtual-datasets) for how to pick. Xdas should automatically detect the correct file format. You can still specify which one you want in the `engine` argument.

Xdas support the following DAS formats:

| Constructor       | Instrument        | `engine` argument | Virtualization    | Default   |
|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:---------:|
| AP Sensing        | DAS N5*           | `"apsensing"`     | HDF5, tiles       | `hdf5`    |
| ASN               | OptoDAS           | `"asn"`           | HDF5, tiles       | `hdf5`    |
| FEBUS             | A1                | `"febus"`         | HDF5, tiles       | `tiles`   |
| OptaSense         | OLA, ODH*, ...    | `"optasense"`     | HDF5, tiles       | `hdf5`    |
| Silixa            | iDAS              | `"silixa"`        | tiles             | `tiles`   |
| SINTELA           | ONYX              | `"sintela"`       | HDF5, tiles       | `hdf5`    |
| Terra15           | Treble            | `"terra15"`       | HDF5, tiles       | `hdf5`    |

It also implements its own format and support ProdML and miniSEED:

| Format            | `engine` argument | Virtualization    | Default   |
|:-----------------:|:-----------------:|:-----------------:|:---------:|
| Xdas              | `None`            | HDF5, tiles       | `hdf5`    |
| ProdML            | `"prodml"`        | HDF5, tiles       | `hdf5`    |
| miniSEED          | `"miniseed"`      | tiles             | `tiles`   |

```{note}
A Febus file stores a stack of overlapping blocks rather than one contiguous array. The
HDF5 backing needs a separate mapping for every block, so the manifest of a Febus dataset
grows with the number of blocks as well as the number of files; a tile array needs one
tile per file whatever the block count. That is why `febus` defaults to `tiles`.
```

```{warning}
Due to poor documentation of the various version of the Febus format, it is recommended to manually provide the required trimming and the position of the timestamps within each block. For example to trim 100 samples on both side of each block and to set the timestamp location at the center of the block for a block of 2000 samples:
`xdas.open("path.h5", engine="febus", overlaps=(100, 100), offset=1000)`
```

### Engine parameters

Every open function ({py:func}`xdas.open`, {py:func}`xdas.open_dataarray`,
{py:func}`xdas.open_mfdataarray`, {py:func}`xdas.open_mfdatatree`) takes the same
engine-related arguments. `engine` selects the file format; `vtype` and `ctype`
select the virtualization backend and the coordinate types, and exist for every
engine. Some formats take additional parameters (the trimming of Febus blocks
shown above, the timezone of Terra15 timestamps, ...). Those are engine
constructor parameters: when `engine` is given by name, any extra keyword
argument is forwarded to the engine constructor; alternatively you can configure
an engine instance yourself and pass it as `engine`. The three calls below are
equivalent:

```python
import xdas as xd
from xdas.io.febus import FebusEngine

da = xd.open("path.h5", engine="febus", vtype="tiles", overlaps=(100, 100), offset=1000)
da = xd.open(
    "path.h5", engine=FebusEngine(vtype="tiles", overlaps=(100, 100), offset=1000)
)
engine = FebusEngine(vtype="tiles", overlaps=(100, 100), offset=1000)  # reusable
da = xd.open("path.h5", engine=engine)
```

Misspelled or unsupported parameters raise a `TypeError` from the engine
constructor. A configured instance is a complete specification: combining it
with `vtype`, `ctype` or extra keyword arguments raises an error. Format
auto-detection (`engine=None`) accepts `vtype` and `ctype` but no
format-specific parameters, since those require knowing the format.

## Extending *xdas* with your file format

*xdas* insists on its extensibility, the power is in the hands of the users. Extending *xdas* usually consists of writing a few-line-of-code-long engine class. The process consists in dealing with the two main aspects of a {py:class}`xarray.DataArray`: unpacking the data and coordinates objects, eventually processing them and packing them back into a Database object. 

### Writing an engine

To add a new file format, create your own engine by inheriting from the `xdas.io.Engine` abstract class. Note that when the class is defined, the `name` keyword argument allows to register the new engine along with the `aliases` one that is useful when several instruments share the same data format. This allows to add your engine to the `Engine._registry` and to retrieve it by doing `Engine[name]`. The `_supported_vtypes` and `_supported_ctypes` class attributes allow to determine which kind of virtualization backend and type of coordinates can be used with this file format. When you open any file, you can additionally provide the `vtype` and `ctype` keyword arguments to specify which backends to use. The `Engine` class defines the `__init__` method that checks those passed kwargs and stores in `self.vtype` and `self.ctype` the chosen backends. If your format needs parameters of its own, define an `__init__` taking them after `vtype` and `ctype` and calling `super().__init__(vtype, ctype)`: they then become available next to the engine name in the open functions, like the built-in ones described above.

```{code-cell}
import h5py
import numpy as np
import xdas as xd
from xdas import DataArray
from xdas.coordinates import Coordinate
from xdas.io import Engine
from xdas.virtual import VirtualSource

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

### Tile-backed engines

Beside the `hdf5` vtype shown above (an HDF5 virtual source), an engine can
offer the `tiles` vtype: `open_dataarray` then backs the data array with a lazy
{py:class}`xdas.tiles.TileArray` describing the file, and the engine implements
the decoding half as a `load_tile` static method — called once per tile
touched, with one source-local slice per axis and the manifest's engine
specification as keyword arguments, returning exactly the selected sub-box:

```{code-cell}
from xdas.tiles import TileArray

class MyTileEngine(Engine, name="my_tile_engine"):
    _supported_vtypes = ["hdf5", "tiles"]
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
            if self.vtype == "tiles":
                data = TileArray.from_tiles(
                    str(fname),
                    file["dataset"].shape,
                    file["dataset"].dtype,
                    {"name": "my_tile_engine"},
                )
            else:
                data = VirtualSource(file["dataset"])
        nt, nx = data.shape
        t = Coordinate[self.ctype["time"]].from_block(t0, nt, dt, dim="time")
        x = Coordinate[self.ctype["distance"]].from_block(x0, nx, dx, dim="distance")
        return DataArray(data, {"time": t, "distance": x})

    @staticmethod
    def load_tile(path, selection):
        with h5py.File(path, "r") as file:
            return file["dataset"][selection]
```

`load_tile` must depend only on its arguments — never on engine instance
state — so that saved tile views decode identically everywhere.

This is the backing used by default for the formats HDF5 virtual datasets cannot
serve (Silixa TDMS, MiniSEED) and for Febus, whose files hold many blocks each; it
is available on request from every other built-in engine. The order of
`_supported_vtypes` decides the default, so listing `"tiles"` first is all it
takes for a new engine to prefer it.
