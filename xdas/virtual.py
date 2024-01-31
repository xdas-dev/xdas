import os
from copy import deepcopy
from tempfile import TemporaryDirectory

import h5py
import numpy as np


class DataSource:
    """
    A lazy array object pointing toward a netCDF4/HDF5 file.
    """

    def __init__(
        self, path_or_dataset, name=None, shape=None, dtype=None, maxshape=None
    ):
        self._vsource = h5py.VirtualSource(
            path_or_dataset, name, shape, dtype, maxshape
        )
        self._sel = Selection(self._vsource.shape)

    def __getitem__(self, key):
        self = deepcopy(self)
        self._sel = self._sel.__getitem__(key)
        return self

    def __array__(self):
        return self.to_layout().__array__()

    def __repr__(self):
        return f"DataSource: {to_human(self.nbytes)} ({self.dtype})"

    @property
    def vsource(self):
        return self._vsource.__getitem__(self._sel.get_key())

    @property
    def shape(self):
        return self.vsource.shape

    @property
    def dtype(self):
        return self.vsource.dtype

    @property
    def path(self):
        return self.vsource.path

    @property
    def name(self):
        return self.vsource.name

    @property
    def id(self):
        return self.vsource.id

    @property
    def sel(self):
        return self.vsource.sel

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    @property
    def ndim(self):
        return len(self.vsource.shape)

    def to_layout(self):
        layout = DataLayout(self.shape, self.dtype)
        layout[...] = self.vsource
        return layout

    def to_dataset(self, file, name):
        self.to_layout().to_dataset(file, name)


class DataLayout(h5py.VirtualLayout):
    """
    A composite lazy array pointing toward multiple netCDF4/HDF5 files.
    """

    def __array__(self):
        with TemporaryDirectory() as tmpdirname:
            fname = os.path.join(tmpdirname, "vds.h5")
            with h5py.File(fname, "w") as file:
                dataset = file.create_virtual_dataset(
                    "__values__", self, fillvalue=np.nan
                )
            with h5py.File(fname, "r") as file:
                dataset = file["__values__"]
                out = dataset[...]
        return out

    def __repr__(self):
        return f"DataSource: {to_human(self.nbytes)} ({self.dtype})"

    def __getitem__(self, key):
        raise NotImplementedError(
            "Cannot slice DataLayout. Use `self.to_netcdf(fname, virtual=True)` to "
            "write to disk and reopen it with `xdas.open_database(fname)`"
        )

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def to_dataset(self, file, name):
        return file.create_virtual_dataset(name, self, fillvalue=np.nan)


def to_human(size):
    unit = {0: "B", 1: "KB", 2: "MB", 3: "GB", 4: "TB"}
    n = 0
    while size > 1024:
        size /= 1024
        n += 1
    return f"{size:.1f}{unit[n]}"


class Selection:
    def __init__(self, shape):
        self.shape = shape
        self.selectors = [SliceSelector(length) for length in shape]

    def __getitem__(self, key):
        sel = deepcopy(self)
        if not isinstance(key, tuple):
            key = (key,)
        dim = 0
        for k in key:
            while isinstance(sel.selectors[dim], SingleSelector):
                dim += 1
                if dim >= sel.ndim:
                    raise IndexError(
                        f"too many indices for array: array is {sel.ndim}-dimensional"
                    )
            sel.selectors[dim] = sel.selectors[dim][k]
            dim += 1
        return sel

    @property
    def ndim(self):
        return len(self.shape)

    def get_key(self):
        return tuple(selector.get_key() for selector in self.selectors)


class SingleSelector:
    def __init__(self, index):
        self.index = index

    def get_key(self):
        return self.index


class SliceSelector:
    def __init__(self, length):
        self.range = range(length)

    def __getitem__(self, key):
        sel = deepcopy(self)
        item = sel.range[key]
        if isinstance(item, int):
            return SingleSelector(item)
        else:
            sel.range = item
            return sel

    def get_key(self):
        if len(self.range) == 0:
            return slice(0)
        elif self.range.stop < 0:
            return slice(self.range.start, None, self.range.step)
        else:
            return slice(self.range.start, self.range.stop, self.range.step)
