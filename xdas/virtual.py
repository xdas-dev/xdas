import os
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
        _vsource = h5py.VirtualSource(
            path_or_dataset, name=None, shape=None, dtype=None, maxshape=None
        )
        _slices = tuple([slice(None)] for axis in range(len(_vsource.shape)))
        self._slices = _slices
        self._vsource = _vsource

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        for k in key:
            if not isinstance(k, slice):
                raise ValueError("only slicing is allowed.")
        for axis, k in enumerate(key):
            self._slices[axis].append(k)
        return self

    def __array__(self):
        return self.to_layout().__array__()

    def __repr__(self):
        return f"DataSource: {to_human(self.nbytes)} ({self.dtype})"

    @property
    def slices(self):
        return tuple(
            combine_slices(length, *slices)
            for length, slices in zip(self._vsource.shape, self._slices)
        )

    @property
    def vsource(self):
        return self._vsource.__getitem__(self.slices)

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
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def to_dataset(self, file, name):
        return file.create_virtual_dataset(name, self, fillvalue=np.nan)


def to_human(size):
    unit = {0: "B", 1: "K", 2: "M", 3: "G", 4: "T"}
    n = 0
    while size > 1024:
        size /= 1024
        n += 1
    return f"{size:.1f}{unit[n]}"


def combine_slices(length, *slices):
    r = range(length)
    for s in slices:
        r = r[s]
    if len(r) == 0:
        return slice(0)
    elif r.stop < 0:
        return slice(r.start, None, r.step)
    else:
        return slice(r.start, r.stop, r.step)
