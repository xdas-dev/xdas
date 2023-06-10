import os
from tempfile import TemporaryDirectory

import h5py
import numpy as np


class DataSource(h5py.VirtualSource):
    """
    A lazy array object pointing toward a netCDF4/HDF5 file.
    """

    def __array__(self):
        return self.to_layout().__array__()

    def __repr__(self):
        return f"DataSource: {to_human(self.nbytes)} ({self.dtype})"

    @property
    def nbytes(self):
        return np.prod(self.shape) * self.dtype.itemsize

    def to_layout(self):
        layout = DataLayout(self.shape, self.dtype)
        layout[...] = self
        return layout

    def to_dataset(self, file, name):
        self.to_layout().to_dataset(self, file, name)

    def to_dict(self):
        return {
            "path": self.path,
            "name": self.name,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "maxshape": self.maxshape,
            "sel": self.sel._sel,
        }

    @classmethod
    def from_dict(cls, dtc):
        vsource = cls(
            dtc["path"], dtc["name"], dtc["shape"], dtc["dtype"], dtc["maxshape"]
        )
        vsource.sel._sel = dtc["sel"]
        return vsource


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
