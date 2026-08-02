"""I/O engine for Terra15 HDF5 files (:class:`Terra15Engine`)."""

from typing import ClassVar

import h5py
import pandas as pd

from ..coordinates import Coordinate
from ..core import DataArray
from ..tiles import TileArray
from ..virtual import VirtualSource
from .core import Engine


class Terra15Engine(Engine, name="terra15"):
    """
    Engine for reading Terra15 HDF5 files.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Default to "hdf5".
    ctype : str or dict, optional
        The coordinate type(s) to use. Default to "interpolated".
    tz : str, optional
        The timezone of the GPS timestamps stored in the file. Default to "UTC".

    """

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def __init__(self, vtype=None, ctype=None, tz="UTC"):
        super().__init__(vtype, ctype)
        self.tz = tz

    def open_dataarray(self, fname):
        """Read a Terra15 HDF5 file *fname* and return a virtual :class:`DataArray`."""
        tz = self.tz
        with h5py.File(fname, "r") as file:
            ti = (
                pd.Timestamp(file["data_product"]["gps_time"][0], unit="s", tz=tz)
                .tz_convert("UTC")
                .tz_localize(None)
                .to_numpy()
            )
            tf = (
                pd.Timestamp(file["data_product"]["gps_time"][-1], unit="s", tz=tz)
                .tz_convert("UTC")
                .tz_localize(None)
                .to_numpy()
            )
            d0 = file.attrs["sensing_range_start"]
            dx = file.attrs["dx"]
            source = file["data_product"]["data"]
            if self.vtype == "tiles":
                data = TileArray.from_tiles(
                    str(fname), source.shape, source.dtype, "terra15"
                )
            else:
                data = VirtualSource(source)
        nt, nd = data.shape
        # time (regular by declaration, rate derived from the file's own stamps)
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": [ti, tf],
            "sampling_interval": (tf - ti) / (nt - 1),
        }
        distance = Coordinate[self.ctype["distance"]].from_block(
            d0, nd, dx, dim="distance"
        )
        return DataArray(data, {"time": time, "distance": distance})

    @staticmethod
    def load_tile(path, selection):
        """Read a source selection of the data product of a Terra15 file."""
        with h5py.File(path, "r") as file:
            return file["/data_product/data"][selection]
