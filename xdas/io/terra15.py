"""
I/O engine for Terra15 HDF5 files (:class:`Terra15Engine`).
"""

import h5py
import pandas as pd

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import Engine


class Terra15Engine(Engine, name="terra15"):
    """Engine for reading Terra15 HDF5 files."""

    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "time": ["interpolated"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, tz="UTC"):
        """Read a Terra15 HDF5 file *fname* and return a virtual :class:`DataArray`."""
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
            data = VirtualSource(file["data_product"]["data"])
        nt, nd = data.shape
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": [ti, tf],
        }  # TODO: use from_block
        distance = Coordinate[self.ctype["distance"]].from_block(
            d0, nd, dx, dim="distance"
        )
        return DataArray(data, {"time": time, "distance": distance})
