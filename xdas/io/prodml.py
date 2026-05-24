"""
I/O engine for ProdML HDF5 files (:class:`ProdML`), also known as
OptaSense and Sintela format.
"""

import h5py
import numpy as np
import pandas as pd

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import Engine


class ProdML(Engine, name="prodml", aliases=["optasense", "sintela"]):
    """Engine for reading ProdML / OptaSense / Sintela HDF5 files."""

    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "time": ["interpolated"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, swapped_dims=False):
        """Read a ProdML HDF5 file *fname* and return a virtual :class:`DataArray`."""
        with h5py.File(fname, "r") as file:
            acquisition = file["Acquisition"]
            dx = acquisition.attrs["SpatialSamplingInterval"]
            x0 = dx * acquisition.attrs["StartLocusIndex"]
            rawdata = acquisition["Raw[0]"]["RawData"]
            tstart = (
                pd.Timestamp(rawdata.attrs["PartStartTime"].decode())
                .tz_convert("UTC")
                .tz_localize(None)
                .to_numpy()
            )
            tend = (
                pd.Timestamp(rawdata.attrs["PartEndTime"].decode())
                .tz_convert("UTC")
                .tz_localize(None)
                .to_numpy()
            )
            data = VirtualSource(rawdata)

        if swapped_dims:
            nd, nt = data.shape
        else:
            nt, nd = data.shape

        # time
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": [tstart, tend],
        }  # TODO: use from_block

        # distance
        distance = Coordinate[self.ctype["distance"]].from_block(
            x0, nd, dx, dim="distance"
        )

        coords = (
            {"distance": distance, "time": time}
            if swapped_dims
            else {"time": time, "distance": distance}
        )
        return DataArray(data, coords)
