"""I/O engine for Silixa TDMS files (:class:`SilixaEngine`)."""

import dask
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from .core import Engine
from .tdms import TdmsReader


class SilixaEngine(Engine, name="silixa"):
    """Engine for reading Silixa iDAS TDMS files as lazy dask-backed DataArrays."""

    _supported_vtypes = ["dask"]
    _supported_ctypes = {
        "time": ["interpolated", "sampled", "dense"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname):
        """Return a lazy dask-backed :class:`DataArray` for the TDMS file *fname*."""
        shape, dtype, coords = self.read_header(fname)
        data = dask.array.from_delayed(
            dask.delayed(self.read_data)(fname), shape, dtype
        )
        return DataArray(data, coords)

    def read_header(self, fname):
        """Read TDMS header and return ``(shape, dtype, coords)``."""
        with TdmsReader(fname) as tdms:
            props = tdms.get_properties()
            shape = tdms.channel_length, tdms.fileinfo["n_channels"]
            dtype = tdms._data_type

        # time
        t0 = np.datetime64(props["GPSTimeStamp"])
        dt = np.timedelta64(round(1e9 / props["SamplingFrequency[Hz]"]), "ns")
        time = Coordinate[self.ctype["time"]].from_block(t0, shape[0], dt, dim="time")

        # distance
        x0 = props["Start Distance (m)"]
        dx = props["Fibre Length Multiplier"] * props["SpatialResolution[m]"]
        distance = Coordinate[self.ctype["distance"]].from_block(
            x0, shape[1], dx, dim="distance"
        )

        coords = {"time": time, "distance": distance}

        return shape, dtype, coords

    def read_data(self, fname):
        """Read and return the raw data array from the TDMS file *fname*."""
        with TdmsReader(fname) as tdms:
            data = tdms.get_data()
        return data
