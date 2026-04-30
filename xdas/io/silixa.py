import dask
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from .core import Engine
from .tdms import TdmsReader


class SilixaEngine(Engine, name="silixa"):
    _supported_vtypes = ["dask"]
    _supported_ctypes = {
        "distance": ["interpolated"],
        "time": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname):
        shape, dtype, coords = self.read_header(fname)
        data = dask.array.from_delayed(
            dask.delayed(self.read_data)(fname), shape, dtype
        )
        return DataArray(data, coords)

    def read_header(self, fname):
        with TdmsReader(fname) as tdms:
            props = tdms.get_properties()
            shape = tdms.channel_length, tdms.fileinfo["n_channels"]
            dtype = tdms._data_type
        t0 = np.datetime64(props["GPSTimeStamp"])
        dt = np.timedelta64(round(1e9 / props["SamplingFrequency[Hz]"]), "ns")
        time = Coordinate[self.ctype["time"]].from_block(t0, shape[0], dt, dim="time")
        distance = {
            "tie_indices": [0, shape[1] - 1],
            "tie_values": [props["Start Distance (m)"], props["Stop Distance (m)"]],
        }  # TODO: use from_block
        coords = {"time": time, "distance": distance}
        return shape, dtype, coords

    def read_data(self, fname):
        with TdmsReader(fname) as tdms:
            data = tdms.get_data()
        return data
