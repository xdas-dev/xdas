import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import Engine


class SintelaEngine(Engine, name="sintela"):
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "distance": ["interpolated", "sampled", "dense"],
        "time": ["interpolated"],
    }

    def open_dataarray(self, fname):
        with h5py.File(fname, "r") as file:
            acquisition = file["Acquisition"]
            dx = acquisition.attrs["SpatialSamplingInterval"]
            rawdata = acquisition["Raw[0]"]["RawData"]
            tstart = np.datetime64(
                rawdata.attrs["PartStartTime"].decode().split("+")[0]
            )
            tend = np.datetime64(rawdata.attrs["PartEndTime"].decode().split("+")[0])
            data = VirtualSource(rawdata)
        nt, nd = data.shape
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": [tstart, tend],
        }  # TODO: use from_block
        distance = Coordinate[self.ctype["distance"]].from_block(
            0.0, nd, dx, dim="distance"
        )
        return DataArray(data, {"time": time, "distance": distance})
