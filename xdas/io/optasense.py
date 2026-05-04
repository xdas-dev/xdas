import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import Engine


class OptaSenseEngine(Engine, name="optasense"):
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "time": ["interpolated"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, swapped_dims=False):
        with h5py.File(fname, "r") as file:
            acquisition = file["Acquisition"]
            dx = acquisition.attrs["SpatialSamplingInterval"]
            rawdata = acquisition["Raw[0]"]["RawData"]
            tstart = np.datetime64(rawdata.attrs["PartStartTime"][:-1])
            tend = np.datetime64(rawdata.attrs["PartEndTime"][:-1])
            data = VirtualSource(rawdata)
        if swapped_dims:
            nd, nt = data.shape
        else:
            nt, nd = data.shape
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": [tstart, tend],
        }  # TODO: use from_block
        distance = Coordinate[self.ctype["distance"]].from_block(
            0.0, nd, dx, dim="distance"
        )
        if swapped_dims:
            return DataArray(data, {"distance": distance, "time": time})
        else:
            return DataArray(data, {"time": time, "distance": distance})
