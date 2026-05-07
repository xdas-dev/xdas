import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import Engine


class ProdML(Engine, name="prodml", aliases=["optasense", "sintela"]):
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "time": ["interpolated"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, swapped_dims=False):
        with h5py.File(fname, "r") as file:
            acquisition = file["Acquisition"]
            dx = acquisition.attrs["SpatialSamplingInterval"]
            x0 = dx * acquisition.attrs["StartLocusIndex"]
            rawdata = acquisition["Raw[0]"]["RawData"]
            tstart = np.datetime64(
                rawdata.attrs["PartStartTime"].decode().split("+")[0]
            )
            tend = np.datetime64(rawdata.attrs["PartEndTime"].decode().split("+")[0])
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
