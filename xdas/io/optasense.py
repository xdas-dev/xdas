import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import parse_ctype


def read(fname, ctype=None):
    ctype = parse_ctype(ctype)
    with h5py.File(fname, "r") as file:
        acquisition = file["Acquisition"]
        dx = acquisition.attrs["SpatialSamplingInterval"]
        rawdata = acquisition["Raw[0]"]["RawData"]
        tstart = np.datetime64(rawdata.attrs["PartStartTime"][:-1])
        tend = np.datetime64(rawdata.attrs["PartEndTime"][:-1])
        data = VirtualSource(rawdata)
    nd, nt = data.shape
    time = {
        "tie_indices": [0, nt - 1],
        "tie_values": [tstart, tend],
    }  # TODO: use from_block
    distance = Coordinate[ctype["distance"]].from_block(0.0, nd, dx, dim="distance")
    return DataArray(data, {"distance": distance, "time": time})
