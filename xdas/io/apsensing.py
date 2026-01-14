import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import parse_ctype


def read(fname, ctype=None):
    ctype = parse_ctype(ctype)
    with h5py.File(fname, "r") as file:
        t0 = file["Metadata"]["Timestamp"][()].item().decode()
        fs = file["DAQ"]["RepetitionFrequency"][()].item()
        dx = file["ProcessingServer"]["SpatialSampling"][()].item()
        data = VirtualSource(file["DAS"])
    if t0.endswith("Z"):
        t0 = t0[:-1]
    else:
        raise NotImplementedError("Only UTC timezone is supported")
    t0 = np.datetime64(t0)
    dt = np.timedelta64(round(1e9 / fs), "ns")
    nt, nd = data.shape
    time = Coordinate[ctype["time"]].from_block(t0, nt, dt, dim="time")
    distance = Coordinate[ctype["distance"]].from_block(0.0, nd, dx, dim="distance")
    return DataArray(data, {"time": time, "distance": distance})
