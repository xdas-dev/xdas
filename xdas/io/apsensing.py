import h5py
import numpy as np

from ..core.dataarray import DataArray
from ..virtual import VirtualSource


def read(fname):
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
    t = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    d = {"tie_indices": [0, nd - 1], "tie_values": [0.0, (nd - 1) * dx]}
    return DataArray(data, {"time": t, "distance": d})
