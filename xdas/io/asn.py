import h5py
import numpy as np

from ..database import Database
from ..virtual import DataSource


def read(fname):
    with h5py.File(fname, "r") as file:
        header = file["header"]
        t0 = np.datetime64(round(header["time"][()] * 1e9), "ns")
        dt = np.timedelta64(round(1e9 * header["dt"][()]), "ns")
        dx = header["dx"][()] * np.median(np.diff(header["channels"]))
        data = DataSource(file["data"])
    nt, nd = data.shape
    time = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    distance = {"tie_indices": [0, nd - 1], "tie_values": [0.0, (nd - 1) * dx]}
    return Database(data, {"time": time, "distance": distance})
