import h5py
import numpy as np

from ..core.database import Database
from ..virtual import DataSource


def read(fname):
    with h5py.File(fname, "r") as file:
        header = file["header"]
        t0 = np.datetime64(round(header["time"][()] * 1e9), "ns")
        dt = np.timedelta64(round(1e9 * header["dt"][()]), "ns")
        dx = header["dx"][()] * np.median(np.diff(header["channels"]))
        data = DataSource(file["data"])
    nt, nx = data.shape
    time = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    distance = {"tie_indices": [0, nx - 1], "tie_values": [0.0, (nx - 1) * dx]}
    return Database(data, {"time": time, "distance": distance})
