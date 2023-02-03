import h5py
import numpy as np

from ..core import Coordinate, Database, DataSource


def read(fname):
    with h5py.File(fname, "r") as file:
        header = file["header"]
        t0 = np.datetime64(round(header["time"][()] * 1e6), "us")
        dt = np.timedelta64(round(1e6 * header["dt"][()]), "us")
        dx = header["dx"][()] * np.median(np.diff(header["channels"]))
        data = DataSource(file["data"])
        nt, nd = data.shape
        time = Coordinate([0, nt - 1], [t0, t0 + (nt - 1) * dt])
        distance = Coordinate([0, nd - 1], [0.0, (nd - 1) * dx])
    return Database(data, {"time": time, "distance": distance})
