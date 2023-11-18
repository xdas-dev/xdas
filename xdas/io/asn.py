import h5py
import numpy as np

from ..coordinates import InterpolatedCoordinate
from ..database import Database
from ..virtual import DataSource


def read(fname):
    with h5py.File(fname, "r") as file:
        header = file["header"]
        t0 = np.datetime64(round(header["time"][()] * 1e6), "us")
        dt = np.timedelta64(round(1e6 * header["dt"][()]), "us")
        dx = header["dx"][()] * np.median(np.diff(header["channels"]))
        data = DataSource(file["data"])
    nt, nd = data.shape
    time = InterpolatedCoordinate([0, nt - 1], [t0, t0 + (nt - 1) * dt])
    distance = InterpolatedCoordinate([0, nd - 1], [0.0, (nd - 1) * dx])
    return Database(data, {"time": time, "distance": distance})
