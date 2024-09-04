from datetime import UTC, datetime

import h5py
import numpy as np

from ..core.dataarray import DataArray
from ..virtual import VirtualSource


def read(fname):
    with h5py.File(fname, "r") as file:
        ti = np.datetime64(
            datetime.fromtimestamp(file["data_product"]["gps_time"][0]), UTC
        ).astype("datetime64[ms]")
        tf = np.datetime64(
            datetime.fromtimestamp(file["data_product"]["gps_time"][-1]), UTC
        ).astype("datetime64[ms]")
        d0 = file.attrs["sensing_range_start"]
        dx = file.attrs["dx"]
        data = VirtualSource(file["data_product"]["data"])
    nt, nd = data.shape
    t = {"tie_indices": [0, nt - 1], "tie_values": [ti, tf]}
    d = {"tie_indices": [0, nd - 1], "tie_values": [d0, d0 + (nd - 1) * dx]}
    return DataArray(data, {"time": t, "distance": d})
