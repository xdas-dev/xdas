from datetime import datetime, timezone

import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource


def read(fname, tz=timezone.utc, ctype="interpolated"):
    with h5py.File(fname, "r") as file:
        ti = np.datetime64(
            datetime.fromtimestamp(file["data_product"]["gps_time"][0], tz=tz)
        ).astype("datetime64[ms]")
        tf = np.datetime64(
            datetime.fromtimestamp(file["data_product"]["gps_time"][-1], tz=tz)
        ).astype("datetime64[ms]")
        d0 = file.attrs["sensing_range_start"]
        dx = file.attrs["dx"]
        data = VirtualSource(file["data_product"]["data"])
    nt, nd = data.shape
    time = {"tie_indices": [0, nt - 1], "tie_values": [ti, tf]}  # TODO: use from_block
    ctype = Coordinate[ctype].from_block(d0, nd, dx, dim="distance")
    return DataArray(data, {"time": time, "distance": ctype})
