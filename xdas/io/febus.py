import h5py
import numpy as np

from ..core.dataarray import DataArray
from ..core.routines import concatenate
from ..virtual import VirtualSource


def read(fname):
    """Open a febus file into a xdas DataArray"""
    with h5py.File(fname, "r") as file:
        (device_name,) = list(file.keys())
        source = file[device_name]["Source1"]
        times = np.asarray(source["time"])
        zone = source["Zone1"]
        (name,) = list(zone.keys())
        chunks = VirtualSource(zone[name])
        delta = [zone.attrs["Spacing"][1] / 1000.0, zone.attrs["Spacing"][0]]
    name = "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")
    # Detect overlap method
    if "BlockOverlap" in zone.attrs:
        noverlap = zone.attrs["BlockOverlap"][0]
    else:
        noverlap = chunks.shape[1] // 4
    chunks = chunks[:, noverlap:-noverlap, :]
    times = times + noverlap * delta[0]
    dt, dx = delta
    _, nt, nx = chunks.shape
    dc = []
    for t0, chunk in zip(times, chunks):
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": np.rint(1e9 * np.array([t0, t0 + (nt - 1) * dt])).astype(
                "M8[ns]"
            ),
        }
        distance = {"tie_indices": [0, nx - 1], "tie_values": [0.0, (nx - 1) * dx]}
        da = DataArray(chunk, {"time": time, "distance": distance}, name=name)
        dc.append(da)
    return concatenate(dc, "time")
