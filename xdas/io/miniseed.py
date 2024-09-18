import dask
import numpy as np
import obspy

from ..core.coordinates import Coordinates
from ..core.dataarray import DataArray


def read(fname):
    shape, dtype, coords = read_header(fname)
    data = dask.array.from_delayed(dask.delayed(read_data)(fname), shape, dtype)
    return DataArray(data, coords)


def read_header(path):
    st = obspy.read(path, headonly=True)

    dtype = uniquifiy(tr.data.dtype for tr in st)
    if not isinstance(dtype, np.dtype):
        raise ValueError("All traces must have the same dtype")

    time = get_time_coord(st[0])
    if not all(get_time_coord(tr) == time for tr in st):
        raise ValueError("All traces must be synchronized")

    network = uniquifiy(tr.stats.network for tr in st)
    stations = uniquifiy(tr.stats.station for tr in st)
    locations = uniquifiy(tr.stats.location for tr in st)
    channels = uniquifiy(tr.stats.channel for tr in st)

    coords = Coordinates(
        {
            "network": network,
            "station": stations,
            "location": locations,
            "channel": channels,
            "time": time,
        }
    )

    shape = tuple(len(coord) for coord in coords.values() if not coord.isscalar())
    return shape, dtype, coords


def read_data(path):
    st = obspy.read(path)
    return np.array(st)


def get_time_coord(tr):
    return {
        "tie_indices": [0, tr.stats.npts - 1],
        "tie_values": [
            np.datetime64(tr.stats.starttime),
            np.datetime64(tr.stats.endtime),
        ],
    }


def uniquifiy(seq):
    seen = set()
    seq = list(x for x in seq if x not in seen and not seen.add(x))
    if len(seq) == 1:
        return seq[0]
    else:
        return seq
