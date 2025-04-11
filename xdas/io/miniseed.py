import dask
import numpy as np
import obspy

from ..core.coordinates import Coordinates, Coordinate
from ..core.dataarray import DataArray

def read(fname, ignore_last_sample=False):
    shape, dtype, coords, method = read_header(fname, ignore_last_sample)
    data = dask.array.from_delayed(dask.delayed(read_data)(fname, method, ignore_last_sample), shape, dtype)
    return DataArray(data, coords)


def read_header(path, ignore_last_sample):
    st = obspy.read(path, headonly=True)

    dtype = uniquifiy(tr.data.dtype for tr in st)
    if not isinstance(dtype, np.dtype):
        raise ValueError("All traces must have the same dtype")

    channels = [tr.stats.channel for tr in st]
    if len(st) > 1 and len(np.unique(channels)) == 1:
        method = "unsynchronized"
        time = get_time_coord(st[0], ignore_last_sample)
        for tr in st[1:]:
            time.append(get_time_coord(tr, ignore_last_sample))
    else:
        method = "synchronized"
        time = get_time_coord(st[0], ignore_last_sample)

        if not all(get_time_coord(tr, ignore_last_sample).equals(time) for tr in st):
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
    return shape, dtype, coords, method


def read_data(path, method, ignore_last_sample):
    st = obspy.read(path)
    if ignore_last_sample:
        for tr in st:
            tr.data = tr.data[:-1]
    if method == "synchronized":
        return np.array(st[0])
    else:
        data = [tr.data for tr in st]
        return np.concatenate((data), axis=1)


def get_time_coord(tr, ignore_last_sample):
    if ignore_last_sample:
        return Coordinate({
            "tie_indices": [0, tr.stats.npts - 2],
            "tie_values": [
                np.datetime64(tr.stats.starttime),
                np.datetime64(tr.stats.endtime - tr.stats.delta),
            ],
        })
    else:
        return Coordinate({
            "tie_indices": [0, tr.stats.npts - 1],
            "tie_values": [
                np.datetime64(tr.stats.starttime),
                np.datetime64(tr.stats.endtime),
            ],
        })


def uniquifiy(seq):
    seen = set()
    seq = list(x for x in seq if x not in seen and not seen.add(x))
    if len(seq) == 1:
        return seq[0]
    else:
        return seq
