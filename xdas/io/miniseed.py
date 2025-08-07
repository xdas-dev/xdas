import dask
import numpy as np
import obspy

from ..core.coordinates import Coordinates, Coordinate
from ..core.dataarray import DataArray


def read(fname, ignore_last_sample=False):
    shape, dtype, coords, method = read_header(fname, ignore_last_sample)
    data = dask.array.from_delayed(
        dask.delayed(read_data)(fname, method, ignore_last_sample), shape, dtype
    )
    return DataArray(data, coords)


def read_header(path, ignore_last_sample):
    st = obspy.read(path, headonly=True)

    dtype = uniquifiy(tr.data.dtype for tr in st)
    if not isinstance(dtype, np.dtype):
        raise ValueError("All traces must have the same dtype")

    stations = [tr.stats.station for tr in st]
    channels = [tr.stats.channel for tr in st]
    starttimes = [tr.stats.starttime for tr in st]
    cond1 = (len(np.unique(stations)) == 1) & (len(st) > len(np.unique(channels)))
    cond2 = (len(np.unique(stations)) == 1) & (
        all(element == starttimes[0] for element in starttimes) == False
    )
    if cond1 or cond2:
        method = "unsynchronized"
        tmp_st = st.select(channel=channels[0])
        for n, tr in enumerate(tmp_st):
            if n == 0:
                time = get_time_coord(tr, ignore_last_sample=False)
            elif n == len(tmp_st) - 1:
                time = time.append(get_time_coord(tr, ignore_last_sample))
            else:
                time = time.append(get_time_coord(tr, ignore_last_sample=False))
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
    if method == "synchronized":
        if ignore_last_sample:
            for tr in st:
                tr.data = tr.data[:-1]
        return np.array(st)
    else:
        channels = [tr.stats.channel for tr in st]
        data = []
        for channel in np.unique(channels):
            tmp_st = st.select(channel=channel)
            channel_data = []
            for n, tr in enumerate(tmp_st):
                if ignore_last_sample and n == len(tmp_st) - 1:
                    tr.data = tr.data[:-1]
                channel_data.append(tr.data)
            data.append(np.concatenate(channel_data))
        return np.array(data)


def get_time_coord(tr, ignore_last_sample):
    if ignore_last_sample:
        return Coordinate(
            {
                "tie_indices": [0, tr.stats.npts - 2],
                "tie_values": [
                    np.datetime64(tr.stats.starttime),
                    np.datetime64(tr.stats.endtime - tr.stats.delta),
                ],
            }
        )
    else:
        return Coordinate(
            {
                "tie_indices": [0, tr.stats.npts - 1],
                "tie_values": [
                    np.datetime64(tr.stats.starttime),
                    np.datetime64(tr.stats.endtime),
                ],
            }
        )


def uniquifiy(seq):
    seen = set()
    seq = list(x for x in seq if x not in seen and not seen.add(x))
    if len(seq) == 1:
        return seq[0]
    else:
        return seq
