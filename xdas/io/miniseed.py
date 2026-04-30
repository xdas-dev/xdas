import dask
import numpy as np
import obspy

from ..coordinates.core import Coordinate, Coordinates, get_sampling_interval
from ..core.dataarray import DataArray
from .core import Engine


class MiniSEEDEngine(Engine, name="miniseed"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        return read(fname, **kwargs)


def to_stream(
    da,
    network="NET",
    station="DAS{:05}",
    location="00",
    channel="{:1}N1",
    dim={"last": "first"},
):
    dimdist, dimtime = dim.copy().popitem()
    if not da.ndim == 2:
        raise ValueError("the data array must be 2D")
    starttime = obspy.UTCDateTime(str(da[dimtime][0].values))
    delta = get_sampling_interval(da, dimtime)
    band_code = get_band_code(1.0 / delta)
    if "{" in channel and "}" in channel:
        channel = channel.format(band_code)
    header = {
        "network": network,
        "location": location,
        "channel": channel,
        "starttime": starttime,
        "delta": delta,
    }
    return obspy.Stream(
        [
            obspy.Trace(
                data=np.ascontiguousarray(da.isel({dimdist: idx}).values),
                header=header | {"station": station.format(idx + 1)},
            )
            for idx in range(len(da[dimdist]))
        ]
    )


def from_stream(st, dims=("channel", "time")):
    data = np.stack([tr.data for tr in st])
    channel = [tr.id for tr in st]
    time = {
        "tie_indices": [0, st[0].stats.npts - 1],
        "tie_values": [
            np.datetime64(st[0].stats.starttime.datetime),
            np.datetime64(st[0].stats.endtime.datetime),
        ],
    }
    return DataArray(data, {dims[0]: channel, dims[1]: time})


def read(fname, ignore_last_sample=False, ctype="interpolated"):
    shape, dtype, coords, method = read_header(fname, ignore_last_sample, ctype)
    data = dask.array.from_delayed(
        dask.delayed(read_data)(fname, method, ignore_last_sample), shape, dtype
    )
    return DataArray(data, coords)


def read_header(path, ignore_last_sample, ctype):
    st = obspy.read(path, headonly=True)

    dtype = uniquifiy(tr.data.dtype for tr in st)
    if not isinstance(dtype, np.dtype):
        raise ValueError("All traces must have the same dtype")

    stations = [tr.stats.station for tr in st]
    channels = [tr.stats.channel for tr in st]
    starttimes = [tr.stats.starttime for tr in st]
    cond1 = (len(np.unique(stations)) == 1) & (len(st) > len(np.unique(channels)))
    cond2 = (len(np.unique(stations)) == 1) & (
        not all(element == starttimes[0] for element in starttimes)
    )
    if cond1 or cond2:
        method = "unsynchronized"
        tmp_st = st.select(channel=channels[0])
        for n, tr in enumerate(tmp_st):
            if n == 0:
                time = get_time_coord(tr, ignore_last_sample=False, ctype=ctype)
            elif n == len(tmp_st) - 1:
                time = time.append(get_time_coord(tr, ignore_last_sample, ctype=ctype))
            else:
                time = time.append(
                    get_time_coord(tr, ignore_last_sample=False, ctype=ctype)
                )
    else:
        method = "synchronized"
        time = get_time_coord(st[0], ignore_last_sample, ctype)

        if not all(
            get_time_coord(tr, ignore_last_sample, ctype).equals(time) for tr in st
        ):
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


def get_time_coord(tr, ignore_last_sample, ctype):
    t0 = np.datetime64(tr.stats.starttime)
    dt = np.rint(1e6 * tr.stats.delta).astype("m8[us]").astype("m8[ns]")
    nt = tr.stats.npts - int(ignore_last_sample)
    return Coordinate[ctype].from_block(t0, nt, dt, dim="time")


def uniquifiy(seq):
    seen = set()
    seq = list(x for x in seq if x not in seen and not seen.add(x))
    if len(seq) == 1:
        return seq[0]
    else:
        return seq


def get_band_code(sampling_rate):
    band_code = ["T", "P", "R", "U", "V", "L", "M", "B", "H", "C", "F"]
    limits = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 80, 250, 1000, 5000]
    index = np.searchsorted(limits, sampling_rate, "right") - 1
    if index < 0 or index >= len(band_code):
        return "X"
    else:
        return band_code[index]
