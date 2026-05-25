"""I/O engine for MiniSEED files via ObsPy (:class:`MiniSEEDEngine`)."""

import dask
import numpy as np
import obspy

from ..coordinates.core import Coordinate, Coordinates, get_sampling_interval
from ..core.dataarray import DataArray
from ..core.routines import concat_coords
from .core import Engine


class MiniSEEDEngine(Engine, name="miniseed"):
    """Engine for reading MiniSEED files via ObsPy as lazy dask-backed DataArrays."""

    _supported_vtypes = ["dask"]
    _supported_ctypes = {
        "time": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, ignore_last_sample=False, ctype="interpolated"):
        """Return a lazy dask-backed :class:`DataArray` for the MiniSEED file *fname*."""
        shape, dtype, coords, method = self.read_header(
            fname, ignore_last_sample, ctype
        )
        data = dask.array.from_delayed(
            dask.delayed(self.read_data)(fname, method, ignore_last_sample),
            shape,
            dtype,
        )
        return DataArray(data, coords)

    def read_header(self, path, ignore_last_sample, ctype):
        """Read metadata from *path* and return ``(shape, dtype, coords, method)``."""
        st = obspy.read(path, headonly=True)

        dtype = uniquifiy(tr.data.dtype for tr in st)
        if not isinstance(dtype, np.dtype):  # pragma: no cover
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
            first_channel_stream = st.select(channel=channels[0])
            time = [
                get_time_coord(
                    tr,
                    ignore_last_sample and idx == len(first_channel_stream) - 1,
                    ctype=ctype,
                )
                for idx, tr in enumerate(first_channel_stream)
            ]
            time = concat_coords(time)
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

    def read_data(self, path, method, ignore_last_sample):
        """Load and return the raw data array from *path* using *method*."""
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


def to_stream(
    da,
    network="NET",
    station="DAS{:05}",
    location="00",
    channel="{:1}N1",
    dim={"last": "first"},
):
    """
    Convert a 2-D :class:`DataArray` to an :class:`obspy.Stream`.

    Parameters
    ----------
    da : DataArray
        2-D array with one time and one distance/channel dimension.
    network, station, location, channel : str
        SEED identifiers.  *station* and *channel* may contain ``{:...}``
        format specs that are filled with the channel index.
    dim : dict, optional
        ``{distance_dim: time_dim}`` mapping.  Defaults to ``{"last": "first"}``.

    Returns
    -------
    obspy.Stream
    """
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
    """
    Convert an :class:`obspy.Stream` to a :class:`DataArray`.

    Parameters
    ----------
    st : obspy.Stream
        Homogeneous stream (all traces must share start time and sample rate).
    dims : tuple of str, optional
        Dimension names for the output array.

    Returns
    -------
    DataArray
    """
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


def get_time_coord(tr, ignore_last_sample, ctype):
    """Build a :class:`Coordinate` for the time axis of trace *tr*."""
    t0 = np.datetime64(tr.stats.starttime)
    dt = np.rint(1e6 * tr.stats.delta).astype("m8[us]").astype("m8[ns]")
    nt = tr.stats.npts - int(ignore_last_sample)
    return Coordinate[ctype].from_block(t0, nt, dt, dim="time")


def uniquifiy(seq):
    """Return the unique elements of *seq* in order; unwrap to scalar if only one."""
    seen = set()
    seq = list(x for x in seq if x not in seen and not seen.add(x))
    if len(seq) == 1:
        return seq[0]
    else:
        return seq


def get_band_code(sampling_rate):
    """Return the SEED band code character for *sampling_rate* (Hz)."""
    band_code = ["T", "P", "R", "U", "V", "L", "M", "B", "H", "C", "F"]
    limits = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 80, 250, 1000, 5000]
    index = np.searchsorted(limits, sampling_rate, "right") - 1
    if index < 0 or index >= len(band_code):
        return "X"
    else:
        return band_code[index]
