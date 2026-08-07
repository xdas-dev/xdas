"""Legacy MiniSEED engine (:class:`MiniSEEDEngine`), kept for stored views.

This is the engine `engine="miniseed"` named before :mod:`xdas.io.obspy`
replaced it, preserved verbatim so that manifests written by it keep decoding
and code written against it keeps running. It describes a whole file as one
tile of stacked channels, which it classifies at scan time as *synchronized*
(all traces share one time coordinate) or *unsynchronized* (the time axis is
the concatenation of the first channel's segments, every other channel assumed
to match), and refuses anything else — a file holding two sampling rates, for
instance.

New code should use the `"obspy"` engine, which reads all of those and emits
one lazy data array per ObsPy trace. Both engines take part in format
auto-detection, `"obspy"` first: this one is reached only for a file the new
engine cannot describe as a single data array, which is exactly the shape
:func:`xdas.open_dataarray` was asked for.
"""

from typing import ClassVar

import numpy as np
import obspy

from ..coordinates import AxisCoordinate, Coordinate, Coordinates
from ..core import DataArray, concat_coords
from ..virtual import TileArray
from .core import Engine

# the stream converters and the band-code table were never miniSEED-specific
# and now live with the obspy engine; re-exported so that imports from this
# module keep working
from .obspy import from_stream, get_band_code, to_stream

__all__ = [
    "MiniSEEDEngine",
    "from_stream",
    "get_band_code",
    "get_time_coord",
    "to_stream",
    "uniquifiy",
]


class MiniSEEDEngine(Engine, name="miniseed"):
    """
    Engine for reading MiniSEED files via ObsPy as lazy tile-backed DataArrays.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Default to "tiles".
    ctype : str or dict, optional
        The coordinate type to use for the time axis. Default to "interpolated".
    ignore_last_sample : bool, optional
        Whether to drop the last sample of each contiguous segment. Useful for
        files whose last sample overlaps the first one of the next file.
        Default to False.

    """

    _supported_vtypes: ClassVar[list] = ["tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated", "sampled", "dense"],
    }

    def __init__(self, vtype=None, ctype=None, ignore_last_sample=False):
        super().__init__(vtype, ctype)
        self.ignore_last_sample = bool(ignore_last_sample)

    def open_dataarray(self, fname):
        """Return a lazy tile-backed :class:`DataArray` for the MiniSEED file *fname*."""
        shape, dtype, coords, method = self.read_header(fname)
        engine = {
            "name": "miniseed",
            "method": method,
            "ignore_last_sample": self.ignore_last_sample,
        }
        data = TileArray.from_tiles(str(fname), shape, np.dtype(dtype), engine)
        return DataArray(data, coords)

    def read_header(self, path):
        """Read metadata from *path* and return ``(shape, dtype, coords, method)``."""
        ignore_last_sample = self.ignore_last_sample
        ctype = self.ctype["time"]
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

        shape = tuple(
            len(coord) for coord in coords.values() if isinstance(coord, AxisCoordinate)
        )
        return shape, dtype, coords, method

    @staticmethod
    def read_data(path, method, ignore_last_sample):
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

    @staticmethod
    def load_tile(path, selection, *, method="synchronized", ignore_last_sample=False):
        """Read a source selection of a MiniSEED file, decoding with ObsPy.

        Decodes the whole file with ObsPy (as the legacy dask path did)
        and crops to *selection*. The decoded rank is squeezed when a
        scalar channel folded an axis out of the scanned shape.
        """
        data = MiniSEEDEngine.read_data(path, method, ignore_last_sample)
        if data.ndim > len(selection):
            data = data.reshape(data.shape[data.ndim - len(selection) :])
        return data[selection]


def get_time_coord(tr, ignore_last_sample, ctype):
    """Build a :class:`Coordinate` for the time axis of trace *tr*."""
    t0 = np.datetime64(tr.stats.starttime)
    dt = np.rint(1e6 * tr.stats.delta).astype("m8[us]").astype("m8[ns]")
    nt = tr.stats.npts - int(ignore_last_sample)
    return Coordinate[ctype].from_block(t0, nt, dt, dim="time")


def uniquifiy(seq):
    """Return the unique elements of *seq* in order; unwrap to scalar if only one."""
    seen = set()
    seq = [x for x in seq if x not in seen and not seen.add(x)]
    if len(seq) == 1:
        return seq[0]
    else:
        return seq
