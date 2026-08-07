"""I/O engine for the formats ObsPy reads (:class:`ObsPyEngine`).

The engine is named for the library, not for a format: decoding is
:func:`obspy.read`, so every format ObsPy supports — MiniSEED, SAC, GSE2,
SEG-2 and the rest — goes through it. It replaces
:mod:`xdas.io.miniseed`, which is kept alongside it for the views it wrote.
"""

from typing import ClassVar

import numpy as np
import obspy

from ..coordinates import Coordinate, get_sampling_interval
from ..core import DataArray, DataCollection
from ..virtual import TileArray
from .core import Engine

#: The blank location code, as FDSN spells it. ObsPy returns ``""``, which
#: cannot be a netCDF group name, and ObsPy's own FDSN client performs the
#: same mapping.
BLANK_LOCATION = "--"

#: The levels of the SEED hierarchy, outermost first.
LEVELS = ("network", "station", "location", "channel")

#: Element type each MiniSEED encoding *decodes to*, which is not the type it
#: was written from: libmseed unpacks every integer encoding to ``int32``,
#: whatever its on-disk width. Needed because ``headonly=True`` leaves
#: ``tr.data`` an empty ``float64`` array, so the decoded type cannot be read
#: off it.
MSEED_DTYPES = {
    "ASCII": np.dtype("S1"),
    "INT16": np.dtype("int32"),
    "INT32": np.dtype("int32"),
    "FLOAT32": np.dtype("float32"),
    "FLOAT64": np.dtype("float64"),
    "STEIM1": np.dtype("int32"),
    "STEIM2": np.dtype("int32"),
    "GEOSCOPE24": np.dtype("float32"),
    "GEOSCOPE16_3": np.dtype("float32"),
    "GEOSCOPE16_4": np.dtype("float32"),
    "CDSN": np.dtype("int32"),
    "SRO": np.dtype("int32"),
    "DWWSSN": np.dtype("int32"),
}


class ObsPyEngine(Engine, name="obspy"):
    """
    Engine for the file formats ObsPy reads, as lazy tile-backed data arrays.

    The engine mirrors :func:`obspy.read`: each contiguous
    :class:`obspy.Trace` becomes one lazy one-dimensional
    :class:`~xdas.DataArray`, and the collection mirrors the
    :class:`obspy.Stream`, nested on the four levels of the SEED hierarchy:

    .. code-block:: text

        network -> station -> location -> channel -> trace -> DataArray

    Merging contiguous traces, separating acquisition epochs and moving gaps
    into the time coordinate are not the engine's job:
    :func:`~xdas.combine_by_coords` does all three, and
    :func:`~xdas.open` calls it. Overlaps are resolved by
    :func:`~xdas.trim_overlaps`.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Default to "tiles".
    ctype : str or dict, optional
        The coordinate type to use for the time axis. Default to "interpolated".

    """

    _supported_vtypes: ClassVar[list] = ["tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated", "sampled", "dense"],
    }

    def open_datacollection(self, fname):
        """Return the traces of *fname* as a collection nested on the SEED hierarchy."""
        st = obspy.read(fname, headonly=True)
        # libmseed's trace-list assembly returns traces in no useful order;
        # `sort` puts them in (id, starttime) order
        st.sort()
        tree = {}
        pointers = set()
        for tr in st:
            pointer = get_pointer(tr)
            key = tuple(pointer.values())
            if key in pointers:
                raise ValueError(
                    f"{fname} holds two traces sharing every identifier and both "
                    f"time bounds ({pointer}); nothing content-free separates them"
                )
            pointers.add(key)
            branch = tree
            for level in LEVELS[:-1]:
                branch = branch.setdefault(pointer[level], {})
            branch.setdefault(pointer["channel"], []).append(
                self._open_trace(fname, tr, pointer)
            )
        return nest(tree, LEVELS)

    def open_dataarray(self, fname):
        """Return the unique trace of *fname* as a lazy tile-backed data array."""
        st = obspy.read(fname, headonly=True)
        if len(st) != 1:
            raise ValueError(
                f"{fname} holds {len(st)} traces, not one; open it with "
                "`open_datacollection` (or `open`, which combines the result)"
            )
        return self._open_trace(fname, st[0], get_pointer(st[0]))

    def _open_trace(self, fname, tr, pointer):
        """Build the lazy data array of the single trace *tr* of *fname*."""
        data = TileArray.from_tiles(
            str(fname),
            (tr.stats.npts,),
            get_dtype(tr),
            {"name": "obspy"},
            **pointer,
        )
        coords = {level: (None, pointer[level]) for level in LEVELS}
        coords["time"] = get_time_coord(tr, self.ctype["time"])
        # the id ObsPy prints, with the location code normalized as the tree
        # keys have it
        name = ".".join(pointer[level] for level in LEVELS)
        return DataArray(data, coords, name=name)

    @staticmethod
    def load_tile(
        path,
        selection,
        *,
        network,
        station,
        location,
        channel,
        starttime,
        endtime,
    ):
        """Read a source selection of *path*, decoding with :func:`obspy.read`.

        The tile is addressed by the data's own address — the four SEED
        identifiers plus both time bounds — never by a position in the stream:
        ObsPy's trace *count* comes from libmseed's segmentation policy, so an
        index would designate a different trace after any re-segmenting version
        bump, silently. The contiguous runs of the selected channel are joined
        before the span is looked up, so the pointer resolves whatever record
        boundaries the reader drew.

        Manifests written before the engine was renamed name the `"miniseed"`
        engine instead, and are decoded by
        :meth:`~xdas.io.miniseed.MiniSEEDEngine.load_tile`.
        """
        st = obspy.read(path).select(
            network=network,
            station=station,
            location="" if location == BLANK_LOCATION else location,
            channel=channel,
        )
        runs = join_contiguous(st)
        # a run may legitimately be longer than the tile — a re-segmenting
        # reader joins the same span from different records, and two traces may
        # share a start and differ in length. The smallest covering run is the
        # one the pointer named; ties mean genuine duplicates.
        covering = [
            run
            for run in runs
            if run["start"] - run["delta"] // 2 <= starttime
            and endtime <= run["end"] + run["delta"] // 2
        ]
        if covering:
            shortest = min(len(run["data"]) for run in covering)
            covering = [run for run in covering if len(run["data"]) == shortest]
        if len(covering) != 1:
            raise ValueError(
                f"{len(covering)} contiguous runs of "
                f"{network}.{station}.{location}.{channel} cover "
                f"[{np.datetime64(starttime, 'ns')}, {np.datetime64(endtime, 'ns')}] "
                f"in {path}; exactly one is required"
            )
        (run,) = covering
        offset = round((starttime - run["start"]) / run["delta"])
        npts = round((endtime - starttime) / run["delta"]) + 1
        return run["data"][offset : offset + npts][selection]


def nest(tree, levels):
    """Wrap the nested dict *tree* of trace lists into named collection levels."""
    name, *rest = levels
    if rest:
        data = {key: nest(value, rest) for key, value in tree.items()}
    else:
        # one element per ObsPy `Trace`: this is the faithful `obspy.read`
        # mirror. Once combined, each element is an acquisition epoch instead
        # and `combine_by_coords` renames the level accordingly.
        data = {key: DataCollection(value, "trace") for key, value in tree.items()}
    return DataCollection(data, name)


def get_pointer(tr):
    """Return the columns that address *tr* in its file.

    The four SEED identifiers are kept apart rather than joined into one
    ``"NET.STA.LOC.CHA"`` string so that the tile manifest folds each
    independently — a `concat` along `channel` unfolds only that field — and so
    that they map one to one onto :meth:`obspy.Stream.select`.

    ``starttime`` alone does not identify a trace: two traces can share all
    four identifiers *and* a start time and still hold different data.
    ``endtime`` completes the key. Neither is recoverable from the tile
    geometry, which describes the *view* once the array is sliced while the
    pointer must keep naming the source.
    """
    return {
        "network": tr.stats.network,
        "station": tr.stats.station,
        "location": tr.stats.location or BLANK_LOCATION,
        "channel": tr.stats.channel,
        "starttime": tr.stats.starttime.ns,
        "endtime": tr.stats.endtime.ns,
    }


def get_dtype(tr):
    """Return the element type *tr* decodes to.

    Under ``headonly=True`` ``tr.data`` is an empty ``float64`` array whatever
    the file holds, so the MiniSEED encoding is authoritative when present.
    The fallback is right for the formats that ignore ``headonly`` and decode
    fully anyway; those simply pay a slower scan.
    """
    encoding = getattr(tr.stats, "mseed", {}).get("encoding")
    if encoding in MSEED_DTYPES:
        return MSEED_DTYPES[encoding]
    return tr.data.dtype


def get_time_coord(tr, ctype):
    """Build the time :class:`~xdas.Coordinate` of trace *tr*.

    Regular by construction: ObsPy already splits a trace at its gaps.
    """
    t0 = np.datetime64(tr.stats.starttime.ns, "ns")
    dt = np.rint(1e6 * tr.stats.delta).astype("m8[us]").astype("m8[ns]")
    return Coordinate[ctype].from_block(t0, tr.stats.npts, dt, dim="time")


def join_contiguous(traces):
    """Group *traces* into sample-exact contiguous runs.

    The legitimate half of :meth:`obspy.Stream._cleanup`, implemented here
    rather than called through that private method: traces that continue each
    other to the sample are concatenated and nothing else is touched — no gap
    filling, no overlap arbitration, no masking.

    Returns
    -------
    list of dict
        One entry per run, with its ``start`` and ``end`` in nanoseconds, its
        sampling ``delta`` in nanoseconds, and its ``data``.
    """
    runs = []
    for tr in sorted(traces, key=lambda tr: (tr.stats.starttime.ns, tr.stats.npts)):
        delta = round(tr.stats.delta * 1e9)
        start = tr.stats.starttime.ns
        if runs and runs[-1]["delta"] == delta and start == runs[-1]["stop"]:
            runs[-1]["stop"] += tr.stats.npts * delta
            runs[-1]["chunks"].append(tr.data)
        else:
            runs.append(
                {
                    "start": start,
                    "stop": start + tr.stats.npts * delta,
                    "delta": delta,
                    "chunks": [tr.data],
                }
            )
    return [
        {
            "start": run["start"],
            "end": run["stop"] - run["delta"],
            "delta": run["delta"],
            "data": (
                run["chunks"][0]
                if len(run["chunks"]) == 1
                else np.concatenate(run["chunks"])
            ),
        }
        for run in runs
    ]


def to_stream(
    da,
    network="NET",
    station="DAS{:05}",
    location="00",
    channel="{:1}N1",
    dim=None,
):
    """
    Convert a 2-D :class:`DataArray` to an :class:`obspy.Stream`.

    Parameters
    ----------
    da : DataArray
        2-D array with one time and one distance/channel dimension.
    network, station, location, channel : str
        SEED identifiers.  *station* and *channel* may contain ``{:...}``
        format specs that are filled with the channel index. The blank
        location code is spelled ``"--"`` in xdas and ``""`` in ObsPy; either
        is accepted here.
    dim : dict, optional
        ``{distance_dim: time_dim}`` mapping.  Defaults to ``{"last": "first"}``.

    Returns
    -------
    obspy.Stream
    """
    if dim is None:
        dim = {"last": "first"}
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
        "location": "" if location == BLANK_LOCATION else location,
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
    # Regular by construction from the stream's own sample rate, at ns
    # resolution so a `to_stream` round trip preserves the coordinate.
    t0 = np.datetime64(st[0].stats.starttime.datetime)
    dt = np.rint(1e6 * st[0].stats.delta).astype("m8[us]").astype("m8[ns]")
    time = Coordinate["interpolated"].from_block(t0, st[0].stats.npts, dt, dim=dims[1])
    return DataArray(data, {dims[0]: channel, dims[1]: time})


def get_band_code(sampling_rate):
    """Return the SEED band code character for *sampling_rate* (Hz)."""
    band_code = ["T", "P", "R", "U", "V", "L", "M", "B", "H", "C", "F"]
    limits = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 80, 250, 1000, 5000]
    index = np.searchsorted(limits, sampling_rate, "right") - 1
    if index < 0 or index >= len(band_code):
        return "X"
    else:
        return band_code[index]
