"""I/O engine for Aragon Photonics HDAS HDF5 files (:class:`AragonEngine`)."""

from typing import ClassVar

import h5py
import numpy as np
from xinterp import simplify_points

from ..coordinates import Coordinate
from ..core import DataArray
from ..virtual import TileArray, VirtualSource
from .core import Engine

_DATA = "StrainRate/StrainRate_Data"
_TIME = "Timestamps/Timestamps_Data"
_HEADER = "Header/Header_Data"


class AragonEngine(Engine, name="aragon"):
    """Engine for Aragon Photonics HDAS 3.0 ``HDAS_StrainRate`` HDF5 files.

    Samples are returned as stored (no nanostrain conversion). The time axis
    keeps the device's per-sample stamps: those their neighbours reproduce
    within the float64 resolution of the stamps (about 240 ns) are dropped, the
    rest become the tie points. The default ``interpolated`` time axis declares
    the header's nominal sampling interval with the smallest tolerance that the
    stamps allow (milliseconds on NTP-locked files). ``ctype={"time":
    "sampled"}`` steps at the header interval from each tie point instead, and
    ``ctype={"time": "dense"}`` keeps every stamp. The distance axis comes from
    the header's processed-fiber start point and spatial sampling.

    When several files are opened together, the declared tolerance is also the
    budget used to thin the tie points: values may then move by up to that
    tolerance (the sampling interval is kept). Pass ``tolerance=0`` to
    :func:`xdas.open` to keep the exact stamps. A one-sample gap cannot be told
    apart from clock jitter in the stamps.
    """

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated", "sampled", "dense"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname):
        """Read an Aragon HDAS HDF5 file *fname* into a virtual :class:`DataArray`."""
        with h5py.File(fname, "r") as file:
            for path in (_DATA, _TIME, _HEADER):
                if path not in file:
                    raise NotImplementedError(
                        f"{fname} is not an HDAS file (no '/{path}')"
                    )
            h = np.asarray(file[_HEADER], dtype="f8").reshape(-1)
            if round(h[199]) != 1:
                raise ValueError(f"{fname} is not an HDAS 3.0 file")
            if round(h[129]) != 0:
                raise NotImplementedError(
                    "the aragon engine only reads HDAS StrainRate"
                )
            timestamps = np.asarray(file[_TIME], dtype="f8").reshape(-1)
            source = file[_DATA]
            if self.vtype == "tiles":
                data = TileArray.from_tiles(
                    str(fname), source.shape, source.dtype, "aragon"
                )
            else:
                data = VirtualSource(source)

        nt, nd = data.shape
        if timestamps.size != nt:
            raise ValueError(f"{fname}: {timestamps.size} timestamps for {nt} samples")
        if self.ctype["time"] == "dense":
            time = Coordinate["dense"](_to_datetime(timestamps), dim="time")
        else:
            # drop the stamps their neighbours reproduce within the float64
            # resolution of the stamps (rounding noise only)
            index = np.arange(nt)
            keep = simplify_points(
                index, timestamps, np.spacing(np.abs(timestamps).max()), 1.0
            )
            tie_indices, tie_values = index[keep], _to_datetime(timestamps[keep])
            dt_ns = round(h[49] * h[76] * h[101] / h[1] * 1e9)
            if self.ctype["time"] == "interpolated":
                # smallest tolerance that lets the header dt validate on every
                # continuous segment (adjacent-index tie pairs are discontinuities)
                span = np.diff(tie_indices)
                drift = (
                    np.diff(tie_values.view("i8"))[span > 1] - dt_ns * span[span > 1]
                )
                tolerance = (int(np.abs(drift).max()) + 1) // 2 if drift.size else 0
                time = Coordinate["interpolated"](
                    {
                        "tie_indices": tie_indices,
                        "tie_values": tie_values,
                        "sampling_interval": np.timedelta64(dt_ns, "ns"),
                        "tolerance": np.timedelta64(tolerance, "ns"),
                    },
                    dim="time",
                )
            else:
                # one segment per tie, the last tie closing the final segment
                starts = tie_indices[:-1] if tie_indices.size > 1 else tie_indices
                time = Coordinate["sampled"](
                    {
                        "tie_values": tie_values[: starts.size],
                        "tie_lengths": np.diff(np.r_[starts, nt]),
                        "sampling_interval": np.timedelta64(dt_ns, "ns"),
                    },
                    dim="time",
                )

        distance = Coordinate[self.ctype["distance"]].from_block(
            float(h[72]), nd, float(h[44]), dim="distance"
        )
        return DataArray(data, {"time": time, "distance": distance})

    @staticmethod
    def load_tile(path, selection):
        """Read a source selection of the ``/StrainRate/StrainRate_Data`` dataset."""
        with h5py.File(path, "r") as file:
            return file[_DATA][selection]


def _to_datetime(seconds):
    """Round float64 epoch *seconds* to ``datetime64[ns]``."""
    return (seconds * 1e9).round().astype("int64").astype("datetime64[ns]")
