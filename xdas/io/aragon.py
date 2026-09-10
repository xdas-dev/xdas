"""I/O engine for Aragon Photonics HDAS HDF5 files (:class:`AragonEngine`)."""

import warnings
from typing import ClassVar

import h5py
import numpy as np

from ..coordinates import Coordinate
from ..core import DataArray
from ..virtual import TileArray, VirtualSource
from .core import Engine

_DATA = "StrainRate/StrainRate_Data"
_TIME = "Timestamps/Timestamps_Data"
_HEADER = "Header/Header_Data"


class AragonEngine(Engine, name="aragon"):
    """Engine for Aragon Photonics HDAS 3.0 ``HDAS_StrainRate`` HDF5 files.

    Samples are returned as stored (no nanostrain conversion). The regular time
    axis steps at the header's nominal sampling interval, anchored to minimise
    the largest departure of a stamp from the grid; that departure is the
    coordinate tolerance, and a value over half a sample (a data gap, or a
    partly-filled file) triggers a warning. ``ctype={"time": "dense"}`` keeps the
    raw per-sample stamps instead. The distance axis comes from the header's
    processed-fiber start point and spatial sampling.
    """

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated", "dense"],
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
        stamps = (timestamps * 1e9).round().astype("int64")

        if self.ctype["time"] == "dense":
            time = Coordinate["dense"](stamps.astype("datetime64[ns]"), dim="time")
        else:
            # regular grid at the header's exact rate, anchored on the residual
            # midpoint so its tolerance is the smallest departure it must allow
            dt_ns = round(h[49] * h[76] * h[101] / h[1] * 1e9)
            residual = stamps - dt_ns * np.arange(nt)
            spread = int(residual.max() - residual.min())
            t0 = np.datetime64(int(residual.min()) + spread // 2, "ns")
            tolerance = np.timedelta64((spread + 1) // 2, "ns")
            if spread > dt_ns:
                warnings.warn(
                    f"{fname}: time stamps depart from the {dt_ns / 1e6:g} ms grid "
                    f"by {tolerance / np.timedelta64(1, 'ms'):.1f} ms; the file may "
                    f"have a gap. Use ctype={{'time': 'dense'}} for the raw stamps.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            time = Coordinate["interpolated"].from_block(
                t0, nt, np.timedelta64(dt_ns, "ns"), dim="time", tolerance=tolerance
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
