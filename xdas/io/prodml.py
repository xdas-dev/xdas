"""
I/O engine for ProdML HDF5 files (:class:`ProdML`).

Also known as OptaSense and Sintela format.
"""

from typing import ClassVar

import h5py
import pandas as pd

from ..coordinates import Coordinate
from ..core import DataArray
from ..tiles import TileArray
from ..virtual import VirtualSource
from .core import Engine

_RAWDATA = "/Acquisition/Raw[0]/RawData"


class ProdML(Engine, name="prodml", aliases=["optasense", "sintela"]):
    """
    Engine for reading ProdML / OptaSense / Sintela HDF5 files.

    Parameters
    ----------
    vtype : str, optional
        The virtualization type to use. Default to "hdf5".
    ctype : str or dict, optional
        The coordinate type(s) to use. Default to "interpolated".
    swapped_dims : bool, optional
        Whether the on-disk array is (distance, time) instead of the usual
        (time, distance). Default to False.

    """

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def __init__(self, vtype=None, ctype=None, swapped_dims=False):
        super().__init__(vtype, ctype)
        self.swapped_dims = bool(swapped_dims)

    def open_dataarray(self, fname):
        """Read a ProdML HDF5 file *fname* and return a virtual :class:`DataArray`."""
        swapped_dims = self.swapped_dims
        with h5py.File(fname, "r") as file:
            acquisition = file["Acquisition"]
            dx = acquisition.attrs["SpatialSamplingInterval"]
            x0 = dx * acquisition.attrs["StartLocusIndex"]
            rawdata = acquisition["Raw[0]"]["RawData"]
            tstart = (
                pd.Timestamp(rawdata.attrs["PartStartTime"].decode())
                .tz_convert("UTC")
                .tz_localize(None)
                .to_numpy()
            )
            tend = (
                pd.Timestamp(rawdata.attrs["PartEndTime"].decode())
                .tz_convert("UTC")
                .tz_localize(None)
                .to_numpy()
            )
            if self.vtype == "tiles":
                # the manifest keeps the on-disk layout, whichever way the
                # dims are labeled, so the spec needs no `transpose`
                data = TileArray.from_tiles(
                    str(fname), rawdata.shape, {"name": "prodml"}, rawdata.dtype
                )
            else:
                data = VirtualSource(rawdata)

        if swapped_dims:
            nd, nt = data.shape
        else:
            nt, nd = data.shape

        # time (regular by declaration, rate derived from the file's own stamps)
        time = {
            "tie_indices": [0, nt - 1],
            "tie_values": [tstart, tend],
            "sampling_interval": (tend - tstart) / (nt - 1),
        }

        # distance
        distance = Coordinate[self.ctype["distance"]].from_block(
            x0, nd, dx, dim="distance"
        )

        coords = (
            {"distance": distance, "time": time}
            if swapped_dims
            else {"time": time, "distance": distance}
        )
        return DataArray(data, coords)

    @staticmethod
    def load_tile(path, selection, *, transpose=False):
        """Read a source selection of the raw data of a ProdML file.

        With ``transpose`` the on-disk layout is distance-major
        ``(distance, time)``; rows are then columns on disk and are
        transposed on the way out.
        """
        with h5py.File(path, "r") as file:
            data = file[_RAWDATA]
            if transpose:
                return data[selection[1], selection[0]].T
            return data[selection]
