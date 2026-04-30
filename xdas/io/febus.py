import warnings

import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..core.routines import concatenate
from ..virtual import VirtualSource
from .core import Engine


class FebusEngine(Engine, name="febus"):
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "distance": ["interpolated", "sampled", "dense"],
        "time": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname, overlaps=None, offset=None):
        """
        Open a Febus file into a xdas DataArray object.

        The Febus file format contains a 3D array which is a stack of 2D (time, distance)
        chunks of data that overlaps with each other. The overlaps must be trimmed and the
        chunks concatenated to form a seamless dataset. Each chunk is associated with a
        timestamp that is located at a fixed offset from the beginning of the chunk.

        Because of poor documentation of the evolution of the Febus file format, it is
        recommended to manually specify the overlap and offset parameters. If not provided,
        the function will attempt to determine the correct values at your own risk.

        Parameters:
        -----------
        fname : str
            The filename of the Febus file to read.
        overlaps : tuple of int, optional
            A tuple specifying the overlap in number of sample to trim on both side of each
            chunk of the data. If not provided, the function will attempt to determine the
            correct overlap at your own risk.
        offset : int, optional
            The location of the timestamp within each block given as the number of samples
            from the beginning. If not provided, the function will attempt to determine the
            correct offset at you own risk.

        Returns:
        --------
        DataArray
            A data array containing the data from the Febus file.

        """
        with h5py.File(fname, "r") as file:
            (device_name,) = list(file.keys())
            source = file[device_name]["Source1"]
            times = np.asarray(source["time"])
            zone = source["Zone1"]
            if "BlockRate" in zone.attrs:
                blockrate = zone.attrs["BlockRate"][0] / 1000.0
            elif "FreqRes" in zone.attrs:
                blockrate = zone.attrs["FreqRes"][0] / 1000.0
            else:
                raise KeyError(
                    "Could not find the block size, please check file header"
                )
            (name,) = list(zone.keys())
            chunks = VirtualSource(zone[name])
            delta = (zone.attrs["Spacing"][1] / 1000.0, zone.attrs["Spacing"][0])
        name = "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip(
            "_"
        )

        match overlaps:
            case None:
                warnings.warn(
                    "No overlap specified, Xdas will try its best to find the correct trimming"
                )
                noverlap = chunks.shape[1] - round((1 / blockrate) / delta[0])
                before = noverlap // 2
                after = noverlap - before
                overlaps = (before, after)
            case (int(), int()):
                pass
            case _:
                raise ValueError(
                    "overlaps must be a integer or a tuple of two integers"
                )

        match offset:
            case None:
                warnings.warn(
                    "No offset specified, Xdas will try its best to place the timestamps"
                )
                offset = chunks.shape[1] // 2
            case int():
                pass
            case _:
                raise ValueError("offset must be an integer")

        chunks = chunks[:, overlaps[0] : -overlaps[-1], :]
        times = times + (overlaps[0] - offset) * delta[0]

        dt, dx = delta
        _, nt, nx = chunks.shape

        dt = np.rint(1e6 * dt).astype("m8[us]").astype("m8[ns]")

        dc = []
        for t0, chunk in zip(times, chunks):

            t0 = np.rint(1e6 * t0).astype("M8[us]").astype("M8[ns]")
            time = Coordinate[self.ctype["time"]].from_block(t0, nt, dt, dim="time")
            distance = Coordinate[self.ctype["distance"]].from_block(
                0.0, nx, dx, dim="distance"
            )
            da = DataArray(chunk, {"time": time, "distance": distance}, name=name)
            dc.append(da)

        return concatenate(dc, "time")
