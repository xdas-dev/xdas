"""I/O engine for Febus HDF5 files (:class:`FebusEngine`)."""

import warnings
from typing import ClassVar

import h5py
import numpy as np

from ..coordinates import Coordinate
from ..core import DataArray, concat, concat_coords
from ..tiles import TileArray
from ..virtual import VirtualSource
from .core import Engine


class FebusEngine(Engine, name="febus"):
    """Engine for reading Febus HDF5 files."""

    _supported_vtypes: ClassVar[list] = ["hdf5", "tiles"]
    _supported_ctypes: ClassVar[dict] = {
        "time": ["interpolated", "sampled", "dense"],
        "distance": ["interpolated", "sampled", "dense"],
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

        Parameters
        ----------
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

        Returns
        -------
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
            dataset_path = zone[name].name
            chunks = VirtualSource(zone[name])
            delta = (zone.attrs["Spacing"][1] / 1000.0, zone.attrs["Spacing"][0])
            x0 = zone.attrs["Extent"][0] * delta[1] + zone.attrs["Origin"][0]
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

        times = times + (overlaps[0] - offset) * delta[0]

        dt, dx = delta
        nblocks, block_size, nx = chunks.shape
        nt = block_size - overlaps[0] - overlaps[1]

        dt = np.rint(1e6 * dt).astype("m8[us]").astype("m8[ns]")

        if self.vtype == "tiles":
            # one tile spans the whole file: the block arithmetic (trimming
            # and 3-D to 2-D fusing) lives in `load_tile`, not the manifest
            time = concat_coords(
                [
                    Coordinate[self.ctype["time"]].from_block(
                        np.rint(1e6 * t0).astype("M8[us]").astype("M8[ns]"),
                        nt,
                        dt,
                        dim="time",
                    )
                    for t0 in times
                ]
            )
            distance = Coordinate[self.ctype["distance"]].from_block(
                x0, nx, dx, dim="distance"
            )
            engine = {
                "name": "febus",
                "dataset": dataset_path,
                "block_size": int(block_size),
                "overlaps": [int(overlaps[0]), int(overlaps[1])],
            }
            data = TileArray(str(fname), (nblocks * nt, nx), engine, chunks.dtype)
            return DataArray(data, {"time": time, "distance": distance}, name=name)

        chunks = chunks[:, overlaps[0] : -overlaps[-1], :]

        dc = []
        for t0, chunk in zip(times, chunks):
            t0 = np.rint(1e6 * t0).astype("M8[us]").astype("M8[ns]")
            time = Coordinate[self.ctype["time"]].from_block(t0, nt, dt, dim="time")
            distance = Coordinate[self.ctype["distance"]].from_block(
                x0, nx, dx, dim="distance"
            )
            da = DataArray(chunk, {"time": time, "distance": distance}, name=name)
            dc.append(da)

        return concat(dc, "time")

    @staticmethod
    def load_tile(path, selection, *, dataset, block_size, overlaps):
        """Read a post-trim source selection of a Febus file.

        Febus files store a 3-D stack of overlapping ``(time, distance)``
        blocks. Rows are counted post-trim: each block contributes
        ``block_size - sum(overlaps)`` rows. Only the blocks overlapping the
        requested rows are read; overlap rows are sliced away without being
        copied.

        Parameters
        ----------
        path : str
            Path of the Febus HDF5 file.
        selection : tuple of slice
            The source selection to read, one possibly strided slice per
            axis, post-trim rows along axis 0.
        dataset : str
            Location of the block stack within the file.
        block_size : int
            Rows of one untrimmed block.
        overlaps : tuple of int
            Rows trimmed at the start and end of each block.
        """
        rows = selection[0]
        start, stop = rows.start, rows.stop
        keep = block_size - overlaps[0] - overlaps[1]
        with h5py.File(path, "r") as file:
            blocks = file[dataset]
            parts = []
            for block in range(start // keep, (stop - 1) // keep + 1):
                lo = max(start - block * keep, 0) + overlaps[0]
                hi = min(stop - block * keep, keep) + overlaps[0]
                parts.append(blocks[block, lo:hi])
        rows = np.concatenate(parts) if len(parts) > 1 else parts[0]
        return rows[(slice(None, None, selection[0].step), *selection[1:])]
