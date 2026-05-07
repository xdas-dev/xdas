import h5py
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from ..virtual import VirtualSource
from .core import Engine


class APSensingEngine(Engine, name="apsensing"):
    _supported_vtypes = ["hdf5"]
    _supported_ctypes = {
        "time": ["interpolated", "sampled", "dense"],
        "distance": ["interpolated", "sampled", "dense"],
    }

    def open_dataarray(self, fname):
        with h5py.File(fname, "r") as file:
            t0 = file["Metadata"]["Timestamp"][()].item().decode()
            fs = file["ProcessingServer"]["DataRate"][()].item()
            dx = file["ProcessingServer"]["SpatialSampling"][()].item()
            x0 = file["DAQ"]["PositionStart"][()].item()
            data = VirtualSource(file["DAS"])

        nt, nd = data.shape

        # time
        if t0.endswith("Z"):
            t0 = t0[:-1]
        else:
            raise NotImplementedError("Only UTC timezone is supported")
        t0 = np.datetime64(t0)
        dt = np.timedelta64(round(1e9 / fs), "ns")
        time = Coordinate[self.ctype["time"]].from_block(t0, nt, dt, dim="time")

        # distance
        dx
        distance = Coordinate[self.ctype["distance"]].from_block(
            x0, nd, dx, dim="distance"
        )
        return DataArray(data, {"time": time, "distance": distance})

        # NOTE: Distance sample are left aligned. The original number of samples is
        # `round((xend - xstart) / dx) + 1` with xstart / xend located  in
        # "DAQ/PositionStart" / "DAQ/PositionEnd" and dx located in "DAQ/SamplingInterval".
        # Then a slice is applied when the gauge length is applied. This sliding opperation
        # is done with a interval of "ProcessingServer/GaugeLengthPoints" and a step of
        # "ProcessingServer/SpatialSamplingPoints". Now the reference distance is attached
        # to the beginning of the gauge length. We could have attached it to the center
        # of the gauge length, but that would complexify things.
