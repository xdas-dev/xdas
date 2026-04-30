import dask
import numpy as np

from ..coordinates.core import Coordinate
from ..core.dataarray import DataArray
from .core import Engine, parse_ctype
from .tdms import TdmsReader


class SilixaEngine(Engine, name="silixa"):
    @staticmethod
    def open_dataarray(fname, **kwargs):
        return read(fname, **kwargs)


def read(fname, ctype=None):
    ctype = parse_ctype(ctype)
    shape, dtype, coords = read_header(fname, ctype)
    data = dask.array.from_delayed(dask.delayed(read_data)(fname), shape, dtype)
    return DataArray(data, coords)


def read_header(fname, ctype):
    with TdmsReader(fname) as tdms:
        props = tdms.get_properties()
        shape = tdms.channel_length, tdms.fileinfo["n_channels"]
        dtype = tdms._data_type
    t0 = np.datetime64(props["GPSTimeStamp"])
    dt = np.timedelta64(round(1e9 / props["SamplingFrequency[Hz]"]), "ns")
    time = Coordinate[ctype["time"]].from_block(t0, shape[0], dt, dim="time")
    distance = {
        "tie_indices": [0, shape[1] - 1],
        "tie_values": [props["Start Distance (m)"], props["Stop Distance (m)"]],
    }  # TODO: use from_block
    coords = {"time": time, "distance": distance}
    return shape, dtype, coords


def read_data(fname):
    with TdmsReader(fname) as tdms:
        data = tdms.get_data()
    return data
