import dask
import numpy as np

from ..core.dataarray import DataArray
from .tdms import TdmsReader


def read(fname):
    shape, dtype, coords = read_header(fname)
    data = dask.array.from_delayed(dask.delayed(read_data)(fname), shape, dtype)
    return DataArray(data, coords)


def read_header(fname):
    with TdmsReader(fname) as tdms:
        props = tdms.get_properties()
        shape = tdms.channel_length, tdms.fileinfo["n_channels"]
        dtype = tdms._data_type
    t0 = np.datetime64(props["GPSTimeStamp"])
    dt = np.timedelta64(round(1e9 / props["SamplingFrequency[Hz]"]), "ns")
    time = {
        "tie_indices": [0, shape[0] - 1],
        "tie_values": [t0, t0 + dt * (shape[0] - 1)],
    }
    distance = {
        "tie_indices": [0, shape[1] - 1],
        "tie_values": [props["Start Distance (m)"], props["Stop Distance (m)"]],
    }
    coords = {"time": time, "distance": distance}
    return shape, dtype, coords


def read_data(fname):
    with TdmsReader(fname) as tdms:
        data = tdms.get_data()
    return data
