import numpy as np

from .tdms import TdmsReader
from ..core.dataarray import DataArray

def read(fname):
    # Read the TDMS file
    tdms = TdmsReader(fname)
    props = tdms.get_properties()
    data = tdms.get_data()
    nt, nd = data.shape

    # Create the time tie points
    t0 = np.datetime64(props["GPSTimeStamp"])
    dt = np.timedelta64(round(1e9 /  props["SamplingFrequency[Hz]"]), "ns")
    time = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + dt * (nt - 1)]}

    # Create the distance tie points
    distance = {"tie_indices": [0, nd - 1], "tie_values": [props["Start Distance (m)"], props["Stop Distance (m)"]]}

    # Pack the data into a DataArray
    return DataArray(data, {"time": time, "distance": distance})