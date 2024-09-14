from glob import glob
from tempfile import TemporaryDirectory

import xdas as xd
import obspy
import numpy as np


def make_network(dirpath):
    for idx in range(1, 11):
        st = make_station(idx)
        st.write(f"{dirpath}/{st[0].id[:-4]}.mseed")


def make_station(idx):
    st = obspy.Stream()
    for component in ["Z", "N", "E"]:
        st.append(make_trace(idx, component))
    return st


def make_trace(idx, component):
    data = np.random.rand(100)
    header = {
        "delta": 0.01,
        "starttime": obspy.UTCDateTime(0),
        "network": "DX",
        "station": f"CH{idx:03d}",
        "location": "00",
        "channel": f"HH{component}",
    }
    tr = obspy.Trace(data, header)
    return tr


def test_miniseed():
    with TemporaryDirectory() as dirpath:
        make_network(dirpath)
        paths = sorted(glob(f"{dirpath}/*.mseed"))

        # read one file
        da = xd.open_dataarray(paths[0], engine="miniseed")
        assert da.shape == (3, 100)
        assert da.dims == ("channel", "time")
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:00:00.990")
        assert da.coords["network"].values == "DX"
        assert da.coords["station"].values == "CH001"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

        # manually concatenate several files
        objs = [xd.open_dataarray(path, engine="miniseed") for path in paths]
        da = xd.concatenate(objs, "station")
        assert da.shape == (10, 3, 100)
        assert da.dims == ("station", "channel", "time")
        assert da.coords["station"].values.tolist() == [
            f"CH{i:03d}" for i in range(1, 11)
        ]
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:00:00.990")
        assert da.coords["network"].values == "DX"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]
