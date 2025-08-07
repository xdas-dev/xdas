from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import obspy

import xdas as xd


def make_network(dirpath, gap=False, samples=100):
    for idx in range(1, 11):
        st = make_station(idx, gap, samples)
        if gap:
            st.write(f"{dirpath}/{st[0].id[:-4]}_gap.mseed")
        else:
            st.write(f"{dirpath}/{st[0].id[:-4]}.mseed")
    return st


def make_station(idx, gap, samples):
    st = obspy.Stream()
    for component in ["Z", "N", "E"]:
        all_tr = make_trace(idx, component, gap, samples)
        for tr in all_tr:
            st.append(tr)
    return st


def make_trace(idx, component, gap, samples):
    if gap:
        data1 = np.random.rand(int(samples / 2))
        data2 = np.random.rand(int(samples / 2 - 10))
        header1 = make_header(idx, component, 0)
        header2 = make_header(idx, component, len(data1) + 10)
        tr1 = obspy.Trace(data1, header1)
        tr2 = obspy.Trace(data2, header2)
        return [tr1, tr2]
    else:
        data = np.random.rand(samples)
        header = make_header(idx, component, 0)
        tr = obspy.Trace(data, header)
        return [tr]


def make_header(idx, component, starttime):
    header = {
        "delta": 0.01,
        "starttime": obspy.UTCDateTime(starttime),
        "network": "DX",
        "station": f"CH{idx:03d}",
        "location": "00",
        "channel": f"HH{component}",
    }
    return header


def test_miniseed():
    with TemporaryDirectory() as dirpath:
        st = make_network(dirpath, samples=100)
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

        # read one file without the last sample
        da = xd.open_dataarray(paths[0], engine="miniseed", ignore_last_sample=True)
        assert da.shape == (3, 99)
        assert da.dims == ("channel", "time")
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:00:00.980")
        assert da.coords["network"].values == "DX"
        assert da.coords["station"].values == "CH001"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

        # read one file with gaps
        st = make_network(dirpath, gap=True, samples=100)
        paths = sorted(glob(f"{dirpath}/*_gap.mseed"))
        da = xd.open_dataarray(paths[0], engine="miniseed")
        assert da.shape == (3, 90)
        assert da.dims == ("channel", "time")
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:01:00.390")
        assert da.coords["network"].values == "DX"
        assert da.coords["station"].values == "CH001"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

        # read one file with gaps and ignore the last sample
        da = xd.open_dataarray(paths[0], engine="miniseed", ignore_last_sample=True)
        assert da.shape == (3, 89)
        assert da.dims == ("channel", "time")
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:01:00.380")
        assert da.coords["network"].values == "DX"
        assert da.coords["station"].values == "CH001"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

        # manually concatenate several files (without gaps)
        paths = sorted(glob(f"{dirpath}/*00.mseed"))
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

        # manually concatenate several files with gaps
        paths = sorted(glob(f"{dirpath}/*gap.mseed"))
        objs = [xd.open_dataarray(path, engine="miniseed") for path in paths]
        da = xd.concatenate(objs, "station")
        assert da.shape == (10, 3, 90)
        assert da.dims == ("station", "channel", "time")
        assert da.coords["station"].values.tolist() == [
            f"CH{i:03d}" for i in range(1, 11)
        ]
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:01:00.390")
        assert da.coords["network"].values == "DX"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

        # automatically open multiple files (without gaps)
        da = xd.open_mfdataarray(
            f"{dirpath}/*00.mseed", dim="station", engine="miniseed"
        )
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

        # automatically open multiple files (with gaps)
        da = xd.open_mfdataarray(
            f"{dirpath}/*gap.mseed", dim="station", engine="miniseed"
        )
        assert da.shape == (10, 3, 90)
        assert da.dims == ("station", "channel", "time")
        assert da.coords["station"].values.tolist() == [
            f"CH{i:03d}" for i in range(1, 11)
        ]
        assert da.coords["time"].isinterp()
        assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:01:00.390")
        assert da.coords["network"].values == "DX"
        assert da.coords["location"].values == "00"
        assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]
