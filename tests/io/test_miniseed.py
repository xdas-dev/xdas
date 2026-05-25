import numpy as np
import obspy
import pytest

import xdas as xd
from xdas.io.miniseed import MiniSEEDEngine, get_band_code, to_stream


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


def test_miniseed(tmp_path):
    make_network(tmp_path, samples=100)
    paths = sorted(tmp_path.glob("*.mseed"))

    # read one file
    da = xd.open(paths[0], engine="miniseed")
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
    da = xd.open(paths[0], engine="miniseed", ignore_last_sample=True)
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
    make_network(tmp_path, gap=True, samples=100)
    paths = sorted(tmp_path.glob("*_gap.mseed"))
    da = xd.open(paths[0], engine="miniseed")
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
    da = xd.open(paths[0], engine="miniseed", ignore_last_sample=True)
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
    paths = sorted(tmp_path.glob("*00.mseed"))
    objs = [xd.open(path, engine="miniseed") for path in paths]
    da = xd.concat(objs, "station")
    assert da.shape == (10, 3, 100)
    assert da.dims == ("station", "channel", "time")
    assert da.coords["station"].values.tolist() == [f"CH{i:03d}" for i in range(1, 11)]
    assert da.coords["time"].isinterp()
    assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
    assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:00:00.990")
    assert da.coords["network"].values == "DX"
    assert da.coords["location"].values == "00"
    assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

    # manually concatenate several files with gaps
    paths = sorted(tmp_path.glob("*gap.mseed"))
    objs = [xd.open(path, engine="miniseed") for path in paths]
    da = xd.concat(objs, "station")
    assert da.shape == (10, 3, 90)
    assert da.dims == ("station", "channel", "time")
    assert da.coords["station"].values.tolist() == [f"CH{i:03d}" for i in range(1, 11)]
    assert da.coords["time"].isinterp()
    assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
    assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:01:00.390")
    assert da.coords["network"].values == "DX"
    assert da.coords["location"].values == "00"
    assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

    # automatically open multiple files (without gaps)
    da = xd.open(tmp_path / "*00.mseed", dim="station", engine="miniseed")
    assert da.shape == (10, 3, 100)
    assert da.dims == ("station", "channel", "time")
    assert da.coords["station"].values.tolist() == [f"CH{i:03d}" for i in range(1, 11)]
    assert da.coords["time"].isinterp()
    assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
    assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:00:00.990")
    assert da.coords["network"].values == "DX"
    assert da.coords["location"].values == "00"
    assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

    # automatically open multiple files (with gaps)
    da = xd.open(tmp_path / "*gap.mseed", dim="station", engine="miniseed")
    assert da.shape == (10, 3, 90)
    assert da.dims == ("station", "channel", "time")
    assert da.coords["station"].values.tolist() == [f"CH{i:03d}" for i in range(1, 11)]
    assert da.coords["time"].isinterp()
    assert da.coords["time"][0].values == np.datetime64("1970-01-01T00:00:00")
    assert da.coords["time"][-1].values == np.datetime64("1970-01-01T00:01:00.390")
    assert da.coords["network"].values == "DX"
    assert da.coords["location"].values == "00"
    assert da.coords["channel"].values.tolist() == ["HHZ", "HHN", "HHE"]

    # trigger read_data by loading values (synchronized case)
    sync_paths = sorted(tmp_path.glob("*00.mseed"))
    da_sync = xd.open(sync_paths[0], engine="miniseed")
    values = da_sync.values
    assert values.shape == (3, 100)

    # trigger read_data synchronized with ignore_last_sample
    da_sync_trimmed = xd.open(sync_paths[0], engine="miniseed", ignore_last_sample=True)
    values_trimmed = da_sync_trimmed.values
    assert values_trimmed.shape == (3, 99)

    # trigger read_data for unsynchronized (gapped) case
    gapped_paths = sorted(tmp_path.glob("*gap.mseed"))
    da_gap = xd.open(gapped_paths[0], engine="miniseed")
    values_gap = da_gap.values
    assert values_gap.shape == (3, 90)

    # trigger read_data unsynchronized with ignore_last_sample
    da_gap_trimmed = xd.open(
        gapped_paths[0], engine="miniseed", ignore_last_sample=True
    )
    values_gap_trimmed = da_gap_trimmed.values
    assert values_gap_trimmed.shape == (3, 89)


def test_miniseed_helpers(tmp_path):
    # get_band_code with out-of-range sampling rate
    assert get_band_code(0.0) == "X"
    assert get_band_code(6000.0) == "X"

    # to_stream raises on non-2D data
    da_3d = xd.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "c"))
    with pytest.raises(ValueError, match="2D"):
        to_stream(da_3d)


def test_miniseed_unsynchronized_traces(tmp_path):
    path = tmp_path / "unsync.mseed"
    st = obspy.Stream()
    st.append(
        obspy.Trace(
            data=np.zeros(100, dtype=np.float32),
            header={"station": "AA", "channel": "HHZ", "delta": 0.01},
        )
    )
    st.append(
        obspy.Trace(
            data=np.zeros(100, dtype=np.float32),
            header={"station": "BB", "channel": "HHZ", "delta": 0.005},
        )
    )
    st.write(str(path), format="MSEED")
    with pytest.raises(ValueError, match="synchronized"):
        MiniSEEDEngine().read_header(str(path), False, "interpolated")
