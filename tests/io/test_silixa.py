import numpy as np
import numpy.testing as npt

from xdas.io import silixa
from xdas.tiles import TileArray


class FakeTdms:
    """In-memory stand-in for :class:`~xdas.io.tdms.TdmsReader`."""

    data = np.arange(20.0 * 4).reshape(20, 4)

    channel_length = 20
    fileinfo = {"n_channels": 4}
    _data_type = np.dtype("float64")

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get_properties(self):
        return {
            "GPSTimeStamp": "2020-01-01T00:00:00",
            "SamplingFrequency[Hz]": 1000.0,
            "Start Distance (m)": 0.0,
            "Fibre Length Multiplier": 1.0,
            "SpatialResolution[m]": 4.0,
        }

    def get_data(self, first_s=None, last_s=None):
        first_s = 0 if first_s is None else first_s
        last_s = len(self.data) - 1 if last_s is None else last_s
        return self.data[first_s : last_s + 1]


def test_tile_load(monkeypatch):
    monkeypatch.setattr(silixa, "TdmsReader", FakeTdms)
    expected = FakeTdms.data
    manifest = TileArray.from_tiles("fake.tdms", (20, 4), {"name": "silixa"}, "float64")
    npt.assert_array_equal(np.asarray(manifest), expected)
    npt.assert_array_equal(np.asarray(manifest[3:15:2, 1:3]), expected[3:15:2, 1:3])
    expanded = np.expand_dims(manifest, 0)
    assert isinstance(expanded, TileArray)
    npt.assert_array_equal(np.asarray(expanded), expected[None])


def test_read_data(monkeypatch):
    monkeypatch.setattr(silixa, "TdmsReader", FakeTdms)
    npt.assert_array_equal(silixa.SilixaEngine().read_data("fake.tdms"), FakeTdms.data)


def test_open_dataarray(monkeypatch):
    monkeypatch.setattr(silixa, "TdmsReader", FakeTdms)
    da = silixa.SilixaEngine().open_dataarray("fake.tdms")
    assert isinstance(da.data, TileArray)
    assert da.dims == ("time", "distance")
    assert da.shape == (20, 4)
    assert da.coords["time"][0].values == np.datetime64("2020-01-01T00:00:00")
    npt.assert_allclose(da.coords["distance"].values, 4.0 * np.arange(4))
    npt.assert_array_equal(da.values, FakeTdms.data)
