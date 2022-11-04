import dask.array as da
import numpy as np
import scipy.signal as sp

from xdas.database import Coordinate, Coordinates, Database
from xdas.signal import LFilter, SignalProcessingChain, SignalProcessingUnit


class TestSignal:
    def test_signal(self):
        fs = 125
        ks = 1 / 10
        data = da.random.normal(size=(10000, 1000))
        time = Coordinate([0, data.shape[0] - 1], [0.0, (data.shape[0] - 1) / fs])
        distance = Coordinate([0, data.shape[1] - 1], [0.0, (data.shape[1] - 1) / ks])
        xarr = Database(data, {"time": time, "distance": distance})
        b, a = sp.iirfilter(4, 0.5, btype="lowpass")
        lfilter = LFilter(b, a, "time")
        result = lfilter(xarr)
        assert result.shape == xarr.shape
        assert result.coords == xarr.coords

