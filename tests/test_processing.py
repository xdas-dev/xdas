import tempfile

import dask.array as da
import numpy as np
import scipy.signal as sp

from xdas.core import Coordinate, Database
from xdas.processing import ChunkWriter, LFilter, SignalProcessingChain, SOSFilter
from xdas.virtual import DataLayout


class TestProcessing:
    def generate(self):
        fs = 125
        ks = 1 / 10
        data = da.random.normal(size=(1000, 100))
        time = Coordinate([0, data.shape[0] - 1], [0.0, (data.shape[0] - 1) / fs])
        distance = Coordinate([0, data.shape[1] - 1], [0.0, (data.shape[1] - 1) / ks])
        return Database(data, {"time": time, "distance": distance})

    def test_lfilter(self):
        db = self.generate()
        b, a = sp.iirfilter(4, 0.5, btype="lowpass")
        lfilter = LFilter(b, a, "time")
        result_direct = lfilter(db)
        chunk_size = 100
        lfilter.reset()
        result_chunks = db.copy()
        for k in range(db.shape[0] // chunk_size):
            query = {"time": slice(k * chunk_size, (k + 1) * chunk_size)}
            result_chunks[query] = lfilter(db[query]).data
        lfilter.reset()
        chain = SignalProcessingChain([lfilter])
        out = chain.process(db, "time", chunk_size, parallel=False)
        lfilter.reset()
        assert chain.filters[0].zi is None
        out_parallel = chain.process(db, "time", chunk_size, parallel=True)
        assert np.allclose(result_chunks.data, result_direct.data)
        assert np.allclose(out.data, result_direct.data)
        assert np.allclose(out_parallel.data, result_direct.data)

    def test_sosfilter(self):
        db = self.generate()
        sos = sp.iirfilter(4, 0.5, btype="lowpass", output="sos")
        sosfilter = SOSFilter(sos, "time")
        result_direct = sosfilter(db)
        chunk_size = 100
        sosfilter.reset()
        result_chunks = db.copy()
        for k in range(db.shape[0] // chunk_size):
            query = {"time": slice(k * chunk_size, (k + 1) * chunk_size)}
            result_chunks[query] = sosfilter(db[query]).data
        sosfilter.reset()
        chain = SignalProcessingChain([sosfilter])
        out = chain.process(db, "time", chunk_size, parallel=False)
        sosfilter.reset()
        assert chain.filters[0].zi is None
        out_parallel = chain.process(db, "time", chunk_size, parallel=True)
        assert np.allclose(result_chunks.data, result_direct.data)
        assert np.allclose(out.data, result_direct.data)
        assert np.allclose(out_parallel.data, result_direct.data)

    def test_chunkwriter(self):
        db = self.generate()
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = tmpdirname + "/test"
            chain = SignalProcessingChain([ChunkWriter(path)])
            out = chain.process(db, "time", 100)
            assert np.array_equal(db.values, out.values)
            print(out["time"].tie_indices)
            print(out["time"].tie_values)
            for dim in db.dims:
                assert out[dim].equals(db[dim])
            assert isinstance(out.data, DataLayout)
