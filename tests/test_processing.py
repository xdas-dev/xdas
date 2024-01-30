import tempfile

import numpy as np
import scipy.signal as sp

from xdas.core import Database
from xdas.processing import ProcessingChain, SOSFilter


class TestProcessing:
    def generate(self):
        fs = 10
        ks = 1 / 10
        data = np.random.randn(1000, 100)
        return Database(
            data,
            {
                "time": {
                    "tie_indices": [0, data.shape[0] - 1],
                    "tie_values": np.array(
                        [round(0.0), round(1e9 * (data.shape[0] - 1) / fs)],
                        dtype="datetime64[ns]",
                    ),
                },
                "distance": {
                    "tie_indices": [0, data.shape[1] - 1],
                    "tie_values": [0.0, (data.shape[1] - 1) / ks],
                },
            },
        )

    def test_all(self):
        db = self.generate()
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        dim = "time"
        axis = db.get_axis_num(dim)
        parallel = 4

        expected = db.copy(data=sp.sosfilt(sos, db.values, axis=axis))

        sosfilter = SOSFilter(sos, dim, parallel)
        result_filter = sosfilter(db)

        chain = ProcessingChain([sosfilter])
        chain.reset()
        result_chain = chain(db)

        with tempfile.TemporaryDirectory() as tempdir:
            result_process = chain.process(db, {dim: 100}, tempdir).load()

        assert result_filter.equals(expected)
        assert result_chain.equals(expected)
        assert np.allclose(result_process.values, expected.values)
