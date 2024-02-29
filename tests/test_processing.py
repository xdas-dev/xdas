import tempfile

import numpy as np
import scipy.signal as sp

from xdas.core import Database
from xdas.processing import DatabaseLoader, DatabaseWriter, ProcessingChain, SOSFilter


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
        with tempfile.TemporaryDirectory() as tempdir:
            db = self.generate()
            dim = "time"

            data_loader = DatabaseLoader(db, {dim: 1000})

            sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
            sosfilter = SOSFilter(sos, dim, parallel=4)

            chain = ProcessingChain([sosfilter])

            data_writer = DatabaseWriter(tempdir)

            sosfilter.reset()
            result_filter = sosfilter(db)
            chain.reset()
            result_chain = chain(db)
            chain.reset()
            result_process = chain.process(data_loader, data_writer).load()
            axis = db.get_axis_num(dim)
            restult_expected = db.copy(data=sp.sosfilt(sos, db.values, axis=axis))

            assert result_filter.equals(restult_expected)
            assert result_chain.equals(restult_expected)
            assert np.allclose(result_process.values, restult_expected.values)
