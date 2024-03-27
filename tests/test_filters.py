import os
from tempfile import TemporaryDirectory

import scipy.signal as sp

from xdas.core import chunk, concatenate
from xdas.filters import IIRFilter
from xdas.synthetics import generate


class TestFilters:
    def test_iirfilter(self):
        db = generate()
        chunks = chunk(db, 6, "time")

        sos = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0, output="sos")
        data = sp.sosfilt(sos, db.values, axis=0)
        expected = db.copy(data=data)

        filter = IIRFilter(4, 10.0, "lowpass", dim="time")
        monolithic = filter(db)
        chunked = concatenate([filter(chunk, chunk="time") for chunk in chunks], "time")

        assert monolithic.equals(expected)
        assert chunked.equals(expected)

        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "state.nc")

            filter_a = IIRFilter(4, 10.0, "lowpass", dim="time")
            chunks_a = [filter_a(chunk, chunk="time") for chunk in chunks[:3]]
            filter_a.save_state(path)

            filter_b = IIRFilter(4, 10.0, "lowpass", dim="time")
            # filter_b(chunks[0], chunk="time")  # TODO``
            filter_b.load_state(path)
            chunks_b = [filter_b(chunk, chunk="time") for chunk in chunks[3:]]

            result = concatenate(chunks_a + chunks_b, "time")
            assert result.equals(expected)
