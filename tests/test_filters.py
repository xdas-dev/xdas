import scipy.signal as sp

from xdas.core import chunk, concatenate
from xdas.filters import IIRFilter
from xdas.synthetics import generate


class TestFilters:
    def test_all(self):
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
