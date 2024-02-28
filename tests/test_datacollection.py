import xdas
from  xdas.synthetics import generate
from tempfile import TemporaryDirectory
import os

class TestDataCollection:
    def test_io(self):
        db = generate()
        dc = xdas.DataCollection(
            {
                "das1": db,
                "das2": db,
            },
            "instrument",
        )
        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "tmp.nc")
            dc.to_netcdf(path)
            result = xdas.DataCollection.from_netcdf(path)
            assert result.equals(dc)
