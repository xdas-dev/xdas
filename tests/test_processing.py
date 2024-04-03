import os
import tempfile

import scipy.signal as sp

import xdas
from xdas.processing import DatabaseLoader, DatabaseWriter, process
from xdas.signal import sosfilt
from xdas.synthetics import generate


class TestProcessing:
    def test_stateful(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # generate test database
            generate().to_netcdf(os.path.join(tempdir, "sample.nc"))
            db = xdas.open_database(os.path.join(tempdir, "sample.nc"))

            # declare processing sequence
            sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
            sequence = xdas.Sequential(
                [xdas.StatePartial(sosfilt, sos, ..., dim="time", state="zi")]
            )

            # monolithic processing
            result1 = sequence(db)

            # chunked processing
            data_loader = DatabaseLoader(db, chunks={"time": 100})
            data_writer = DatabaseWriter(tempdir)
            result2 = process(
                sequence, data_loader, data_writer
            )  # resets the sequence by default

            # test
            assert result1.equals(result2)
