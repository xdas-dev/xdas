import os
import tempfile

import scipy.signal as sp

import xdas
from xdas.atoms import Partial, Sequential
from xdas.processing.core import DataArrayLoader, DataArrayWriter, process
from xdas.signal import sosfilt
from xdas.synthetics import wavelet_wavefronts


class TestProcessing:
    def test_stateful(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # generate test dataarray
            wavelet_wavefronts().to_netcdf(os.path.join(tempdir, "sample.nc"))
            da = xdas.open_dataarray(os.path.join(tempdir, "sample.nc"))

            # declare processing sequence
            sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
            sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])

            # monolithic processing
            result1 = sequence(da)

            # chunked processing
            data_loader = DataArrayLoader(da, chunks={"time": 100})
            data_writer = DataArrayWriter(tempdir)
            result2 = process(
                sequence, data_loader, data_writer
            )  # resets the sequence by default

            # test
            assert result1.equals(result2)
