import os
import tempfile

import numpy as np
import scipy.signal as sp

import xdas
from xdas.core import Database
from xdas.synthetics import generate
from xdas.processing import ProcessingChain, SOSFilter, DatabaseLoader, DatabaseWriter


class TestProcessing:

    def test_sequence(self):
        """
        Objective: to test a sequence of NumPy functions
        """

        with tempfile.TemporaryDirectory() as tempdir:

            # Generate a temporary dataset
            generate(tempdir)

            # Load the database
            db = xdas.open_database(os.path.join(tempdir, "sample.nc"))

            # Wrapper for numpy.mean
            def my_mean(db, dim):
                dim = xdas.signal.parse_dim(db, dim)
                axis = db.get_axis_num(dim)
                result = np.mean(db.values, axis=axis)
                return db.copy(data=result)

            # Sequence to execute
            sequence = xdas.Sequence(
                xdas.Atom(np.abs),
                xdas.Atom(np.square, name="some square"),
                xdas.Atom(my_mean, dim="time"),
            )

            # Process using sequence.execute
            result1 = sequence.execute(db)
            # Process manually
            result2 = my_mean(np.abs(db) ** 2, dim="time")

            # Check that automatic and manual results are the same
            assert np.allclose(result1.values, result2.values)

    def test_stateful(self):
        """
        Objective: to test state updates of the StateAtom class
        """

        with tempfile.TemporaryDirectory() as tempdir:

            # Generate a temporary dataset
            generate(tempdir)

            # Load the database
            db = xdas.open_database(os.path.join(tempdir, "sample.nc"))

            # Initialise the SOS filter
            corners = 4
            freq = 1.0
            btype = "lowpass"
            fs = 50.0
            sos = sp.iirfilter(
                corners, freq, btype=btype, ftype="butter", output="sos", fs=fs
            )
            zi = np.zeros((sos.shape[0], 2, db.shape[1]))

            # Wrapper around the SciPy sosfilt
            def sosfilter(db, zi, sos, dim="last"):
                dim = xdas.signal.parse_dim(db, dim)
                axis = db.get_axis_num(dim)
                result, zi = sp.sosfilt(sos, db.values, axis=axis, zi=zi)
                return db.copy(data=result), zi

            # Sequence to execute
            sequence = xdas.Sequence(
                xdas.StateAtom(
                    sosfilter,
                    sos=sos,
                    state_arg="zi",
                    state=zi.copy(),
                    dim="time",
                    name="sosfilt",
                ),
            )

            # Get the result for the entire dataset processed at once
            # Make sure to use .copy() to prevent state updates
            # associated with the StateAtom
            result1 = sequence.copy().execute(db)
            # Initialise the data loader/writer
            data_loader = xdas.processing.DatabaseLoader(db, chunks={"time": 100})
            data_writer = xdas.processing.DatabaseWriter(tempdir)
            # Perform chunked processing
            result2 = sequence.copy().execute(data_loader, data_writer)

            # Check that the chunked results are the same as the monolithic result
            assert np.allclose(result1.values, result2.values)

    def test_all_old(self):
        with tempfile.TemporaryDirectory() as tempdir:
            generate(tempdir)
            db = xdas.open_database(os.path.join(tempdir, "sample.nc"))
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
