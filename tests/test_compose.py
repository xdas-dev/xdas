import os
import tempfile
import numpy as np

import xdas
import xdas.signal as xp
from xdas.synthetics import generate
from xdas.xarray import mean


class TestCompose:
    def test_init(self):
        sequence = xdas.Sequence(
            [
                xdas.Atom(xp.taper, dim="time"),
                xdas.Atom(xp.taper, dim="distance"),
                xdas.Atom(np.abs),
                xdas.Atom(np.square),
            ]
        )


class TestProcessing:
    def test_sequence(self):
        with tempfile.TemporaryDirectory() as tempdir:

            # Generate and load a temporary dataset
            generate(tempdir)
            db = xdas.open_database(os.path.join(tempdir, "sample.nc"))

            # Declare sequence to execute
            sequence = xdas.Sequence(
                [
                    xdas.Atom(np.abs),
                    xdas.Atom(np.square, name="some square"),
                    xdas.Atom(mean, dim="time"),
                ]
            )

            # Sequence processing
            result1 = sequence(db)
            # Manual processing
            result2 = mean(np.abs(db) ** 2, dim="time")

            # Test
            assert np.allclose(result1.values, result2.values)
