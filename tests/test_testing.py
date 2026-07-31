import numpy as np
import pytest

import xdas as xd


class TestDummy:
    def test_defaults(self):
        da = xd.testing.dummy()
        assert da.shape == (100, 10)
        assert da.dims == ("time", "distance")
        assert da["time"].isregular()
        assert da["distance"].isregular()

    def test_mismatched_shape(self):
        with pytest.raises(ValueError, match="must equal len\\(shape\\)"):
            xd.testing.dummy(dims=("time",), shape=(10, 10))

    def test_mismatched_step(self):
        with pytest.raises(ValueError, match="must equal len\\(dims\\)"):
            xd.testing.dummy(step=(1.0,))

    def test_datetime_step_passthrough(self):
        da = xd.testing.dummy(step=(np.timedelta64(10, "ms"), 10.0))
        assert da["time"].get_sampling_interval() == 0.01
