import re

import numpy as np

import xdas as xd

# Release segment, plus the optional PEP 440 pre/post/dev markers (e.g. 0.2.9.dev0).
VERSION_PATTERN = re.compile(r"^\d+(\.\d+)*((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?$")


def test_version():
    version = xd.__version__
    assert isinstance(version, str)
    assert VERSION_PATTERN.match(version)


class TestSynthetics:
    def test_randn_wavefronts_contract(self):
        from xdas.synthetics import randn_wavefronts

        da = randn_wavefronts()
        assert da.dims == ("time", "distance")
        assert da.sizes == {"time": 20000, "distance": 1001}
        assert da["time"][0].values == np.datetime64("2024-01-01T00:00:00", "ns")
        assert float(da["distance"][-1].values) == 100000.0
        # seeded: two calls give the same wavefronts
        assert np.array_equal(da.values[:100], randn_wavefronts().values[:100])
