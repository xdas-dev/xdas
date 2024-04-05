import numpy as np

import xdas.core.methods as xp
from xdas.core.database import DataArray
from xdas.synthetics import generate


class TestXarray:
    def test_returns_database(self):
        db = generate()
        for name, func in xp.HANDLED_METHODS.items():
            if callable(func):
                if name in [
                    "percentile",
                    "quantile",
                ]:
                    result = func(db, 0.5)
                    assert isinstance(result, DataArray)
                    result = getattr(db, name)(0.5)
                    assert isinstance(result, DataArray)
                elif name == "diff":
                    result = func(db, "time")
                    assert isinstance(result, DataArray)
                    result = getattr(db, name)("time")
                    assert isinstance(result, DataArray)
                else:
                    result = func(db)
                    assert isinstance(result, DataArray)
                    result = getattr(db, name)()
                    assert isinstance(result, DataArray)

    def test_mean(self):
        db = generate()
        result = xp.mean(db, "time")
        result_method = db.mean("time")
        expected = np.mean(db, 0)
        assert result.equals(expected)
        assert result_method.equals(expected)
