import numpy as np

import xdas.xarray as xp
from xdas.database import Database
from xdas.synthetics import generate


class TestXarray:
    def test_returns_database(self):
        db = generate()
        for name, func in xp.XARRAY_HANDLED_METHODS.items():
            if callable(func):
                if func in [
                    xp.percentile,
                    xp.quantile,
                ]:
                    result = func(db, 0.5)
                    assert isinstance(result, Database)
                    result = getattr(db, name)(0.5)
                    assert isinstance(result, Database)
                elif func == xp.diff:
                    result = func(db, "time")
                    assert isinstance(result, Database)
                    result = getattr(db, name)("time")
                    assert isinstance(result, Database)
                else:
                    result = func(db)
                    assert isinstance(result, Database)
                    result = getattr(db, name)()
                    assert isinstance(result, Database)

    def test_mean(self):
        db = generate()
        result = xp.mean(db, "time")
        result_method = db.mean("time")
        expected = np.mean(db, 0)
        assert result.equals(expected)
        assert result_method.equals(expected)
