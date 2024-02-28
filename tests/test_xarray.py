import xdas.xarray as xp
from xdas.database import Database
from xdas.synthetics import generate


class TestXarray:
    def test_returns_database(self):
        db = generate()
        for name, obj in xp.__dict__.items():
            if callable(obj):
                if obj in [
                    xp.percentile,
                    xp.quantile,
                ]:
                    result = obj(db, 0.5)
                    assert isinstance(result, Database)
                else:
                    result = obj(db)
                    assert isinstance(result, Database)
