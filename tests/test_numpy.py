import numpy as np
import pytest

from xdas.database import HANDLED_FUNCTIONS, Database
from xdas.synthetics import generate


class TestUfuncs:
    def test_unitary_operators(self):
        db = generate()
        result = np.abs(db)
        expected = db.copy(data=np.abs(db.data))
        db_out = db.copy()
        np.abs(db, out=db_out)
        db_where = db.copy()
        np.abs(db, out=db_where, where=db.copy(data=db.data > 0))
        assert result.equals(expected)
        assert db_out.equals(expected)
        assert db_where.equals(db)

    def test_binary_operators(self):
        db1 = generate()
        db2 = generate()
        result = np.add(db1, db2)
        expected = db1.copy(data=db1.data + db2.data)
        db_out = db1.copy()
        np.add(db1, db2, out=db_out)
        db_where = db1.copy()
        np.abs(db1, out=db_where, where=db1.copy(data=np.zeros(db1.shape, "bool")))
        assert result.equals(expected)
        assert db_out.equals(expected)
        assert db_where.equals(db1)
        with pytest.raises(ValueError):
            np.add(db1, db2[1:])

    def test_multiple_outputs(self):
        db = generate()
        result1, result2 = np.divmod(db, db)
        expected1 = db.copy(data=np.ones(db.shape))
        expected2 = db.copy(data=np.zeros(db.shape))
        assert result1.equals(expected1)
        assert result2.equals(expected2)
        with pytest.raises(ValueError):
            np.add(db, db[1:])


class TestFunc:
    def test_returns_database(self):
        db = generate()
        for numpy_function in HANDLED_FUNCTIONS:
            if numpy_function == np.clip:
                result = numpy_function(db, -1, 1)
                assert isinstance(result, Database)
            elif numpy_function in [
                np.percentile,
                np.nanpercentile,
                np.quantile,
                np.nanquantile,
            ]:
                result = numpy_function(db, 0.5)
                assert isinstance(result, Database)
            else:
                result = numpy_function(db)
                assert isinstance(result, Database)

    def test_reduce(self):
        db = generate()
        result = np.sum(db)
        assert result.shape == ()
        result = np.sum(db, axis=0)
        assert result.dims == ("distance",)
        assert result.coords["distance"].equals(db.coords["distance"])
        result = np.sum(db, axis=1)
        assert result.dims == ("time",)
        assert result.coords["time"].equals(db.coords["time"])
        with pytest.raises(np.AxisError):
            np.sum(db, axis=2)

    def test_out(self):
        db = generate()
        out = db.copy()
        np.cumsum(db, axis=-1, out=out)
        assert not out.equals(db)
