import numpy as np
import pytest

import xdas.ufuncs as xp
from xdas.synthetics import generate


class TestUfuncs:
    def test_unitary_operators(self):
        db = generate()
        result = xp.abs(db)
        expected = db.copy(data=np.abs(db.data))
        db_out = db.copy()
        xp.abs(db, out=db_out)
        db_where = db.copy()
        xp.abs(db, out=db_where, where=db.copy(data=db.data > 0))
        assert result.equals(expected)
        assert db_out.equals(expected)
        assert db_where.equals(db)

    def test_binary_operators(self):
        db1 = generate()
        db2 = generate()
        result = xp.add(db1, db2)
        expected = db1.copy(data=db1.data + db2.data)
        db_out = db1.copy()
        xp.add(db1, db2, out=db_out)
        db_where = db1.copy()
        xp.abs(db1, out=db_where, where=db1.copy(data=np.zeros(db1.shape, "bool")))
        assert result.equals(expected)
        assert db_out.equals(expected)
        assert db_where.equals(db1)
        with pytest.raises(ValueError):
            xp.add(db1, db2[1:])

    def test_multiple_outputs(self):
        db = generate()
        result1, result2 = xp.divmod(db, db)
        expected1 = db.copy(data=np.ones(db.shape))
        expected2 = db.copy(data=np.zeros(db.shape))
        assert result1.equals(expected1)
        assert result2.equals(expected2)
        with pytest.raises(ValueError):
            xp.add(db, db[1:])
