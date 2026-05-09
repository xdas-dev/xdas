import numpy as np
import pytest

import xdas as xd


class TestFromBlock:
    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_generic(self, dtype, ctype):
        start = np.array(0, dtype)
        size = 10
        step = np.array(
            1, "timedelta64" if np.issubdtype(dtype, np.datetime64) else dtype
        )
        coord = xd.Coordinate[ctype].from_block(start, size, step, "dim")


@pytest.fixture
def coord(dtype, ctype):
    starts = np.array(
        [
            0,  # 0 - initial block
            10,  # 10 - continuous
            18,  # 20 - 2 overlap
            30,  # 30 - 2 gap
            48,  # 40 - 8 gap
            50,  # 50 - 8 overlap
        ],
        dtype,
    )
    size = 10
    step = np.array(1, "timedelta64" if np.issubdtype(dtype, np.datetime64) else dtype)
    out = xd.Coordinate[ctype](data=None, dim="dim", dtype=float)
    for start in starts:
        out = out.append(xd.Coordinate[ctype].from_block(start, size, step, "dim"))
    return out


class TestAppend:
    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_generic(self, coord, dtype, ctype):
        assert coord.dtype == dtype
        assert isinstance(coord, xd.Coordinate[ctype])


class TestGetSplitIndices:
    @pytest.mark.parametrize("ctype", ["interpolated"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    @pytest.mark.parametrize(
        "kind,tolerance,expected",
        [
            ("discontinuities", False, [10, 20, 30, 40, 50]),
            ("discontinuities", None, [20, 30, 40, 50]),
            ("discontinuities", 1, [20, 30, 40, 50]),
            ("discontinuities", 2, [40, 50]),
            ("discontinuities", 4, [40, 50]),
            ("discontinuities", 8, []),
            ("discontinuities", 20, []),
            ("gap", False, [10, 30, 40]),
            ("gap", None, [30, 40]),
            ("gap", 1, [30, 40]),
            ("gap", 2, [40]),
            ("gap", 4, [40]),
            ("gap", 8, []),
            ("gap", 20, []),
            ("overlap", False, [20, 50]),
            ("overlap", None, [20, 50]),
            ("overlap", 1, [20, 50]),
            ("overlap", 2, [50]),
            ("overlap", 4, [50]),
            ("overlap", 8, []),
            ("overlap", 20, []),
        ],
    )
    def test_generic(self, coord, kind, tolerance, expected):
        indices = coord.get_split_indices(kind=kind, tolerance=tolerance)
        np.testing.assert_array_equal(indices, expected)

    @pytest.mark.parametrize("ctype", ["interpolated"])
    def test_wrong_kind(self, ctype):
        with pytest.raises(ValueError):
            xd.Coordinate[ctype]().get_split_indices("wrong_kind")
