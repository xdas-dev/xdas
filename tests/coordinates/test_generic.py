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
        assert isinstance(coord, xd.Coordinate[ctype])
        assert coord[0].values == start
        assert len(coord) == size
        assert coord.get_sampling_interval(cast=False) == step


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
    return xd.concat_coords(
        [xd.Coordinate[ctype].from_block(start, size, step, "dim") for start in starts]
    )


class TestAppend:
    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_generic(self, coord, dtype, ctype):
        assert isinstance(coord, xd.Coordinate[ctype])
        assert coord.dtype == dtype
        assert len(coord) == 60


class TestGetSplitIndices:
    # kind, tolerance, split_indices
    CASES = [
        ("discontinuities", False, [10, 20, 30, 40, 50]),
        ("discontinuities", None, [20, 30, 40, 50]),
        ("discontinuities", 1, [20, 30, 40, 50]),
        ("discontinuities", 2, [40, 50]),
        ("discontinuities", 4, [40, 50]),
        ("discontinuities", 8, []),
        ("discontinuities", 20, []),
        ("gaps", False, [10, 30, 40]),
        ("gaps", None, [30, 40]),  # continuity is a gaps
        ("gaps", 1, [30, 40]),
        ("gaps", 2, [40]),
        ("gaps", 4, [40]),
        ("gaps", 8, []),
        ("gaps", 20, []),
        ("overlaps", False, [20, 50]),
        ("overlaps", None, [20, 50]),  # continuity is not an overlaps
        ("overlaps", 1, [20, 50]),
        ("overlaps", 2, [50]),
        ("overlaps", 4, [50]),
        ("overlaps", 8, []),
        ("overlaps", 20, []),
    ]

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    @pytest.mark.parametrize("dtype", [int, float, "datetime64[s]"])
    def test_generic(self, coord):
        for kind, tolerance, expected in self.CASES:
            indices = coord.get_split_indices(kind=kind, tolerance=tolerance)
            np.testing.assert_array_equal(indices, expected)

    @pytest.mark.parametrize("ctype", ["interpolated", "sampled"])
    def test_wrong_kind(self, ctype):
        with pytest.raises(ValueError):
            xd.Coordinate[ctype]().get_split_indices("wrong_kind")
