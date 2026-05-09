import numpy as np
import pytest

import xdas as xd


class TestGetSplitIndices:
    @pytest.fixture
    def float_coord(self, ctype):
        starts = [
            0.0,  # 0 - initial block
            3.0,  # 3 - continuous
            5.5,  # 6 - 0.5 overlap
            9.0,  # 9 - 0.5 gap
            14.0,  # 12 - 2 gap
            15.0,  # 15 - 2 overlap
        ]
        size = 3
        step = 1.0
        coord = xd.Coordinate[ctype](data=None, dim="dim", dtype=float)
        for start in starts:
            coord = coord.append(
                xd.Coordinate[ctype].from_block(start, size, step, "dim")
            )
        return coord

    @pytest.mark.parametrize("ctype", ["interpolated"])
    @pytest.mark.parametrize(
        "kind,tolerance,expected",
        [
            ("discontinuities", False, [3, 6, 9, 12, 15]),
            ("discontinuities", None, [6, 9, 12, 15]),
            ("discontinuities", 0.25, [6, 9, 12, 15]),
            ("discontinuities", 0.5, [12, 15]),
            ("discontinuities", 1.0, [12, 15]),
            ("discontinuities", 2.0, []),
            ("discontinuities", 5.0, []),
            ("gap", False, [3, 9, 12]),
            ("gap", None, [9, 12]),
            ("gap", 0.25, [9, 12]),
            ("gap", 0.5, [12]),
            ("gap", 1.0, [12]),
            ("gap", 2.0, []),
            ("gap", 5.0, []),
            ("overlap", False, [6, 15]),
            ("overlap", None, [6, 15]),
            ("overlap", 0.25, [6, 15]),
            ("overlap", 0.5, [15]),
            ("overlap", 1.0, [15]),
            ("overlap", 2.0, []),
            ("overlap", 5.0, []),
        ],
    )
    def test_float(self, float_coord, kind, tolerance, expected):
        indices = float_coord.get_split_indices(kind=kind, tolerance=tolerance)
        np.testing.assert_array_equal(indices, expected)
