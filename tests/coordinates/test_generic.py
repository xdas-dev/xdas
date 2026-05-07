import numpy as np
import pytest

import xdas as xd


class TestGetSplitIndices:
    @pytest.mark.parametrize("ctype", ["interpolated"])
    def test_float(self, ctype):
        blocks = [
            xd.Coordinate[ctype].from_block(0.0, 3, 1.0, "time"),  # 0 - initial block
            xd.Coordinate[ctype].from_block(3.0, 3, 1.0, "time"),  # 3 - continuous
            xd.Coordinate[ctype].from_block(5.5, 3, 1.0, "time"),  # 6 - 0.5 overlap
            xd.Coordinate[ctype].from_block(9.0, 3, 1.0, "time"),  # 9 - 0.5 gap
            xd.Coordinate[ctype].from_block(14.0, 3, 1.0, "time"),  # 12 - 2 gap
            xd.Coordinate[ctype].from_block(15.0, 3, 1.0, "time"),  # 15 - 2 overlap
        ]
        coord = xd.Coordinate[ctype](data=None, dim="time", dtype=float)
        for block in blocks:
            coord = coord.append(block)

        # discontinuities
        indices = coord.get_split_indices(kind="discontinuities", tolerance=False)
        assert np.array_equal(indices, [3, 6, 9, 12, 15])
        indices = coord.get_split_indices(kind="discontinuities", tolerance=None)
        assert np.array_equal(indices, [6, 9, 12, 15])
        indices = coord.get_split_indices(kind="discontinuities", tolerance=0.25)
        assert np.array_equal(indices, [6, 9, 12, 15])
        indices = coord.get_split_indices(kind="discontinuities", tolerance=0.5)
        assert np.array_equal(indices, [12, 15])
        indices = coord.get_split_indices(kind="discontinuities", tolerance=1.0)
        assert np.array_equal(indices, [12, 15])
        indices = coord.get_split_indices(kind="discontinuities", tolerance=2.0)
        assert np.array_equal(indices, [])
        indices = coord.get_split_indices(kind="discontinuities", tolerance=5.0)
        assert np.array_equal(indices, [])

        # gap
        indices = coord.get_split_indices(kind="gap", tolerance=False)
        assert np.array_equal(indices, [3, 9, 12])  # continuity is gap
        indices = coord.get_split_indices(kind="gap", tolerance=None)
        assert np.array_equal(indices, [9, 12])
        indices = coord.get_split_indices(kind="gap", tolerance=0.25)
        assert np.array_equal(indices, [9, 12])
        indices = coord.get_split_indices(kind="gap", tolerance=0.5)
        assert np.array_equal(indices, [12])
        indices = coord.get_split_indices(kind="gap", tolerance=1.0)
        assert np.array_equal(indices, [12])
        indices = coord.get_split_indices(kind="gap", tolerance=2.0)
        assert np.array_equal(indices, [])
        indices = coord.get_split_indices(kind="gap", tolerance=5.0)
        assert np.array_equal(indices, [])

        # overlap
        indices = coord.get_split_indices(kind="overlap", tolerance=False)
        assert np.array_equal(indices, [6, 15])  # continuity is not overlap
        indices = coord.get_split_indices(kind="overlap", tolerance=None)
        assert np.array_equal(indices, [6, 15])
        indices = coord.get_split_indices(kind="overlap", tolerance=0.25)
        assert np.array_equal(indices, [6, 15])
        indices = coord.get_split_indices(kind="overlap", tolerance=0.5)
        assert np.array_equal(indices, [15])
        indices = coord.get_split_indices(kind="overlap", tolerance=1.0)
        assert np.array_equal(indices, [15])
        indices = coord.get_split_indices(kind="overlap", tolerance=2.0)
        assert np.array_equal(indices, [])
        indices = coord.get_split_indices(kind="overlap", tolerance=5.0)
        assert np.array_equal(indices, [])
