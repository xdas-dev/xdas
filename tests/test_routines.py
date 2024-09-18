import numpy as np
import pytest

from xdas.core.coordinates import Coordinates
from xdas.core.dataarray import DataArray
from xdas.core.routines import Bag, CompatibilityError, combine_by_coords


class TestBag:
    def test_bag_initialization(self):
        bag = Bag(dim="time")
        assert bag.dim == "time"
        assert bag.objs == []

    def test_bag_append_initializes(self):
        da = DataArray(
            np.random.rand(10, 5), {"time": np.arange(10), "space": np.arange(5)}
        )
        bag = Bag(dim="time")
        bag.append(da)
        assert len(bag.objs) == 1
        assert bag.objs[0] is da
        assert bag.subcoords.equals(Coordinates({"space": np.arange(5)}))
        assert bag.subshape == (5,)
        assert bag.dims == ("time", "space")
        assert bag.delta

    def test_bag_append_compatible(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        bag.append(da2)
        assert len(bag.objs) == 2
        assert bag.objs[1] is da2
        da1 = DataArray(
            np.random.rand(10, 5), {"time": np.arange(10), "space": np.arange(5)}
        )
        da2 = DataArray(
            np.random.rand(10, 5), {"time": np.arange(10, 20), "space": np.arange(5)}
        )
        bag = Bag(dim="time")
        bag.append(da1)
        bag.append(da2)
        assert len(bag.objs) == 2
        assert bag.objs[1] is da2

    def test_bag_append_incompatible_dims(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 5), dims=("space", "time"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_shape(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 6), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_dtype(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.randint(0, 10, size=(10, 5)), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_coords(self):
        da1 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5)},
        )
        da2 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5) + 1},
        )
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)

    def test_bag_append_incompatible_sampling_interval(self):
        da1 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"time": np.arange(10)},
        )
        da2 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"time": np.arange(10) * 2},
        )
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(CompatibilityError):
            bag.append(da2)


class TestCombineByCoords:
    def test_basic(self):
        # without coords
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        combined = combine_by_coords([da1, da2], dim="time", squeeze=True)
        assert combined.shape == (20, 5)

        # with coords
        da1 = DataArray(
            np.random.rand(10, 5),
            coords={"time": np.arange(10), "space": np.arange(5)},
        )
        da2 = DataArray(
            np.random.rand(10, 5),
            coords={"time": np.arange(10, 20), "space": np.arange(5)},
        )
        combined = combine_by_coords([da1, da2], dim="time", squeeze=True)
        assert combined.shape == (20, 5)

    def test_incompatible_shape(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 6), dims=("time", "space"))
        dc = combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_dims(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 5), dims=("space", "time"))
        dc = combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_dtype(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.randint(0, 10, size=(10, 5)), dims=("time", "space"))
        dc = combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_coords(self):
        da1 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5)},
        )
        da2 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"space": np.arange(5) + 1},
        )
        dc = combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_incompatible_sampling_interval(self):
        da1 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"time": np.arange(10)},
        )
        da2 = DataArray(
            np.random.rand(10, 5),
            dims=("time", "space"),
            coords={"time": np.arange(10) * 2},
        )
        dc = combine_by_coords([da1, da2], dim="time")
        assert len(dc) == 2
        assert dc[0].equals(da1)
        assert dc[1].equals(da2)

    def test_expand_scalar_coordinate(self):
        da1 = DataArray(
            np.random.rand(10),
            dims=("time",),
            coords={"time": np.arange(10), "space": 0},
        )
        da2 = DataArray(
            np.random.rand(10),
            dims=("time",),
            coords={"time": np.arange(10), "space": 1},
        )
        dc = combine_by_coords([da1, da2], dim="space", squeeze=True)
        assert dc.shape == (2, 10)
        assert dc.dims == ("space", "time")
        assert dc.coords["space"].values.tolist() == [0, 1]
