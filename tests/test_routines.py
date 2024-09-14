import numpy as np
import pytest
from xdas.core.routines import Bag, SplitError
from xdas.core.dataarray import DataArray
from xdas.core.coordinates import Coordinates


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
        with pytest.raises(SplitError):
            bag.append(da2)

    def test_bag_append_incompatible_shape(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.rand(10, 6), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(SplitError):
            bag.append(da2)

    def test_bag_append_incompatible_dtype(self):
        da1 = DataArray(np.random.rand(10, 5), dims=("time", "space"))
        da2 = DataArray(np.random.randint(0, 10, size=(10, 5)), dims=("time", "space"))
        bag = Bag(dim="time")
        bag.append(da1)
        with pytest.raises(SplitError):
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
        with pytest.raises(SplitError):
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
        with pytest.raises(SplitError):
            bag.append(da2)
