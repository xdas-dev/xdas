import numpy as np
import pytest

import xdas as xd


class TestCoordinate:
    def test_new(self):
        assert xd.Coordinate(1).isscalar()
        assert xd.Coordinate([1]).isdense()
        assert xd.Coordinate({"tie_values": [], "tie_indices": []}).isinterp()
        coord = xd.Coordinate(xd.Coordinate([1]), "dim")
        assert coord.isdense()
        assert coord.dim == "dim"

    def test_to_dataarray(self):
        coord = xd.Coordinate([1, 2, 3], "dim")
        result = coord.to_dataarray()
        expected = xd.DataArray([1, 2, 3], {"dim": [1, 2, 3]}, name="dim")
        assert result.equals(expected)

    def test_empty(self):
        with pytest.raises(TypeError, match="cannot infer coordinate type"):
            xd.Coordinate()

    def test_isdim(self):
        coord = xd.Coordinate([1, 2, 3])
        assert coord.isdim() is None
        coord = xd.Coordinate([1, 2, 3], "dim")
        assert coord.isdim() is None
        coords = xd.Coordinates({"dim": coord})
        assert coords["dim"].isdim()
        coords = xd.Coordinates({"other_dim": coord})
        assert not coords["other_dim"].isdim()

    def test_name(self):
        coord = xd.Coordinate([1, 2, 3])
        assert coord.name is None
        coord = xd.Coordinate([1, 2, 3], "dim")
        assert coord.name == "dim"
        coords = xd.Coordinates({"dim": coord})
        assert coords["dim"].name == "dim"
        coords = xd.Coordinates({"other_dim": coord})
        assert coords["other_dim"].name == "other_dim"

    def test_to_dataarray(self):
        coord = xd.Coordinate([1, 2, 3])
        with pytest.raises(ValueError, match="unnamed coordinate"):
            coord.to_dataarray()
        coord = xd.Coordinate([1, 2, 3], "dim")
        result = coord.to_dataarray()
        expected = xd.DataArray([1, 2, 3], {"dim": [1, 2, 3]}, name="dim")
        assert result.equals(expected)
        coords = xd.Coordinates({"dim": coord})
        result = coords["dim"].to_dataarray()
        assert result.equals(expected)
        coords = xd.Coordinates({"other_dim": coord})
        result = coords["other_dim"].to_dataarray()
        expected = xd.DataArray(
            [1, 2, 3], coords={"other_dim": coord}, dims=["dim"], name="other_dim"
        )
        assert result.equals(expected)
        coords["dim"] = [4, 5, 6]
        result = coords["dim"].to_dataarray()
        expected = xd.DataArray(
            [4, 5, 6],
            coords={"dim": [4, 5, 6], "other_dim": ("dim", [1, 2, 3])},
            dims=["dim"],
            name="dim",
        )
        assert result.equals(expected)
        result = coords["other_dim"].to_dataarray()
        expected = xd.DataArray(
            [1, 2, 3],
            coords={"dim": [4, 5, 6], "other_dim": ("dim", [1, 2, 3])},
            dims=["dim"],
            name="other_dim",
        )
        assert result.equals(expected)


class TestCoordinates:
    def test_init(self):
        coords = xd.Coordinates(
            {"dim": ("dim", {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]})}
        )
        coord = coords["dim"]
        assert coord.isinterp()
        assert np.allclose(coord.tie_indices, [0, 8])
        assert np.allclose(coord.tie_values, [100.0, 900.0])
        assert coords.isdim("dim")
        coords = xd.Coordinates({"dim": [1.0, 2.0, 3.0]})
        coord = coords["dim"]
        assert coord.isdense()
        assert np.allclose(coord.values, [1.0, 2.0, 3.0])
        assert coords.isdim("dim")
        coords = xd.Coordinates(
            {
                "dim_0": (
                    "dim_0",
                    {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
                ),
                "dim_1": (
                    "dim_0",
                    {"tie_indices": [0, 8], "tie_values": [100.0, 900.0]},
                ),
            }
        )
        assert coords.isdim("dim_0")
        assert not coords.isdim("dim_1")
        coords = xd.Coordinates()
        assert coords == dict()
        assert coords.dims == tuple()

    def test_first_last(self):
        coords = xd.Coordinates({"dim_0": [1.0, 2.0, 3.0], "dim_1": [1.0, 2.0, 3.0]})
        assert coords["first"].dim == "dim_0"
        assert coords["last"].dim == "dim_1"

    def test_setitem(self):
        coords = xd.Coordinates()
        coords["dim_0"] = [1, 2, 4]
        assert coords.dims == ("dim_0",)
        coords["dim_1"] = {"tie_indices": [0, 10], "tie_values": [0.0, 100.0]}
        assert coords.dims == ("dim_0", "dim_1")
        coords["dim_0"] = [1, 2, 3]
        assert coords.dims == ("dim_0", "dim_1")
        coords["metadata"] = 0
        assert coords.dims == ("dim_0", "dim_1")
        coords["non-dimensional"] = ("dim_0", [-1, -1, -1])
        assert coords.dims == ("dim_0", "dim_1")
        coords["other_dim"] = ("dim_2", [0])
        assert coords.dims == ("dim_0", "dim_1", "dim_2")
        with pytest.raises(TypeError, match="must be of type str"):
            coords[0] = ...

    def test_to_from_dict(self):
        starttime = np.datetime64("2020-01-01T00:00:00.000")
        endtime = np.datetime64("2020-01-01T00:00:10.000")
        coords = {
            "time": {"tie_indices": [0, 999], "tie_values": [starttime, endtime]},
            "distance": np.linspace(0, 1000, 3),
            "channel": ("distance", ["DAS01", "DAS02", "DAS03"]),
            "interrogator": (None, "SRN"),
        }
        coords = xd.Coordinates(coords)
        assert xd.Coordinates.from_dict(coords.to_dict()).equals(coords)
