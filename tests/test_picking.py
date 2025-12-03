import numpy as np
import pandas as pd
import pytest

import xdas as xd
from xdas.picking import WaveFront, WaveFrontCollection, tapered_selection


class TestWaveFront:
    def test_init(self):
        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
            xd.DataArray(
                data=[2.0, 3.0, 2.0],
                coords={"distance": [4.0, 5.0, 6.0]},
            ),
        ]
        wavefront = WaveFront(horizons)
        assert wavefront.dim == "distance"
        assert wavefront.dtype == np.dtype(float)
        assert wavefront.name == "horizon"
        assert len(wavefront) == 2
        assert wavefront[0].equals(horizons[0])
        assert wavefront[1].equals(horizons[1])

    def test_init_errors(self):
        horizons = [
            xd.DataArray(
                data=[[1.0, 2.0], [1.0, 2.0]],
            ),
        ]
        with pytest.raises(ValueError, match="All horizons must be 1D"):
            WaveFront(horizons)

        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
            xd.DataArray(
                data=[2.0, 3.0, 2.0],
                coords={"time": [4.0, 5.0, 6.0]},
            ),
        ]
        with pytest.raises(
            ValueError, match="All horizons must have the same dimension"
        ):
            WaveFront(horizons)

        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
            xd.DataArray(
                data=[2, 3, 2],
                coords={"distance": [4.0, 5.0, 6.0]},
            ),
        ]
        with pytest.raises(ValueError, match="All horizons must have the same dtype"):
            WaveFront(horizons)

        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
            xd.DataArray(
                data=[2.0, 3.0, 2.0],
                coords={"distance": [1.0, 2.0, 3.0]},
            ),
        ]
        with pytest.raises(ValueError, match="Horizons are overlapping"):
            WaveFront(horizons)

    def test_from_picks(self):
        picks = pd.DataFrame(
            {
                "time": [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 7.0, 8.0, 7.0],
                "distance": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0],
            }
        )
        wavefront = WaveFront.from_picks(picks, gap_threshold=1.0)
        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
            xd.DataArray(
                data=[7.0, 8.0, 7.0],
                coords={"distance": [4.0, 5.0, 6.0]},
            ),
        ]
        expected = WaveFront(horizons)
        assert wavefront.equals(expected)

    def test_interp(self):
        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
            xd.DataArray(
                data=[2.0, 3.0, 2.0],
                coords={"distance": [4.0, 5.0, 6.0]},
            ),
        ]
        wavefront = WaveFront(horizons)

        coords = [-0.5, 0.5, 1.5, 3.0, 4.5, 5.5, 6.5]
        result = wavefront.interp(coords)

        expected = xd.DataArray(
            data=[np.nan, 1.5, 1.5, np.nan, 2.5, 2.5, np.nan],
            coords={"distance": coords},
        )

        assert result.equals(expected)

    def test_interp_errors(self):
        horizons = [
            xd.DataArray(
                data=[1.0, 2.0, 1.0],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
        ]
        wavefront = WaveFront(horizons)

        coords = [[0.0, 1.0], [2.0, 3.0]]
        with pytest.raises(
            ValueError, match="`coords` must be an 1D array-like object"
        ):
            wavefront.interp(coords)

    def test_interp_timedelta(self):
        horizons = [
            xd.DataArray(
                data=[
                    np.datetime64("2023-01-01T00:00:01.0"),
                    np.datetime64("2023-01-01T00:00:02.0"),
                    np.datetime64("2023-01-01T00:00:03.0"),
                ],
                coords={"distance": [0.0, 1.0, 2.0]},
            ),
        ]
        wavefront = WaveFront(horizons)

        coords = [0.5, 1.5]
        result = wavefront.interp(coords)

        expected = xd.DataArray(
            data=[
                np.datetime64("2023-01-01T00:00:01.5"),
                np.datetime64("2023-01-01T00:00:02.5"),
            ],
            coords={"distance": coords},
        )

        assert result.equals(expected)


class TestWaveFrontCollection:
    def test_init(self):
        wavefronts = {
            "P": WaveFront(
                [
                    xd.DataArray(
                        data=[1.0, 2.0, 1.0],
                        coords={"distance": [0.0, 1.0, 2.0]},
                    ),
                ],
            ),
            "S": WaveFront(
                [
                    xd.DataArray(
                        data=[2.0, 3.0, 2.0],
                        coords={"distance": [0.0, 1.0, 2.0]},
                    ),
                ],
            ),
        }
        collection = WaveFrontCollection(wavefronts)
        assert len(collection) == 2
        assert "P" in collection
        assert "S" in collection
        assert collection["P"].equals(wavefronts["P"])
        assert collection["S"].equals(wavefronts["S"])
        assert collection.dim == "distance"
        assert collection.dtype == np.dtype(float)

        wavefronts = {
            "P": [
                xd.DataArray(
                    data=[1.0, 2.0, 1.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
            "S": [
                xd.DataArray(
                    data=[2.0, 3.0, 2.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
        }
        collection = WaveFrontCollection(wavefronts)

        assert len(collection) == 2
        assert "P" in collection
        assert "S" in collection
        assert collection["P"].equals(WaveFront(wavefronts["P"]))
        assert collection["S"].equals(WaveFront(wavefronts["S"]))
        assert collection.dim == "distance"
        assert collection.dtype == np.dtype(float)

    def test_init_errors(self):
        wavefronts = {
            "P": [
                xd.DataArray(
                    data=[1.0, 2.0, 1.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
            "S": [
                xd.DataArray(
                    data=[2.0, 3.0, 2.0],
                    coords={"time": [0.0, 1.0, 2.0]},
                ),
            ],
        }
        with pytest.raises(
            ValueError, match="All wavefronts must have the same dimension"
        ):
            WaveFrontCollection(wavefronts)

        wavefronts = {
            "P": [
                xd.DataArray(
                    data=[1.0, 2.0, 1.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
            "S": [
                xd.DataArray(
                    data=[2, 3, 2],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
        }
        with pytest.raises(ValueError, match="All wavefronts must have the same dtype"):
            WaveFrontCollection(wavefronts)

    def test_from_picks(self):
        picks = pd.DataFrame(
            {
                "time": [1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 7.0, 8.0, 7.0],
                "distance": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0],
                "phase": ["P", "P", "P", "S", "S", "S", "S", "S", "S"],
            }
        )
        collection = WaveFrontCollection.from_picks(picks, gap_threshold=1.0)
        wavefronts = {
            "P": [
                xd.DataArray(
                    data=[1.0, 2.0, 1.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
            "S": [
                xd.DataArray(
                    data=[2.0, 3.0, 2.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
                xd.DataArray(
                    data=[7.0, 8.0, 7.0],
                    coords={"distance": [4.0, 5.0, 6.0]},
                ),
            ],
        }
        expected = WaveFrontCollection(wavefronts)
        assert collection.equals(expected)

    def test_interp(self):
        wavefronts = {
            "P": [
                xd.DataArray(
                    data=[1.0, 2.0, 1.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
            "S": [
                xd.DataArray(
                    data=[2.0, 3.0, 2.0],
                    coords={"distance": [0.0, 1.0, 2.0]},
                ),
            ],
        }
        collection = WaveFrontCollection(wavefronts)

        coords = [-0.5, 0.5, 1.5, 2.5]
        result = collection.interp(coords)

        expected = xd.DataCollection(
            {
                "P": xd.DataArray(
                    data=[np.nan, 1.5, 1.5, np.nan],
                    coords={"distance": coords},
                ),
                "S": xd.DataArray(
                    data=[np.nan, 2.5, 2.5, np.nan],
                    coords={"distance": coords},
                ),
            },
            "wavefront",
        )

        assert result["P"].equals(expected["P"])
        assert result["S"].equals(expected["S"])


class TestTaperedSelection:
    def generate(self):
        return xd.DataArray(
            data=np.arange(5 * 10).reshape(5, 10).astype(float),
            coords={
                "distance": {
                    "tie_indices": [0, 4],
                    "tie_values": [0.0, 400.0],
                },
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [
                        np.datetime64("2023-01-01T00:00:00"),
                        np.datetime64("2023-01-01T00:00:09"),
                    ],
                },
            },
        )

    def test_basic_functionality(self):
        da = self.generate()

        start = xd.DataArray(
            data=np.array(
                [np.datetime64("NaT")]
                + [np.datetime64("2023-01-01T00:00:03")] * 2
                + [np.datetime64("NaT")] * 2
            ),
            coords={"distance": da["distance"]},
        )
        end = (
            [np.datetime64("NaT")]
            + [np.datetime64("2023-01-01T00:00:07")] * 2
            + [np.datetime64("NaT")] * 2
        )
        window = [0.5, 1.0, 0.5]

        result = tapered_selection(da, start, end, window, dim="time")

        expected = xd.DataArray(
            data=[
                [6.5, 14.0, 15.0, 16.0, 8.5],
                [11.5, 24.0, 25.0, 26.0, 13.5],
            ],
            coords={
                "distance": [100.0, 200.0],
                "time": {
                    "tie_indices": [0, 4],
                    "tie_values": [0.0, 4.0],
                },
            },
        )

        assert result.equals(expected)

    def test_window_size_error(self):
        da = self.generate()

        start = xd.DataArray(
            data=np.array(
                [np.datetime64("NaT")]
                + [np.datetime64("2023-01-01T00:00:08")] * 2
                + [np.datetime64("NaT")] * 2
            ),
            coords={"distance": da["distance"]},
        )
        end = (
            [np.datetime64("NaT")]
            + [np.datetime64("2023-01-01T00:00:09")] * 2
            + [np.datetime64("NaT")] * 2
        )
        window = [0.5, 1.0, 0.5]

        with pytest.raises(
            ValueError, match="some selected windows are smaller than the window size"
        ):
            tapered_selection(da, start, end, window, dim="time")

    def test_no_window(self):
        da = self.generate()

        start = xd.DataArray(
            data=np.array(
                [np.datetime64("NaT")]
                + [np.datetime64("2023-01-01T00:00:03")] * 2
                + [np.datetime64("NaT")] * 2
            ),
            coords={"distance": da["distance"]},
        )
        end = (
            [np.datetime64("NaT")]
            + [np.datetime64("2023-01-01T00:00:07")] * 2
            + [np.datetime64("NaT")] * 2
        )

        result = tapered_selection(da, start, end, dim="time")

        expected = xd.DataArray(
            data=[
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [23.0, 24.0, 25.0, 26.0, 27.0],
            ],
            coords={
                "distance": [100.0, 200.0],
                "time": {
                    "tie_indices": [0, 4],
                    "tie_values": [0.0, 4.0],
                },
            },
        )

        assert result.equals(expected)

    def test_with_size(self):
        da = self.generate()

        start = xd.DataArray(
            data=np.array(
                [np.datetime64("NaT")]
                + [np.datetime64("2023-01-01T00:00:03")] * 2
                + [np.datetime64("NaT")] * 2
            ),
            coords={"distance": da["distance"]},
        )
        end = (
            [np.datetime64("NaT")]
            + [np.datetime64("2023-01-01T00:00:07")] * 2
            + [np.datetime64("NaT")] * 2
        )

        result = tapered_selection(da, start, end, size=8, dim="time")

        expected = xd.DataArray(
            data=[
                [13.0, 14.0, 15.0, 16.0, 17.0, 0.0, 0.0, 0.0],
                [23.0, 24.0, 25.0, 26.0, 27.0, 0.0, 0.0, 0.0],
            ],
            coords={
                "distance": [100.0, 200.0],
                "time": {
                    "tie_indices": [0, 7],
                    "tie_values": [0.0, 7.0],
                },
            },
        )

        assert result.equals(expected)

    def test_different_selection_lengths(self):
        da = self.generate()

        start = xd.DataArray(
            data=np.array(
                [np.datetime64("NaT")]
                + [np.datetime64("2023-01-01T00:00:03")]
                + [np.datetime64("2023-01-01T00:00:04")]
                + [np.datetime64("NaT")] * 2
            ),
            coords={"distance": da["distance"]},
        )
        end = (
            [np.datetime64("NaT")]
            + [np.datetime64("2023-01-01T00:00:07")] * 2
            + [np.datetime64("NaT")] * 2
        )

        result = tapered_selection(da, start, end, dim="time")

        expected = xd.DataArray(
            data=[
                [13.0, 14.0, 15.0, 16.0, 17.0],
                [24.0, 25.0, 26.0, 27.0, 0.0],
            ],
            coords={
                "distance": [100.0, 200.0],
                "time": {
                    "tie_indices": [0, 4],
                    "tie_values": [0.0, 4.0],
                },
            },
        )

        assert result.equals(expected)

    def test_other_dim(self):
        da = self.generate()

        start = [np.nan] + [100.0] * 2 + [np.nan] * 7
        end = [np.nan] + [300.0] * 2 + [np.nan] * 7
        window = [0.5, 1.0, 0.5]

        result = tapered_selection(da, start, end, window, dim="distance")

        expected = xd.DataArray(
            data=[
                [5.5, 21.0, 15.5],
                [6.0, 22.0, 16.0],
            ],
            coords={
                "time": [
                    np.datetime64("2023-01-01T00:00:01"),
                    np.datetime64("2023-01-01T00:00:02"),
                ],
                "distance": {
                    "tie_indices": [0, 2],
                    "tie_values": [0.0, 200.0],
                },
            },
        )

        assert result.equals(expected)

    def test_no_valid_selections(self):
        da = self.generate()

        start = [np.datetime64("NaT")] * 5
        end = [np.datetime64("NaT")] * 5
        window = [0.5, 1.0, 0.5]

        with pytest.raises(ValueError, match="No valid start/end pairs found"):
            tapered_selection(da, start, end, window, dim="time")
