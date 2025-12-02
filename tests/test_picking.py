import numpy as np

import xdas as xd
from xdas.picking import tapered_selection


class TestTaperedSelection:
    def test_basic_functionality(self):
        da = xd.DataArray(
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
