import numpy as np
import pandas as pd
import pytest

import xdas as xd
from xdas.trigger import Trigger, _concat, _find_picks_numeric, find_picks


def test_trigger():
    # test case
    cft = xd.DataArray(
        data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
        coords={
            "space": [0.0],
            "time": {"tie_indices": [0, 9], "tie_values": [0.0, 9.0]},
        },
    )

    # test monolithic processing
    picks = Trigger(thresh=0.5, dim="time")(cft)
    expected = pd.DataFrame(
        {"space": [0.0, 0.0], "time": [2.0, 7.0], "value": [0.9, 0.7]}
    )
    assert picks.equals(expected)

    # test chunked processing
    trigger = Trigger(thresh=0.5, dim="time")
    chunks = xd.split(cft, 3, dim="time")
    result = []
    for chunk in chunks:
        picks = trigger(chunk, chunk_dim="time")
        result.append(picks)
    result = pd.concat(result, ignore_index=True)
    assert result.equals(expected)


def test_find_picks_numeric():
    # test case
    cft = np.array([[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]])
    thresh = 0.5

    # test axis = -1
    axis = -1
    expected_coords = (np.array([0, 0]), np.array([2, 7]))
    expected_values = np.array([0.9, 0.7])

    coords, values = _find_picks_numeric(cft, thresh, axis)
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # test axis = 0
    axis = 0
    cft_transposed = cft.T
    expected_coords = (np.array([2, 7]), np.array([0, 0]))
    expected_values = np.array([0.9, 0.7])

    coords, values = _find_picks_numeric(cft_transposed, thresh, axis)
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # test 1d array
    cft_squeezed = cft[0]
    expected_coords = (np.array([2, 7]),)
    expected_values = np.array([0.9, 0.7])

    coords, values = _find_picks_numeric(cft_squeezed, thresh, axis)
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # test chunked processing without pick in previous chunk
    axis = -1
    split = 7
    first_chunk, last_chunk = cft[:, :split], cft[:, split:]

    coords, values, buffer, offset = _find_picks_numeric(
        first_chunk, thresh, axis, buffer=..., offset=...
    )
    expected_coords = (np.array([0]), np.array([2]))
    expected_values = np.array([0.9])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    coords, values, buffer, offset = _find_picks_numeric(
        last_chunk, thresh, axis, buffer=buffer, offset=offset
    )
    expected_coords = (np.array([0]), np.array([7]))
    expected_values = np.array([0.7])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # test chunked processing with pick in previous chunk
    axis = -1
    split = 3
    first_chunk, last_chunk = cft[:, :split], cft[:, split:]

    coords, values, buffer, offset = _find_picks_numeric(
        first_chunk, thresh, axis, buffer=..., offset=...
    )
    expected_coords = (np.array([]), np.array([]))
    expected_values = np.array([])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    coords, values, buffer, offset = _find_picks_numeric(
        last_chunk, thresh, axis, buffer=buffer, offset=offset
    )
    expected_coords = (np.array([0, 0]), np.array([2, 7]))
    expected_values = np.array([0.9, 0.7])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # test chunked processing with pick in two chunk ago
    cft = [1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]
    chunk_0, chunk_1, chunk_2 = cft[:3], cft[3:6], cft[6:]

    coords, values, buffer, offset = _find_picks_numeric(
        chunk_0, thresh, axis, buffer=..., offset=...
    )
    coords, values, buffer, offset = _find_picks_numeric(
        chunk_1, thresh, axis, buffer=buffer, offset=offset
    )
    coords, values, buffer, offset = _find_picks_numeric(
        chunk_2, thresh, axis, buffer=buffer, offset=offset
    )

    expected_coords = (np.array([0]),)
    expected_values = np.array([1.0])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)


def test_find_picks():
    # test case
    cft = xd.DataArray(
        data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
        coords={
            "space": [0.0],
            "time": {"tie_indices": [0, 9], "tie_values": [0.0, 9.0]},
        },
    )

    # test monolithic processing
    picks = find_picks(cft, thresh=0.5, dim="time")
    expected = pd.DataFrame(
        {"space": [0.0, 0.0], "time": [2.0, 7.0], "value": [0.9, 0.7]}
    )
    assert picks.equals(expected)

    # test chunked processing
    chunks = xd.split(cft, 3, dim="time")
    state_dict = ...
    result = []
    for chunk in chunks:
        picks, state_dict = find_picks(
            chunk, thresh=0.5, dim="time", state_dict=state_dict
        )
        result.append(picks)
    result = pd.concat(result, ignore_index=True)
    assert result.equals(expected)

    # test atomic processing
    atom = find_picks(..., thresh=0.5, dim="time", state_dict=...)
    result = []
    for chunk in chunks:
        result.append(atom(chunk, chunk_dim="time"))
    result = pd.concat(result, ignore_index=True)
    assert result.equals(expected)


def test_trigger_1d():
    """1D input (no spatial dimension) covers the coords=() branch in _call_numeric."""
    cft = xd.DataArray(
        data=[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2],
        coords={
            "time": {"tie_indices": [0, 9], "tie_values": [0.0, 9.0]},
        },
    )
    picks = Trigger(thresh=0.5, dim="time")(cft)
    assert len(picks) == 2
    assert list(picks["time"]) == [2.0, 7.0]


def test_concat_non_interp_coord():
    """_concat raises ValueError for non-interpolated coordinates."""
    from xdas.coordinates.sampled import SampledCoordinate

    coord1 = xd.Coordinate({"tie_indices": [0, 2], "tie_values": [10, 30]}, dim="dim")
    coord_bad = SampledCoordinate(
        {"tie_values": [0.0], "tie_lengths": [3], "sampling_interval": 1.0}, "dim"
    )
    with pytest.raises(ValueError, match="interpolated"):
        _concat([coord1, coord_bad])


def test_concat_different_dims():
    """_concat raises ValueError when coords have different dims."""
    coord1 = xd.Coordinate({"tie_indices": [0, 2], "tie_values": [10, 30]}, dim="dim1")
    coord2 = xd.Coordinate({"tie_indices": [0, 2], "tie_values": [40, 60]}, dim="dim2")
    with pytest.raises(ValueError, match="same dimension"):
        _concat([coord1, coord2])
