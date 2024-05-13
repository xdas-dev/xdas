import numpy as np
import pandas as pd

import xdas as xd
from xdas.trigger import _find_picks_numeric, find_picks


def test_find_picks_numeric():
    # Test case
    cft = np.array([[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]])
    thresh = 0.5

    # Test axis = -1
    axis = -1
    expected_coords = (np.array([0, 0]), np.array([2, 7]))
    expected_values = np.array([0.9, 0.7])

    coords, values = _find_picks_numeric(cft, thresh, axis)
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # Test axis = 0
    axis = 0
    cft_transposed = cft.T
    expected_coords = (np.array([2, 7]), np.array([0, 0]))
    expected_values = np.array([0.9, 0.7])

    coords, values = _find_picks_numeric(cft_transposed, thresh, axis)
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # Test 1d array
    cft_squeezed = cft[0]
    expected_coords = (np.array([2, 7]),)
    expected_values = np.array([0.9, 0.7])

    coords, values = _find_picks_numeric(cft_squeezed, thresh, axis)
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # Test chunked processing without pick in previous chunk
    axis = -1
    split = 7
    first_chunk, last_chunk = cft[:, :split], cft[:, split:]

    coords, values, state = _find_picks_numeric(first_chunk, thresh, axis, state=...)
    expected_coords = (np.array([0]), np.array([2]))
    expected_values = np.array([0.9])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    coords, values, state = _find_picks_numeric(last_chunk, thresh, axis, state=state)
    expected_coords = (np.array([0]), np.array([7]) - split)
    expected_values = np.array([0.7])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # Test chunked processing with pick in previous chunk
    axis = -1
    split = 3
    first_chunk, last_chunk = cft[:, :split], cft[:, split:]

    coords, values, state = _find_picks_numeric(first_chunk, thresh, axis, state=...)
    expected_coords = (np.array([]), np.array([]))
    expected_values = np.array([])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    coords, values, state = _find_picks_numeric(last_chunk, thresh, axis, state=state)
    expected_coords = (np.array([0, 0]), np.array([-1, 7 - split]))
    expected_values = np.array([0.9, 0.7])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)

    # Test chunked processing with pick in two chunk ago
    cft = [1.0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.0]
    chunk_0, chunk_1, chunk_2 = cft[:3], cft[3:6], cft[6:]

    coords, values, state = _find_picks_numeric(chunk_0, thresh, axis, state=...)
    coords, values, state = _find_picks_numeric(chunk_1, thresh, axis, state=state)
    coords, values, state = _find_picks_numeric(chunk_2, thresh, axis, state=state)

    expected_coords = (np.array([-6]),)
    expected_values = np.array([1.0])
    assert np.array_equal(coords, expected_coords)
    assert np.array_equal(values, expected_values)


def test_find_picks():
    cft = xd.DataArray(
        data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
        coords={"space": [0.0], "time": np.linspace(0, 9.0, 10)},
    )
    picks = find_picks(cft, thresh=0.5, dim="time")
    expected = pd.DataFrame(
        {"space": [0.0, 0.0], "time": [2.0, 7.0], "value": [0.9, 0.7]}
    )
    assert picks.equals(expected)
