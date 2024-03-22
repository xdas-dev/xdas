import numpy as np

from xdas.parallel import multithreaded_concatenate


class TestParallel:
    def test_multithreaded_concatenate(self):
        arrays = [np.random.rand(100, 20) for _ in range(100)]
        expected = np.concatenate(arrays)
        result = multithreaded_concatenate(arrays)
        assert np.array_equal(expected, result)
        expected = np.concatenate(arrays, axis=1)
        result = multithreaded_concatenate(arrays, axis=1)
        assert np.array_equal(expected, result)
