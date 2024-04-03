import numpy as np
import scipy.signal as sp

from xdas.parallel import concatenate, parallelize


class TestParallelize:
    def test_ufunc(self):
        func = parallelize()(np.square)
        x = np.random.rand(2, 3, 5)
        expected = np.square(x)
        result = func(x)
        assert np.array_equal(result, expected)

    def test_two_in_one_out(self):
        func = parallelize(split_axis=(0, 0))(np.multiply)
        x = np.random.rand(2, 3, 5)
        y = np.random.rand(2, 3, 5)
        expected = np.multiply(x, y)
        result = func(x, y)
        assert np.array_equal(result, expected)

    def test_lfilter(self):
        # axis=-1
        func = parallelize(
            split_axis=(None, None, 0, None, 0),
            concat_axis=(0, 0),
        )(sp.lfilter)
        b = [1, 1]
        a = [1, 1]
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((20, 3, 1))
        y_exp, zf_exp = sp.lfilter(b, a, x, -1, zi)
        y_res, zf_res = func(b, a, x, -1, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)
        # axis=0
        func = parallelize(
            split_axis=(None, None, 1, None, 1),
            concat_axis=(1, 1),
        )(sp.lfilter)
        b = [1, 1]
        a = [1, 1]
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((1, 3, 50))
        y_exp, zf_exp = sp.lfilter(b, a, x, 0, zi)
        y_res, zf_res = func(b, a, x, 0, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)

    def test_sosfilter(self):
        # axis=-1
        func = parallelize(
            split_axis=(None, 0, None, 1),
            concat_axis=(0, 1),
        )(sp.sosfilt)
        sos = np.ones((5, 6))
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((5, 20, 3, 2))
        y_exp, zf_exp = sp.sosfilt(sos, x, -1, zi)
        y_res, zf_res = func(sos, x, -1, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)
        # axis=0
        func = parallelize(
            split_axis=(None, 1, None, 2),
            concat_axis=(1, 2),
        )(sp.sosfilt)
        sos = np.ones((5, 6))
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((5, 2, 3, 50))
        y_exp, zf_exp = sp.sosfilt(sos, x, 0, zi)
        y_res, zf_res = func(sos, x, 0, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)

    def test_ignore_one_output(self):
        func = parallelize()(sp.resample)
        x = np.random.rand(20, 3, 50)
        t = np.arange(50)
        y_exp, t_exp = sp.resample(x, 30, t, axis=-1)
        y_res, t_res = func(x, 30, t, axis=-1)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(t_res, t_exp)


class TestConcatenate:
    def test_concatenate(self):
        arrays = [np.random.rand(100, 20) for _ in range(100)]
        expected = np.concatenate(arrays)
        result = concatenate(arrays)
        assert np.array_equal(expected, result)
        expected = np.concatenate(arrays, axis=1)
        result = concatenate(arrays, axis=1)
        assert np.array_equal(expected, result)
