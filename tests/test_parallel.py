import numpy as np
import pytest
import scipy.signal as sp

from xdas.parallel import concatenate, get_workers_count, parallelize


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


    def test_parallel_multiple_outputs(self):
        # Force 2 workers to hit the parallel output path (line 104)
        func = parallelize(split_axis=(0, 0), concat_axis=(0, 0), parallel=2)(
            lambda x, y: (x + y, x - y)
        )
        x = np.random.rand(10, 5)
        y = np.random.rand(10, 5)
        result = func(x, y)
        assert len(result) == 2

    def test_parallel_size_mismatch(self):
        # Two inputs with different sizes along split axis → raises ValueError
        func = parallelize(split_axis=(0, 0), parallel=2)(np.add)
        x = np.random.rand(10, 5)
        y = np.random.rand(8, 5)  # different size on axis 0
        with pytest.raises(ValueError, match="mismatch in size"):
            func(x, y)

    def test_parallel_single_output(self):
        # parallel=2 + single-output function → covers fn tuplize path and output[0] return
        func = parallelize(parallel=2)(np.square)
        x = np.random.rand(10, 5)
        result = func(x)
        np.testing.assert_array_equal(result, np.square(x))

    def test_input_ndim_less_than_split_axis(self):
        # array ndim <= split_axis → early return from fn
        func = parallelize(split_axis=2)(np.square)
        x = np.random.rand(5)  # ndim=1, split_axis=2: 1 <= 2 → early exit
        result = func(x)
        np.testing.assert_array_equal(result, np.square(x))


class TestConcatenate:
    def test_concatenate(self):
        arrays = [np.random.rand(100, 20) for _ in range(100)]
        expected = np.concatenate(arrays)
        result = concatenate(arrays)
        assert np.array_equal(expected, result)
        expected = np.concatenate(arrays, axis=1)
        result = concatenate(arrays, axis=1)
        assert np.array_equal(expected, result)

    def test_different_ndims(self):
        with pytest.raises(ValueError, match="same number of dimensions"):
            concatenate([np.ones((3, 4)), np.ones((3,))])

    def test_different_dtypes(self):
        with pytest.raises(ValueError, match="same dtype"):
            concatenate([np.ones((3,), dtype=np.float32), np.ones((3,), dtype=np.float64)])

    def test_different_shape_other_axis(self):
        with pytest.raises(ValueError, match="same shape"):
            concatenate([np.ones((3, 4)), np.ones((3, 5))])

    def test_out_parameter(self):
        arrays = [np.ones((5, 3)), np.ones((5, 3))]
        out = np.empty((10, 3))
        result = concatenate(arrays, out=out)
        assert np.array_equal(result, np.ones((10, 3)))
        assert result is out

    def test_out_wrong_shape(self):
        arrays = [np.ones((5, 3)), np.ones((5, 3))]
        out = np.empty((9, 3))  # wrong shape
        with pytest.raises(ValueError, match="does not match"):
            concatenate(arrays, out=out)


class TestGetWorkersCount:
    def test_none_uses_config(self):
        # conftest sets n_workers=1
        assert get_workers_count(None) == 1

    def test_bool_true(self):
        import os
        assert get_workers_count(True) == os.cpu_count()

    def test_bool_false(self):
        assert get_workers_count(False) == 1

    def test_int(self):
        assert get_workers_count(4) == 4
