"""Tests for xdas/core/methods.py — covers the uncovered branches."""

import numpy as np
import pytest

import xdas as xd


@pytest.fixture
def float_da():
    return xd.DataArray(
        data=np.arange(12.0).reshape(3, 4),
        coords={
            "x": {"tie_indices": [0, 2], "tie_values": [0.0, 2.0]},
            "y": {"tie_indices": [0, 3], "tie_values": [0.0, 3.0]},
        },
    )


@pytest.fixture
def int_da():
    return xd.DataArray(
        data=np.arange(12).reshape(3, 4),
        coords={
            "x": {"tie_indices": [0, 2], "tie_values": [0.0, 2.0]},
            "y": {"tie_indices": [0, 3], "tie_values": [0.0, 3.0]},
        },
    )


class TestMethods:
    # --- cumprod / cumsum float+skipna=True (nancumprod/nancumsum) ---

    def test_cumprod_float(self, float_da):
        result = float_da.cumprod("y")
        np.testing.assert_array_equal(result.data, np.nancumprod(float_da.data, axis=1))

    def test_cumsum_float(self, float_da):
        result = float_da.cumsum("y")
        np.testing.assert_array_equal(result.data, np.nancumsum(float_da.data, axis=1))

    # --- cumprod / cumsum with integer dtype (non-skipna branch) ---

    def test_cumprod_integer(self, int_da):
        result = int_da.cumprod("y")
        assert result.shape == int_da.shape
        np.testing.assert_array_equal(result.data, np.cumprod(int_da.data, axis=1))

    def test_cumsum_integer(self, int_da):
        result = int_da.cumsum("y")
        assert result.shape == int_da.shape
        np.testing.assert_array_equal(result.data, np.cumsum(int_da.data, axis=1))

    # --- skipna=False branches ---

    def test_cumprod_skipna_false(self, float_da):
        result = float_da.cumprod("y", skipna=False)
        np.testing.assert_array_equal(result.data, np.cumprod(float_da.data, axis=1))

    def test_cumsum_skipna_false(self, float_da):
        result = float_da.cumsum("y", skipna=False)
        np.testing.assert_array_equal(result.data, np.cumsum(float_da.data, axis=1))

    # --- dim=None (axis=None) branches ---

    def test_all_no_dim(self, int_da):
        result = int_da.all()
        assert result.values == np.all(int_da.data)

    def test_all_with_dim(self, int_da):
        result = int_da.all("x")
        assert result.shape == (4,)

    def test_any_no_dim(self, int_da):
        result = int_da.any()
        assert result.values == np.any(int_da.data)

    def test_any_with_dim(self, int_da):
        result = int_da.any("x")
        assert result.shape == (4,)

    def test_max_no_dim(self, float_da):
        assert float_da.max().values == pytest.approx(11.0)

    def test_max_skipna_false(self, int_da):
        int_da.max("x", skipna=False)

    def test_min_no_dim(self, float_da):
        assert float_da.min().values == pytest.approx(0.0)

    def test_min_skipna_false(self, int_da):
        int_da.min("x", skipna=False)

    def test_argmax_no_dim(self, float_da):
        assert float_da.argmax().values == 11

    def test_argmax_skipna_false(self, int_da):
        int_da.argmax("x", skipna=False)

    def test_argmin_no_dim(self, float_da):
        assert float_da.argmin().values == 0

    def test_argmin_skipna_false(self, int_da):
        int_da.argmin("x", skipna=False)

    def test_median_no_dim(self, float_da):
        assert float_da.median().values == pytest.approx(5.5)

    def test_median_skipna_false(self, int_da):
        int_da.median("x", skipna=False)

    def test_ptp_no_dim(self, float_da):
        assert float_da.ptp().values == pytest.approx(11.0)

    def test_ptp_with_dim(self, float_da):
        result = float_da.ptp("x")
        assert result.shape == (4,)

    def test_mean_no_dim(self, float_da):
        assert float_da.mean().values == pytest.approx(5.5)

    def test_mean_skipna_false(self, int_da):
        int_da.mean("x", skipna=False)

    def test_prod_no_dim(self, float_da):
        result = float_da.prod()
        assert result.values == pytest.approx(0.0)

    def test_prod_skipna_false(self, int_da):
        int_da.prod("x", skipna=False)

    def test_std_no_dim(self, float_da):
        float_da.std()

    def test_std_skipna_false(self, int_da):
        int_da.std("x", skipna=False)

    def test_sum_no_dim(self, float_da):
        assert float_da.sum().values == pytest.approx(66.0)

    def test_sum_skipna_false(self, int_da):
        int_da.sum("x", skipna=False)

    def test_var_no_dim(self, float_da):
        float_da.var()

    def test_var_skipna_false(self, int_da):
        int_da.var("x", skipna=False)

    def test_percentile_no_dim(self, float_da):
        float_da.percentile(50)

    def test_percentile_skipna_false(self, int_da):
        int_da.percentile(50, "x", skipna=False)

    def test_quantile_no_dim(self, float_da):
        float_da.quantile(0.5)

    def test_quantile_skipna_false(self, int_da):
        int_da.quantile(0.5, "x", skipna=False)

    def test_average_no_dim(self, float_da):
        float_da.average()

    def test_average_with_dim(self, float_da):
        result = float_da.average("x")
        assert result.shape == (4,)

    def test_count_nonzero_no_dim(self, float_da):
        result = float_da.count_nonzero()
        assert result.values == 11  # element 0 is zero, rest are non-zero

    def test_count_nonzero_with_dim(self, float_da):
        result = float_da.count_nonzero("x")
        assert result.shape == (4,)

    # --- diff ---

    def test_diff_label_upper(self, float_da):
        result = float_da.diff("y", label="upper")
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result.data, np.diff(float_da.data, axis=1))
        # upper: coords come from index 1:
        assert result.coords["y"].values[-1] == pytest.approx(3.0)

    def test_diff_label_lower(self, float_da):
        result = float_da.diff("y", label="lower")
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result.data, np.diff(float_da.data, axis=1))
        assert result.coords["y"].values[0] == pytest.approx(0.0)

    def test_diff_label_invalid(self, float_da):
        with pytest.raises(ValueError, match="label"):
            float_da.diff("y", label="bad")

    def test_diff_dim_none(self, float_da):
        # axis=None flattens the array, making coords inconsistent — expected to fail
        with pytest.raises((TypeError, ValueError)):
            float_da.diff(None)
