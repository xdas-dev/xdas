import inspect

import numpy as np
import numpy.testing as npt
import pytest

import xdas as xd
import xdas.signal as xs
import xdas.spectral
from xdas.atoms import (
    STFT,
    Differentiate,
    Filter,
    Integrate,
    Partial,
    Resample,
    ResamplePoly,
    Sequential,
)
from xdas.atoms.tasks import (
    _edge_resample,
    _snap_factors,
    _solve_ratio,
    _target_ratio,
)
from xdas.synthetics import wavelet_wavefronts
from xdas.testing import dummy


def through_chunks(atom, da, nchunk=6, dim="time"):
    chunks = xd.split(da, nchunk, dim)
    return xd.concat([atom(chunk, chunk_dim=dim) for chunk in chunks], dim)


class TestFunctionForms:
    def test_seed_returns_atom(self):
        atom = xd.filter(..., (1.0, 10.0))
        assert isinstance(atom, Filter)
        assert atom.freq == (1.0, 10.0)

    def test_seed_returns_partial_for_functions(self):
        atom = xd.taper(...)
        assert isinstance(atom, Partial)

    def test_atom_input_composes(self):
        head = xd.resample(..., 25.0)
        pipeline = xd.filter(head, (1.0, 10.0))
        assert isinstance(pipeline, Sequential)
        assert len(pipeline) == 2
        assert not isinstance(head, Sequential)

    def test_signature(self):
        parameters = list(inspect.signature(xd.resample).parameters)
        assert parameters == [
            "da",
            "rate",
            "interval",
            "up",
            "down",
            "method",
            "snap",
            "maxup",
            "maxdown",
            "tolerance",
            "window",
            "numtaps",
            "order",
            "zerophase",
            "edge",
            "dim",
        ]

    def test_docstring(self):
        assert "Resample" in xd.resample.__doc__
        assert "rate" in xd.resample.__doc__

    def test_names(self):
        assert xd.filter.__name__ == "filter"
        assert xd.sliding_mean_removal.__name__ == "sliding_mean_removal"

    def test_time_defaults(self):
        assert xd.filter(..., (1.0, 10.0)).dim == "time"
        assert xd.resample(..., 25.0).dim == "time"

    def test_seed_idiom_matches_eager(self):
        da = wavelet_wavefronts()

        def workflow(da):
            da = xd.resample(da, 25.0)
            da = xd.filter(da, (1.0, 10.0))
            return np.square(da)

        preview = workflow(da)
        result = workflow(...)(da)
        assert np.allclose(result.values, preview.values)


class TestFilter:
    def test_bandpass_matches_signal(self):
        da = wavelet_wavefronts()
        result = xd.filter(da, (1.0, 10.0))
        expected = xs.filter(da, (1.0, 10.0), "bandpass", corners=4, dim="time")
        assert result.equals(expected)

    def test_lowpass_matches_signal(self):
        da = wavelet_wavefronts()
        result = xd.filter(da, (None, 10.0))
        expected = xs.filter(da, 10.0, "lowpass", corners=4, dim="time")
        assert result.equals(expected)

    def test_highpass_matches_signal(self):
        da = wavelet_wavefronts()
        result = xd.filter(da, (1.0, None))
        expected = xs.filter(da, 1.0, "highpass", corners=4, dim="time")
        assert result.equals(expected)

    def test_zerophase_matches_signal(self):
        da = wavelet_wavefronts()
        result = xd.filter(da, (1.0, 10.0), zerophase=True)
        expected = xs.filter(
            da, (1.0, 10.0), "bandpass", corners=4, zerophase=True, dim="time"
        )
        assert result.equals(expected)

    def test_iir_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Filter((1.0, 10.0))
        expected = atom(da)
        result = through_chunks(atom, da)
        assert result.equals(expected)

    def test_fir_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Filter((None, 10.0), ftype="fir")
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)

    def test_fir_compensates_lag(self):
        # The FIR filter is linear-phase with the group delay compensated on
        # the coordinate: a lowpassed wavelet must not shift in time.
        da = wavelet_wavefronts()
        result = xd.filter(da, (None, 10.0), ftype="fir", transition=5.0)
        reference = xd.filter(da, (None, 10.0), zerophase=True)
        assert result.sizes == da.sizes
        trace = result.isel(distance=100)
        reference_trace = reference.isel(distance=100)
        peak = trace["time"].values[int(np.argmax(trace.values))]
        reference_peak = reference_trace["time"].values[
            int(np.argmax(reference_trace.values))
        ]
        assert abs(peak - reference_peak) <= np.timedelta64(40, "ms")

    def test_zerophase_iir_chunked_raises(self):
        da = wavelet_wavefronts()
        atom = Filter((1.0, 10.0), zerophase=True)
        chunk, *_ = xd.split(da, 6, "time")
        with pytest.raises(ValueError, match="whole record"):
            atom(chunk, chunk_dim="time")

    def test_zerophase_iir_chunked_along_other_dim_passes(self):
        da = wavelet_wavefronts()
        atom = Filter((1.0, 10.0), zerophase=True)
        expected = atom(da)
        result = through_chunks(atom, da, dim="distance")
        assert result.equals(expected)

    def test_distance_dim(self):
        da = wavelet_wavefronts()
        result = xd.filter(da, (None, 0.005), dim="distance")
        expected = xs.filter(da, 0.005, "lowpass", corners=4, dim="distance")
        assert result.equals(expected)

    def test_scalar_freq_raises(self):
        with pytest.raises(TypeError, match="pair of corner frequencies"):
            Filter(10.0)

    def test_open_both_ends_raises(self):
        with pytest.raises(ValueError, match="at least one corner"):
            Filter((None, None))

    def test_invalid_ftype_raises(self):
        with pytest.raises(ValueError, match="ftype"):
            Filter((1.0, 10.0), ftype="cheby")


class TestFilterState:
    def test_initialize_from_state_is_noop_for_iir(self):
        # Only the FIR path has a design to rebuild from restored state.
        atom = Filter((1.0, 10.0))
        assert atom.initialize_from_state() is None
        assert not atom.filter.initialized


class TestSolveRatio:
    """`_solve_ratio(ratio, tolerance, maxup, maxdown)`: phase A."""

    def test_das_case_raises_at_default_tolerance(self):
        ratio = 1.021975 / 10
        with pytest.raises(ValueError, match="up=9, down=88") as info:
            _solve_ratio(ratio, 1e-5, 10, None)
        assert "0.1022" in str(info.value)
        up, down, deviation = info.value.closest
        assert (up, down) == (9, 88)
        assert deviation == pytest.approx(abs(9 / 88 - ratio))

    def test_das_case_at_loose_tolerance_returns_9_88(self):
        ratio = 1.021975 / 10
        assert _solve_ratio(ratio, 1e-3, 10, None) == (9, 88)

    @pytest.mark.parametrize("drift", [1e-10, 1e-7, 1e-5])
    def test_drift_within_default_tolerance_gives_1_10(self, drift):
        assert _solve_ratio(0.1 * (1 + drift), 1e-5, 10, None) == (1, 10)

    def test_1e4_drift_raises_at_default_tolerance(self):
        with pytest.raises(ValueError, match="up=1, down=10"):
            _solve_ratio(0.1 * (1 + 1e-4), 1e-5, 10, None)

    def test_1e4_drift_passes_at_looser_tolerance(self):
        assert _solve_ratio(0.1 * (1 + 1e-4), 1e-3, 10, None) == (1, 10)

    @pytest.mark.parametrize(
        "ratio, expected",
        [
            (0.4, (2, 5)),
            (0.15, (3, 20)),
            (1 / 441, (1, 441)),
            (1 / 1000, (1, 1000)),
            (1 / 250, (1, 250)),
        ],
    )
    def test_exact_ratios_come_out_without_special_casing(self, ratio, expected):
        assert _solve_ratio(ratio, 1e-5, 10, None) == expected

    @pytest.mark.parametrize("ratio", [0.333, 0.441])
    def test_no_good_rational_raises(self, ratio):
        with pytest.raises(ValueError):
            _solve_ratio(ratio, 1e-5, 10, None)

    def test_reduction_prefers_the_simplest_equal_valued_ratio(self):
        # up=3, down=30 is also exact, but (1, 10) is simpler and found first.
        assert _solve_ratio(0.1, 1e-5, 10, None) == (1, 10)

    def test_ratio_greater_than_one_is_the_interpolation_branch(self):
        assert _solve_ratio(2.5, 1e-5, 10, None) == (5, 2)

    def test_maxdown_drops_candidates(self):
        # Uncapped, up=5, down=2 is exact; capped at maxdown=1 it is unreachable
        # and the closest surviving candidate must respect the cap.
        assert _solve_ratio(2.5, 1e-5, 10, None) == (5, 2)
        with pytest.raises(ValueError):
            _solve_ratio(2.5, 1e-5, 10, maxdown=1)
        try:
            _solve_ratio(2.5, 1e-5, 10, maxdown=1)
        except ValueError as error:
            _, down, _ = error.closest
            assert down <= 1

    def test_no_candidate_within_maxdown_raises_distinct_message(self):
        with pytest.raises(ValueError, match="no ratio with down"):
            _solve_ratio(1 / 1000, 1e-5, 1, maxdown=5)

    @pytest.mark.parametrize("ratio", [0, -1.0])
    def test_non_positive_ratio_raises(self, ratio):
        with pytest.raises(ValueError, match="positive"):
            _solve_ratio(ratio, 1e-5, 10, None)

    def test_tolerance_boundary_is_inclusive(self):
        # deviation exactly equal to tolerance * ratio must be accepted.
        ratio = 0.1
        deviation = 1e-5 * ratio
        drifted = ratio + deviation
        assert _solve_ratio(drifted, 1e-5 * (1 + 1e-12), 10, None) == (1, 10)


class TestSnapFactors:
    """`_snap_factors(ratio)` / `_round_half_up`: phase A."""

    def test_das_snap_lands_on_down_ten(self):
        assert _snap_factors(1.021975 / 10) == (1, 10)

    def test_twice_coarser_instrument_lands_on_same_grid(self):
        # delta=2.04395 is exactly twice delta=1.021975: down halves too, so
        # both nest on the same 10.2195 m output grid.
        assert _snap_factors(2.04395 / 10) == (1, 5)

    def test_symmetric_interpolation_branch(self):
        # f = 1/ratio < 1: the input grid nests inside the output's instead.
        assert _snap_factors(4.0) == (4, 1)

    def test_round_half_up_tie_beats_bankers_rounding(self):
        # f = 1/ratio = 2.5 exactly; Python's round() would give 2 (banker's).
        assert _snap_factors(0.4) == (1, 3)

    def test_a_factor_rounding_low_becomes_one(self):
        # interval = 1.2 * delta: nearest nested grid is delta itself.
        assert _snap_factors(1 / 1.2) == (1, 1)

    def test_ratio_one_is_identity(self):
        assert _snap_factors(1.0) == (1, 1)


class TestTargetRatio:
    """`_target_ratio(rate, interval, up, down, delta)`: phase B."""

    def test_rate_spelling_float_delta(self):
        ratio, spelling, up, down = _target_ratio(25.0, None, None, None, 0.02)
        assert spelling == "rate"
        assert ratio == pytest.approx(0.5)
        assert up is None and down is None

    def test_rate_spelling_timedelta_delta(self):
        ratio, spelling, _, _ = _target_ratio(
            25.0, None, None, None, np.timedelta64(20, "ms")
        )
        assert spelling == "rate"
        assert ratio == pytest.approx(0.5)

    def test_interval_spelling_float(self):
        ratio, spelling, _, _ = _target_ratio(None, 0.04, None, None, 0.02)
        assert spelling == "interval"
        assert ratio == pytest.approx(0.5)

    def test_interval_spelling_float_on_datetime_means_seconds(self):
        ratio, spelling, _, _ = _target_ratio(
            None, 0.04, None, None, np.timedelta64(20, "ms")
        )
        assert spelling == "interval"
        assert ratio == pytest.approx(0.5)

    def test_interval_spelling_timedelta64_is_exact(self):
        ratio, spelling, _, _ = _target_ratio(
            None, np.timedelta64(20, "ms"), None, None, np.timedelta64(10, "ms")
        )
        assert spelling == "interval"
        assert ratio == 0.5

    def test_interval_spelling_datetime_timedelta_is_exact(self):
        import datetime

        ratio, _, _, _ = _target_ratio(
            None,
            datetime.timedelta(milliseconds=20),
            None,
            None,
            np.timedelta64(10, "ms"),
        )
        assert ratio == 0.5

    def test_interval_timedelta_on_float_delta_raises(self):
        with pytest.raises(ValueError, match="datetime coordinate"):
            _target_ratio(None, np.timedelta64(20, "ms"), None, None, 0.02)

    def test_up_down_spelling_both(self):
        ratio, spelling, up, down = _target_ratio(None, None, 3, 7, 0.02)
        assert spelling == "factor"
        assert (up, down) == (3, 7)
        assert ratio == pytest.approx(3 / 7)

    def test_down_alone_defaults_up_to_one(self):
        _, _, up, down = _target_ratio(None, None, None, 16, 0.02)
        assert (up, down) == (1, 16)

    def test_up_alone_defaults_down_to_one(self):
        _, _, up, down = _target_ratio(None, None, 3, None, 0.02)
        assert (up, down) == (3, 1)

    def test_no_spelling_raises_type_error(self):
        with pytest.raises(TypeError, match="one of"):
            _target_ratio(None, None, None, None, 0.02)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rate": 25.0, "interval": 0.04},
            {"rate": 25.0, "up": 2},
            {"interval": 0.04, "down": 2},
        ],
    )
    def test_two_spellings_raises_type_error(self, kwargs):
        with pytest.raises(TypeError, match="only one of"):
            _target_ratio(
                kwargs.get("rate"),
                kwargs.get("interval"),
                kwargs.get("up"),
                kwargs.get("down"),
                0.02,
            )

    def test_non_integer_up_raises(self):
        with pytest.raises(ValueError, match="integer"):
            _target_ratio(None, None, 2.5, None, 0.02)

    def test_non_integer_down_raises(self):
        with pytest.raises(ValueError, match="integer"):
            _target_ratio(None, None, None, 2.5, 0.02)

    def test_non_positive_up_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _target_ratio(None, None, -1, None, 0.02)

    def test_non_positive_down_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _target_ratio(None, None, None, -1, 0.02)

    def test_non_positive_rate_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _target_ratio(-1.0, None, None, None, 0.02)

    def test_non_positive_interval_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _target_ratio(None, 0.0, None, None, 0.02)

    def test_non_positive_timedelta_interval_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _target_ratio(
                None, np.timedelta64(0, "ms"), None, None, np.timedelta64(10, "ms")
            )


class TestEdgeResample:
    """`_edge_resample(da, num, dim, window, edge)`: phase C."""

    def test_invalid_edge_raises(self):
        da = dummy(shape=(100, 3))
        with pytest.raises(ValueError, match="edge"):
            _edge_resample(da, 50, "time", None, "quadratic")

    def test_none_matches_plain_resample(self):
        da = wavelet_wavefronts()
        result = _edge_resample(da, 100, "time", None, "none")
        expected = xs.resample(da, 100, dim="time")
        assert result.equals(expected)

    def test_mirror_crop_is_exact_first_half_and_preserves_x0(self):
        da = dummy(shape=(200, 3), step=(0.01, 10.0), datetime=False)
        n, num = da.sizes["time"], 80
        result = _edge_resample(da, num, "time", None, "mirror")
        # Built independently of `_edge_resample`, to check its mirror branch
        # against the same construction rather than against itself.
        extended_coord = type(da["time"]).from_block(
            da["time"].values[0],
            2 * n,
            xd.get_sampling_interval(da, "time"),
            dim="time",
        )
        extended = xd.DataArray(
            np.concatenate([da.values, da.values[::-1]]),
            {"time": extended_coord, "distance": da["distance"]},
            da.dims,
        )
        full = xs.resample(extended, 2 * num, dim="time")
        assert result.sizes["time"] == num
        assert result["time"].values[0] == da["time"].values[0]
        np.testing.assert_array_equal(
            result.values, full.isel(time=slice(0, num)).values
        )
        np.testing.assert_array_equal(
            result["time"].values, full.isel(time=slice(0, num))["time"].values
        )

    def test_linear_detrend_exactly_restores_a_ramp(self):
        # A pure ramp has zero deviation from its own trend, so resampling
        # the (exactly-zero) detrended signal and adding the trend back must
        # reproduce the ramp exactly, whatever the resampling ratio.
        n, num = 200, 77
        da = dummy(shape=(n, 3), step=(0.01, 10.0), datetime=False)
        ramp = np.linspace(0.0, 1.0, n)[:, None] * np.ones((1, 3))
        da = da.copy(data=ramp)
        result = _edge_resample(da, num, "time", None, "linear")
        # Output sample k sits at fraction k/(num-1) along the record.
        expected = np.linspace(0.0, 1.0, num)[:, None] * np.ones((1, 3))
        np.testing.assert_allclose(result.values, expected, atol=1e-10)

    def test_mirror_rings_less_than_none_on_a_ramp_with_a_jump(self):
        # A ramp resets to zero at the end: a real discontinuity across the
        # assumed period. "none" rings from it; "mirror" removes the jump.
        n = 200
        da = dummy(shape=(n, 3), step=(0.01, 10.0), datetime=False)
        ramp = np.linspace(0.0, 1.0, n)[:, None] * np.ones((1, 3))
        da = da.copy(data=ramp)
        none_result = _edge_resample(da, 100, "time", None, "none")
        mirror_result = _edge_resample(da, 100, "time", None, "mirror")
        # near the end of the record, "none" overshoots from the wrap-around
        # discontinuity; "mirror" tracks the ramp closely.
        expected_end = 1.0
        assert abs(mirror_result.values[-1, 0] - expected_end) < abs(
            none_result.values[-1, 0] - expected_end
        )


class TestResample:
    """The `Resample` atom: phase D."""

    def test_default_method_is_fir(self):
        assert Resample(25.0, dim="time").method == "fir"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            Resample(25.0, method="iiir")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"numtaps": 5, "method": "iir"},
            {"order": 4, "method": "fir"},
            {"zerophase": True, "method": "fir"},
            {"edge": "mirror", "method": "fir"},
            {"window": "hann", "method": "iir"},
        ],
    )
    def test_method_specific_parameter_on_wrong_method_raises(self, kwargs):
        with pytest.raises(ValueError):
            Resample(25.0, **kwargs)

    def test_snap_with_fft_raises(self):
        with pytest.raises(ValueError, match="fft"):
            Resample(25.0, method="fft", snap=True)

    def test_snap_with_up_down_raises(self):
        with pytest.raises(ValueError, match="snap"):
            Resample(up=1, down=2, snap=True)

    def test_fir_matches_resample_poly(self):
        da = wavelet_wavefronts()
        result = xd.resample(da, 20.0)
        expected = ResamplePoly(20.0, dim="time")(da)
        assert result.equals(expected)

    def test_fir_matches_xs_resample_poly_once_the_removed_lag_is_undone(self):
        # xd.resample removes the FIR group delay from the coordinate (and
        # crops it from the values); xs.resample_poly does neither, so the
        # two agree once shifted by the lag xd.resample removed.
        da = wavelet_wavefronts()  # 50 Hz
        result = xd.resample(da, down=2, dim="time")
        expected = xs.resample_poly(da, 1, 2, "time")
        numtaps = 20 * 2 + 1
        lag = ((numtaps - 1) // 2) // 2  # group delay, in output samples
        n = expected.sizes["time"]
        np.testing.assert_allclose(
            result.values[lag:], expected.values[: n - lag], atol=1e-10
        )

    def test_three_spellings_agree(self):
        da = wavelet_wavefronts()  # 50 Hz
        by_rate = xd.resample(da, rate=25.0, dim="time")
        by_interval = xd.resample(da, interval=0.04, dim="time")
        by_factor = xd.resample(da, down=2, dim="time")
        assert by_rate.equals(by_interval)
        assert by_rate.equals(by_factor)

    def test_up_down_bypasses_the_solver_even_off_tolerance(self):
        # up/down is authoritative: no tolerance check applies to it.
        da = wavelet_wavefronts()
        result = xd.resample(da, up=1, down=2, dim="time", tolerance=0)
        assert result.sizes["time"] == 150

    def test_iir_matches_scipy_decimate_causal(self):
        da = wavelet_wavefronts()
        result = xd.resample(da, down=2, method="iir", dim="time")
        expected = xs.decimate(da, 2, ftype="iir", zero_phase=False, dim="time")
        assert np.allclose(result.values, expected.values)

    def test_iir_zerophase_matches_scipy_decimate_zerophase(self):
        da = wavelet_wavefronts()
        result = xd.resample(da, down=2, method="iir", zerophase=True, dim="time")
        expected = xs.decimate(da, 2, ftype="iir", zero_phase=True, dim="time")
        assert np.allclose(result.values, expected.values)

    def test_iir_zerophase_with_up_greater_than_one_and_down_one(self):
        # Exercises both the UpSample branch (up > 1) and the DownSample
        # branch being skipped (down == 1) of the zerophase call path.
        da = wavelet_wavefronts()
        result = xd.resample(da, up=2, down=1, method="iir", zerophase=True, dim="time")
        assert result.sizes["time"] == 2 * da.sizes["time"]

    def test_fft_initialize_from_state_is_a_noop(self):
        # Stateless: the design happens fresh in `call`, not from restored
        # state, so this is a no-op reachable directly (e.g. a state round
        # trip that calls it regardless of method).
        atom = Resample(25.0, method="fft", dim="time")
        assert atom.initialize_from_state() is None

    def test_fft_ratio_one_is_identity(self):
        da = wavelet_wavefronts()
        assert xd.resample(da, up=1, down=1, method="fft", dim="time").equals(da)

    def test_fft_explicit_tolerance_above_the_floor_passes(self):
        da = dummy(shape=(20, 10000), step=(0.02, 1.021975))
        result = xd.resample(
            da, interval=10.0, method="fft", dim="distance", tolerance=1e-3
        )
        assert result.sizes["distance"] == 1022

    def test_fft_matches_scipy_resample(self):
        da = wavelet_wavefronts()
        result = xd.resample(da, down=2, method="fft", edge="none", dim="time")
        expected = xs.resample(da, 150, dim="time")
        assert np.allclose(result.values, expected.values)

    def test_fir_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Resample(down=2, dim="time")
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)

    def test_iir_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Resample(down=2, method="iir", dim="time")
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)

    def test_iir_up_greater_than_one_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Resample(up=2, down=5, method="iir", dim="time")
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values, atol=1e-12, rtol=1e-9)
        assert result.coords.equals(expected.coords)

    def test_fft_refuses_chunking_along_its_own_dim(self):
        da = wavelet_wavefronts()
        atom = Resample(down=2, method="fft", dim="time")
        chunk, *_ = xd.split(da, 6, "time")
        with pytest.raises(ValueError, match="whole record"):
            atom(chunk, chunk_dim="time")

    def test_fft_along_distance_streams_fine_when_chunked_along_time(self):
        da = wavelet_wavefronts()
        atom = Resample(down=2, method="fft", dim="distance")
        expected = atom(da)
        result = through_chunks(atom, da, dim="time")
        assert np.allclose(result.values, expected.values, atol=1e-10)

    def test_iir_zerophase_refuses_chunking(self):
        da = wavelet_wavefronts()
        atom = Resample(down=2, method="iir", zerophase=True, dim="time")
        chunk, *_ = xd.split(da, 6, "time")
        with pytest.raises(ValueError, match="whole record"):
            atom(chunk, chunk_dim="time")

    def test_state_round_trip_redesigns_from_delta_alone(self):
        # A heterogeneous fleet: the same atom instance, called on records
        # with unrelated spacings, must redesign its ratio from each one's
        # own measured delta rather than reusing a previous run's factors.
        atom = Resample(interval=10.0, snap=True, dim="distance")
        first = dummy(shape=(20, 500), step=(0.01, 1.021975))
        second = dummy(shape=(20, 500), step=(0.01, 2.04395))
        out1 = atom(first)
        assert atom.up_, atom.down_ == (1, 10)
        out2 = atom(second)
        assert (atom.up_, atom.down_) == (1, 5)
        np.testing.assert_allclose(
            np.diff(out1["distance"].values)[0], np.diff(out2["distance"].values)[0]
        )

    def test_achieved_factors_are_readable_after_init(self):
        da = wavelet_wavefronts()
        atom = Resample(rate=25.0, dim="time")
        atom(da)
        assert (atom.up_, atom.down_) == (1, 2)

    def test_long_fir_warns_above_down_100(self):
        da = dummy(shape=(2000, 3), step=(0.001, 10.0))  # 1000 Hz
        with pytest.warns(UserWarning, match="tap"):
            xd.resample(da, down=200, dim="time")

    def test_no_warning_at_or_below_down_100(self):
        da = dummy(shape=(2000, 3), step=(0.001, 10.0))
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            xd.resample(da, down=100, dim="time")

    def test_numtaps_override(self):
        da = wavelet_wavefronts()
        atom = Resample(down=2, numtaps=11, dim="time")
        atom(da)
        assert atom.child.numtaps == 11

    def test_grid_property_is_exact_under_snap_chunked_and_eager(self):
        da = dummy(shape=(300, 5000), step=(0.02, 1.021975))
        eager = xd.resample(da, down=10, dim="distance")
        atom = Resample(down=10, dim="distance")
        chunks = xd.split(da, 6, "time")
        chunked = xd.concat([atom(c, chunk_dim="time") for c in chunks], "time")
        assert eager.coords.equals(chunked.coords)

    def test_no_false_refusal_at_deep_integer_decimation(self):
        # p=1 covers any depth: no maxdown cap applies by default.
        da = dummy(shape=(2000, 3), step=(0.001, 10.0))  # 1000 Hz
        result = xd.resample(da, 1.0, dim="time")  # -> down=1000
        assert result.sizes["time"] == 2

    def test_maxdown_turns_deep_decimation_into_a_refusal(self):
        da = dummy(shape=(2000, 3), step=(0.001, 10.0))
        with pytest.raises(ValueError):
            xd.resample(da, 1.0, dim="time", maxdown=100)

    def test_solve_ratio_failure_message_reaches_the_caller(self):
        da = dummy(shape=(20, 10000), step=(0.02, 1.021975))
        with pytest.raises(ValueError, match="Ways out"):
            xd.resample(da, interval=10.0, dim="distance")

    def test_up_equals_down_equals_one_is_identity(self):
        da = wavelet_wavefronts()
        assert xd.resample(da, up=1, down=1, dim="time").equals(da)

    def test_fft_tolerance_floor_does_not_raise_by_default(self):
        da = dummy(shape=(20, 10000), step=(0.02, 1.021975))
        result = xd.resample(da, interval=10.0, method="fft", dim="distance")
        assert result.sizes["distance"] == 1022

    def test_fft_explicit_tolerance_below_the_floor_raises(self):
        da = dummy(shape=(20, 10000), step=(0.02, 1.021975))
        with pytest.raises(ValueError, match="quantisation floor"):
            xd.resample(da, interval=10.0, method="fft", dim="distance", tolerance=1e-6)


class TestResamplePolyDeprecated:
    def test_warns(self):
        with pytest.warns(DeprecationWarning, match="Resample"):
            ResamplePoly(50.0, dim="time")

    def test_still_works(self):
        da = wavelet_wavefronts()
        with pytest.warns(DeprecationWarning):
            result = ResamplePoly(20.0, dim="time")(da)
        expected = xd.resample(da, 20.0, dim="time")
        assert result.equals(expected)


class TestIntegrate:
    def test_matches_signal(self):
        da = wavelet_wavefronts()
        result = xd.integrate(da)
        expected = xs.integrate(da, dim="time")
        assert result.equals(expected)

    def test_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Integrate()
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values)
        assert result.coords.equals(expected.coords)

    def test_midpoints(self):
        # Midpoints on the distance dim: xs.integrate cannot shift datetime
        # coordinates by a float half-step, so time is not testable here.
        da = wavelet_wavefronts()
        result = xd.integrate(da, midpoints=True, dim="distance")
        expected = xs.integrate(da, midpoints=True, dim="distance")
        assert result.equals(expected)


class TestDifferentiate:
    def test_matches_signal(self):
        da = wavelet_wavefronts()
        result = xd.differentiate(da)
        expected = xs.differentiate(da, dim="time")
        assert result.equals(expected)

    def test_chunked_equals_monolithic(self):
        da = wavelet_wavefronts()
        atom = Differentiate()
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values)
        assert result.coords.equals(expected.coords)


class TestWholeRecordFunctions:
    def test_detrend_matches_signal(self):
        da = wavelet_wavefronts()
        assert xd.detrend(da).equals(xs.detrend(da, dim="time"))

    def test_taper_matches_signal(self):
        da = wavelet_wavefronts()
        assert xd.taper(da).equals(xs.taper(da, dim="time"))

    def test_hilbert_matches_signal(self):
        da = wavelet_wavefronts()
        assert xd.hilbert(da).equals(xs.hilbert(da, dim="time"))

    def test_sliding_mean_removal_matches_signal(self):
        da = wavelet_wavefronts()
        assert xd.sliding_mean_removal(da, 1.0).equals(
            xs.sliding_mean_removal(da, 1.0, dim="time")
        )

    def test_medfilt_physical_units(self):
        da = wavelet_wavefronts()
        dt = xd.get_sampling_interval(da, "time")
        dx = xd.get_sampling_interval(da, "distance")
        result = xd.medfilt(da, {"time": 7 * dt, "distance": 5 * dx})
        expected = xs.medfilt(da, {"time": 7, "distance": 5})
        assert result.equals(expected)

    def test_chunked_raises(self):
        da = wavelet_wavefronts()
        chunk, *_ = xd.split(da, 6, "time")
        for atom in [
            xd.detrend(...),
            xd.taper(...),
            xd.hilbert(...),
            xd.sliding_mean_removal(..., 1.0),
            xd.medfilt(..., {"time": 0.1}),
        ]:
            with pytest.raises(ValueError, match="whole record"):
                atom(chunk, chunk_dim="time")

    def test_chunked_along_other_dim_passes(self):
        da = wavelet_wavefronts()
        atom = xd.taper(...)
        expected = atom(da)
        result = through_chunks(atom, da, dim="distance")
        assert result.equals(expected)


class TestSTFT:
    # dummy is sampled at 100 Hz: 0.32 s windows are 32 samples (a fast FFT
    # size, so the target is not snapped) and 0.16 s hops are 16 samples.

    def test_matches_legacy_spectral(self):
        da = dummy(shape=(200, 5))
        result = xd.stft(da, 0.32, hop=0.16)
        expected = xdas.spectral.stft(
            da, window="hann", nperseg=32, noverlap=16, dim={"time": "frequency"}
        )
        assert np.allclose(result.values, expected.values)
        assert np.array_equal(result["time"].values, expected["time"].values)
        assert np.allclose(result["frequency"].values, expected["frequency"].values)
        assert result["distance"].equals(expected["distance"])

    def test_default_hop_is_half_window(self):
        da = dummy(shape=(200, 5))
        result = xd.stft(da, 0.32)
        expected = xd.stft(da, 0.32, hop=0.16)
        assert np.allclose(result.values, expected.values)

    def test_wlen_snaps_to_fast_length(self):
        # 1.27 s at 100 Hz is 127 samples, a prime: the next fast size is 128.
        da = dummy(shape=(400, 5))
        result = xd.stft(da, 1.27)
        assert result.sizes["frequency"] == 128 // 2 + 1

    def test_psd_scaling_matches_legacy(self):
        da = dummy(shape=(200, 5))
        result = xd.stft(da, 0.32, hop=0.16, scaling="psd")
        expected = xdas.spectral.stft(
            da,
            window="hann",
            nperseg=32,
            noverlap=16,
            scaling="psd",
            dim={"time": "frequency"},
        )
        assert np.allclose(result.values, expected.values)

    def test_nfft_zero_padding(self):
        da = dummy(shape=(200, 5))
        result = xd.stft(da, 0.32, hop=0.16, nfft=64)
        expected = xdas.spectral.stft(
            da,
            window="hann",
            nperseg=32,
            noverlap=16,
            nfft=64,
            dim={"time": "frequency"},
        )
        assert result.sizes["frequency"] == 64 // 2 + 1
        assert np.allclose(result.values, expected.values)
        assert np.allclose(result["frequency"].values, expected["frequency"].values)

    def test_nfft_smaller_than_window_raises(self):
        da = dummy(shape=(200, 5))
        with pytest.raises(ValueError, match="nfft"):
            xd.stft(da, 0.32, nfft=16)

    def test_complex_input_two_sided(self):
        da = dummy(shape=(200, 5), dtype=complex)
        result = xd.stft(da, 0.32, hop=0.16)
        expected = xdas.spectral.stft(
            da,
            window="hann",
            nperseg=32,
            noverlap=16,
            return_onesided=False,
            dim={"time": "frequency"},
        )
        assert result.sizes["frequency"] == 32
        assert np.allclose(result.values, expected.values)
        assert np.allclose(result["frequency"].values, expected["frequency"].values)

    def test_invalid_parameters(self):
        with pytest.raises(ValueError, match="wlen"):
            STFT(0.0)
        with pytest.raises(ValueError, match="hop"):
            STFT(1.0, hop=2.0)
        with pytest.raises(ValueError, match="scaling"):
            STFT(1.0, scaling="power")

    def test_record_shorter_than_window_raises(self):
        da = dummy(shape=(50, 5))
        with pytest.raises(ValueError, match="shorter"):
            xd.stft(da, 1.0)

    def test_chunked_along_other_dim(self):
        da = dummy(shape=(200, 5))
        expected = xd.stft(da, 0.32, hop=0.16)
        result = through_chunks(STFT(0.32, hop=0.16), da, 2, "distance")
        assert np.allclose(result.values, expected.values)

    def test_non_dimensional_coords(self):
        da = dummy(shape=(200, 5))
        da["latitude"] = ("distance", np.arange(5.0))
        da["quality"] = ("time", np.arange(200.0))
        result = xd.stft(da, 0.32, hop=0.16)
        # coords along other dimensions are kept; those along the transformed
        # dimension are dropped (TODO in STFT._transform, as in spectral.stft)
        assert result["latitude"].equals(da["latitude"])
        assert "quality" not in result.coords

    def test_function_form_seed(self):
        atom = xd.stft(..., 1.0)
        assert isinstance(atom, STFT)
        assert atom.dim == "time"

    def test_pipeline_chunk_invariant_over_cuts_and_gaps(self):
        da = dummy(shape=(400, 5))
        pipeline = xd.filter(..., (None, 20.0)) >> xd.stft(..., 0.32, hop=0.16)
        xd.testing.assert_chunk_invariant(
            pipeline, da, {"time": 100}, cuts=2, gaps=2, atol=1e-12
        )


class TestLabelsFollowTheSamples:
    """
    A resampled dimension carries its other coordinates onto the output grid.

    A DAS channel is named by a non-dimensional ``station`` coordinate
    attached to ``distance``, and picking answers with those names: a stage
    that changes the number of samples has to say what became of them, or the
    output is labelled with the wrong lanes. They follow the *samples* —
    output ``k`` is drawn from input ``k * down / up`` — so unlike the
    dimension coordinate they carry no group-delay shift, and the mapping
    cannot depend on the chunking.
    """

    def labelled(self, n=120, dim="distance", step=(0.01, 10.0)):
        da = dummy(dims=("time", "distance"), shape=(200, n), step=step)
        labels = np.array([f"S{index:04d}" for index in range(da.sizes[dim])])
        return da.assign_coords(station=(dim, labels))

    def test_decimation_subsamples_them(self):
        da = self.labelled()
        result = xd.resample(da, down=2, dim="distance")
        assert result.sizes["distance"] == 60
        npt.assert_array_equal(result["station"].values, da["station"].values[::2])

    def test_rational_resampling_lands_on_the_output_grid(self):
        da = self.labelled(n=8)
        result = xd.resample(da, 1 / 25.0, dim="distance")  # up=2, down=5
        assert result.sizes["distance"] == len(result["station"].values)
        npt.assert_array_equal(
            result["station"].values, ["S0000", "S0002", "S0005", "S0007"]
        )

    def test_upsampling_repeats_them(self):
        from xdas.atoms import UpSample

        da = self.labelled(n=4)
        result = UpSample(3, dim="distance")(da)
        assert result.sizes["distance"] == 12
        npt.assert_array_equal(
            result["station"].values[:6], ["S0000"] * 3 + ["S0001"] * 3
        )

    def test_the_labels_do_not_depend_on_the_chunking(self):
        # chunked along the very dimension being decimated: every chunk must
        # label its output with the same input samples the eager call does.
        da = dummy(dims=("time", "distance"), shape=(120, 3))
        labels = np.array([f"T{index:04d}" for index in range(120)])
        da = da.assign_coords(label=("time", labels))
        eager = xd.resample(..., 25.0, dim="time")(da)
        atom = xd.resample(..., 25.0, dim="time")
        chunks = list(atom.iter_chunks(xd.split(da, 7, "time"), "time"))
        streamed = np.concatenate([chunk["label"].values for chunk in chunks])
        npt.assert_array_equal(streamed, eager["label"].values)

    def test_an_untouched_dimension_keeps_its_labels(self):
        da = self.labelled()
        result = xd.resample(da, 25.0, dim="time")
        npt.assert_array_equal(result["station"].values, da["station"].values)
