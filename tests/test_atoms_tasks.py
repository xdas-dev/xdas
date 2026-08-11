import inspect

import numpy as np
import pytest

import xdas as xd
import xdas.signal as xs
from xdas.atoms import (
    Decimate,
    Differentiate,
    Filter,
    Integrate,
    Partial,
    ResamplePoly,
    Sequential,
)
from xdas.synthetics import wavelet_wavefronts


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
        head = xd.decimate(..., 25.0)
        pipeline = xd.filter(head, (1.0, 10.0))
        assert isinstance(pipeline, Sequential)
        assert len(pipeline) == 2
        assert not isinstance(head, Sequential)

    def test_signature(self):
        parameters = list(inspect.signature(xd.decimate).parameters)
        assert parameters == ["da", "target", "window", "dim"]

    def test_docstring(self):
        assert "Decimate" in xd.decimate.__doc__
        assert "target" in xd.decimate.__doc__

    def test_names(self):
        assert xd.filter.__name__ == "filter"
        assert xd.sliding_mean_removal.__name__ == "sliding_mean_removal"

    def test_time_defaults(self):
        assert xd.filter(..., (1.0, 10.0)).dim == "time"
        assert xd.decimate(..., 25.0).dim == "time"

    def test_seed_idiom_matches_eager(self):
        da = wavelet_wavefronts()

        def workflow(da):
            da = xd.decimate(da, 25.0)
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


class TestDecimate:
    def test_matches_resample_poly(self):
        # For an integer factor Decimate shares its design with ResamplePoly.
        da = wavelet_wavefronts()
        result = xd.decimate(da, 25.0)
        expected = ResamplePoly(25.0, dim="time")(da)
        assert result.equals(expected)

    def test_chunked_equals_monolithic(self):
        # Chunk sizes must be multiples of the factor: draining the trailing
        # remainder needs the flush() lifecycle.
        da = wavelet_wavefronts()
        atom = Decimate(25.0)
        expected = atom(da)
        result = through_chunks(atom, da)
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)

    def test_non_integer_factor_raises(self):
        da = wavelet_wavefronts()
        with pytest.raises(ValueError, match="integer multiple"):
            xd.decimate(da, 30.0)

    def test_upsampling_raises(self):
        da = wavelet_wavefronts()
        with pytest.raises(ValueError, match="integer multiple"):
            xd.decimate(da, 60.0)

    def test_factor_one_is_identity(self):
        da = wavelet_wavefronts()  # already at 50 Hz
        assert xd.decimate(da, 50.0).equals(da)


class TestResample:
    def test_matches_resample_poly(self):
        da = wavelet_wavefronts()
        result = xd.resample(da, 20.0)
        expected = ResamplePoly(20.0, dim="time")(da)
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
