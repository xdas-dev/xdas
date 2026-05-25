import pickle

import numpy as np
import pytest
import scipy.signal as sp

import xdas as xd
import xdas.signal as xs
from xdas.atoms import (
    DownSample,
    FIRFilter,
    IIRFilter,
    MLPicker,
    Partial,
    ResamplePoly,
    Sequential,
    UpSample,
)
from xdas.signal import lfilter
from xdas.synthetics import randn_wavefronts, wavelet_wavefronts


class TestPartialAtom:
    def test_init(self):
        Sequential(
            [
                Partial(xs.taper, dim="time"),
                Partial(xs.taper, dim="distance"),
                Partial(np.abs),
                Partial(np.square),
            ]
        )

    def test_pickable(self, tmp_path):
        atom = xs.integrate(..., dim="dim")
        tmpfile_path = tmp_path / "tempfile.pkl"
        with open(tmpfile_path, "wb") as tmpfile:
            pickle.dump(atom, tmpfile)
        with open(tmpfile_path, "rb") as tmpfile:
            result = pickle.load(tmpfile)
        assert result.func.__module__ == atom.func.__module__
        assert result.func.__name__ == atom.func.__name__
        assert result.args == atom.args
        assert result.kwargs == atom.kwargs
        assert result.name == atom.name
        assert result._state == atom._state
        assert result.state == atom.state


class TestProcessing:
    def test_sequence(self):
        # Generate a temporary dataset
        da = wavelet_wavefronts()

        # Declare sequence to execute
        seq = Sequential(
            [
                Partial(np.abs),
                Partial(np.square, name="some square"),
                Partial(xd.mean, dim="time"),
            ]
        )

        # Sequence processing
        result1 = seq(da)
        # Manual processing
        result2 = xd.mean(np.abs(da) ** 2, dim="time")

        # Test
        assert np.allclose(result1.values, result2.values)


class TestDecorator:
    def test_decorator(self):
        a = [1, 1]
        b = [1, 1]
        atom = lfilter(b, a, ..., "time")
        statefull = lfilter(b, a, ..., "time", zi=...)
        assert isinstance(atom, Partial)
        assert isinstance(statefull, Partial)
        assert statefull.state == {"zi": ...}

    def test_passing_atom(self):
        a = [1, 1]
        b = [1, 1]
        atom = lfilter(b, a, ..., "time")
        atom = lfilter(b, a, atom, "time")
        assert isinstance(atom, Sequential)
        assert len(atom) == 2


class TestFilters:
    def test_lfilter(self):
        da = wavelet_wavefronts()
        chunks = xd.split(da, 6, "time")

        b, a = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0)
        data = sp.lfilter(b, a, da.values, axis=0)
        expected = da.copy(data=data)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        monolithic = atom(da)

        chunked = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")

        assert monolithic.equals(expected)
        assert chunked.equals(expected)

        # TODO: make clean save/load state
        # with TemporaryDirectory() as dirpath:
        #     path = os.path.join(dirpath, "state.nc")

        #     atom_a = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        #     chunks_a = [atom_a(chunk, chunk_dim="time") for chunk in chunks[:3]]
        #     atom_a.save_state(path)

        #     atom_b = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        #     atom_b.load_state(path)
        #     chunks_b = [atom_b(chunk, chunk_dim="time") for chunk in chunks[3:]]

        #     result = xd.concat(chunks_a + chunks_b, "time")
        #     assert result.equals(expected)

    def test_sosfilter(self):
        da = wavelet_wavefronts()
        chunks = xd.split(da, 6, "time")

        sos = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0, output="sos")
        data = sp.sosfilt(sos, da.values, axis=0)
        expected = da.copy(data=data)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time")
        monolithic = atom(da)

        chunked = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")

        assert monolithic.equals(expected)
        assert chunked.equals(expected)

        # TODO: make clean save/load state
        # with TemporaryDirectory() as dirpath:
        #     path = os.path.join(dirpath, "state.nc")

        #     atom_a = IIRFilter(4, 10.0, "lowpass", dim="time")
        #     chunks_a = [atom_a(chunk, chunk_dim="time") for chunk in chunks[:3]]
        #     atom_a.save_state(path)

        #     atom_b = IIRFilter(4, 10.0, "lowpass", dim="time")
        #     atom_b.load_state(path)
        #     chunks_b = [atom_b(chunk, chunk_dim="time") for chunk in chunks[3:]]

        #     result = xd.concat(chunks_a + chunks_b, "time")
        #     assert result.equals(expected)

    def test_downsample(self):
        da = wavelet_wavefronts()
        chunks = xd.split(da, 6, "time")
        expected = da.isel(time=slice(None, None, 3))
        atom = DownSample(3, "time")
        result = atom(da)
        assert result.equals(expected)
        atom.reset()
        result = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")
        assert result.equals(expected)

    def test_upsample(self):
        da = xd.DataArray(
            [1, 1, 1], {"time": {"tie_indices": [0, 2], "tie_values": [0.0, 6.0]}}
        )
        expected = xd.DataArray(
            [3, 0, 0, 3, 0, 0, 3, 0, 0],
            {"time": {"tie_indices": [0, 8], "tie_values": [0.0, 8.0]}},
        )
        atom = UpSample(3, dim="time")
        result = atom(da)
        assert result.equals(expected)

        da = wavelet_wavefronts()
        chunks = xd.split(da, 6, "time")
        expected = atom(da)
        result = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")
        assert result.equals(expected)

    def test_firfilter(self):
        da = wavelet_wavefronts()
        chunks = xd.split(da, 6, "time")
        taps = sp.firwin(11, 0.4, pass_zero="lowpass")
        expected = xs.lfilter(taps, 1.0, da, "time")
        expected["time"] -= np.timedelta64(20, "ms") * 5
        atom = FIRFilter(11, 10.0, "lowpass", dim="time")
        result = atom(da)
        assert result.equals(expected)

        result = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)
        assert result.attrs == expected.attrs
        assert result.name == expected.name


class TestResamplePoly:
    def test_up_down(self):
        da = wavelet_wavefronts()
        chunks = xd.split(da, 6, "time")

        expected = xs.resample_poly(da, 5, 2, "time")
        atom = ResamplePoly(125, maxfactor=10, dim="time")
        result = atom(da)
        result_chunked = xd.concat(
            [atom(chunk, chunk_dim="time") for chunk in chunks], "time"
        )

        assert np.allclose(result.values, result_chunked.values, atol=1e-15, rtol=1e-12)
        assert result.coords.equals(result_chunked.coords)
        assert result.attrs == result_chunked.attrs
        assert result.name == result_chunked.name

        result = result.sel(time=slice("2023-01-01T00:00:01", "2023-01-01T00:00:05"))
        expected = expected.sel(
            time=slice("2023-01-01T00:00:01", "2023-01-01T00:00:05")
        )
        assert np.allclose(result.values, expected.values, atol=1e-15, rtol=1e-12)
        assert result.coords.equals(expected.coords)
        assert result.attrs == expected.attrs
        assert result.name == expected.name

    def test_nothing_to_do(self):
        da = wavelet_wavefronts()
        fs = 1 / xd.get_sampling_interval(da, "time")
        atom = ResamplePoly(fs, maxfactor=10, dim="time")
        result = atom(da)
        assert result.equals(da)


class TestMLPicker:
    @pytest.mark.slow
    def test_picker(self):
        from seisbench.models import PhaseNet

        model = PhaseNet.from_pretrained("diting")
        picker = MLPicker(model, "time", device="cpu", component_strategy="Z")
        da = randn_wavefronts()
        # da = da.isel(time=slice(0, 5000)) TODO: why not faster ?
        expected = picker(da)
        chunks = xd.split(da, 4, "time")
        result = xd.concat([picker(chunk, chunk_dim="time") for chunk in chunks])
        assert result.equals(expected)

    @pytest.mark.slow
    def test_compare_with_seisbench(self):
        import obspy
        from seisbench.models import PhaseNet

        model = PhaseNet.from_pretrained("original")  # works at 100 Hz
        model.to_preferred_device()
        picker = MLPicker(model, "time", component_strategy="clone")

        # generate one trace
        da = randn_wavefronts()  # 100 Hz
        da = da.isel(distance=slice(0, 1))

        # xdas
        result = picker(da)

        # convert to one stream with clonning
        st = da.to_stream()
        tr = st[0]
        st = obspy.Stream()
        for component in model.component_order:
            _tr = tr.copy()
            _tr.stats.component = component
            st.append(_tr)

        # seisbench
        expected = model.annotate(st)
        expected = xd.DataArray.from_stream(expected)

        # align because of different overlap managment
        _result = result.sel(time=slice(expected["time"][0].values, None))
        _result = _result.isel(distance=0)
        _expected = expected.sel(time=slice(None, result["time"][-1].values))
        _expected = _expected.transpose("time", "channel")

        # remove unfinished end part
        _result = _result[:-1000]
        _expected = _expected[:-1000]

        # check equal by removing the
        np.testing.assert_allclose(
            _result.values, _expected.values, rtol=1e-5, atol=1e-7
        )
        np.testing.assert_array_max_ulp(_result.values, _expected.values, maxulp=300)


class TestAtomCoreMissingBranches:
    def test_repr_with_nested_atoms(self):
        from xdas.atoms.core import Atom, State

        a = [1, 1]
        b = [1, 1]
        atom = IIRFilter(a, b, 10.0, "lowpass", dim="time")
        s = repr(atom)
        assert "IIRFilter" in s

    def test_sequential_wraps_non_atom(self):
        seq = Sequential([np.abs, np.square])
        assert all(isinstance(a, Partial) for a in seq)

    def test_partial_non_callable_raises(self):
        with pytest.raises(TypeError, match="`func` should be callable"):
            Partial(42)

    def test_partial_multiple_ellipsis_raises(self):
        with pytest.raises(ValueError, match="at most one Ellipsis"):
            Partial(np.abs, ..., ...)

    def test_partial_state_kwarg(self):
        from xdas.atoms.core import State

        p = Partial(np.abs, key=State(42))
        assert "key" in p._state

    def test_partial_stateful_call(self):
        da = wavelet_wavefronts()
        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        da_out = atom(da, chunk_dim="time")
        assert da_out.shape == da.shape

    def test_save_and_load_state(self, tmp_path):
        from xdas.atoms.core import Atom, State

        class SimpleAtom(Atom):
            def __init__(self):
                super().__init__()
                self.buf = State(...)

            def initialize(self, x, **flags):
                self.buf = State(x.copy())

            def initialize_from_state(self):
                pass

            def call(self, x, **flags):
                return x

        atom = SimpleAtom()
        da = xd.DataArray(np.ones((10, 5)), dims=("x", "y"))
        atom(da, chunk_dim="x")
        path = tmp_path / "state.nc"
        atom.save_state(path)
        recovered = SimpleAtom()
        recovered.load_state(path)
        # TODO: should be Dataarray.equals comparison
        np.testing.assert_array_equal(recovered.buf, atom.buf)

    def test_atomized_two_atom_args_raises(self):
        da = wavelet_wavefronts()
        atom1 = xs.integrate(...)
        atom2 = xs.integrate(...)
        with pytest.raises(ValueError, match="Only one Atom"):
            xs.integrate(atom1, atom2)

    def test_atomized_sequential_input(self):
        atom = xs.integrate(...)
        seq = Sequential([atom])
        initial_len = len(seq)
        xs.integrate(seq)
        assert len(seq) == initial_len + 1

    def test_set_state_nested_atom(self):
        from xdas.atoms.core import Atom, State

        class InnerAtom(Atom):
            def __init__(self):
                super().__init__()
                self.val = State(np.zeros(3))

            def call(self, x, **flags):
                return x

        class OuterAtom(Atom):
            def __init__(self):
                super().__init__()
                self.inner = InnerAtom()

            def call(self, x, **flags):
                return x

        outer = OuterAtom()
        state = xd.DataArray(np.ones(3))
        outer.set_state({"inner": {"val": state}})
        # TODO: should be Dataarray.equals comparison
        np.testing.assert_array_equal(outer.inner.val, state)

    def test_partial_repr_long_kwarg(self):
        atom = Partial(np.abs, axis=np.arange(10))
        r = repr(atom)
        assert "<ndarray>" in r


class TestAtomSignalMissingBranches:
    def test_iirfilter_invalid_stype(self):
        with pytest.raises(ValueError):
            IIRFilter(4, 10.0, "lowpass", dim="time", stype="invalid")

    def test_iirfilter_initialize_from_state_zpk_stype(self):
        da = wavelet_wavefronts()
        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        atom(da, chunk_dim="time")
        atom.stype = "zpk"
        with pytest.raises(ValueError):
            atom.initialize_from_state()

    def test_downsample_factor_one(self):
        da = wavelet_wavefronts()
        atom = DownSample(1, dim="time")
        result = atom(da)
        assert result.equals(da)

    def test_upsample_no_scale(self):
        da = wavelet_wavefronts().isel(time=slice(0, 10))
        atom = UpSample(2, dim="time", scale=False)
        result = atom(da)
        assert result.sizes["time"] == 2 * da.sizes["time"]


class TestMLPickerMissingBranches:
    def test_lazy_module_import_error(self):
        from xdas.atoms.ml import LazyModule

        mod = LazyModule("nonexistent_module_xdas_test")
        with pytest.raises(ImportError, match="is not installed by default"):
            _ = mod.something

    def test_mlpicker_invalid_component_strategy(self):
        import seisbench.models as sbm

        model = sbm.PhaseNet.from_pretrained("geofon")
        with pytest.raises(ValueError, match="component_strategy must be one of"):
            MLPicker(model, dim="time", component_strategy="invalid")
