import pickle

import numpy as np
import pytest
import scipy.signal as sp

import xdas as xd
import xdas.signal as xs
from xdas.atoms import (
    Atom,
    DownSample,
    FIRFilter,
    IIRFilter,
    LFilter,
    MLPicker,
    Partial,
    Polyphase,
    ResamplePoly,
    Sequential,
    UpSample,
    atomized,
)
from xdas.atoms.core import _whole_record
from xdas.signal import lfilter
from xdas.synthetics import randn_wavefronts


class TestAbstractAtom:
    def test(self):
        atom = Atom()
        assert atom.initialize(None) is NotImplemented
        assert atom.initialize_from_state() is NotImplemented
        assert atom.call(None) is NotImplemented


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
        da = xd.testing.dummy()

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
        da = xd.testing.dummy()
        chunks = xd.split(da, 6, "time")

        b, a = sp.iirfilter(4, 10.0, btype="lowpass", fs=100.0)
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
        da = xd.testing.dummy()
        chunks = xd.split(da, 6, "time")

        sos = sp.iirfilter(4, 10.0, btype="lowpass", fs=100.0, output="sos")
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
        # size must be a multiple of the decimation factor: on a partial trailing
        # phase the chunked path drops one sample that the monolithic one keeps
        da = xd.testing.dummy(shape=(102, 10))
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
            [1, 1, 1],
            {
                "time": {
                    "tie_indices": [0, 2],
                    "tie_values": [0.0, 6.0],
                    "sampling_interval": 3.0,
                }
            },
        )
        expected = xd.DataArray(
            [3, 0, 0, 3, 0, 0, 3, 0, 0],
            {
                "time": {
                    "tie_indices": [0, 8],
                    "tie_values": [0.0, 8.0],
                    "sampling_interval": 1.0,
                }
            },
        )
        atom = UpSample(3, dim="time")
        result = atom(da)
        assert result.equals(expected)

        da = xd.testing.dummy()
        chunks = xd.split(da, 6, "time")
        expected = atom(da)
        result = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")
        assert result.equals(expected)

        assert UpSample(1, dim="time")(da).equals(da)

    def test_firfilter(self):
        da = xd.testing.dummy()
        chunks = xd.split(da, 6, "time")
        taps = sp.firwin(11, 0.2, pass_zero="lowpass")
        expected = xs.lfilter(taps, 1.0, da, "time")
        expected["time"] -= np.timedelta64(10, "ms") * 5
        atom = FIRFilter(11, 10.0, "lowpass", dim="time")
        result = atom(da)
        # The polyphase form accumulates the taps in a different order than
        # `lfilter`, so the two agree to rounding rather than exactly.
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)

        result = xd.concat([atom(chunk, chunk_dim="time") for chunk in chunks], "time")
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)
        assert result.attrs == expected.attrs
        assert result.name == expected.name


class TestResamplePoly:
    def test_up_down(self):
        da = xd.testing.dummy(shape=(300, 10), step=(0.02, 25.0))  # 50 Hz, 6 s
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

        result = result.sel(time=slice("2024-05-21T00:00:01", "2024-05-21T00:00:05"))
        expected = expected.sel(
            time=slice("2024-05-21T00:00:01", "2024-05-21T00:00:05")
        )
        assert np.allclose(result.values, expected.values, atol=1e-15, rtol=1e-12)
        assert result.coords.equals(expected.coords)
        assert result.attrs == expected.attrs
        assert result.name == expected.name

    def test_nothing_to_do(self):
        da = xd.testing.dummy()
        fs = 1 / xd.get_sampling_interval(da, "time")
        atom = ResamplePoly(fs, maxfactor=10, dim="time")
        result = atom(da)
        assert result.equals(da)


class TestPolyphase:
    """The fused kernel against the upsample/filter/downsample chain it replaces."""

    @staticmethod
    def chain(taps, up, down, dim="time"):
        """The unfused formulation, kept here as the reference."""

        def apply(da):
            # UpSample already carries the `up` energy scaling of the taps.
            da = UpSample(up, dim=dim)(da) if up > 1 else da
            da = LFilter(taps, [1.0], dim)(da)
            da[dim] -= xd.get_sampling_interval(da, dim, cast=False) * (
                (len(taps) - 1) // 2
            )
            return DownSample(down, dim)(da) if down > 1 else da

        return apply

    @pytest.mark.parametrize("up, down", [(1, 1), (1, 2), (1, 5), (2, 1), (2, 5)])
    def test_equals_the_explicit_chain(self, up, down):
        da = xd.testing.dummy(shape=(101, 5))
        taps = sp.firwin(20 * max(up, down) + 1, 0.4 / max(up, down))
        expected = self.chain(taps, up, down)(da)
        result = Polyphase(up * taps, up, down, "time")(da)
        assert result.sizes["time"] == expected.sizes["time"]
        np.testing.assert_allclose(result.values, expected.values, atol=1e-15)
        assert result.coords.equals(expected.coords)

    @pytest.mark.parametrize("up, down", [(1, 2), (2, 5), (3, 10)])
    def test_pinned_against_upfirdn(self, up, down):
        # An eager call emits ceil(size * up / down) samples, the leading ones
        # of scipy's own upfirdn output; the group delay moves the coordinate,
        # never the values.
        da = xd.testing.dummy(shape=(101, 5))
        taps = sp.firwin(21, 0.4 / max(up, down))
        result = Polyphase(taps, up, down, "time")(da)
        full = sp.upfirdn(taps, da.values, up, down, axis=0)
        assert result.sizes["time"] == -(-101 * up // down)
        np.testing.assert_allclose(
            result.values, full[: result.sizes["time"]], atol=1e-15
        )

    @pytest.mark.parametrize("up, down", [(1, 2), (2, 5), (2, 3)])
    @pytest.mark.parametrize("nchunk", [3, 7])
    def test_chunked_equals_eager(self, up, down, nchunk):
        da = xd.testing.dummy(shape=(101, 5))
        taps = sp.firwin(21, 0.4 / max(up, down))
        eager = Polyphase(taps, up, down, "time")(da)
        atom = Polyphase(taps, up, down, "time")
        outs = [atom(chunk, chunk_dim="time") for chunk in xd.split(da, nchunk, "time")]
        chunked = xd.concat(outs, "time")
        np.testing.assert_allclose(chunked.values, eager.values, atol=1e-15)
        assert chunked.coords.equals(eager.coords)

    def test_keeps_the_data_precision(self):
        da = xd.testing.dummy(shape=(101, 5), dtype="float32")
        taps = sp.firwin(21, 0.2)
        assert taps.dtype == np.float64
        result = Polyphase(taps, 1, 2, "time")(da)
        assert result.dtype == np.float32
        expected = Polyphase(taps, 1, 2, "time")(da.copy(data=da.values.astype(float)))
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-6)

    def test_rate_the_coordinate_cannot_represent_exactly(self):
        # 100 Hz resampled by 3/10 is 10/3 nanoseconds per output sample: the
        # truncated step must be declared as jitter, not silently drift.
        da = xd.testing.dummy(shape=(101, 5))
        taps = sp.firwin(31, 0.4 / 10)
        result = Polyphase(3 * taps, 3, 10, "time")(da)
        coord = result.coords["time"]
        assert coord.isregular()
        assert coord.tolerance > np.timedelta64(0, "ns")

    def test_too_few_taps(self):
        da = xd.testing.dummy(shape=(20, 5))
        atom = Polyphase(sp.firwin(3, 0.4), 5, 1, "time")
        with pytest.raises(ValueError, match="at least 5 taps"):
            atom(da)

    def test_empty_input_emits_nothing(self):
        da = xd.testing.dummy(shape=(101, 5))
        atom = Polyphase(sp.firwin(21, 0.2), 1, 2, "time")
        assert atom(da.isel(time=slice(0, 0))) == xd.DataCollection([])

    def test_upsample_single_sample(self):
        # a one-sample chunk has a single tie: the upsampled block still
        # spans `factor` samples, which takes a second tie to say.
        da = xd.testing.dummy(shape=(101, 5))
        result = UpSample(3, dim="time")(da.isel(time=slice(0, 1)))
        assert result.sizes["time"] == 3


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
        da = xd.testing.dummy()
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
        atom1 = xs.integrate(...)
        atom2 = xs.integrate(...)
        with pytest.raises(ValueError, match="Only one Atom"):
            xs.integrate(atom1, atom2)

    def test_atomized_sequential_input(self):
        # Composition has value semantics: the input pipeline is not mutated.
        atom = xs.integrate(...)
        seq = Sequential([atom])
        result = xs.integrate(seq)
        assert isinstance(result, Sequential)
        assert len(result) == 2
        assert len(seq) == 1

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
        da = xd.testing.dummy()
        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        atom(da, chunk_dim="time")
        atom.stype = "zpk"
        with pytest.raises(ValueError):
            atom.initialize_from_state()

    def test_downsample_factor_one(self):
        da = xd.testing.dummy()
        atom = DownSample(1, dim="time")
        result = atom(da)
        assert result.equals(da)

    def test_upsample_no_scale(self):
        da = xd.testing.dummy().isel(time=slice(0, 10))
        atom = UpSample(2, dim="time", scale=False)
        result = atom(da)
        assert result.sizes["time"] == 2 * da.sizes["time"]


class TestLegacyIrregularCoordinates:
    """Atoms must keep working on coordinates written before 0.2.8."""

    @staticmethod
    def legacy(**kwargs):
        # Data saved by earlier versions declares no rate and no jitter.
        da = xd.testing.dummy(**kwargs)
        da["time"] = xd.Coordinate(
            {
                "tie_indices": da["time"].tie_indices,
                "tie_values": da["time"].tie_values,
            },
            "time",
        )
        return da

    def test_upsample_on_irregular_coordinate(self):
        da = self.legacy(shape=(20, 3))
        assert not da["time"].isregular()
        result = UpSample(3, dim="time")(da)
        assert result.sizes["time"] == 3 * da.sizes["time"]
        # Nothing was declared upstream, so nothing is claimed downstream.
        assert not result["time"].isregular()

    def test_resample_poly_atom_on_irregular_coordinate(self):
        da = self.legacy(shape=(100, 3))
        target = 1.0 / (2.0 * xd.get_sampling_interval(da, "time"))
        result = ResamplePoly(target=target, dim="time")(da)
        assert result.sizes["time"] == da.sizes["time"] // 2
        assert not result["time"].isregular()

    def test_upsample_keeps_declaring_rate_when_input_is_regular(self):
        da = xd.testing.dummy(shape=(20, 3))
        assert da["time"].isregular()
        result = UpSample(3, dim="time")(da)
        assert result["time"].isregular()


class TestSequentialReset:
    def test_reset_clears_stateful_atoms(self):
        da = xd.testing.dummy(shape=(400, 3))

        def stream(sequence, nchunks=4):
            size = da.sizes["time"] // nchunks
            return xd.concat(
                [
                    sequence(
                        da.isel(time=slice(k * size, (k + 1) * size)), chunk_dim="time"
                    )
                    for k in range(nchunks)
                ],
                "time",
            )

        sequence = Sequential([IIRFilter(4, 10.0, "lowpass", dim="time")])
        first = stream(sequence)
        sequence.reset()
        second = stream(sequence)
        np.testing.assert_array_equal(first.values, second.values)


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


class TestCompose:
    def test_rshift_atoms(self):
        pipeline = xs.detrend(...) >> xs.integrate(...)
        assert isinstance(pipeline, Sequential)
        assert len(pipeline) == 2

    def test_rshift_value_semantics(self):
        head = xs.detrend(...)
        pipeline = head >> xs.integrate(...)
        longer = pipeline >> np.square
        assert isinstance(head, Partial)
        assert len(pipeline) == 2
        assert len(longer) == 3

    def test_irshift(self):
        pipeline = xs.detrend(...)
        pipeline >>= xs.integrate(...)
        pipeline >>= np.square
        assert isinstance(pipeline, Sequential)
        assert len(pipeline) == 3

    def test_rshift_callable_wraps(self):
        pipeline = xs.detrend(...) >> np.square
        assert isinstance(pipeline[-1], Partial)
        assert pipeline[-1].func is np.square

    def test_rrshift_callable_prepends(self):
        pipeline = np.square >> xs.detrend(...)
        assert isinstance(pipeline, Sequential)
        assert pipeline[0].func is np.square

    def test_rrshift_applies_to_data(self):
        da = xd.testing.dummy()
        result = da >> Partial(np.square)
        assert np.allclose(result.values, np.square(da.values))

    def test_rrshift_applies_pipeline_to_data(self):
        da = xd.testing.dummy()
        pipeline = Partial(np.abs) >> Partial(np.square)
        result = da >> pipeline
        assert np.allclose(result.values, np.square(np.abs(da.values)))

    def test_named_sequential_kept_nested_on_right(self):
        named = Sequential([Partial(np.square)], name="named")
        pipeline = xs.detrend(...) >> named
        assert len(pipeline) == 2
        assert pipeline[1] is named

    def test_named_sequential_extended_keeps_name(self):
        named = Sequential([Partial(np.square)], name="named")
        pipeline = named >> Partial(np.abs)
        assert pipeline.name == "named"
        assert len(pipeline) == 2
        assert len(named) == 1

    def test_unnamed_sequentials_flatten(self):
        left = Partial(np.abs) >> Partial(np.square)
        right = Partial(np.sqrt) >> Partial(np.abs)
        pipeline = left >> right
        assert len(pipeline) == 4

    def test_rshift_with_data_on_right_raises(self):
        with pytest.raises(TypeError):
            xs.detrend(...) >> 1.0


class TestTracing:
    def test_ufunc_appends(self):
        atom = xs.detrend(...)
        traced = np.square(atom)
        assert isinstance(traced, Sequential)
        assert len(traced) == 2

    def test_expression_matches_eager(self):
        da = xd.testing.dummy()
        atom = xs.detrend(...)
        traced = 20 * np.log10(np.abs(atom) + 1e-12)
        expected = 20 * np.log10(np.abs(xs.detrend(da)) + 1e-12)
        assert np.allclose(traced(da).values, expected.values)

    def test_reflected_scalar(self):
        da = xd.testing.dummy()
        traced = 2.0 * xs.detrend(...)
        assert np.allclose(traced(da).values, 2.0 * xs.detrend(da).values)

    def test_untraceable_attribute_raises(self):
        atom = xs.detrend(...)
        with pytest.raises(AttributeError):
            _ = atom.values

    def test_fan_in_raises(self):
        atom1 = xs.detrend(...)
        atom2 = xs.detrend(...)
        with pytest.raises(TypeError, match="fan-in"):
            np.add(atom1, atom2)

    def test_same_atom_twice_raises(self):
        atom = xs.detrend(...)
        with pytest.raises(TypeError, match="fan-in"):
            np.add(atom, atom)

    def test_equality_is_identity(self):
        atom1 = xs.detrend(...)
        atom2 = xs.detrend(...)
        alias = atom1
        assert atom1 == alias
        assert atom1 != atom2
        assert len({atom1, atom2}) == 2

    def test_inplace_operator_traces_out_of_place(self):
        da = xd.testing.dummy()
        atom = xs.detrend(...)
        atom *= 2.0
        assert isinstance(atom, Sequential)
        assert np.allclose(atom(da).values, 2.0 * xs.detrend(da).values)

    def test_out_to_another_atom_raises(self):
        atom1 = xs.detrend(...)
        atom2 = xs.detrend(...)
        with pytest.raises(TypeError):
            np.multiply(atom1, 2.0, out=atom2)

    def test_non_call_ufunc_method_raises(self):
        atom = xs.detrend(...)
        with pytest.raises(TypeError):
            np.add.reduce(atom)

    def test_right_shift_as_data_traces(self):
        # np.right_shift with the atom on the *left* is an ordinary traced
        # ufunc, not the `da >> atom` application path.
        traced = np.right_shift(xs.detrend(...) >> Partial(np.abs), 1)
        assert isinstance(traced, Sequential)


class TestWholeRecordRefusal:
    """Whole-record functions carry their own guard at the definition site."""

    @staticmethod
    def whole_record_atom(*args, **kwargs):
        @atomized
        @_whole_record()
        def whole_record(da, dim="time"):
            return da

        return whole_record(*args, **kwargs)

    def test_chunked_along_dim_raises(self):
        da = xd.testing.dummy()
        atom = self.whole_record_atom(...)
        with pytest.raises(ValueError, match="whole record"):
            atom(da, chunk_dim="time")

    def test_chunked_along_other_dim_passes(self):
        da = xd.testing.dummy()
        atom = self.whole_record_atom(...)
        atom(da, chunk_dim="distance")

    def test_unchunked_passes(self):
        da = xd.testing.dummy()
        atom = self.whole_record_atom(...)
        atom(da)

    def test_positional_dim_resolved(self):
        da = xd.testing.dummy()
        atom = self.whole_record_atom(..., "distance")
        atom(da, chunk_dim="time")
        with pytest.raises(ValueError, match="whole record"):
            atom(da, chunk_dim="distance")

    def test_alias_dim_resolved_against_the_data(self):
        da = xd.testing.dummy()  # dims ("time", "distance")
        atom = self.whole_record_atom(..., "last")
        atom(da, chunk_dim="time")
        with pytest.raises(ValueError, match="whole record"):
            atom(da, chunk_dim="distance")

    def test_streaming_class_unaffected(self):
        da = xd.testing.dummy()
        chunks = xd.split(da, 3, "time")
        atom = IIRFilter(4, 10.0, "lowpass", dim="time")
        for chunk in chunks:
            atom(chunk, chunk_dim="time")


class TestFresh:
    def test_fresh_is_stateless_and_config_shared(self):
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        atom = Partial(xs.sosfilt, sos, ..., dim="time", zi=...)
        da = xd.testing.dummy()
        atom(da, chunk_dim="time")
        assert atom.initialized
        clone = atom.fresh()
        assert not clone.initialized
        assert clone.func is atom.func
        assert atom.initialized  # the original is untouched
        assert clone(da).equals(Partial(xs.sosfilt, sos, ..., dim="time", zi=...)(da))

    def test_fresh_recurses_into_sequences(self):
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        seq = Sequential(
            [Partial(xs.sosfilt, sos, ..., dim="time", zi=...), Partial(np.square)],
            name="energy",
        )
        da = xd.testing.dummy()
        seq(da, chunk_dim="time")
        clone = seq.fresh()
        assert not clone.initialized
        assert clone.name == "energy"
        assert len(clone) == len(seq)
        assert clone[0] is not seq[0]
        assert clone[0].func is seq[0].func


class TestFreshNested:
    def test_fresh_recurses_into_nested_class_atoms(self):
        da = xd.testing.dummy()
        atom = xd.atoms.Filter((1.0, 10.0))
        atom(da, chunk_dim="time")
        clone = atom.fresh()
        assert not clone.initialized
        assert clone.filter is not atom.filter
        assert atom.initialized


class TestRefusalHelper:
    def test_alias_without_data_is_conservative(self):
        atom = Partial(np.square)
        atom._refuse_chunked_along("distance", "time", None)  # distinct: passes
        with pytest.raises(ValueError, match="whole record"):
            atom._refuse_chunked_along("last", "time", None)

    def test_no_chunking_passes(self):
        Partial(np.square)._refuse_chunked_along("time", None, None)

    def test_kernel_dict_checks_its_keys(self):
        da = xd.testing.dummy()
        atom = Partial(np.square)
        atom._refuse_chunked_along({"distance": 5}, "time", da)
        with pytest.raises(ValueError, match="whole record"):
            atom._refuse_chunked_along({"time": 5}, "time", da)


class TestInitializedRecurses:
    def test_a_fresh_nested_atom_reports_uninitialized(self):
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        seq = Sequential([Partial(xs.sosfilt, sos, ..., dim="time", zi=...)])
        assert not seq.initialized
        seq(xd.testing.dummy(), chunk_dim="time")
        assert seq.initialized
