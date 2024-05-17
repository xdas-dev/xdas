import numpy as np
import scipy.signal as sp

import xdas
import xdas as xd
import xdas.signal as xp
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
                Partial(xp.taper, dim="time"),
                Partial(xp.taper, dim="distance"),
                Partial(np.abs),
                Partial(np.square),
            ]
        )


class TestProcessing:
    def test_sequence(self):
        # Generate a temporary dataset
        da = wavelet_wavefronts()

        # Declare sequence to execute
        seq = Sequential(
            [
                Partial(np.abs),
                Partial(np.square, name="some square"),
                Partial(xdas.mean, dim="time"),
            ]
        )

        # Sequence processing
        result1 = seq(da)
        # Manual processing
        result2 = xdas.mean(np.abs(da) ** 2, dim="time")

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
        chunks = xdas.split(da, 6, "time")

        b, a = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0)
        data = sp.lfilter(b, a, da.values, axis=0)
        expected = da.copy(data=data)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        monolithic = atom(da)

        chunked = xdas.concatenate(
            [atom(chunk, chunk_dim="time") for chunk in chunks], "time"
        )

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

        #     result = xdas.concatenate(chunks_a + chunks_b, "time")
        #     assert result.equals(expected)

    def test_sosfilter(self):
        da = wavelet_wavefronts()
        chunks = xdas.split(da, 6, "time")

        sos = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0, output="sos")
        data = sp.sosfilt(sos, da.values, axis=0)
        expected = da.copy(data=data)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time")
        monolithic = atom(da)

        chunked = xdas.concatenate(
            [atom(chunk, chunk_dim="time") for chunk in chunks], "time"
        )

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

        #     result = xdas.concatenate(chunks_a + chunks_b, "time")
        #     assert result.equals(expected)

    def test_downsample(self):
        da = wavelet_wavefronts()
        chunks = xdas.split(da, 6, "time")
        expected = da.isel(time=slice(None, None, 3))
        atom = DownSample(3, "time")
        result = atom(da)
        assert result.equals(expected)
        atom.reset()
        result = xdas.concatenate(
            [atom(chunk, chunk_dim="time") for chunk in chunks], "time"
        )
        assert result.equals(expected)

    def test_upsample(self):
        da = xdas.DataArray(
            [1, 1, 1], {"time": {"tie_indices": [0, 2], "tie_values": [0.0, 6.0]}}
        )
        expected = xdas.DataArray(
            [3, 0, 0, 3, 0, 0, 3, 0, 0],
            {"time": {"tie_indices": [0, 8], "tie_values": [0.0, 8.0]}},
        )
        atom = UpSample(3, dim="time")
        result = atom(da)
        assert result.equals(expected)

        da = wavelet_wavefronts()
        chunks = xdas.split(da, 6, "time")
        expected = atom(da)
        result = xdas.concatenate(
            [atom(chunk, chunk_dim="time") for chunk in chunks], "time"
        )
        assert result.equals(expected)

    def test_firfilter(self):
        da = wavelet_wavefronts()
        chunks = xdas.split(da, 6, "time")
        taps = sp.firwin(11, 0.4, pass_zero="lowpass")
        expected = xp.lfilter(taps, 1.0, da, "time")
        expected["time"] -= np.timedelta64(20, "ms") * 5
        atom = FIRFilter(11, 10.0, "lowpass", dim="time")
        result = atom(da)
        assert result.equals(expected)

        result = xdas.concatenate(
            [atom(chunk, chunk_dim="time") for chunk in chunks], "time"
        )
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)
        assert result.attrs == expected.attrs
        assert result.name == expected.name

    def test_resample_poly(self):
        da = wavelet_wavefronts()
        chunks = xdas.split(da, 6, "time")

        expected = xp.resample_poly(da, 5, 2, "time")
        atom = ResamplePoly(125, maxfactor=10, dim="time")
        result = atom(da)
        result_chunked = xdas.concatenate(
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


class TestMLPicker:
    def test_picker(self):
        from seisbench.models import PhaseNet

        model = PhaseNet.from_pretrained("diting")
        picker = MLPicker(model, "time", device="cpu")
        da = randn_wavefronts()
        expected = picker(da)
        chunks = xd.split(da, 4, "time")
        result = xd.concatenate([picker(chunk, chunk_dim="time") for chunk in chunks])
        assert result.equals(expected)
