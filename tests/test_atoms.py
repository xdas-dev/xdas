import os
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import scipy.signal as sp

import xdas
import xdas.atoms as atoms
import xdas.signal as xp
from xdas.atoms import FIRFilter, IIRFilter, Partial, ResamplePoly, StatePartial
from xdas.core import chunk, concatenate
from xdas.signal import lfilter
from xdas.synthetics import generate
from xdas.xarray import mean


class TestPartialAtom:
    def test_init(self):
        sequence = xdas.Sequential(
            [
                xdas.Partial(xp.taper, dim="time"),
                xdas.Partial(xp.taper, dim="distance"),
                xdas.Partial(np.abs),
                xdas.Partial(np.square),
            ]
        )


class TestProcessing:
    def test_sequence(self):
        # Generate a temporary dataset
        db = generate()

        # Declare sequence to execute
        sequence = xdas.Sequential(
            [
                xdas.Partial(np.abs),
                xdas.Partial(np.square, name="some square"),
                xdas.Partial(mean, dim="time"),
            ]
        )

        # Sequence processing
        result1 = sequence(db)
        # Manual processing
        result2 = mean(np.abs(db) ** 2, dim="time")

        # Test
        assert np.allclose(result1.values, result2.values)


class TestDecorator:
    def test_decorator(self):
        a = [1, 1]
        b = [1, 1]
        atom = lfilter(b, a, ..., "time")
        statefull = lfilter(b, a, ..., "time", zi=...)
        assert isinstance(atom, Partial)
        assert isinstance(statefull, StatePartial)
        assert statefull.state == {"zi": ...}


class TestFilters:
    def test_lfilter(self):
        db = generate()
        chunks = chunk(db, 6, "time")

        b, a = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0)
        data = sp.lfilter(b, a, db.values, axis=0)
        expected = db.copy(data=data)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        monolithic = atom(db)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
        chunked = concatenate([atom(chunk, chunk="time") for chunk in chunks], "time")

        assert monolithic.equals(expected)
        assert chunked.equals(expected)

        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "state.nc")

            atom_a = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
            chunks_a = [atom_a(chunk, chunk="time") for chunk in chunks[:3]]
            atom_a.save_state(path)

            atom_b = IIRFilter(4, 10.0, "lowpass", dim="time", stype="ba")
            atom_b.load_state(path)
            chunks_b = [atom_b(chunk, chunk="time") for chunk in chunks[3:]]

            result = concatenate(chunks_a + chunks_b, "time")
            assert result.equals(expected)

    def test_sosfilter(self):
        db = generate()
        chunks = chunk(db, 6, "time")

        sos = sp.iirfilter(4, 10.0, btype="lowpass", fs=50.0, output="sos")
        data = sp.sosfilt(sos, db.values, axis=0)
        expected = db.copy(data=data)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time")
        monolithic = atom(db)

        atom = IIRFilter(4, 10.0, "lowpass", dim="time")
        chunked = concatenate([atom(chunk, chunk="time") for chunk in chunks], "time")

        assert monolithic.equals(expected)
        assert chunked.equals(expected)

        with TemporaryDirectory() as dirpath:
            path = os.path.join(dirpath, "state.nc")

            atom_a = IIRFilter(4, 10.0, "lowpass", dim="time")
            chunks_a = [atom_a(chunk, chunk="time") for chunk in chunks[:3]]
            atom_a.save_state(path)

            atom_b = IIRFilter(4, 10.0, "lowpass", dim="time")
            atom_b.load_state(path)
            chunks_b = [atom_b(chunk, chunk="time") for chunk in chunks[3:]]

            result = concatenate(chunks_a + chunks_b, "time")
            assert result.equals(expected)

    def test_downsample(self):
        db = generate()
        chunks = chunk(db, 6, "time")
        expected = db.isel(time=slice(None, None, 3))
        atom = atoms.DownSample(3, "time")
        result = atom(db)
        assert result.equals(expected)
        result = concatenate([atom(chunk, chunk="time") for chunk in chunks], "time")
        assert result.equals(expected)

    def test_upsample(self):
        db = xdas.Database(
            [1, 1, 1], {"time": {"tie_indices": [0, 2], "tie_values": [0.0, 6.0]}}
        )
        expected = xdas.Database(
            [3, 0, 0, 3, 0, 0, 3, 0, 0],
            {"time": {"tie_indices": [0, 8], "tie_values": [0.0, 8.0]}},
        )
        atom = atoms.UpSample(3, dim="time")
        result = atom(db)
        assert result.equals(expected)

        db = generate()
        chunks = chunk(db, 6, "time")
        atom = atoms.UpSample(3, dim="time")
        expected = atom(db)
        result = concatenate([atom(chunk, chunk="time") for chunk in chunks], "time")
        assert result.equals(expected)

    def test_firfilter(self):
        db = generate()
        chunks = chunk(db, 6, "time")
        taps = sp.firwin(11, 0.4, pass_zero="lowpass")
        expected = xp.lfilter(taps, 1.0, db, "time")
        expected["time"] -= np.timedelta64(20, "ms") * 5
        atom = FIRFilter(11, 10.0, "lowpass", dim="time")
        result = atom(db)
        assert result.equals(expected)

        atom = FIRFilter(11, 10.0, "lowpass", dim="time")
        result = concatenate([atom(chunk, chunk="time") for chunk in chunks], "time")
        assert np.allclose(result.values, expected.values, atol=1e-16, rtol=1e-11)
        assert result.coords.equals(expected.coords)
        assert result.attrs == expected.attrs
        assert result.name == expected.name

    def test_resample_poly(self):
        db = generate()
        chunks = chunk(db, 6, "time")

        expected = xp.resample_poly(db, 5, 2, "time")
        atom = ResamplePoly(125, maxfactor=10, dim="time")
        result = atom(db)
        atom = ResamplePoly(125, maxfactor=10, dim="time")
        result_chunked = concatenate(
            [atom(chunk, chunk="time") for chunk in chunks], "time"
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
