import os
import tempfile

import numpy as np
import pytest
import xarray as xr

import xdas
from xdas.synthetics import generate
from xdas.virtual import *


class TestFunctional:
    def test_all(self):
        with tempfile.TemporaryDirectory() as dirpath:
            expected = generate()
            chunks = xdas.chunk(expected, 3)
            for index, chunk in enumerate(chunks, start=1):
                chunk.to_netcdf(os.path.join(dirpath, f"{index:03d}.nc"))

            db = xdas.open_database(os.path.join(dirpath, "002.nc"))
            datasource = db.data
            assert np.allclose(np.asarray(datasource[0]), db.load().values[0])
            assert np.allclose(np.asarray(datasource[0][1]), db.load().values[0][1])
            assert np.allclose(
                np.asarray(datasource[:, 0][1]), db.load().values[:, 0][1]
            )
            assert np.allclose(
                np.asarray(datasource[:, 0][1]), db.load().values[:, 0][1]
            )
            assert np.allclose(np.asarray(datasource[10:][1]), db.load().values[10:][1])
            with pytest.raises(IndexError):
                datasource[1, 2, 3]
            assert np.allclose(np.asarray(datasource[10:][1]), db.load().values[10:][1])
            assert np.array_equal(db.load().data, db.load().data)
            db1 = db.sel(
                time=slice("2023-01-01T00:00:03", None),
                distance=slice(1000, None),
            ).load()
            db2 = db.load().sel(
                time=slice("2023-01-01T00:00:03", None),
                distance=slice(1000, None),
            )
            assert db1.equals(db2)


class TestDataLayout:
    def test_init(self):
        ...


class TestDataSource:
    def test_init(self):
        ...


class TestSelection:
    def test_init(self):
        shape = (2, 3, 5)
        sel = Selection(shape)
        assert sel._shape == shape
        for size, selector in zip(shape, sel._selectors):
            assert isinstance(selector, SliceSelector)
            assert selector._range == range(size)
        assert sel._whole == True

    def test_usecases(self):
        arr = np.arange(2 * 3 * 5).reshape(2, 3, 5)
        sel = Selection(arr.shape)

        expected = arr[0][:, 1:-1][::2]
        sub = sel[0][:, 1:-1][::2]
        slc = sub.get_indexer()
        assert np.array_equal(arr[slc], expected)
        assert sub.shape == expected.shape
        assert sub.ndim == expected.ndim

        expected = arr[:][1:0][:, 1]
        sub = sel[:][1:0][:, 1]
        slc = sub.get_indexer()
        assert np.array_equal(arr[slc], expected)
        assert sub.shape == expected.shape
        assert sub.ndim == expected.ndim


class TestSingleSelector:
    def test_init(self):
        sel = SingleSelector(0)
        assert sel._index == 0

    def test_get_indexer(self):
        sel = SingleSelector(0)
        assert sel.get_indexer() == 0


class TestSliceSelector:
    def test_init(self):
        sel = SliceSelector(3)
        assert sel._range == range(3)

    def test_getitem_int(self):
        sel = SliceSelector(3)
        assert isinstance(sel[0], SingleSelector)
        with pytest.raises(IndexError):
            sel[-4]
        assert sel[-3]._index == 0
        assert sel[-2]._index == 1
        assert sel[-1]._index == 2
        assert sel[0]._index == 0
        assert sel[1]._index == 1
        assert sel[2]._index == 2
        with pytest.raises(IndexError):
            sel[3]

    def test_getitem_slice(self):
        sel = SliceSelector(5)
        assert isinstance(sel[0:1], SliceSelector)
        assert sel[:]._range == range(5)
        assert sel[0:1]._range == range(0, 1)
        assert sel[1:0]._range == range(1, 0)
        assert sel._range == range(5)
        sel = sel[1:-1]
        assert sel._range == range(1, 4)
        assert sel[0]._index == 1
        assert sel[-1]._index == 3
        sel = sel[::2]
        assert sel._range == range(1, 4, 2)
        assert sel[0]._index == 1
        assert sel[-1]._index == 3

    def test_get_indexer(self):
        sel = SliceSelector(5)
        assert sel.get_indexer() == slice(0, 5, 1)
        assert sel[:].get_indexer() == slice(0, 5, 1)
        assert sel[0:0].get_indexer() == slice(None, 0, None)
        assert sel[1:0].get_indexer() == slice(None, 0, None)
        assert sel[0:1].get_indexer() == slice(0, 1, 1)
        assert sel[::2].get_indexer() == slice(0, 5, 2)
        assert sel[:-1].get_indexer() == slice(0, 4, 1)
        assert sel[-1:].get_indexer() == slice(4, 5, 1)
        arr = [0, 1, 2, 3, 4]
        assert arr[:] == arr[sel[:].get_indexer()]
        assert arr[0:0] == arr[sel[0:0].get_indexer()]
        assert arr[1:0] == arr[sel[1:0].get_indexer()]
        assert arr[:1] == arr[sel[:1].get_indexer()]
        assert arr[1:] == arr[sel[1:].get_indexer()]
        assert arr[1:4:2] == arr[sel[1:4:2].get_indexer()]
        assert arr[1:-1] == arr[sel[1:-1].get_indexer()]
        assert arr[1:-1][1:-1] == arr[sel[1:-1][1:-1].get_indexer()]
        assert arr[1:-1][::2] == arr[sel[1:-1][::2].get_indexer()]
