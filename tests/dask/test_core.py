import os
from glob import glob
from tempfile import TemporaryDirectory

import dask
import numpy as np

from xdas.dask.core import from_dict, fuse, iskey, to_dict


class TestIsKey:
    def test_valid(self):
        keys = [("a", 0), ("a", 0, 1), "name-s0d9us-df63ij"]
        for key in keys:
            assert iskey(key)

    def test_invalid(self):
        keys = ["", (sum, 0, 1), ("a",), ("a", "b")]
        for key in keys:
            assert not iskey(key)


class TestFuse:
    def test_simple(self):
        graph = {
            "a": "b",
            "b": (np.sum, [1, 2, 3]),
        }
        assert fuse(graph) == {"a": (np.sum, [1, 2, 3])}

    def test_recursive(self):
        graph = {
            "a": "b",
            "b": "c",
            "c": (np.sum, [1, 2, 3]),
        }
        assert fuse(graph) == {"a": (np.sum, [1, 2, 3])}

    def test_tuple(self):
        graph = {
            ("a", 0): ("b", 1),
            ("b", 1): (np.sum, [1, 2, 3]),
        }
        assert fuse(graph) == {("a", 0): (np.sum, [1, 2, 3])}

    def test_ignore(self):
        graph = {
            "a": (sum, 1, 2),
            "b": (sum, 3),
            "c": (sum, "a", "b"),
        }
        assert fuse(graph) == graph


class TestToFromDict:
    def test(self):
        import types

        import xdas.io

        def read_data(path):
            return np.load(path)

        xdas.io.numpy = types.ModuleType("numpy")
        xdas.io.numpy.read_data = read_data
        xdas.io.numpy.read_data.__module__ = "xdas.io.numpy"

        with TemporaryDirectory() as tmpdir:
            values = np.random.rand(3, 10)
            chunks = np.split(values, 5, axis=1)
            for idx, chunk in enumerate(chunks):
                np.save(os.path.join(tmpdir, f"chunk_{idx}.npy"), chunk)
            paths = glob(os.path.join(tmpdir, "*.npy"))
            chunks = [dask.delayed(xdas.io.numpy.read_data)(path) for path in paths]
            chunks = [
                dask.array.from_delayed(chunk, shape=(3, 2), dtype=values.dtype)
                for chunk in chunks
            ]
            data = dask.array.concatenate(chunks, axis=1)
            assert np.array_equal(data.compute(), values)
            assert np.array_equal(data.compute(), from_dict(to_dict(data)).compute())
            sliced = data[:, 0]
            assert np.array_equal(sliced.compute(), values[:, 0])
