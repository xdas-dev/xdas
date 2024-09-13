import os
from glob import glob
from tempfile import TemporaryDirectory

import dask
import numpy as np

from xdas.dask import dumps, loads
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


class TestIO:
    def generate(self, tmpdir):
        expected = np.random.rand(3, 10)
        chunks = np.split(expected, 5, axis=1)
        for idx, chunk in enumerate(chunks):
            np.save(os.path.join(tmpdir, f"chunk_{idx}.npy"), chunk)
        paths = glob(os.path.join(tmpdir, "*.npy"))
        chunks = [dask.delayed(np.load)(path) for path in paths]
        chunks = [
            dask.array.from_delayed(chunk, shape=(3, 2), dtype=expected.dtype)
            for chunk in chunks
        ]
        data = dask.array.concatenate(chunks, axis=1)
        assert np.array_equal(data.compute(), expected)
        return expected, data

    def test_dict(self):
        with TemporaryDirectory() as tmpdir:
            expected, data = self.generate(tmpdir)
            result = from_dict(to_dict(data))
            assert np.array_equal(result.compute(), expected)
            sliced = result[:, 0]
            assert np.array_equal(sliced.compute(), expected[:, 0])

    def test_serial(self):
        with TemporaryDirectory() as tmpdir:
            expected, data = self.generate(tmpdir)
            result = loads(dumps(data))
            assert np.array_equal(result.compute(), expected)
            sliced = result[:, 0]
            assert np.array_equal(sliced.compute(), expected[:, 0])
