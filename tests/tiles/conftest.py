"""Shared fixtures for the tile-backed virtual array tests."""

import h5py
import numpy as np
import pytest

from xdas.io import Engine
from xdas.tiles import TileArray

NX = 5

ENGINE = {"name": "h5py", "dataset": "data"}


class H5pyEngine(Engine, name="h5py"):
    """Read any HDF5 dataset — the engine of the synthetic test files.

    The format engines each read their own layout; test files belong to
    no format, so they are described by this generic load-only engine
    (its opening half stays abstract). Extra leading selection axes
    (virtually expanded arrays) pad the output rank, as the production
    engines do.
    """

    @staticmethod
    def load_tile(path, selection, *, dataset):
        with h5py.File(path, "r") as file:
            source = file[dataset]
            extra = len(selection) - source.ndim
            data = source[selection[extra:]]
        return data.reshape((1,) * extra + data.shape)


@pytest.fixture
def stack(tmp_path):
    """Three gzip-compressed HDF5 files with junk edge rows to trim.

    Emulates overlap trimming: each file carries one junk row at its start
    and end that the tile's start row (plus the row ``size``) cuts out.
    Returns the manifest and the expected stacked values.
    """
    paths = []
    sizes = []
    parts = []
    row = 0
    for k, raw_nt in enumerate([12, 9, 14]):
        path = str(tmp_path / f"src{k}.h5")
        useful = raw_nt - 2
        data = np.full((raw_nt, NX), -999.0)
        data[1:-1] = (row + np.arange(useful))[:, None] + np.arange(NX) / 10
        with h5py.File(path, "w") as file:
            file.create_dataset("data", data=data, chunks=(4, NX), compression="gzip")
        paths.append(path)
        sizes.append(useful)
        parts.append(data[1:-1])
        row += useful
    manifest = TileArray.from_tiles(
        paths, (sizes, NX), "float64", ENGINE, attrs={"units": "strain"}
    )
    # per-tile source origins are view state: assigned through the manifest
    manifest = TileArray(
        manifest.dataset.assign(starts_0=("tile_0", np.array([1, 1, 1]))),
        manifest.dtype,
        manifest.engine,
    )
    return manifest, np.concatenate(parts)


@pytest.fixture
def windowed(tmp_path):
    """Three files whose rows contribute blob-local windows via ``starts_0``.

    Each file holds junk rows around the useful window; the manifest
    exposes blob rows ``[start, start + size)``. The middle file has a
    zero start (window at the top of the blob).
    """
    paths, sizes, starts, parts = [], [], [], []
    row = 0
    for k, raw_nt in enumerate([12, 9, 14]):
        path = str(tmp_path / f"win{k}.h5")
        useful = raw_nt - 4
        first = 0 if k == 1 else 2
        data = np.full((raw_nt, NX), -999.0)
        good = (row + np.arange(useful))[:, None] + np.arange(NX) / 10
        data[first : first + useful] = good
        with h5py.File(path, "w") as file:
            file.create_dataset("data", data=data)
        paths.append(path)
        sizes.append(useful)
        starts.append(first)
        parts.append(good)
        row += useful
    manifest = TileArray.from_tiles(
        paths, (sizes, NX), "float64", {"name": "h5py", "dataset": "data"}
    )
    manifest = TileArray(
        manifest.dataset.assign(starts_0=("tile_0", np.array(starts))),
        manifest.dtype,
        manifest.engine,
    )
    return manifest, np.concatenate(parts)


@pytest.fixture
def engine_calls(monkeypatch):
    """Record the path of every h5py engine read, delegating to the real one."""
    calls = []
    original = Engine["h5py"].load_tile

    def counting(path, selection, **params):
        calls.append(path)
        return original(path, selection, **params)

    monkeypatch.setattr(Engine["h5py"], "load_tile", counting)
    return calls
