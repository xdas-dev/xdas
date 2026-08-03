"""The tile-backed virtual array and its integration in the DataArray and native format."""

import math
import os

import dask.array as da_
import h5py
import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

import xdas as xd
from xdas.io import Engine
from xdas.tiles import TileArray

NX = 5

DIMS = ("time", "distance")

ENGINE = {"name": "h5py", "dataset": "data"}


class H5pyEngine(Engine, name="h5py"):
    """Read any HDF5 dataset — the engine of the synthetic test files.

    The format engines each read their own layout; test files belong to
    no format, so they are described by this generic load-only engine
    (its opening half stays abstract). The selection always has one
    slice per source axis, in source order, whatever the virtual
    arrangement.
    """

    @staticmethod
    def load_tile(path, selection, *, dataset):
        with h5py.File(path, "r") as file:
            return file[dataset][selection]


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


def _tile_file(path, data, **kwargs):
    """Write *data* to an HDF5 file at *path*."""
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=data, **kwargs)


def _with_starts(manifest, *starts):
    """Rebuild *manifest* with per-axis tile origins inside their sources.

    ``starts_k`` is view state the :meth:`TileArray.from_tiles` encoder
    does not take: windowed manifests assign it through the dataset and
    the canonical constructor.
    """
    assign = {
        f"starts_{k}": (f"tile_{k}", np.asarray(entry, dtype=np.int64))
        for k, entry in enumerate(starts)
        if entry is not None
    }
    return TileArray(manifest.dataset.assign(assign), manifest.dtype, manifest.engine)


def _random_key(rng, shape, max_step=1):
    """A random non-empty positive-step slice per axis."""
    key = []
    for extent in shape:
        a = int(rng.integers(0, extent))
        b = int(rng.integers(a + 1, extent + 1))
        s = int(rng.integers(1, max_step + 1))
        key.append(slice(a, b, s))
    return tuple(key)


def _random_grid(tmp_path, rng, ndim):
    """A random rectilinear grid of junk-padded files, one per tile.

    Per-axis margins model source cropping (axis-separable, as the grid
    requires): every tile at index ``i`` along axis ``k`` starts at
    ``margins[k][i]`` inside its own file.
    """
    counts = tuple(int(rng.integers(1, 4)) for _ in range(ndim))
    sizes = [rng.integers(2, 6, count).astype(np.int64) for count in counts]
    margins = [rng.integers(0, 3, count).astype(np.int64) for count in counts]
    shape = tuple(int(entry.sum()) for entry in sizes)
    edges = [np.concatenate(([0], np.cumsum(entry))) for entry in sizes]
    reference = np.empty(shape)
    paths = np.empty(counts, dtype=object)
    for number, index in enumerate(np.ndindex(counts)):
        extents = tuple(int(sizes[k][i]) for k, i in enumerate(index))
        raw = tuple(
            int(margins[k][i]) + extent + int(rng.integers(0, 2))
            for (k, i), extent in zip(enumerate(index), extents)
        )
        data = np.full(raw, -1.0)
        block = 1000.0 * number + np.arange(math.prod(extents)).reshape(extents)
        inner = tuple(
            slice(int(margins[k][i]), int(margins[k][i]) + extent)
            for (k, i), extent in zip(enumerate(index), extents)
        )
        data[inner] = block
        placed = tuple(
            slice(int(edges[k][i]), int(edges[k][i + 1])) for k, i in enumerate(index)
        )
        reference[placed] = block
        path = str(tmp_path / f"grid{number}.h5")
        _tile_file(path, data)
        paths[index] = path
    manifest = TileArray.from_tiles(
        paths, sizes, "float64", {"name": "h5py", "dataset": "data"}
    )
    return _with_starts(manifest, *margins), reference


class TestManifest:
    def test_shape_and_geometry(self, stack):
        manifest, reference = stack
        assert manifest.shape == reference.shape
        assert manifest.ntiles == 3
        npt.assert_array_equal(manifest._edges[0], [0, 10, 17, 29])

    def test_reads_across_sources(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest), reference)
        npt.assert_array_equal(np.asarray(manifest[9:13]), reference[9:13])
        npt.assert_array_equal(np.asarray(manifest[3:5]), reference[3:5])

    def test_dataset_model(self, stack, tmp_path):
        manifest, _ = stack
        dataset = manifest.dataset
        assert tuple(dataset["sizes_0"].dims) == ("tile_0",)
        assert tuple(dataset["sizes_1"].dims) == ("tile_1",)
        # per-file paths vary along tile_0 only: the trailing axis folds
        assert tuple(dataset["paths"].dims) == ("tile_0",)
        # the common directory splits off: 0-d root, root-relative paths
        assert dataset["root"].ndim == 0
        assert os.fsdecode(dataset["root"].values[()]) == str(tmp_path)
        # strings are held as fixed-width bytes, not str objects
        assert dataset["paths"].dtype.kind == "S"
        assert dataset["paths"].values.tolist() == [b"src0.h5", b"src1.h5", b"src2.h5"]
        npt.assert_array_equal(dataset["starts_0"].values, [1, 1, 1])
        # all-default geometry columns are not stored
        assert "starts_1" not in dataset and "steps_0" not in dataset

    def test_param_folding(self, stack):
        manifest, _ = stack
        path = manifest._full_paths().item(0)
        uniform = TileArray.from_tiles(
            path, ([10, 10, 10], NX), "float64", ENGINE, record=0, nbytes=80
        )
        # one path everywhere: 0-d; uniform per-tile params: 0-d
        assert uniform.dataset["paths"].ndim == 0
        assert uniform.dataset["record"].ndim == 0
        assert uniform.shape == (30, NX)
        varying = TileArray.from_tiles(
            path, ([10, 10], NX), "float64", ENGINE, record=[[0], [80]]
        )
        assert tuple(varying.dataset["record"].dims) == ("tile_0",)

    def test_validation(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="at least one axis"):
            TileArray.from_tiles("a", (), "f8", ENGINE)
        with pytest.raises(ValueError, match="little-endian"):
            TileArray.from_tiles("a", (5, NX), ">f8", ENGINE)
        with pytest.raises(ValueError, match="strictly positive"):
            TileArray.from_tiles("a", (0, NX), "f8", ENGINE)
        with pytest.raises(ValueError, match="does not match the grid"):
            TileArray.from_tiles(
                np.array(["a", "b"], dtype=object), ([1, 2, 3], NX), "f8", ENGINE
            )
        with pytest.raises(ValueError, match="reserved"):
            TileArray.from_tiles("a", (5, NX), "f8", ENGINE, sizes_0=[5])
        with pytest.raises(ValueError, match="reserved"):
            TileArray.from_tiles("a", (5, NX), "f8", ENGINE, starts_0=[0])
        with pytest.raises(ValueError, match="reserved"):
            TileArray.from_tiles("a", (5, NX), "f8", ENGINE, root=["r"])
        bad_root = manifest.dataset.copy()
        bad_root["root"] = (("tile_0",), np.array(["a", "b", "c"], dtype=object))
        with pytest.raises(ValueError, match="0-d"):
            TileArray(bad_root, manifest.dtype, manifest.engine)
        dataset = manifest.dataset.copy()
        with pytest.raises(ValueError, match="`sizes_0`"):
            TileArray(
                dataset.drop_vars(["sizes_0", "sizes_1"]),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="`paths`"):
            TileArray(dataset.drop_vars("paths"), manifest.dtype, manifest.engine)
        bad_starts = dataset.assign(starts_0=("tile_0", np.array([-1, 0, 0])))
        with pytest.raises(ValueError, match="non-negative"):
            TileArray(bad_starts, manifest.dtype, manifest.engine)

    def test_extra_variables_are_params(self, stack):
        """Any non-geometry manifest variable is a per-tile engine parameter."""
        manifest, _ = stack
        arr = TileArray(
            manifest.dataset.assign(record=(("tile_0",), np.arange(3))),
            manifest.dtype,
            manifest.engine,
        )
        assert arr._params == ("record",)

    def test_string_params_decode_to_str(self, tmp_path):
        """Per-tile string parameters store as bytes but reach the engine as str."""
        paths, parts = [], []
        for k in range(2):
            path = str(tmp_path / f"named{k}.h5")
            data = 100.0 * k + np.arange(3.0 * NX).reshape(3, NX)
            with h5py.File(path, "w") as file:
                file.create_dataset(f"data{k}", data=data)
            paths.append(path)
            parts.append(data)
        manifest = TileArray.from_tiles(
            paths, ([3, 3], NX), "float64", "h5py", dataset=["data0", "data1"]
        )
        assert manifest.dataset["dataset"].dtype.kind == "S"
        npt.assert_array_equal(np.asarray(manifest), np.concatenate(parts))

    def test_engine_validation(self):
        with pytest.raises(KeyError, match="no engine registered"):
            TileArray.from_tiles("a", (5, NX), "f8", {"name": "bogus"})
        with pytest.raises(ValueError, match="`name` key"):
            TileArray.from_tiles("a", (5, NX), "f8", {"dataset": "data"})
        with pytest.raises(ValueError, match="`name` key"):
            TileArray.from_tiles("a", (5, NX), "f8", None)

    def test_engine_string_shorthand(self):
        arr = TileArray.from_tiles("a", (5, NX), "f8", "h5py")
        assert arr.engine == {"name": "h5py"}

    def test_engine_registration(self):
        class DummyEngine(Engine, name="dummy"):
            @staticmethod
            def load_tile(path, selection):
                return np.zeros((1, 1))

        try:
            assert Engine["dummy"] is DummyEngine
        finally:
            del Engine._registry["dummy"]

    def test_engine_without_tile_loader(self):
        # a registered engine that predates the tiles machinery resolves
        # but fails loudly when a manifest asks it to decode
        class NoTilesEngine(Engine, name="notiles"):
            pass

        try:
            arr = TileArray.from_tiles("a", (5, NX), "f8", {"name": "notiles"})
            with pytest.raises(NotImplementedError):
                np.asarray(arr)
        finally:
            del Engine._registry["notiles"]

    def test_repr(self, stack):
        manifest, _ = stack
        assert repr(manifest) == "TileArray[h5py] 1kB (float64) 3 tiles"
        assert manifest._repr_inline_(40) == "TileArray[h5py] (3 tiles)"
        assert manifest._repr_inline_(10) == "TileArray"

    def test_repr_of_a_single_tile(self, tmp_path):
        """One tile reads as one tile, and the volume scales with the array."""
        path = str(tmp_path / "one.h5")
        _tile_file(path, np.zeros((250, NX)))
        arr = TileArray.from_tiles(path, (250, NX), "float64", ENGINE)
        assert repr(arr) == "TileArray[h5py] 10kB (float64) 1 tile"
        assert arr._repr_inline_(40) == "TileArray[h5py] (1 tile)"

    def test_relative_paths_are_anchored(self, tmp_path, monkeypatch):
        """Relative paths absolutize at construction and survive a chdir."""
        data = np.arange(4.0 * NX).reshape(4, NX)
        _tile_file(tmp_path / "rel.h5", data)
        monkeypatch.chdir(tmp_path)
        manifest = TileArray.from_tiles("rel.h5", (4, NX), "f8", ENGINE)
        assert manifest.root == str(tmp_path)
        assert os.path.isabs(manifest._full_paths().item(0))
        monkeypatch.chdir(tmp_path.parent)
        npt.assert_array_equal(np.asarray(manifest), data)

    def test_attrs(self, stack):
        manifest, _ = stack
        assert manifest.attrs == {"units": "strain"}


class TestSourcePaths:
    """Paths are stored split: a common 0-d root and root-relative values."""

    def make(self, path):
        # the file's first row is skipped: sliced away, as views are made
        return TileArray.from_tiles([str(path)], ([5], NX), "<f4", ENGINE)[1:5]

    def stored(self, manifest):
        # a single tile folds its path to a 0-d variable
        return manifest.to_dataset()["paths"].values.ravel().tolist()

    def round_trip(self, manifest):
        return TileArray(manifest.to_dataset(), manifest.dtype, manifest.engine)

    def test_root_splits_off(self, tmp_path):
        manifest = self.make(tmp_path / "sources" / "f.h5")
        assert manifest.root == str(tmp_path / "sources")
        assert self.stored(manifest) == [b"f.h5"]

    def test_paths_round_trip(self, tmp_path):
        path = tmp_path / "sources" / "f.h5"
        restored = self.round_trip(self.make(path))
        assert restored.root == str(tmp_path / "sources")
        assert self.stored(restored) == [b"f.h5"]
        assert os.fsdecode(restored._full_paths().item(0)) == str(path)

    def test_stored_paths_read(self, tmp_path):
        # the tile's start row skips the first row of the file
        data = np.arange(5 * NX, dtype="<f4").reshape(5, NX)
        (tmp_path / "sources").mkdir()
        _tile_file(tmp_path / "sources" / "f.h5", data)
        restored = self.round_trip(self.make(tmp_path / "sources" / "f.h5"))
        npt.assert_array_equal(np.asarray(restored), data[1:])

    def test_rootless_manifest_reads(self, tmp_path):
        """Manifests without a `root` (the pre-split stored form) still work."""
        data = np.arange(5 * NX, dtype="<f4").reshape(5, NX)
        _tile_file(tmp_path / "f.h5", data)
        manifest = self.make(tmp_path / "f.h5")
        dataset = manifest.to_dataset().drop_vars("root")
        dataset["paths"] = xr.Variable(
            (), np.asarray(str(tmp_path / "f.h5"), dtype=object)
        )
        legacy = TileArray(dataset, manifest.dtype, manifest.engine)
        assert legacy.root == ""
        npt.assert_array_equal(np.asarray(legacy), data[1:])
        # same files, however split between root and relative paths
        assert legacy.equals(manifest) and manifest.equals(legacy)

    def test_no_common_directory_keeps_paths_whole(self):
        from xdas.tiles import _split_root

        mixed = np.array([b"rel/f.h5", b"/abs/g.h5"], dtype=object)
        root, kept = _split_root(mixed)
        assert root == b"" and kept is mixed
        empty = np.array([], dtype=object)
        root, kept = _split_root(empty)
        assert root == b"" and kept is empty

    def test_concat_with_unrelatable_roots_stores_whole_paths(self):
        """Roots `commonpath` cannot relate (absolute vs relative) fuse rootless."""

        def make(root, path):
            dataset = xr.Dataset(
                {
                    "sizes_0": ("tile_0", np.array([3])),
                    "paths": ((), np.asarray(os.fsencode(path))),
                    "root": ((), np.asarray(os.fsencode(root))),
                }
            )
            return TileArray(dataset, "float64", ENGINE)

        fused = TileArray.concat([make("/a/b", "f.h5"), make("rel", "g.h5")])
        assert fused.root == ""
        assert fused.dataset["paths"].values.tolist() == [
            os.fsencode(os.path.join("/a/b", "f.h5")),
            os.fsencode(os.path.join("rel", "g.h5")),
        ]

    def test_no_common_directory_falls_back_rootless(self, tmp_path, monkeypatch):
        """Paths sharing no directory (several drives) store whole, rootless."""
        import xdas.tiles

        data = np.arange(4.0 * NX).reshape(4, NX)
        _tile_file(tmp_path / "d.h5", data)
        monkeypatch.setattr(xdas.tiles, "_split_root", lambda paths: ("", paths))
        manifest = TileArray.from_tiles(str(tmp_path / "d.h5"), (4, NX), "f8", ENGINE)
        assert manifest.root == "" and "root" not in manifest.dataset
        npt.assert_array_equal(np.asarray(manifest), data)


class TestStarts:
    def test_start_windows_read(self, windowed):
        manifest, reference = windowed
        npt.assert_array_equal(manifest.dataset["starts_0"].values, [2, 0, 2])
        npt.assert_array_equal(np.asarray(manifest), reference)
        npt.assert_array_equal(np.asarray(manifest[6:15]), reference[6:15])

    def test_two_windows_of_one_blob(self, windowed):
        manifest, reference = windowed
        path = manifest._full_paths().item(0)
        split = _with_starts(
            TileArray.from_tiles(path, ([4, 4], NX), manifest.dtype, manifest.engine),
            [2, 6],
        )
        npt.assert_array_equal(np.asarray(split), reference[:8])

    def test_repeated_window_reads_twice(self, windowed):
        """The same source sub-box placed at two virtual positions is legal."""
        manifest, reference = windowed
        path = manifest._full_paths().item(0)
        repeated = _with_starts(
            TileArray.from_tiles(path, ([4, 4], NX), manifest.dtype, manifest.engine),
            [2, 2],
        )
        npt.assert_array_equal(
            np.asarray(repeated), np.concatenate([reference[:4], reference[:4]])
        )

    def test_getitem_composes_starts(self, windowed):
        manifest, reference = windowed
        sliced = manifest[6:15]
        npt.assert_array_equal(np.asarray(sliced), reference[6:15])
        assert sliced.dataset["starts_0"].values[0] == 2 + 6  # fixture 2, slice at 6


@pytest.fixture
def grid(tmp_path):
    """A 2 x 2 grid of files, one tile per grid cell."""
    heights = [6, 4]
    widths = [3, 3]
    paths = np.empty((2, 2), dtype=object)
    blocks = {}
    for i, nt in enumerate(heights):
        for j, nx in enumerate(widths):
            path = str(tmp_path / f"grid{i}_{j}.h5")
            data = 100.0 * i + 10.0 * j + np.arange(nt * nx).reshape(nt, nx)
            _tile_file(path, data)
            paths[i, j] = path
            blocks[i, j] = data
    manifest = TileArray.from_tiles(
        paths, (heights, widths), "float64", {"name": "h5py", "dataset": "data"}
    )
    reference = np.block([[blocks[0, 0], blocks[0, 1]], [blocks[1, 0], blocks[1, 1]]])
    return manifest, reference


class TestGrid:
    def test_2d_grid_reads(self, grid):
        manifest, reference = grid
        assert manifest.shape == (10, 6)
        assert manifest.ntiles == 4
        assert tuple(manifest.dataset["paths"].dims) == ("tile_0", "tile_1")
        npt.assert_array_equal(np.asarray(manifest), reference)
        npt.assert_array_equal(np.asarray(manifest[3:8, 2:5]), reference[3:8, 2:5])

    def test_concat_grid_columns(self, grid):
        manifest, reference = grid
        fused = TileArray.concat([manifest[:, 0:3], manifest[:, 3:6]], dim=1)
        assert fused.shape == reference.shape
        npt.assert_array_equal(np.asarray(fused), reference)

    def test_3d_grid(self, tmp_path):
        paths, parts = [], []
        for k in range(2):
            path = str(tmp_path / f"v{k}.h5")
            data = 100.0 * k + np.arange(6 * 4 * 3).reshape(6, 4, 3)
            _tile_file(path, data)
            paths.append(path)
            parts.append(data)
        manifest = TileArray.from_tiles(
            paths, (6, 4, 3), "float64", {"name": "h5py", "dataset": "data"}
        )
        assert manifest.shape == (12, 4, 3)
        npt.assert_array_equal(np.asarray(manifest), np.concatenate(parts))

    def test_1d_stack(self, tmp_path):
        paths, parts = [], []
        for k in range(2):
            path = str(tmp_path / f"one{k}.h5")
            data = 10.0 * k + np.arange(7.0)
            _tile_file(path, data)
            paths.append(path)
            parts.append(data)
        manifest = TileArray.from_tiles(
            paths, ([7, 7],), "float64", {"name": "h5py", "dataset": "data"}
        )
        assert manifest.shape == (14,)
        npt.assert_array_equal(np.asarray(manifest), np.concatenate(parts))


class TestStreaming:
    """The tiling is the only blocking: reductions walk it row by row."""

    @pytest.fixture
    def rows(self, tmp_path):
        """One source of 29 rows, to be tiled in various ways."""
        path = str(tmp_path / "rows.h5")
        data = np.arange(29.0 * NX).reshape(29, NX)
        _tile_file(path, data)
        return path, data

    @pytest.mark.parametrize("sizes", [[29], [8, 21], [10, 7, 12], [1] * 29])
    def test_reductions_reproduce_values_over_any_tiling(self, rows, sizes):
        path, data = rows
        starts = np.cumsum([0, *sizes[:-1]])
        manifest = _with_starts(
            TileArray.from_tiles(
                [path] * len(sizes),
                (sizes, NX),
                "float64",
                {"name": "h5py", "dataset": "data"},
            ),
            starts,
        )
        npt.assert_array_equal(np.asarray(manifest), data)
        npt.assert_allclose(np.mean(manifest, axis=0), data.mean(0))
        npt.assert_allclose(np.sum(manifest), data.sum())
        npt.assert_allclose(np.max(manifest, axis=1), data.max(1))

    def test_blocks_are_tile_rows(self, stack, monkeypatch):
        manifest, reference = stack
        boxes = []
        original = TileArray.__getitem__
        monkeypatch.setattr(
            TileArray,
            "__getitem__",
            lambda self, key: boxes.append(key) or original(self, key),
        )
        npt.assert_allclose(np.mean(manifest, axis=0), reference.mean(0))
        assert boxes == [
            (slice(0, 10), slice(0, NX)),
            (slice(10, 17), slice(0, NX)),
            (slice(17, 29), slice(0, NX)),
        ]

    def test_reduction_never_holds_the_array_whole(self, stack):
        manifest, reference = stack
        npt.assert_allclose(np.mean(manifest, axis=0), reference.mean(0))
        assert manifest._cache is None


class TestConcat:
    def test_concat(self, stack):
        manifest, reference = stack
        fused = TileArray.concat([manifest, manifest])
        assert fused.ntiles == 6
        npt.assert_array_equal(fused._edges[0], [0, 10, 17, 29, 39, 46, 58])
        npt.assert_array_equal(
            np.asarray(fused), np.concatenate([reference, reference])
        )

    def test_concat_requires_compatibility(self, stack):
        manifest, _ = stack
        other = TileArray.from_tiles(
            list(manifest.dataset["paths"].values),
            ([10, 7, 12], NX),
            manifest.dtype,
            {"name": "h5py", "dataset": "other"},
        )
        with pytest.raises(ValueError, match="compatible"):
            TileArray.concat([manifest, other])
        with pytest.raises(ValueError, match="compatible"):
            TileArray.concat([manifest, manifest[:, 0:3]], dim=0)
        with pytest.raises(ValueError, match="compatible"):
            TileArray.concat([manifest, manifest[0:10]], dim=1)
        with pytest.raises(ValueError, match="no axis"):
            TileArray.concat([manifest, manifest], dim=2)

    def test_concat_channel_sections(self, stack):
        """Differently-trimmed subviews of one archive fuse along a
        non-time axis, entirely virtually."""
        manifest, reference = stack
        fused = TileArray.concat([manifest[:, 0:3], manifest[:, 3:5]], dim=1)
        assert fused.shape == reference.shape
        npt.assert_array_equal(np.asarray(fused), reference)

    def test_folded_params_stay_folded(self, tmp_path):
        path = str(tmp_path / "shared.h5")
        _tile_file(path, np.arange(20.0 * NX).reshape(20, NX))
        engine = {"name": "h5py", "dataset": "data"}
        base = TileArray.from_tiles(path, ([5, 5], NX), "float64", engine)
        a = _with_starts(base, [0, 5])
        b = _with_starts(base, [10, 15])
        fused = TileArray.concat([a, b])
        assert fused.dataset["paths"].ndim == 0  # equal 0-d paths stay folded
        npt.assert_array_equal(np.asarray(fused), np.arange(20.0 * NX).reshape(20, NX))

    def test_differing_params_promote(self, tmp_path):
        parts = []
        manifests = []
        engine = {"name": "h5py", "dataset": "data"}
        for k in range(2):
            path = str(tmp_path / f"p{k}.h5")
            data = 100.0 * k + np.arange(5.0 * NX).reshape(5, NX)
            _tile_file(path, data)
            parts.append(data)
            manifests.append(TileArray.from_tiles(path, (5, NX), "float64", engine))
        fused = TileArray.concat(manifests)
        assert tuple(fused.dataset["paths"].dims) == ("tile_0",)
        npt.assert_array_equal(np.asarray(fused), np.concatenate(parts))

    def test_concat_rebases_differing_roots(self, tmp_path):
        """Arrays rooted apart fuse under the deepest shared directory."""
        engine = {"name": "h5py", "dataset": "data"}
        parts, manifests = [], []
        for k, sub in enumerate(["a", "b"]):
            (tmp_path / sub).mkdir()
            path = str(tmp_path / sub / f"p{k}.h5")
            data = 100.0 * k + np.arange(5.0 * NX).reshape(5, NX)
            _tile_file(path, data)
            parts.append(data)
            manifests.append(TileArray.from_tiles(path, (5, NX), "float64", engine))
        fused = TileArray.concat(manifests)
        assert fused.root == str(tmp_path)
        assert fused.dataset["paths"].values.tolist() == [
            os.fsencode(os.path.join("a", "p0.h5")),
            os.fsencode(os.path.join("b", "p1.h5")),
        ]
        npt.assert_array_equal(np.asarray(fused), np.concatenate(parts))

    def test_concat_without_common_root_stores_absolute(self, tmp_path):
        """A rootless (legacy) input drags the fusion to absolute paths."""
        engine = {"name": "h5py", "dataset": "data"}
        parts, manifests = [], []
        for k in range(2):
            path = str(tmp_path / f"r{k}.h5")
            data = 100.0 * k + np.arange(5.0 * NX).reshape(5, NX)
            _tile_file(path, data)
            parts.append(data)
            manifests.append(TileArray.from_tiles(path, (5, NX), "float64", engine))
        dataset = manifests[1].to_dataset().drop_vars("root")
        dataset["paths"] = xr.Variable(
            (), np.asarray(str(tmp_path / "r1.h5"), dtype=object)
        )
        legacy = TileArray(dataset, "float64", engine)
        fused = TileArray.concat([manifests[0], legacy])
        assert fused.root == ""
        assert all(os.path.isabs(path) for path in fused.dataset["paths"].values)
        npt.assert_array_equal(np.asarray(fused), np.concatenate(parts))

    def test_concat_chains_per_tile_params(self, stack):
        manifest, _ = stack
        arr = TileArray(
            manifest.dataset.assign(record=(("tile_0",), np.arange(3))),
            manifest.dtype,
            manifest.engine,
        )
        fused = TileArray.concat([arr, arr])
        npt.assert_array_equal(fused.dataset["record"].values, [0, 1, 2, 0, 1, 2])

    def test_mixed_starts_concat(self, tmp_path, windowed):
        """A windowed and an untrimmed manifest fuse: starts promote to 0."""
        trimmed, windowed_reference = windowed
        paths, parts = [], []
        for k in range(2):
            path = str(tmp_path / f"plain{k}.h5")
            data = 1000.0 * (k + 1) + np.arange(5.0 * NX).reshape(5, NX)
            _tile_file(path, data)
            paths.append(path)
            parts.append(data)
        plain = TileArray.from_tiles(paths, (5, NX), trimmed.dtype, trimmed.engine)
        fused = TileArray.concat([trimmed, plain], dim=0)
        npt.assert_array_equal(fused.dataset["starts_0"].values, [2, 0, 2, 0, 0])
        npt.assert_array_equal(
            np.asarray(fused), np.concatenate([windowed_reference, *parts])
        )


class TestEngineContract:
    """The engine receives source-local selections and returns exact boxes."""

    def build(self, tmp_path, rng):
        """One junk-padded file exposing a trimmed, decimated window."""
        nt, nx = 15, 8
        data = rng.standard_normal((nt, nx))
        path = str(tmp_path / "contract.h5")
        _tile_file(path, data)
        manifest = TileArray.from_tiles(path, (nt, nx), "float64", ENGINE)[2:12]
        return manifest[::2, 1:6:2], data[2:12:2, 1:6:2]

    @pytest.mark.parametrize("seed", range(3))
    def test_selection_matches_crop(self, tmp_path, seed):
        """Random boxes read identically to full-read-then-crop."""
        rng = np.random.default_rng(seed)
        manifest, expected = self.build(tmp_path, rng)
        npt.assert_array_equal(np.asarray(manifest), expected)
        for _ in range(5):
            key = _random_key(rng, manifest.shape)
            npt.assert_array_equal(np.asarray(manifest[key]), expected[key])

    def test_per_tile_params_reach_the_engine(self, stack):
        """Per-tile variables land as keyword arguments, one value per tile."""
        manifest, _ = stack
        seen = []

        class ProbeEngine(Engine, name="probe"):
            @staticmethod
            def load_tile(path, selection, *, record, flavor):
                seen.append((path, record, flavor))
                widths = tuple(
                    len(range(entry.start, entry.stop, entry.step or 1))
                    for entry in selection
                )
                return np.zeros(widths)

        try:
            arr = TileArray(
                manifest.dataset.assign(record=(("tile_0",), np.arange(3))),
                manifest.dtype,
                # a per-tile variable shadows a same-named spec constant
                {"name": "probe", "record": -1, "flavor": "spec"},
            )
            np.asarray(arr)
            assert [entry[1] for entry in seen] == [0, 1, 2]
            assert {entry[2] for entry in seen} == {"spec"}
        finally:
            del Engine._registry["probe"]

    def test_wrong_shape_fails_loudly(self, stack):
        manifest, _ = stack

        class BadShapeEngine(Engine, name="badshape"):
            @staticmethod
            def load_tile(path, selection):
                return np.zeros((1, 1))

        try:
            bad = TileArray(
                manifest.dataset.copy(), manifest.dtype, {"name": "badshape"}
            )
            with pytest.raises(ValueError, match="shape"):
                np.asarray(bad[0:5])
        finally:
            del Engine._registry["badshape"]

    def test_wrong_dtype_fails_loudly(self, stack):
        """The recorded dtype is verified against each decoded part, never cast."""
        manifest, _ = stack

        class BadDtypeEngine(Engine, name="baddtype"):
            @staticmethod
            def load_tile(path, selection):
                widths = tuple(
                    len(range(entry.start, entry.stop, entry.step or 1))
                    for entry in selection
                )
                return np.zeros(widths, dtype="float32")

        try:
            bad = TileArray(
                manifest.dataset.copy(), manifest.dtype, {"name": "baddtype"}
            )
            with pytest.raises(ValueError, match="float32"):
                np.asarray(bad[0:5])
        finally:
            del Engine._registry["baddtype"]


class TestEdgeCases:
    def test_size_and_array_conversions(self, stack):
        manifest, reference = stack
        assert manifest.size == reference.size
        cast = np.asarray(manifest, dtype="float32")
        assert cast.dtype == np.float32
        npt.assert_allclose(cast, reference, rtol=1e-6)
        copied = np.asarray(manifest, copy=True)
        assert copied is not manifest._cache

    def test_equals_distinctions(self, stack, windowed):
        manifest, _ = stack
        assert not manifest.equals("not a tile array")
        assert not manifest.equals(windowed[0])  # differing geometry
        other = TileArray(
            manifest.dataset.assign(record=(("tile_0",), np.arange(3))),
            manifest.dtype,
            manifest.engine,
        )
        assert not manifest.equals(other)  # differing param sets
        shifted = TileArray(
            manifest.dataset.drop_vars("starts_0"), manifest.dtype, manifest.engine
        )
        assert not manifest.equals(shifted)  # same shape, differing geometry
        renamed = manifest.dataset.copy()
        renamed["paths"] = renamed["paths"].copy()
        renamed["paths"].values[0] = "elsewhere.h5"
        assert not manifest.equals(TileArray(renamed, manifest.dtype, manifest.engine))
        moved = manifest.dataset.copy()
        moved["root"] = ((), np.asarray("/elsewhere", dtype=object))
        assert not manifest.equals(TileArray(moved, manifest.dtype, manifest.engine))
        withrec = TileArray(
            manifest.dataset.assign(record=(("tile_0",), np.arange(3))),
            manifest.dtype,
            manifest.engine,
        )
        assert withrec.equals(
            TileArray(withrec.dataset.copy(), withrec.dtype, withrec.engine)
        )
        rerecorded = TileArray(
            manifest.dataset.assign(record=(("tile_0",), np.arange(1, 4))),
            manifest.dtype,
            manifest.engine,
        )
        assert not withrec.equals(rerecorded)  # same param, differing values

    def test_index_errors(self, stack):
        manifest, reference = stack
        with pytest.raises(IndexError, match="out of bounds"):
            manifest[100]
        with pytest.raises(IndexError, match="out of bounds"):
            manifest[[0, 100]]
        with pytest.raises(IndexError, match="invalid index array dtype"):
            manifest[np.array([0.5, 1.5])]
        npt.assert_array_equal(manifest[-1], reference[-1])
        npt.assert_array_equal(manifest[[-1, -2]], reference[[-1, -2]])

    def test_new_axis_stays_virtual(self, stack):
        manifest, reference = stack
        expanded = manifest[np.newaxis]
        assert isinstance(expanded, TileArray)
        npt.assert_array_equal(np.asarray(expanded), reference[np.newaxis])
        mixed = manifest[5:20, None, ::2]
        assert isinstance(mixed, TileArray)
        npt.assert_array_equal(np.asarray(mixed), reference[5:20, None, ::2])

    def test_bad_shapes(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="more axes"):
            TileArray.from_tiles(
                np.full((2, 2), "a", dtype=object), ([4, 4],), "f8", ENGINE
            )
        transposed = manifest.dataset.copy()
        transposed["record"] = (("tile_1", "tile_0"), np.zeros((1, 3)))
        with pytest.raises(ValueError, match="ordered subset"):
            TileArray(transposed, manifest.dtype, manifest.engine)
        bad_geometry = manifest.dataset.copy()
        bad_geometry["sizes_1"] = (("tile_0",), np.full(3, NX))
        with pytest.raises(ValueError, match="must have dimensions"):
            TileArray(bad_geometry, manifest.dtype, manifest.engine)

    def test_ufunc_out_not_supported(self, stack):
        manifest, _ = stack
        with pytest.raises(TypeError):
            np.add(manifest, 1, out=manifest)

    def test_unhandled_reduction_kwargs_materialize(self, stack):
        manifest, reference = stack
        out = np.zeros(manifest.shape[1])
        np.sum(manifest, axis=0, out=out)
        npt.assert_allclose(out, reference.sum(0))
        npt.assert_allclose(np.sum(manifest, axis=0, initial=1), reference.sum(0) + 1)


class TestExpandDims:
    """Lazy leading-axis expansion, the 0.2 extension for concat-to-a-new-dim."""

    def test_leading_expansion_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, 0)
        assert isinstance(expanded, TileArray)
        assert expanded.shape == (1, *manifest.shape)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(expanded), reference[np.newaxis])

    def test_negative_leading_axis(self, stack):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, -manifest.ndim - 1)
        assert isinstance(expanded, TileArray)
        npt.assert_array_equal(np.asarray(expanded), reference[np.newaxis])

    def test_repeated_expansion(self, stack):
        manifest, reference = stack
        expanded = np.expand_dims(np.expand_dims(manifest, 0), 0)
        assert isinstance(expanded, TileArray)
        assert expanded.shape == (1, 1, *manifest.shape)
        npt.assert_array_equal(np.asarray(expanded), reference[None, None])

    def test_expanded_geometry_carries_over(self, stack):
        """Expansion appends a synthetic axis: the stored geometry never moves."""
        manifest, _ = stack
        expanded = manifest.expand_dims()
        npt.assert_array_equal(
            expanded.dataset["sizes_0"].values, manifest.dataset["sizes_0"].values
        )
        npt.assert_array_equal(
            expanded.dataset["starts_0"].values, manifest.dataset["starts_0"].values
        )
        npt.assert_array_equal(expanded.dataset["sizes_2"].values, [1])
        npt.assert_array_equal(expanded.dataset["axes"].values, [2, 0, 1])
        assert int(expanded.dataset["source_ndim"].values[()]) == 2

    def test_expanded_slicing_folds(self, stack, engine_calls):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, 0)
        sliced = expanded[:, 9:13]
        assert isinstance(sliced, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(sliced), reference[np.newaxis, 9:13])

    def test_expanded_concat_stacks(self, stack):
        manifest, reference = stack
        parts = [np.expand_dims(manifest, 0) for _ in range(2)]
        fused = np.concatenate(parts, axis=0)
        assert isinstance(fused, TileArray)
        assert fused.shape == (2, *manifest.shape)
        npt.assert_array_equal(np.asarray(fused), np.stack([reference, reference]))

    def test_any_axis_stays_virtual(self, stack):
        manifest, reference = stack
        for axis in [1, 2, -1, -2]:
            expanded = np.expand_dims(manifest, axis)
            assert isinstance(expanded, TileArray)
            npt.assert_array_equal(
                np.asarray(expanded), np.expand_dims(reference, axis)
            )

    def test_tuple_axis_stays_virtual(self, stack):
        manifest, reference = stack
        for axis in [(0, 1), (0, 3), (3, 1), (-1, 0)]:
            expanded = np.expand_dims(manifest, axis)
            assert isinstance(expanded, TileArray)
            npt.assert_array_equal(
                np.asarray(expanded), np.expand_dims(reference, axis)
            )

    def test_out_of_range_method_axis_raises(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="position"):
            manifest.expand_dims(4)

    def test_negative_axis_method(self, stack):
        manifest, reference = stack
        expanded = manifest.expand_dims(-manifest.ndim - 1)
        assert isinstance(expanded, TileArray)
        npt.assert_array_equal(np.asarray(expanded), reference[np.newaxis])

    def test_dispatch_guards(self, stack):
        from xdas.tiles import _expand_dims_virtual

        manifest, _ = stack
        expand = np.expand_dims
        assert (
            _expand_dims_virtual(manifest, expand, (np.zeros(3), 0), {})
            is NotImplemented
        )
        assert (
            _expand_dims_virtual(manifest, expand, (manifest, 0), {"extra": 1})
            is NotImplemented
        )


class TestGetitem:
    def test_keeps_only_overlapping_tiles(self, stack):
        manifest, reference = stack
        sliced = manifest[9:13]
        assert sliced.ntiles == 2  # rows 9..13 live in the first two files
        assert sliced.shape == (4, NX)
        npt.assert_array_equal(np.asarray(sliced), reference[9:13])

    def test_both_axes(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest[9:13, 1:4]), reference[9:13, 1:4])

    def test_composes(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest[:, 1:4][:, 1:3]), reference[:, 2:4])
        npt.assert_array_equal(np.asarray(manifest[5:25][2:12]), reference[7:17])

    def test_stepped_slices_fold(self, stack):
        manifest, reference = stack
        for key in [
            (slice(0, 29, 2),),
            (slice(3, 25, 3), slice(1, 5, 2)),
            (slice(None, None, 7),),
            (slice(None, None, 50),),  # step larger than the whole axis
        ]:
            sliced = manifest[key]
            expected = reference[key]
            assert sliced.shape == expected.shape
            npt.assert_array_equal(np.asarray(sliced), expected)

    def test_coarse_step_drops_tiles(self, stack):
        manifest, reference = stack
        # positions 0, 15 skip the 7-row middle file (rows 10..17)
        sliced = manifest[0:16:15]
        assert sliced.ntiles == 2
        npt.assert_array_equal(np.asarray(sliced), reference[0:16:15])

    def test_steps_compose(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(
            np.asarray(manifest[2:27:2][1:10:3]), reference[2:27:2][1:10:3]
        )

    def test_non_foldable_keys_materialize(self, stack):
        """Empty, integer and reversed keys resolve in memory."""
        manifest, reference = stack
        assert isinstance(manifest[5:5], np.ndarray)
        assert manifest[5:5].shape == (0, NX)
        assert manifest[:, 2:2].shape == (29, 0)
        npt.assert_array_equal(manifest[3], reference[3])
        npt.assert_array_equal(manifest[::-1], reference[::-1])
        with pytest.raises(IndexError, match="too many indices"):
            manifest[:, :, :]

    def test_empty_selection_allocates_nothing(self):
        """An empty result must not size itself on the full array."""
        huge = TileArray.from_tiles(["/nowhere.bin"], ([10**7], [10**6]), "<f8", ENGINE)
        assert huge[0:0].shape == (0, 10**6)

    @pytest.mark.parametrize("ndim", [1, 2, 3])
    @pytest.mark.parametrize("seed", range(3))
    def test_matches_numpy(self, tmp_path, ndim, seed):
        rng = np.random.default_rng(seed + 10 * ndim)
        manifest, reference = _random_grid(tmp_path, rng, ndim)
        npt.assert_array_equal(np.asarray(manifest), reference)
        for _ in range(3):
            key = _random_key(rng, manifest.shape, max_step=4)
            sliced = manifest[key]
            npt.assert_array_equal(np.asarray(sliced), reference[key])
            key2 = _random_key(rng, sliced.shape, max_step=3)
            npt.assert_array_equal(np.asarray(sliced[key2]), reference[key][key2])


@pytest.fixture
def line(tmp_path):
    """A 1-D two-tile manifest and its values."""
    paths, parts = [], []
    for k in range(2):
        path = str(tmp_path / f"line{k}.h5")
        data = 10.0 * k + np.arange(7.0)
        _tile_file(path, data)
        paths.append(path)
        parts.append(data)
    manifest = TileArray.from_tiles(
        paths, ([7, 7],), "float64", {"name": "h5py", "dataset": "data"}
    )
    return manifest, np.concatenate(parts)


class TestManipulationRoutines:
    """Numpy manipulation routines that rewrite the tile geometry lazily."""

    def test_split_stays_lazy(self, stack, engine_calls):
        manifest, reference = stack
        pieces = np.split(manifest, [10, 17])
        assert all(isinstance(piece, TileArray) for piece in pieces)
        assert engine_calls == []
        for piece, expected in zip(pieces, np.split(reference, [10, 17])):
            npt.assert_array_equal(np.asarray(piece), expected)

    def test_split_sections(self, stack):
        manifest, reference = stack
        with pytest.raises(ValueError, match="equal division"):
            np.split(manifest, 2)
        pieces = np.array_split(manifest, 4, axis=1)
        assert all(isinstance(piece, TileArray) for piece in pieces)
        for piece, expected in zip(pieces, np.array_split(reference, 4, axis=1)):
            npt.assert_array_equal(np.asarray(piece), expected)

    def test_split_variants(self, stack, line):
        manifest, reference = stack
        for got, expected in zip(np.vsplit(manifest, [12]), np.vsplit(reference, [12])):
            npt.assert_array_equal(np.asarray(got), expected)
        for got, expected in zip(np.hsplit(manifest, [2]), np.hsplit(reference, [2])):
            npt.assert_array_equal(np.asarray(got), expected)
        with pytest.raises(ValueError, match="3 or more"):
            np.dsplit(manifest, 1)
        # int sections require an equal division for the whole family
        # (numpy parity), array_split being the one lenient spelling
        with pytest.raises(ValueError, match="equal division"):
            np.vsplit(manifest, 2)
        line_manifest, line_reference = line
        with pytest.raises(ValueError, match="2 or more"):
            np.vsplit(line_manifest, 2)
        pieces = np.hsplit(line_manifest, 2)  # 1-D hsplit works on axis 0
        assert all(isinstance(piece, TileArray) for piece in pieces)
        for piece, expected in zip(pieces, np.hsplit(line_reference, 2)):
            npt.assert_array_equal(np.asarray(piece), expected)

    def test_split_empty_pieces_are_plain(self, stack):
        manifest, reference = stack
        pieces = np.split(manifest, [17, 12])  # unsorted: middle piece empty
        assert isinstance(pieces[1], np.ndarray)
        for piece, expected in zip(pieces, np.split(reference, [17, 12])):
            npt.assert_array_equal(np.asarray(piece), expected)

    def test_roll_stays_lazy(self, stack, engine_calls):
        manifest, reference = stack
        rolled = np.roll(manifest, 12, axis=0)
        assert isinstance(rolled, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(rolled), np.roll(reference, 12, axis=0))

    def test_roll_variants(self, stack):
        manifest, reference = stack
        for shift, axis in [(-4, 0), (100, 0), ((3, 2), (0, 1)), (2, (0, 1)), (0, 1)]:
            rolled = np.roll(manifest, shift, axis=axis)
            assert isinstance(rolled, TileArray)
            npt.assert_array_equal(
                np.asarray(rolled), np.roll(reference, shift, axis=axis)
            )

    def test_roll_flat_materializes(self, line):
        manifest, reference = line
        rolled = np.roll(manifest, 3)  # axis=None rolls the flattened array
        assert isinstance(rolled, np.ndarray)
        npt.assert_array_equal(rolled, np.roll(reference, 3))

    def test_tile_stays_lazy(self, stack, engine_calls):
        manifest, reference = stack
        tiled = np.tile(manifest, (2, 3))
        assert isinstance(tiled, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(tiled), np.tile(reference, (2, 3)))

    def test_tile_promotes_rank(self, stack, line):
        manifest, reference = stack
        tiled = np.tile(manifest, (2, 1, 1))
        assert isinstance(tiled, TileArray)
        npt.assert_array_equal(np.asarray(tiled), np.tile(reference, (2, 1, 1)))
        line_manifest, line_reference = line
        tiled = np.tile(line_manifest, 3)
        assert isinstance(tiled, TileArray)
        npt.assert_array_equal(np.asarray(tiled), np.tile(line_reference, 3))

    def test_tile_zero_rep_reads_nothing(self, stack, engine_calls):
        manifest, reference = stack
        tiled = np.tile(manifest, (0, 2))
        assert isinstance(tiled, np.ndarray)
        assert engine_calls == []
        npt.assert_array_equal(tiled, np.tile(reference, (0, 2)))

    def test_delete_lazy_cases(self, stack):
        manifest, reference = stack
        for obj in [4, -1, slice(5, 20), slice(3, 3), slice(20, 5, -1)]:
            deleted = np.delete(manifest, obj, axis=0)
            assert isinstance(deleted, TileArray)
            npt.assert_array_equal(
                np.asarray(deleted), np.delete(reference, obj, axis=0)
            )

    def test_delete_everything(self, stack):
        manifest, _ = stack
        deleted = np.delete(manifest, slice(None), axis=1)
        assert isinstance(deleted, np.ndarray)
        assert deleted.shape == (29, 0)

    def test_delete_fallbacks_and_errors(self, stack):
        manifest, reference = stack
        strided = np.delete(manifest, slice(None, None, 2), axis=0)
        assert isinstance(strided, np.ndarray)
        npt.assert_array_equal(
            strided, np.delete(reference, slice(None, None, 2), axis=0)
        )
        listed = np.delete(manifest, [1, 4], axis=0)
        npt.assert_array_equal(listed, np.delete(reference, [1, 4], axis=0))
        assert isinstance(np.delete(manifest, 3), np.ndarray)  # axis=None flattens
        with pytest.raises(IndexError, match="out of bounds"):
            np.delete(manifest, 40, axis=0)

    def test_append_insert_stay_lazy(self, stack):
        manifest, reference = stack
        appended = np.append(manifest, manifest[0:4], axis=0)
        assert isinstance(appended, TileArray)
        npt.assert_array_equal(
            np.asarray(appended), np.append(reference, reference[0:4], axis=0)
        )
        for pos in [0, 17, 29, -29]:
            inserted = np.insert(manifest, pos, manifest[3:5], axis=0)
            assert isinstance(inserted, TileArray)
            npt.assert_array_equal(
                np.asarray(inserted), np.insert(reference, pos, reference[3:5], axis=0)
            )

    def test_append_insert_fallbacks(self, stack, line):
        manifest, reference = stack
        eager = np.append(manifest, np.ones((1, NX)), axis=0)
        assert isinstance(eager, np.ndarray)
        npt.assert_array_equal(eager, np.append(reference, np.ones((1, NX)), axis=0))
        flat = np.append(manifest, manifest)  # axis=None flattens 2-D inputs
        assert isinstance(flat, np.ndarray)
        npt.assert_array_equal(flat, np.append(reference, reference))
        line_manifest, line_reference = line
        joined = np.append(line_manifest, line_manifest)  # 1-D stays lazy
        assert isinstance(joined, TileArray)
        npt.assert_array_equal(
            np.asarray(joined), np.append(line_reference, line_reference)
        )
        scalar = np.insert(manifest, 3, 0.0, axis=0)  # eager values broadcast
        assert isinstance(scalar, np.ndarray)
        npt.assert_array_equal(scalar, np.insert(reference, 3, 0.0, axis=0))
        with pytest.raises(IndexError, match="out of bounds"):
            np.insert(manifest, 100, manifest[0:1], axis=0)

    def test_stack(self, stack, engine_calls):
        manifest, reference = stack
        stacked = np.stack([manifest, manifest])
        assert isinstance(stacked, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(stacked), np.stack([reference, reference]))
        negative = np.stack([manifest, manifest], axis=-3)
        assert isinstance(negative, TileArray)
        npt.assert_array_equal(
            np.asarray(negative), np.stack([reference, reference], axis=-3)
        )
        for axis in [1, 2, -1]:
            middle = np.stack([manifest, manifest], axis=axis)
            assert isinstance(middle, TileArray)
            npt.assert_array_equal(
                np.asarray(middle), np.stack([reference, reference], axis=axis)
            )

    def test_stack_wrappers(self, stack):
        manifest, reference = stack
        piled = np.vstack([manifest, manifest])
        assert isinstance(piled, TileArray)
        npt.assert_array_equal(np.asarray(piled), np.vstack([reference, reference]))
        wide = np.hstack([manifest, manifest])
        assert isinstance(wide, TileArray)
        npt.assert_array_equal(np.asarray(wide), np.hstack([reference, reference]))
        cols = np.column_stack([manifest, manifest])
        assert isinstance(cols, TileArray)
        npt.assert_array_equal(
            np.asarray(cols), np.column_stack([reference, reference])
        )
        deep = np.dstack([manifest, manifest])
        assert isinstance(deep, TileArray)
        npt.assert_array_equal(np.asarray(deep), np.dstack([reference, reference]))

    def test_stack_wrappers_1d(self, line):
        manifest, reference = line
        rows = np.vstack([manifest, manifest])
        assert isinstance(rows, TileArray)
        npt.assert_array_equal(np.asarray(rows), np.vstack([reference, reference]))
        flat = np.hstack([manifest, manifest])
        assert isinstance(flat, TileArray)
        npt.assert_array_equal(np.asarray(flat), np.hstack([reference, reference]))
        cols = np.column_stack([manifest, manifest])
        assert isinstance(cols, TileArray)
        npt.assert_array_equal(
            np.asarray(cols), np.column_stack([reference, reference])
        )

    def test_atleast(self, stack, line):
        manifest, reference = stack
        assert np.atleast_1d(manifest) is manifest
        assert np.atleast_2d(manifest) is manifest
        line_manifest, line_reference = line
        promoted = np.atleast_2d(line_manifest)
        assert isinstance(promoted, TileArray)
        npt.assert_array_equal(np.asarray(promoted), np.atleast_2d(line_reference))
        deep = np.atleast_3d(manifest)
        assert isinstance(deep, TileArray)
        npt.assert_array_equal(np.asarray(deep), np.atleast_3d(reference))
        line_deep = np.atleast_3d(line_manifest)
        assert isinstance(line_deep, TileArray)
        npt.assert_array_equal(np.asarray(line_deep), np.atleast_3d(line_reference))

    def test_mixed_operands_materialize(self, stack):
        manifest, reference = stack
        result = np.vstack([manifest, np.ones((1, NX))])
        assert isinstance(result, np.ndarray)
        npt.assert_array_equal(result, np.vstack([reference, np.ones((1, NX))]))

    def test_3d_variants_stay_lazy(self, stack):
        manifest, reference = stack
        deep = manifest.expand_dims(0)
        reference = reference[np.newaxis]
        pieces = np.dsplit(deep, [2])
        assert all(isinstance(piece, TileArray) for piece in pieces)
        for piece, expected in zip(pieces, np.dsplit(reference, [2])):
            npt.assert_array_equal(np.asarray(piece), expected)
        stacked = np.dstack([deep, deep])
        assert isinstance(stacked, TileArray)
        npt.assert_array_equal(np.asarray(stacked), np.dstack([reference, reference]))
        columns = np.array_split(deep, 2, axis=-1)
        assert all(isinstance(piece, TileArray) for piece in columns)
        for piece, expected in zip(columns, np.array_split(reference, 2, axis=-1)):
            npt.assert_array_equal(np.asarray(piece), expected)

    def test_numpy_only_variants_materialize(self, stack):
        """Calls the grid cannot express take the numpy path with equal values."""
        manifest, reference = stack
        rolled = np.roll(manifest, 1.5, axis=0)  # non-integer shift
        assert isinstance(rolled, np.ndarray)
        npt.assert_array_equal(rolled, np.roll(reference, 1.5, axis=0))
        multi = np.insert(manifest, [1, 2], manifest[0:2], axis=0)
        assert isinstance(multi, np.ndarray)
        npt.assert_array_equal(
            multi, np.insert(reference, [1, 2], reference[0:2], axis=0)
        )
        flat = np.insert(manifest, 3, 7.0)  # axis=None flattens
        assert isinstance(flat, np.ndarray)
        npt.assert_array_equal(flat, np.insert(reference, 3, 7.0))
        casted = np.stack([manifest, manifest], dtype="float32")
        assert isinstance(casted, np.ndarray) and casted.dtype == np.float32
        mixed = np.stack([manifest, np.asarray(reference)])
        assert isinstance(mixed, np.ndarray)
        piled = np.vstack([manifest, manifest], dtype="float32")
        assert isinstance(piled, np.ndarray) and piled.dtype == np.float32
        first, _ = np.atleast_2d(manifest, manifest)
        assert isinstance(first, np.ndarray)
        npt.assert_array_equal(first, reference)

    def test_error_parity(self, stack, line):
        """Inexpressible or invalid calls raise the numpy errors."""
        manifest, _ = stack
        line_manifest, _ = line
        with pytest.raises(ValueError, match="larger than 0"):
            np.split(manifest, 0)
        with pytest.raises(TypeError):
            np.split(manifest, [2.5])
        with pytest.raises(TypeError):
            np.split(manifest, 2, axis="bad")
        with pytest.raises(ValueError):
            np.roll(manifest, (1, 2, 3), axis=(0, 1))
        with pytest.raises(np.exceptions.AxisError):
            np.roll(manifest, 1, axis=5)
        with pytest.raises(TypeError):
            np.tile(manifest, 1.5)
        with pytest.raises(ValueError):
            np.tile(manifest, -1)
        with pytest.raises(np.exceptions.AxisError):
            np.delete(manifest, 1, axis=5)
        with pytest.raises(np.exceptions.AxisError):
            np.append(manifest, manifest, axis=5)
        with pytest.raises(np.exceptions.AxisError):
            np.insert(manifest, 1, manifest[0:1], axis=5)
        with pytest.raises(TypeError):
            np.stack([manifest, manifest], axis=None)
        with pytest.raises(ValueError, match="same number of dimensions"):
            np.hstack([line_manifest, manifest])

    def test_lazy_rewrite_guards(self, stack):
        """Handlers step aside for calls that are not theirs to rewrite."""
        from xdas import tiles

        manifest, _ = stack
        other = np.zeros(3)
        assert (
            tiles._split_virtual(manifest, np.split, (other, 2), {}) is NotImplemented
        )
        assert tiles._roll_virtual(manifest, np.roll, (manifest,), {}) is NotImplemented
        assert tiles._tile_virtual(manifest, np.tile, (other, 2), {}) is NotImplemented
        assert (
            tiles._append_virtual(manifest, np.append, (manifest,), {})
            is NotImplemented
        )
        assert (
            tiles._insert_virtual(
                manifest, np.insert, (manifest, 1, manifest.expand_dims(0)), {"axis": 0}
            )
            is NotImplemented
        )
        assert tiles._stack_virtual(manifest, np.stack, (5,), {}) is NotImplemented
        assert tiles._stack_virtual(manifest, np.stack, (), {}) is NotImplemented
        assert (
            tiles._stack_like_virtual(manifest, np.vstack, (5,), {}) is NotImplemented
        )


class TestAxisMap:
    """The axis map: transposes, inserted, hidden and pinned axes stay lazy."""

    def test_transpose_stays_lazy(self, stack, engine_calls):
        manifest, reference = stack
        flipped = np.transpose(manifest)
        assert isinstance(flipped, TileArray)
        assert flipped.shape == reference.T.shape
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(flipped), reference.T)

    def test_transpose_variants(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(
            np.asarray(np.swapaxes(manifest, 0, 1)), np.swapaxes(reference, 0, 1)
        )
        npt.assert_array_equal(
            np.asarray(np.moveaxis(manifest, 0, -1)), np.moveaxis(reference, 0, -1)
        )
        npt.assert_array_equal(
            np.asarray(np.matrix_transpose(manifest)), np.matrix_transpose(reference)
        )
        npt.assert_array_equal(
            np.asarray(np.permute_dims(manifest, (1, 0))), reference.T
        )
        npt.assert_array_equal(np.asarray(manifest.transpose()), reference.T)

    def test_transpose_composes_with_slicing(self, stack, engine_calls):
        manifest, reference = stack
        view = np.transpose(manifest)[1:4, 9:20:2]
        assert isinstance(view, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(view), reference.T[1:4, 9:20:2])

    def test_transpose_composes_with_concat(self, stack):
        manifest, reference = stack
        flipped = np.transpose(manifest)
        fused = np.concatenate([flipped, flipped], axis=1)
        assert isinstance(fused, TileArray)
        npt.assert_array_equal(
            np.asarray(fused), np.concatenate([reference.T, reference.T], axis=1)
        )

    def test_engine_receives_source_order(self, stack):
        """The selection reaches the engine in source order, full rank."""
        manifest, _ = stack
        seen = []

        class SelectionProbe(Engine, name="selection-probe"):
            @staticmethod
            def load_tile(path, selection, **params):
                seen.append(selection)
                widths = tuple(
                    len(range(entry.start, entry.stop, entry.step or 1))
                    for entry in selection
                )
                return np.zeros(widths)

        try:
            probe = TileArray(
                manifest.dataset.drop_vars("record", errors="ignore"),
                manifest.dtype,
                "selection-probe",
            )
            np.asarray(np.expand_dims(np.transpose(probe), 1)[:, :, 9:13])
            assert all(len(selection) == 2 for selection in seen)
            # the time trim reaches source axis 0 whatever the virtual order
            rows = [
                len(range(sel[0].start, sel[0].stop, sel[0].step or 1)) for sel in seen
            ]
            columns = [
                len(range(sel[1].start, sel[1].stop, sel[1].step or 1)) for sel in seen
            ]
            assert sum(rows) == 4 and set(columns) == {NX}
        finally:
            del Engine._registry["selection-probe"]

    def test_integer_indexing_stays_lazy(self, stack, engine_calls):
        manifest, reference = stack
        column = manifest[:, 2]
        assert isinstance(column, TileArray)
        assert column.shape == (29,)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(column), reference[:, 2])
        row = manifest[11]
        assert isinstance(row, TileArray)
        npt.assert_array_equal(np.asarray(row), reference[11])
        npt.assert_array_equal(np.asarray(manifest[-1]), reference[-1])
        # a pinned axis composes with later slicing and reads one tile
        npt.assert_array_equal(np.asarray(row[1:4]), reference[11, 1:4])

    def test_scalar_selection_materializes(self, stack):
        manifest, reference = stack
        value = manifest[11, 2]
        assert not isinstance(value, TileArray)
        npt.assert_array_equal(value, reference[11, 2])

    def test_squeeze(self, stack):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, 1)
        squeezed = np.squeeze(expanded, axis=1)
        assert isinstance(squeezed, TileArray)
        npt.assert_array_equal(np.asarray(squeezed), reference)
        # squeezing a real unit axis hides it: the source is still read
        thin = manifest[:, 1:2]
        squeezed = np.squeeze(thin)
        assert isinstance(squeezed, TileArray)
        assert squeezed.shape == (29,)
        npt.assert_array_equal(np.asarray(squeezed), reference[:, 1])
        assert np.squeeze(manifest) is manifest  # nothing to squeeze
        with pytest.raises(ValueError, match="not equal to one"):
            np.squeeze(manifest, axis=1)
        scalar = manifest[0:1, 0:1].squeeze()
        assert isinstance(scalar, np.ndarray) and scalar.shape == ()
        npt.assert_array_equal(scalar, reference[0, 0])

    def test_expand_middle_and_concat_along_it(self, stack):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, 1)
        fused = np.concatenate([expanded, expanded], axis=1)
        assert isinstance(fused, TileArray)
        npt.assert_array_equal(
            np.asarray(fused), np.stack([reference, reference], axis=1)
        )

    def test_mapped_round_trip(self, stack, tmp_path):
        """A transposed, pinned view survives the native format."""
        manifest, reference = stack
        view = np.transpose(manifest)[1:4]
        da = wrap(view)
        path = str(tmp_path / "mapped.nc")
        da.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        assert reopened.data.equals(view)
        npt.assert_array_equal(reopened.values, reference.T[1:4])

    def test_expanded_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, 1)
        da = xd.DataArray(expanded, dims=("time", "extra", "distance"))
        path = str(tmp_path / "expanded.nc")
        da.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        assert reopened.data.equals(expanded)
        npt.assert_array_equal(reopened.values, reference[:, np.newaxis])

    def test_equals_distinguishes_maps(self, stack):
        manifest, _ = stack
        assert not manifest.equals(np.transpose(manifest))
        assert not manifest.equals(np.expand_dims(manifest, 0))
        assert np.transpose(manifest).equals(np.transpose(manifest))

    def test_map_validation(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="distinct geometry axes"):
            TileArray(
                manifest.dataset.assign(axes=("axis", np.array([0, 0]))),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="1-D over its own"):
            TileArray(
                manifest.dataset.assign(axes=("tile_0", np.array([0, 1, 0]))),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="between 1 and"):
            TileArray(
                manifest.dataset.assign(source_ndim=((), np.int64(3))),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="sizes must be 1"):
            TileArray(
                manifest.dataset.assign(axes=("axis", np.array([1]))),
                manifest.dtype,
                manifest.engine,
            )
        two = manifest[9:11]  # two one-row tiles along the time axis
        with pytest.raises(ValueError, match="single tile"):
            TileArray(
                two.dataset.assign(axes=("axis", np.array([1]))),
                two.dtype,
                two.engine,
            )
        with pytest.raises(ValueError, match="sizes must be 1"):
            TileArray(
                manifest.dataset.assign(source_ndim=((), np.int64(1))),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="at least one visible"):
            TileArray(
                manifest.dataset.assign(axes=("axis", np.array([], dtype=np.int64))),
                manifest.dtype,
                manifest.engine,
            )

    def test_transpose_method_validation(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="permute"):
            manifest.transpose((0, 0))

    def test_streaming_reduction_on_transposed(self, stack):
        manifest, reference = stack
        flipped = np.transpose(manifest)
        npt.assert_allclose(np.mean(flipped, axis=1), reference.T.mean(1))
        npt.assert_allclose(np.sum(flipped), reference.sum())

    def test_chunks_follow_the_map(self, stack):
        manifest, _ = stack
        assert np.transpose(manifest).chunks == (
            manifest.chunks[1],
            manifest.chunks[0],
        )
        assert np.expand_dims(manifest, 1).chunks == (
            manifest.chunks[0],
            (1,),
            manifest.chunks[1],
        )

    def test_map_error_parity(self, stack, line):
        """Inexpressible or invalid map calls raise the numpy errors."""
        manifest, _ = stack
        line_manifest, _ = line
        with pytest.raises(ValueError):
            np.transpose(manifest, 0)  # too few axes
        with pytest.raises(ValueError):
            np.matrix_transpose(line_manifest)  # ndim < 2
        with pytest.raises(np.exceptions.AxisError):
            np.swapaxes(manifest, 0, 5)
        with pytest.raises(np.exceptions.AxisError):
            np.moveaxis(manifest, 0, 5)
        with pytest.raises(ValueError):
            np.moveaxis(manifest, (0, 1), (0,))  # length mismatch
        with pytest.raises(ValueError):
            np.moveaxis(manifest, (0, 0), (0, 1))  # repeated source
        with pytest.raises(TypeError):
            np.expand_dims(manifest, "bad")
        with pytest.raises(ValueError):
            np.expand_dims(manifest, (0, 0))  # repeated position
        with pytest.raises(np.exceptions.AxisError):
            np.expand_dims(manifest, 5)
        with pytest.raises(np.exceptions.AxisError):
            np.stack([manifest, manifest], axis=5)
        with pytest.raises(TypeError):
            np.squeeze(manifest, axis="bad")
        with pytest.raises(ValueError, match="no axis"):
            manifest.squeeze(axis=5)
        with pytest.raises(ValueError, match="0-d"):
            TileArray(
                manifest.dataset.assign(source_ndim=("axis", np.array([2]))),
                manifest.dtype,
                manifest.engine,
            )

    def test_map_dispatch_guards(self, stack):
        """Handlers step aside for calls that are not theirs to rewrite."""
        from xdas import tiles

        manifest, _ = stack
        other = np.zeros((3, 4))
        assert (
            tiles._transpose_virtual(manifest, np.transpose, (other,), {})
            is NotImplemented
        )
        assert (
            tiles._squeeze_virtual(manifest, np.squeeze, (other,), {}) is NotImplemented
        )

    def test_masked_selection_on_bounded_path(self, stack):
        """Advanced keys with integers still take the bounded read."""
        manifest, reference = stack
        picked = manifest[[3, 7, 20], -1]
        assert isinstance(picked, np.ndarray)
        npt.assert_array_equal(picked, reference[[3, 7, 20], -1])
        with pytest.raises(IndexError, match="out of bounds"):
            manifest[[3, 7], 99]
        mask = np.zeros(manifest.shape, dtype=bool)
        mask[3, 2] = True
        npt.assert_array_equal(manifest[mask], reference[mask])


class TestSignedSteps:
    """Negative steps: lazy reversal in the geometry, ascending engine reads."""

    def test_reversed_slices_fold(self, stack, engine_calls):
        manifest, reference = stack
        flipped = manifest[::-1]
        assert isinstance(flipped, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(np.asarray(flipped), reference[::-1])

    @pytest.mark.parametrize(
        "key",
        [
            np.s_[::-1, ::-1],
            np.s_[::-2],
            np.s_[20:5:-3],
            np.s_[::-1, 3:1:-1],
            np.s_[25:, ::-2],
        ],
    )
    def test_reversed_slice_values(self, stack, key):
        manifest, reference = stack
        view = manifest[key]
        assert isinstance(view, TileArray)
        npt.assert_array_equal(np.asarray(view), reference[key])

    def test_reversal_composes(self, stack, windowed):
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest[::2][::-1]), reference[::2][::-1])
        npt.assert_array_equal(np.asarray(manifest[::-1][::2]), reference[::-1][::2])
        npt.assert_array_equal(np.asarray(manifest[::-2][3:8]), reference[::-2][3:8])
        windowed_manifest, windowed_reference = windowed
        npt.assert_array_equal(
            np.asarray(windowed_manifest[::-1]), windowed_reference[::-1]
        )
        flipped = np.transpose(manifest)[::-1, 9:20:2]
        assert isinstance(flipped, TileArray)
        npt.assert_array_equal(np.asarray(flipped), reference.T[::-1, 9:20:2])

    def test_double_reversal_leaves_no_trace(self, stack):
        manifest, reference = stack
        back = manifest[::-1][::-1]
        assert isinstance(back, TileArray)
        assert "steps_0" not in back.dataset
        assert back.equals(manifest)
        npt.assert_array_equal(np.asarray(back), reference)

    def test_flip_family(self, stack, engine_calls):
        manifest, reference = stack
        for flip in [
            lambda a: np.flip(a),
            lambda a: np.flip(a, 1),
            lambda a: np.flip(a, (0, 1)),
            np.flipud,
            np.fliplr,
        ]:
            flipped = flip(manifest)
            assert isinstance(flipped, TileArray)
            npt.assert_array_equal(np.asarray(flipped), flip(reference))
        assert engine_calls != []  # reads happened, but only at np.asarray

    def test_rot90(self, stack):
        manifest, reference = stack
        for k in [0, 1, 2, 3, 4, -1]:
            rotated = np.rot90(manifest, k)
            assert isinstance(rotated, TileArray)
            npt.assert_array_equal(np.asarray(rotated), np.rot90(reference, k))
        rotated = np.rot90(manifest, 1, axes=(1, 0))
        assert isinstance(rotated, TileArray)
        npt.assert_array_equal(np.asarray(rotated), np.rot90(reference, 1, axes=(1, 0)))

    def test_engine_reads_stay_ascending(self, stack):
        """Whatever the reversal, the engine sees ascending selections."""
        manifest, _ = stack
        seen = []

        class AscendingProbe(Engine, name="ascending-probe"):
            @staticmethod
            def load_tile(path, selection, **params):
                seen.append(selection)
                widths = tuple(
                    len(range(entry.start, entry.stop, entry.step or 1))
                    for entry in selection
                )
                return np.zeros(widths)

        try:
            probe = TileArray(manifest.dataset, manifest.dtype, "ascending-probe")
            np.asarray(np.flip(probe[::-2, ::-1]))
            assert seen and all(
                entry.start <= entry.stop and (entry.step or 1) >= 1
                for selection in seen
                for entry in selection
            )
        finally:
            del Engine._registry["ascending-probe"]

    def test_flipped_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        view = np.flip(manifest, 0)[5:20]
        da = wrap(view)
        path = str(tmp_path / "flipped.nc")
        da.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        assert reopened.data.equals(view)
        npt.assert_array_equal(reopened.values, reference[::-1][5:20])

    def test_streaming_reduction_on_flipped(self, stack):
        manifest, reference = stack
        flipped = np.flip(manifest)
        npt.assert_allclose(np.mean(flipped, axis=0), reference[::-1, ::-1].mean(0))
        npt.assert_allclose(np.max(flipped), reference.max())

    def test_step_validation(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="nonzero"):
            TileArray(
                manifest.dataset.assign(
                    steps_0=("tile_0", np.zeros(3, dtype=np.int64))
                ),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="walks out of the source"):
            TileArray(
                manifest.dataset.assign(
                    steps_0=("tile_0", np.full(3, -1, dtype=np.int64))
                ),
                manifest.dtype,
                manifest.engine,
            )

    def test_flip_error_parity(self, stack, line):
        manifest, _ = stack
        line_manifest, _ = line
        with pytest.raises(ValueError, match="repeated"):
            np.flip(manifest, (0, 0))
        with pytest.raises(np.exceptions.AxisError):
            np.flip(manifest, 5)
        with pytest.raises(ValueError, match="different"):
            np.rot90(manifest, axes=(0, 0))
        with pytest.raises(ValueError, match="out of range"):
            np.rot90(manifest, axes=(0, 5))
        with pytest.raises(ValueError):
            np.rot90(line_manifest)
        with pytest.raises(ValueError, match=">= 2-d"):
            np.fliplr(line_manifest)

    def test_reversed_slice_on_bounded_path(self, stack):
        """A reversed slice next to an index array takes the bounded read."""
        manifest, reference = stack
        picked = manifest[::-1, [1, 3]]
        assert isinstance(picked, np.ndarray)
        npt.assert_array_equal(picked, reference[::-1, [1, 3]])

    def test_flip_dispatch_guards(self, stack):
        from xdas import tiles

        manifest, _ = stack
        other = np.zeros((3, 4))
        assert tiles._flip_virtual(manifest, np.flip, (other,), {}) is NotImplemented
        assert tiles._rot90_virtual(manifest, np.rot90, (other,), {}) is NotImplemented
        assert (
            tiles._rot90_virtual(manifest, np.rot90, (manifest, 1.5), {})
            is NotImplemented
        )


class TestNumpyProtocols:
    """Direct duck-array protocol behavior, without a wrapping DataArray."""

    def test_ufunc_materializes(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(np.add(manifest, 1), reference + 1)
        npt.assert_array_equal(-manifest, -reference)

    def test_result_type_uses_dtype(self, stack):
        manifest, _ = stack
        assert np.result_type(manifest, np.float32) == np.float64

    def test_ellipsis_key(self, stack):
        manifest, reference = stack
        sliced = manifest[..., 1:3]
        assert isinstance(sliced, TileArray)
        npt.assert_array_equal(np.asarray(sliced), reference[..., 1:3])

    def test_boolean_masks(self, stack):
        manifest, reference = stack
        mask = reference[:, 0] > 10
        npt.assert_array_equal(manifest[mask], reference[mask])
        from xdas.tiles import _bounding_key

        with pytest.raises(NotImplementedError, match="boolean mask"):
            _bounding_key(
                (np.zeros(reference.shape, dtype=bool), slice(None)), reference.shape
            )

    def test_empty_index_array(self, stack):
        manifest, _ = stack
        selected = manifest[np.array([], dtype=np.int64)]
        assert selected.shape == (0, NX)

    def test_chunks_report_the_tiling(self, stack):
        manifest, _ = stack
        assert manifest.chunks == ((10, 7, 12), (NX,))

    def test_transpose_astype_methods(self, stack):
        manifest, reference = stack
        npt.assert_array_equal(manifest.transpose((1, 0)), reference.T)
        assert manifest.astype("float32").dtype == np.float32

    def test_materialize_descends_sequences(self, stack):
        manifest, reference = stack
        out = np.concatenate([manifest, np.ones((1, NX))])
        assert isinstance(out, np.ndarray)
        npt.assert_array_equal(out, np.concatenate([reference, np.ones((1, NX))]))

    def test_concatenate_fallbacks(self, stack):
        manifest, reference = stack
        flat = np.concatenate([manifest, manifest], axis=None)
        npt.assert_array_equal(flat, np.concatenate([reference, reference], axis=None))
        casted = np.concatenate([manifest, manifest], dtype="float32")
        assert casted.dtype == np.float32
        out = np.concatenate([manifest, manifest], 0, None)
        npt.assert_array_equal(out, np.concatenate([reference, reference]))
        from xdas.tiles import _concatenate_virtual

        concat = np.concatenate
        assert _concatenate_virtual(manifest, concat, (), {}) is NotImplemented
        assert _concatenate_virtual(manifest, concat, (5,), {}) is NotImplemented

    def test_incompatible_concat_materializes(self, stack):
        manifest, reference = stack
        a = manifest[:, 0:3]
        b = manifest[:, ::2]  # same shape but differing column geometry
        out = np.concatenate([a, b], axis=0)
        assert isinstance(out, np.ndarray)
        npt.assert_array_equal(
            out, np.concatenate([reference[:, 0:3], reference[:, ::2]], axis=0)
        )

    def test_streaming_dtype_and_keepdims(self, stack):
        manifest, reference = stack
        casted = np.sum(manifest, axis=0, dtype="float32")
        assert casted.dtype == np.float32
        # the stream accumulates per tile row and casts at the end
        npt.assert_allclose(casted, reference.sum(0).astype("float32"))
        kept = np.sum(manifest, axis=0, keepdims=True)
        assert kept.shape == (1, NX)
        npt.assert_allclose(np.sum(manifest, axis=(0, 1)), reference.sum())

    def test_streaming_guards(self, stack):
        manifest, _ = stack
        # a reduction whose first argument is not this array is not streamed
        assert manifest._reduce_streaming(np.sum, (np.zeros(3),), {}) is (
            NotImplemented
        )
        # unbindable arguments fall back to materialization
        assert manifest._reduce_streaming(np.sum, (manifest,), {"bogus": 1}) is (
            NotImplemented
        )


class TestReadScheduling:
    def test_one_read_per_tile(self, stack, engine_calls):
        """A full read calls the engine once per tile."""
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest), reference)
        assert len(engine_calls) == manifest.ntiles

    def test_sliced_reads_touch_only_needed_sources(self, stack, engine_calls):
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest[0:5]), reference[0:5])
        assert len(engine_calls) == 1  # rows 0..5 live in the first file only
        engine_calls.clear()
        npt.assert_array_equal(np.asarray(manifest[9:13]), reference[9:13])
        assert len(engine_calls) == 2

    def test_cache_reads_once(self, stack, engine_calls):
        manifest, reference = stack
        npt.assert_array_equal(np.asarray(manifest), reference)
        first = len(engine_calls)
        assert first > 0
        npt.assert_array_equal(np.asarray(manifest), reference)
        assert len(engine_calls) == first  # served from the cache

    def test_deepcopy_and_pickle(self, stack):
        import copy
        import pickle

        manifest, reference = stack
        copied = copy.deepcopy(manifest)
        assert isinstance(copied, TileArray)
        assert copied.dataset is manifest.dataset
        restored = pickle.loads(pickle.dumps(manifest))
        assert isinstance(restored, TileArray)
        npt.assert_array_equal(np.asarray(restored), reference)


def wrap(manifest):
    """Wrap *manifest* in a DataArray with regular time/distance coordinates."""
    nt, nx = manifest.shape
    # ns resolution: the netCDF round trip casts datetimes to M8[ns]
    time = xd.Coordinate["interpolated"].from_block(
        np.datetime64("2020-01-01T00:00:00", "ns"),
        nt,
        np.timedelta64(10_000_000, "ns"),
        dim="time",
    )
    distance = xd.Coordinate["interpolated"].from_block(0.0, nx, 4.0, dim="distance")
    return xd.DataArray(manifest, {"time": time, "distance": distance})


class TestDataArray:
    def test_data(self, stack):
        manifest, _ = stack
        da = wrap(manifest)
        assert da.data is manifest
        assert "TileArray" in repr(da)

    def test_isel_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        view = da.isel(time=slice(9, 13), distance=slice(1, 4))
        assert isinstance(view.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(view.values, reference[9:13, 1:4])

    def test_sel_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        t0 = da["time"][2].values
        t1 = da["time"][20].values
        view = da.sel(time=slice(t0, t1))
        assert isinstance(view.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(view.values, reference[2:21])

    def test_load_materializes(self, stack):
        manifest, reference = stack
        loaded = wrap(manifest).load()
        assert isinstance(loaded.data, np.ndarray)
        npt.assert_array_equal(loaded.values, reference)

    def test_concat_along_existing_dim_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        head = da.isel(time=slice(0, 10))
        tail = da.isel(time=slice(10, None))
        out = xd.concat([head, tail], "time")
        assert isinstance(out.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(out.values, reference)

    def test_concat_along_new_dim_stays_virtual(self, stack, engine_calls):
        manifest, reference = stack
        objs = [wrap(manifest), wrap(manifest)]
        out = xd.concat(objs, "station")
        assert out.dims == ("station", "time", "distance")
        assert isinstance(out.data, TileArray)
        assert engine_calls == []
        npt.assert_array_equal(out.values, np.stack([reference, reference]))

    def test_mean_streams(self, stack, engine_calls):
        manifest, reference = stack
        da = wrap(manifest)
        npt.assert_allclose(da.mean("time").values, reference.mean(0))
        assert len(engine_calls) > 0
        assert manifest._cache is None


class TestPersistence:
    def test_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        da = wrap(manifest)
        path = str(tmp_path / "view.nc")
        da.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        assert reopened.data.equals(manifest)
        assert reopened.data.root == manifest.root
        assert reopened.coords["time"].equals(da.coords["time"])
        npt.assert_array_equal(reopened.values, reference)

    def test_sliced_view_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        view = wrap(manifest).isel(time=slice(9, 13))
        path = str(tmp_path / "sliced.nc")
        view.to_netcdf(path)
        reopened = xd.open_dataarray(path)
        assert isinstance(reopened.data, TileArray)
        npt.assert_array_equal(reopened.values, reference[9:13])

    def test_grouped_round_trip(self, stack, tmp_path):
        manifest, reference = stack
        da = wrap(manifest)
        path = str(tmp_path / "grouped.nc")
        da.to_netcdf(path, group="acquisition")
        reopened = xd.open_dataarray(path, engine="xdas", group="acquisition")
        assert isinstance(reopened.data, TileArray)
        npt.assert_array_equal(reopened.values, reference)

    def test_eager_save_writes_values(self, stack, tmp_path):
        manifest, reference = stack
        da = wrap(manifest)
        path = str(tmp_path / "eager.nc")
        da.to_netcdf(path, virtual=False)
        reopened = xd.open_dataarray(path)
        assert not isinstance(reopened.data, TileArray)
        npt.assert_array_equal(reopened.values, reference)

    def test_dask_write_deprecated(self, tmp_path):
        import dask

        data = da_.from_delayed(dask.delayed(np.zeros)((4, NX)), (4, NX), np.float64)
        da = xd.DataArray(data, dims=DIMS)
        path = str(tmp_path / "dask.nc")
        with pytest.warns(FutureWarning, match="dask-backed"):
            da.to_netcdf(path, virtual=True)
        reopened = xd.open_dataarray(path)
        npt.assert_array_equal(reopened.values, np.zeros((4, NX)))
