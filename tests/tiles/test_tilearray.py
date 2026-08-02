import math
import os

import h5py
import numpy as np
import numpy.testing as npt
import pytest

from xdas.io import Engine
from xdas.tiles import TileArray

NX = 5

ENGINE = {"name": "h5py", "dataset": "data"}


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

    def test_dataset_model(self, stack):
        manifest, _ = stack
        dataset = manifest.dataset
        assert tuple(dataset["sizes_0"].dims) == ("tile_0",)
        assert tuple(dataset["sizes_1"].dims) == ("tile_1",)
        # per-file paths vary along tile_0 only: the trailing axis folds
        assert tuple(dataset["paths"].dims) == ("tile_0",)
        npt.assert_array_equal(dataset["starts_0"].values, [1, 1, 1])
        # all-default geometry columns are not stored
        assert "starts_1" not in dataset and "steps_0" not in dataset

    def test_param_folding(self, stack):
        manifest, _ = stack
        path = str(manifest.dataset["paths"].values[0])
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
        dataset = manifest.dataset.copy()
        with pytest.raises(ValueError, match="`sizes_0`"):
            TileArray(
                dataset.drop_vars(["sizes_0", "sizes_1"]),
                manifest.dtype,
                manifest.engine,
            )
        with pytest.raises(ValueError, match="`paths`"):
            TileArray(dataset.drop_vars("paths"), manifest.dtype, manifest.engine)

    def test_extra_variables_are_params(self, stack):
        """Any non-geometry manifest variable is a per-tile engine parameter."""
        manifest, _ = stack
        arr = TileArray(
            manifest.dataset.assign(record=(("tile_0",), np.arange(3))),
            manifest.dtype,
            manifest.engine,
        )
        assert arr._params == ("record",)

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
        assert "3 tiles" in repr(manifest)
        assert "'h5py'" in repr(manifest)
        assert manifest._repr_inline_(40) == "TileArray (3 tiles)"
        assert manifest._repr_inline_(10) == "TileArray"

    def test_relative_paths_are_anchored(self, tmp_path, monkeypatch):
        """Relative paths absolutize at construction and survive a chdir."""
        data = np.arange(4.0 * NX).reshape(4, NX)
        _tile_file(tmp_path / "rel.h5", data)
        monkeypatch.chdir(tmp_path)
        manifest = TileArray.from_tiles("rel.h5", (4, NX), "f8", ENGINE)
        assert os.path.isabs(manifest._grid_values("paths").item(0))
        monkeypatch.chdir(tmp_path.parent)
        npt.assert_array_equal(np.asarray(manifest), data)

    def test_attrs(self, stack):
        manifest, _ = stack
        assert manifest.attrs == {"units": "strain"}


class TestSourcePaths:
    """Paths are stored verbatim: an array holds exactly what it was given."""

    def make(self, path):
        # the file's first row is skipped: sliced away, as views are made
        return TileArray.from_tiles([str(path)], ([5], NX), "<f4", ENGINE)[1:5]

    def stored(self, manifest):
        # a single tile folds its path to a 0-d variable
        return manifest.to_dataset()["paths"].values.ravel().tolist()

    def round_trip(self, manifest):
        return TileArray(manifest.to_dataset(), manifest.dtype, manifest.engine)

    def test_paths_round_trip(self, tmp_path):
        path = tmp_path / "sources" / "f.h5"
        assert self.stored(self.round_trip(self.make(path))) == [str(path)]

    def test_stored_paths_read(self, tmp_path):
        # the tile's start row skips the first row of the file
        data = np.arange(5 * NX, dtype="<f4").reshape(5, NX)
        (tmp_path / "sources").mkdir()
        _tile_file(tmp_path / "sources" / "f.h5", data)
        restored = self.round_trip(self.make(tmp_path / "sources" / "f.h5"))
        npt.assert_array_equal(np.asarray(restored), data[1:])


class TestStarts:
    def test_start_windows_read(self, windowed):
        manifest, reference = windowed
        npt.assert_array_equal(manifest.dataset["starts_0"].values, [2, 0, 2])
        npt.assert_array_equal(np.asarray(manifest), reference)
        npt.assert_array_equal(np.asarray(manifest[6:15]), reference[6:15])

    def test_two_windows_of_one_blob(self, windowed):
        manifest, reference = windowed
        path = str(manifest.dataset["paths"].values[0])
        split = _with_starts(
            TileArray.from_tiles(path, ([4, 4], NX), manifest.dtype, manifest.engine),
            [2, 6],
        )
        npt.assert_array_equal(np.asarray(split), reference[:8])

    def test_repeated_window_reads_twice(self, windowed):
        """The same source sub-box placed at two virtual positions is legal."""
        manifest, reference = windowed
        path = str(manifest.dataset["paths"].values[0])
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
        renamed["paths"].values[0] = "/elsewhere.h5"
        assert not manifest.equals(TileArray(renamed, manifest.dtype, manifest.engine))

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

    def test_new_axis_materializes(self, stack):
        manifest, reference = stack
        expanded = manifest[np.newaxis]
        assert isinstance(expanded, np.ndarray)
        npt.assert_array_equal(expanded, reference[np.newaxis])

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
        manifest, _ = stack
        expanded = manifest.expand_dims()
        npt.assert_array_equal(
            expanded.dataset["sizes_1"].values, manifest.dataset["sizes_0"].values
        )
        npt.assert_array_equal(
            expanded.dataset["starts_1"].values, manifest.dataset["starts_0"].values
        )

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

    def test_non_leading_axis_materializes(self, stack):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, 1)
        assert isinstance(expanded, np.ndarray)
        npt.assert_array_equal(expanded, reference[:, np.newaxis])

    def test_tuple_axis_materializes(self, stack):
        manifest, reference = stack
        expanded = np.expand_dims(manifest, (0, 1))
        assert isinstance(expanded, np.ndarray)
        npt.assert_array_equal(expanded, reference[np.newaxis, np.newaxis])

    def test_non_leading_method_axis_raises(self, stack):
        manifest, _ = stack
        with pytest.raises(ValueError, match="leading"):
            manifest.expand_dims(1)

    def test_negative_axis_method(self, stack):
        manifest, reference = stack
        expanded = manifest.expand_dims(-manifest.ndim - 1)
        assert isinstance(expanded, TileArray)
        npt.assert_array_equal(np.asarray(expanded), reference[np.newaxis])

    def test_dispatch_guards(self, stack):
        manifest, _ = stack
        assert manifest._expand_virtual((np.zeros(3), 0), {}) is NotImplemented
        assert manifest._expand_virtual((manifest, 0), {"extra": 1}) is NotImplemented


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
        from xdas.tiles.tilearray import _bounding_key

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
        from xdas.tiles.tilearray import _concatenate_virtual

        assert _concatenate_virtual((), {}) is NotImplemented
        assert _concatenate_virtual((5,), {}) is NotImplemented

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
