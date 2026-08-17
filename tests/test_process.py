"""
Dispatch tests for the `process()` boundary: sources, sinks, guards.

The governing invariant (plan §6) is that `pipeline.process(da, out=None)`
equals `pipeline(da)` whatever the source and chunking, and that sinks are
resolved from *(out spec × first-chunk type)* with writer instantiation
deferred to the first output chunk.
"""

import os
import threading

import numpy as np
import pandas as pd
import pytest

import xdas as xd
import xdas.processing as xp
from xdas.atoms import Partial, State
from xdas.atoms.core import _join_chunks
from xdas.config import Config
from xdas.processing.core import _auto_chunks, _ChainSource, _to_human
from xdas.virtual import TileArray


@pytest.fixture
def da():
    # 101 samples: an awkward length that exercises the flushed tails.
    return xd.testing.dummy(shape=(101, 5))


@pytest.fixture
def pipeline():
    return xd.resample(..., rate=25.0) >> xd.filter(..., (1.0, 10.0)) >> np.square


def gappy(da, at=50, gap=10):
    """Return *da* with a gap of *gap* samples at index *at*."""
    left = da.isel(time=slice(0, at))
    right = da.isel(time=slice(at + gap, None))
    return xd.concat([left, right], "time")


@pytest.fixture
def virtual(da, tmp_path):
    for index, chunk in enumerate(xd.split(da, 4, "time")):
        chunk.to_netcdf(tmp_path / f"{index:03d}.nc")
    return xd.open_mfdataarray(str(tmp_path / "*.nc"))


class TestSourceDispatch:
    def test_in_memory_eager(self, da, pipeline):
        assert pipeline.process(da).equals(pipeline(da))

    def test_in_memory_chunked(self, da, pipeline):
        expected = pipeline(da)
        result = pipeline.process(da, chunks={"time": 30})
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_in_memory_gappy_chunked(self, da, pipeline):
        da = gappy(da)
        expected = pipeline(da)
        result = pipeline.process(da, chunks={"time": 30})
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_virtual_auto(self, da, pipeline, virtual):
        source = xp.get_source(virtual)
        assert isinstance(source, xp.DataArrayLoader)
        result = pipeline.process(virtual)
        expected = pipeline(da)
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_glob_and_directory(self, da, pipeline, virtual, tmp_path):
        expected = pipeline(da)
        for spec in (str(tmp_path / "*.nc"), str(tmp_path)):
            result = pipeline.process(spec)
            assert np.allclose(result.values, expected.values)

    def test_iterable_passthrough(self, da, pipeline):
        chunks = xd.split(da, 4, "time")
        result = pipeline.process(iter(list(chunks)))
        expected = pipeline(da)
        assert np.allclose(result.values, expected.values)

    def test_loader_passthrough(self, da, pipeline):
        loader = xp.DataArrayLoader(da, {"time": 30})
        assert xp.get_source(loader) is loader
        result = pipeline.process(loader)
        assert np.allclose(result.values, pipeline(da).values)

    def test_multi_acquisition_chains_loaders(self, da, tmp_path):
        other = xd.testing.dummy(shape=(40, 3))
        da.to_netcdf(tmp_path / "000.nc")
        other.to_netcdf(tmp_path / "001.nc")
        collection = xd.open_mfdataarray(str(tmp_path / "*.nc"))
        assert isinstance(collection, xd.DataSequence)
        source = xp.get_source(collection)
        assert isinstance(source, _ChainSource)
        assert source.chunk_dim == "time"
        assert source.nbytes == da.nbytes + other.nbytes
        chunks = list(source)
        assert xd.concat(chunks[:1]).equals(da) or len(chunks) >= 2

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="URL scheme"):
            xp.get_source("ftp://somewhere")

    def test_invalid_source_raises(self):
        with pytest.raises(TypeError, match="source"):
            xp.get_source(42)

    def test_tcp_scheme(self, opened):
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        source = opened(xp.get_source(address))
        assert isinstance(source, xp.ZMQSubscriber)
        assert source.unbounded
        assert source.chunk_dim == "time"


class TestAutoChunks:
    def test_file_aligned(self, virtual, monkeypatch):
        # A tiny budget aligns chunk boundaries to the per-file extents.
        monkeypatch.setattr(xp.core, "AUTO_CHUNK_NBYTES", 1)
        dim, divs = _auto_chunks(virtual)
        assert dim == "time"
        assert divs == [0, 26, 51, 76, 101]

    def test_merged_to_budget(self, virtual, monkeypatch):
        # Two files fit the budget: boundaries merge pairwise.
        nbytes_per_slice = virtual.nbytes // virtual.sizes["time"]
        monkeypatch.setattr(xp.core, "AUTO_CHUNK_NBYTES", 52 * nbytes_per_slice)
        _, divs = _auto_chunks(virtual)
        assert divs == [0, 51, 101]

    def test_dense_fallback(self, da, monkeypatch):
        nbytes_per_slice = da.nbytes // da.sizes["time"]
        monkeypatch.setattr(xp.core, "AUTO_CHUNK_NBYTES", 30 * nbytes_per_slice)
        dim, divs = _auto_chunks(da)
        assert dim == "time"
        assert divs == [0, 30, 60, 90, 101]

    def test_loader_accepts_auto(self, virtual, monkeypatch):
        monkeypatch.setattr(xp.core, "AUTO_CHUNK_NBYTES", 1)
        loader = xp.DataArrayLoader(virtual, "auto")
        assert loader.chunk_size is None
        assert len(loader) == 4
        assert xd.concat(list(loader), "time").equals(virtual.load())

    def test_loader_rejects_bad_chunks(self, da):
        with pytest.raises(TypeError, match="auto"):
            xp.DataArrayLoader(da, "automatic")

    def test_tile_aligned(self, da, tmp_path, monkeypatch):
        for index, chunk in enumerate(xd.split(da, 4, "time")):
            chunk.to_netcdf(tmp_path / f"{index:03d}.nc")
        tiled = xd.open_mfdataarray(str(tmp_path / "*.nc"), vtype="tiles")
        monkeypatch.setattr(xp.core, "AUTO_CHUNK_NBYTES", 1)
        dim, divs = _auto_chunks(tiled)
        assert dim == "time"
        assert divs == [0, 26, 51, 76, 101]

    def test_chained_loaders_with_explicit_chunks(self, da, tmp_path):
        other = xd.testing.dummy(shape=(40, 3))
        da.to_netcdf(tmp_path / "000.nc")
        other.to_netcdf(tmp_path / "001.nc")
        collection = xd.open_mfdataarray(str(tmp_path / "*.nc"))
        # The per-run chunk size is clipped to the smallest run.
        source = xp.get_source(collection, {"time": 60})
        chunks = list(source)
        assert [chunk.sizes["time"] for chunk in chunks] == [60, 41, 40]


class TestSinkDispatch:
    def test_directory(self, da, pipeline, tmp_path):
        result = pipeline.process(da, out=str(tmp_path / "out"), chunks={"time": 30})
        assert np.allclose(result.values, pipeline(da).values)
        assert len(os.listdir(tmp_path / "out")) > 0

    def test_dataarray_to_file_raises(self, da, pipeline, tmp_path):
        with pytest.raises(ValueError, match="directory"):
            pipeline.process(da, out=str(tmp_path / "out.nc"), chunks={"time": 30})

    def test_dataframe_to_csv(self, da, tmp_path):
        atom = Partial(lambda da: pd.DataFrame({"mean": [float(np.mean(da.values))]}))
        path = tmp_path / "picks.csv"
        result = atom.process(da, out=str(path), chunks={"time": 30})
        assert path.exists()
        assert len(result) == 4

    def test_dataframe_to_other_suffix_raises(self, da, tmp_path):
        atom = Partial(lambda da: pd.DataFrame({"mean": [0.0]}))
        with pytest.raises(ValueError, match="csv"):
            atom.process(da, out=str(tmp_path / "picks.parquet"), chunks={"time": 30})

    def test_stream_to_directory(self, tmp_path):
        data = np.random.randint(-1000, 1000, size=(1000, 3), dtype=np.int32)
        starttime = np.datetime64("2023-01-01T00:00:00")
        da = xd.DataArray(
            data=data,
            coords={
                "time": {
                    "tie_indices": [0, data.shape[0] - 1],
                    "tie_values": [
                        starttime,
                        starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1),
                    ],
                    "sampling_interval": np.timedelta64(10, "ms"),
                },
                "distance": 5.0 * np.arange(data.shape[1]),
            },
        )
        atom = Partial(
            lambda da: da.to_stream(
                network="NT",
                station="ST{:03}",
                channel="HN1",
                location="00",
                dim={"distance": "time"},
            )
        )
        result = atom.process(da, out=str(tmp_path), chunks={"time": 100})
        assert len(result) == 3
        assert (tmp_path / "2023").exists()

    def test_writer_instance_passthrough(self, da, pipeline, tmp_path):
        writer = xp.DataArrayWriter(tmp_path, create_dirs=True)
        result = pipeline.process(da, out=writer, chunks={"time": 30})
        assert np.allclose(result.values, pipeline(da).values)

    def test_none_with_no_output_returns_none(self, da):
        atom = Partial(lambda da: None)
        assert atom.process(da, chunks={"time": 30}) is None

    def test_eager_with_no_output_returns_none(self, da, tmp_path):
        atom = Partial(lambda da: None)
        assert atom.process(da, out=str(tmp_path / "out")) is None
        assert not (tmp_path / "out").exists()

    def test_empty_chunk_dropped_by_writer(self, da, tmp_path):
        writer = xp.DataArrayWriter(tmp_path)
        writer.submit(da.isel(time=slice(0, 0)))
        assert len(os.listdir(tmp_path)) == 0

    def test_unknown_chunk_type_raises(self, da):
        atom = Partial(lambda da: object())
        with pytest.raises(TypeError, match="no writer"):
            atom.process(da, out="somewhere", chunks={"time": 30})

    def test_invalid_out_raises(self, da, pipeline):
        with pytest.raises(TypeError, match="cannot infer"):
            pipeline.process(da, out=42, chunks={"time": 30})

    def test_unknown_sink_scheme_raises(self, da):
        with pytest.raises(ValueError, match="URL scheme"):
            xp.get_writer("ftp://somewhere", da)

    def test_writer_instance_in_get_writer(self, da, tmp_path):
        writer = xp.DataArrayWriter(tmp_path)
        assert xp.get_writer(writer, da) is writer

    def test_eager_gappy_with_out(self, da, pipeline, tmp_path):
        # The eager result is a collection: each run is written as a chunk.
        result = pipeline.process(gappy(da), out=str(tmp_path / "out"))
        expected = pipeline(gappy(da))
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_tcp_sink(self, da, pipeline):
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        result = pipeline.process(da, out=address, chunks={"time": 30})
        assert result is None

    def test_legacy_signature(self, da, pipeline, tmp_path):
        loader = xp.DataArrayLoader(da, {"time": 30})
        writer = xp.DataArrayWriter(tmp_path)
        result = xp.process(pipeline, loader, writer)
        assert np.allclose(result.values, pipeline(da).values)

    def test_eager_with_out(self, da, pipeline, tmp_path):
        # In-memory source, no chunks: eager call, then sink dispatch.
        result = pipeline.process(da, out=str(tmp_path / "out"))
        assert np.allclose(result.values, pipeline(da).values)
        assert len(os.listdir(tmp_path / "out")) > 0

    def test_none_accumulates_streams(self, tmp_path):
        data = np.random.randint(-1000, 1000, size=(200, 3), dtype=np.int32)
        starttime = np.datetime64("2023-01-01T00:00:00")
        da = xd.DataArray(
            data=data,
            coords={
                "time": {
                    "tie_indices": [0, data.shape[0] - 1],
                    "tie_values": [
                        starttime,
                        starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1),
                    ],
                    "sampling_interval": np.timedelta64(10, "ms"),
                },
                "distance": 5.0 * np.arange(data.shape[1]),
            },
        )
        atom = Partial(
            lambda da: da.to_stream(
                network="NT",
                station="ST{:03}",
                channel="HN1",
                location="00",
                dim={"distance": "time"},
            )
        )
        result = atom.process(da, chunks={"time": 100})
        assert len(result) == 6  # 3 stations x 2 chunks, unmerged


class TestJoinChunks:
    def test_empty_and_single(self, da):
        assert _join_chunks([]) is None
        assert _join_chunks([da]) is da

    def test_dataframes(self):
        parts = [pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})]
        result = _join_chunks(parts)
        assert list(result["a"]) == [1, 2]

    def test_unconcatenatable_falls_back_to_collection(self, da):
        other = xd.testing.dummy(shape=(10, 3))
        result = _join_chunks([da, other], "time")
        assert isinstance(result, xd.DataSequence)

    def test_no_dim_falls_back_to_collection(self, da):
        assert isinstance(_join_chunks([da, da], None), xd.DataSequence)

    def test_mixed_types_fall_back_to_list(self, da):
        parts = [da, pd.DataFrame({"a": [1]})]
        assert isinstance(_join_chunks(parts, "time"), list)

    def test_to_human(self):
        assert _to_human(1) == "1 B"
        assert _to_human(5 * 2**20) == "5.0 MB"
        assert _to_human(2**42) == "4.0 TB"


class TestGuards:
    def test_eager_on_huge_virtual_raises(self, pipeline, virtual, monkeypatch):
        monkeypatch.setitem(Config.config, "memory_limit", 1)
        with pytest.raises(ValueError, match="process"):
            pipeline(virtual)

    def test_process_streams_below_guard(self, da, pipeline, virtual, monkeypatch):
        # Streaming stays legal with the same tiny limit on ingress, since
        # chunks are loaded one at a time; only out=None accumulation trips.
        monkeypatch.setitem(Config.config, "memory_limit", 1)
        with pytest.raises(ValueError, match="memory_limit"):
            pipeline.process(virtual)

    def test_accumulation_guard(self, da, pipeline, monkeypatch):
        monkeypatch.setitem(Config.config, "memory_limit", 1)
        with pytest.raises(ValueError, match="memory_limit"):
            pipeline.process(da, chunks={"time": 30})

    def test_disk_sink_ignores_guard(self, da, pipeline, tmp_path, monkeypatch):
        monkeypatch.setitem(Config.config, "memory_limit", 1)
        pipeline.process(da, out=str(tmp_path / "out"), chunks={"time": 30})


class TestUnbounded:
    class Source:
        """A never-ending source that the user interrupts after 3 chunks."""

        chunk_dim = "time"
        unbounded = True

        def __init__(self, chunks):
            self.chunks = chunks
            self.stopped = False

        def __iter__(self):
            yield from self.chunks
            raise KeyboardInterrupt

        def stop(self):
            self.stopped = True

    def test_keyboard_interrupt_flushes(self, da, pipeline):
        chunks = list(xd.split(da, 4, "time"))
        source = self.Source(chunks)
        result = pipeline.process(source)
        expected = pipeline(da)
        assert source.stopped
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_keyboard_interrupt_propagates_when_bounded(self, da, pipeline):
        class Bounded(self.Source):
            unbounded = False

        with pytest.raises(KeyboardInterrupt):
            pipeline.process(Bounded(list(xd.split(da, 4, "time"))))

    def test_until_truncates(self, da, pipeline):
        until = da["time"][60].values
        expected = pipeline(da.sel(time=slice(None, until)))
        result = pipeline.process(da, chunks={"time": 30}, until=until)
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_until_truncates_an_unchunked_source(self, da, pipeline):
        # Nothing is chunked here, so `until` cuts the source itself; the cut
        # is the same one, inclusive, as `sel` slices.
        until = da["time"][60].values
        result = pipeline.process(da, until=until)
        expected = pipeline(da.sel(time=slice(None, until)))
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_until_skips_late_chunks(self, da, pipeline):
        until = da["time"][30].values
        chunks = list(xd.split(da, [30], "time"))
        result = pipeline.process(iter(chunks), until=until)
        expected = pipeline(da.sel(time=slice(None, until)))
        assert np.allclose(result.values, expected.values)

    def test_until_ignores_chunks_without_the_dim(self, da):
        # Chunks that do not carry the chunked dimension are passed through.
        chunk = xd.testing.dummy(dims=("distance",), shape=(10,), step=(10.0,))
        atom = Partial(np.square)
        result = atom.process(iter([chunk]), until=np.datetime64("2024-05-21"))
        assert np.allclose(result.values, np.square(chunk).values)

    def test_until_as_string(self, da, pipeline):
        until = da["time"][60].values
        result = pipeline.process(da, chunks={"time": 30}, until=str(until))
        expected = pipeline(da.sel(time=slice(None, until)))
        assert np.allclose(result.values, expected.values)

    def test_until_inside_gap_breaks(self, da, pipeline):
        # Chunks split exactly at the gap: the first chunk ends before
        # `until`, the next one starts after it and is skipped entirely.
        left = da.isel(time=slice(0, 50))
        right = da.isel(time=slice(60, None))
        until = left["time"][-1].values + np.timedelta64(50, "ms")
        result = pipeline.process(iter([left, right]), until=until)
        expected = pipeline(left)
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)

    def test_realtime_seam_warns_per_seam(self, da):
        left = da.isel(time=slice(0, 50))
        right = da.isel(time=slice(60, None))
        source = self.Source([left, right])
        atom = Partial(np.square)
        with pytest.warns(UserWarning, match="realtime source has a discontinuity"):
            atom.process(source)

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_realtime_continuous_stream_is_silent(self, da):
        source = self.Source(list(xd.split(da, 4, "time")))
        atom = Partial(np.square)
        atom.process(source)

    def test_chunked_source_announces_its_splits_upfront(self, da, pipeline):
        with pytest.warns(UserWarning, match="1 discontinuity along 'time'"):
            pipeline.process(gappy(da), chunks={"time": 30})

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_upfront_scan_skips_non_axis_coordinates(self):
        # A dense coordinate has no free discontinuity scan: the source is
        # processed without any upfront announcement.
        dense = xd.testing.dummy(shape=(52, 5), ctype="dense")
        atom = Partial(np.square)
        result = atom.process(dense, chunks={"time": 20})
        assert np.allclose(result.values, np.square(dense).values)

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_realtime_chunks_without_the_dim_are_not_judged(self, da):
        # A realtime chunk that does not carry the chunked dimension leaves
        # the seam information untouched rather than resetting it.
        aside = xd.testing.dummy(dims=("distance",), shape=(5,), step=(10.0,))
        left, right = xd.split(da, 2, "time")
        source = self.Source([left, aside, right])
        atom = Partial(np.square)
        atom.process(source)

    @pytest.mark.filterwarnings("error::UserWarning")
    def test_realtime_one_sample_chunk_adopts_the_stream_rate(self):
        # A one-sample chunk of a sampled coordinate declares no rate of its
        # own: continuous with the stream, it inherits the previous chunk's
        # delta so the seam after it is still judged correctly.
        sampled = xd.testing.dummy(shape=(52, 5), ctype="sampled")
        chunks = list(xd.split(sampled, [50, 51], "time"))
        source = self.Source(chunks)
        atom = Partial(np.square)
        atom.process(source)

    def test_watch_is_a_realtime_loader(self, da, tmp_path):
        loader = xd.watch(tmp_path)
        try:
            assert isinstance(loader, xp.RealTimeLoader)
            assert loader.unbounded
        finally:
            loader.stop()

    def test_watch_source_end_to_end(self, da, pipeline, tmp_path):
        # Feed the queue directly (the watchdog handler is tested elsewhere)
        # and close the stream with the None sentinel.
        loader = xd.watch(tmp_path)
        for chunk in xd.split(da, 4, "time"):
            loader.queue.put(chunk)
        loader.queue.put(None)
        result = pipeline.process(loader)
        expected = pipeline(da)
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)


class TestZMQRoundTrip:
    def test_publish_process_subscribe(self, da, opened):
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        packets = list(xd.split(da, 10, "time"))
        # Bind before connecting so the subscription is live for packet one.
        publisher = opened(xp.ZMQPublisher(address))
        source = opened(xp.get_source(address))

        def publish():
            publisher.wait_for_subscribers(timeout=60.0)
            for packet in packets:
                publisher.submit(packet)

        thread = threading.Thread(target=publish)
        thread.start()
        atom = Partial(np.square)
        until = da["time"][-1].values
        result = atom.process(source, until=until)
        thread.join()
        expected = np.square(da)
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values)


def cft(t0=0.0, quiet=False, open_ended=False):
    """
    A characteristic function peaking on P and S at ``t0 + 1``.

    ``open_ended`` leaves the trigger on at the last sample, so the pick is
    only emitted when the atom is flushed.
    """
    lane = [0.0, 0.8, 0.8] if open_ended else [0.0, 0.8, 0.0]
    if quiet:
        lane = [0.0, 0.0, 0.0]
    return xd.DataArray(
        data=[[0.0, 0.0, 0.0], lane, lane],
        coords={
            "phase": ["N", "P", "S"],
            "time": {
                "tie_indices": [0, 2],
                "tie_values": [t0, t0 + 2.0],
                "sampling_interval": 1.0,
            },
        },
    )


def seed_tree():
    """A ``network / station / location`` tree of pick-able leaves."""
    return xd.DataCollection(
        {
            "IA": xd.DataCollection(
                {
                    "DBNFM": xd.DataCollection({"--": cft()}, "location"),
                    "LBFI": xd.DataCollection({"00": cft(10.0)}, "location"),
                },
                "station",
            )
        },
        "network",
    )


def das_tree():
    """A ``node / cable / acquisition`` tree, whose sequence level folds."""
    return xd.DataCollection(
        {
            "N1": xd.DataCollection(
                {"C1": xd.DataCollection([cft(0.0), cft(10.0)], "record")}, "cable"
            )
        },
        "node",
    )


def picker():
    """A fresh table-valued atom, whose tables carry their tree path."""
    return xd.trigger(..., thresh={"P": 0.5, "S": 0.5})


class Resets(xd.atoms.Atom):
    """An identity atom logging the chunk count it held at every reset."""

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.count = State(...)

    def initialize(self, x, **flags):
        self.count = State(0)

    def call(self, x, **flags):
        self.count = State(self.count + 1)
        return x

    def reset(self):
        self.log.append(self.count)
        super().reset()


class TestCollectionWalk:
    """
    W11: `process` walks a collection exactly as `atom(dc)` walks it.

    The defining invariant is `atom.process(dc, out=None) == atom(dc)`: it is
    what stops the streaming and the eager walk drifting apart, and it matters
    most where a leaf is too large to call eagerly at all.
    """

    @pytest.mark.parametrize("chunks", [None, {"time": 2}])
    def test_the_seed_tree_streams_to_the_eager_result(self, chunks):
        dc = seed_tree()
        expected = picker()(dc)
        result = picker().process(dc, out=None, chunks=chunks)
        assert isinstance(result, pd.DataFrame)
        assert result.equals(expected)

    @pytest.mark.parametrize("chunks", [None, {"time": 2}])
    def test_the_das_tree_streams_to_the_eager_result(self, chunks):
        dc = das_tree()
        expected = picker()(dc)
        result = picker().process(dc, out=None, chunks=chunks)
        assert list(result["record"]) == [0, 0, 1, 1]
        assert result.equals(expected)

    def test_an_array_atom_rebuilds_the_walked_tree(self, da, pipeline):
        dc = xd.DataCollection({"a": da, "b": da}, "node")
        expected = pipeline(dc)
        result = pipeline.process(dc, chunks={"time": 30})
        assert isinstance(result, xd.DataMapping)
        assert result.name == "node" and list(result) == ["a", "b"]
        for key in expected:
            assert result[key].coords.equals(expected[key].coords)
            assert np.allclose(result[key].values, expected[key].values)

    def test_a_bare_sequence_folds_like_the_eager_call(self):
        dc = xd.DataCollection([cft(0.0), cft(10.0)], "record")
        expected = picker()(dc)
        result = picker().process(dc, out=None)
        assert result.equals(expected)

    def test_the_flushed_tail_goes_to_the_last_element(self):
        dc = xd.DataCollection([cft(0.0), cft(10.0, open_ended=True)], "record")
        expected = picker()(dc)
        result = picker().process(dc, out=None)
        assert list(result["record"]) == [0, 0, 1, 1]
        assert result.equals(expected)

    def test_a_sequence_with_nothing_to_fold_along_walks_element_by_element(self, da):
        atom = Partial(np.square)
        dc = xd.DataCollection([da, da])
        result = atom.process(dc)
        assert isinstance(result, xd.DataSequence) and len(result) == 2
        assert np.allclose(result[0].values, np.square(da).values)

    def test_a_bare_callable_walks_a_collection_too(self, da):
        dc = xd.DataCollection({"a": xd.DataCollection([da], "record")}, "node")
        result = xp.process(np.square, dc)
        assert np.allclose(result["a"][0].values, np.square(da).values)

    def test_a_mapping_inside_a_folded_sequence_raises(self):
        dc = xd.DataCollection([cft(0.0), xd.DataCollection({"a": cft(10.0)})])
        with pytest.raises(NotImplementedError, match="mapping collections"):
            picker().process(dc)

    def test_merge_false_keeps_the_labelled_tree(self):
        tree = picker().process(seed_tree(), out=None, merge=False)
        assert isinstance(tree, xd.DataCollection)
        assert tree.name == "network"
        leaf = tree["IA"]["DBNFM"]["--"]
        assert list(leaf.columns)[:3] == ["network", "station", "location"]

    def test_the_atom_is_reset_between_leaves(self, da):
        log = []
        dc = xd.DataCollection({key: da for key in "abc"}, "node")
        Resets(log).process(dc, chunks={"time": 30})
        # the walk resets before every leaf, so each of the first two leaves
        # left its own chunk count behind and none of them inherited it
        assert [count for count in log if isinstance(count, int)] == [4, 4]

    def test_state_does_not_leak_from_one_leaf_to_the_next(self, da, pipeline):
        dc = xd.DataCollection({"a": da, "b": da}, "node")
        result = pipeline.process(dc, chunks={"time": 30})
        assert np.allclose(result["a"].values, result["b"].values)


class TestCollectionSinks:
    """W11: one rule per kind of destination, applied leaf by leaf."""

    def test_one_csv_holds_every_leaf(self, tmp_path):
        path = tmp_path / "picks.csv"
        result = picker().process(seed_tree(), out=str(path))
        assert path.exists()
        written = pd.read_csv(path)
        assert list(written["station"]) == ["DBNFM", "DBNFM", "LBFI", "LBFI"]
        assert result.equals(written)

    def test_a_csv_no_leaf_wrote_to_stays_absent(self, tmp_path):
        path = tmp_path / "picks.csv"
        dc = xd.DataCollection({"A": cft(quiet=True)}, "station")
        assert picker().process(dc, out=str(path)).empty
        assert not path.exists()

    def test_a_shared_csv_no_leaf_emitted_to_closes_to_nothing(self, da, tmp_path):
        # Not one leaf emits, so the shared table is never even resolved to a
        # writer: closing it has nothing to report and nothing to write.
        atom = Partial(lambda x: None)
        dc = xd.DataCollection({"a": da, "b": da}, "node")
        path = tmp_path / "picks.csv"
        assert atom.process(dc, out=str(path)) is None
        assert not path.exists()

    def test_a_leaf_emitting_nothing_answers_with_an_empty_collection(
        self, da, tmp_path
    ):
        atom = Partial(lambda x: None)
        dc = xd.DataCollection({"a": da, "b": da}, "node")
        out = tmp_path / "results"
        result = atom.process(dc, out=str(out), chunks={"time": 30})
        assert all(len(leaf) == 0 for leaf in result.values())
        assert not out.exists()  # no directory is created for nothing

    def test_a_directory_fans_out_to_the_tree_path(self, da, pipeline, tmp_path):
        dc = xd.DataCollection(
            {"N1": xd.DataCollection({"C1": da, "C2": da}, "cable")}, "node"
        )
        out = tmp_path / "results"
        result = pipeline.process(dc, out=str(out), chunks={"time": 30})
        assert sorted(path.name for path in (out / "N1").iterdir()) == ["C1", "C2"]
        expected = pipeline(dc)
        assert np.allclose(result["N1"]["C1"].values, expected["N1"]["C1"].values)

    def test_a_folded_sequence_writes_to_one_directory(self, da, pipeline, tmp_path):
        dc = xd.DataCollection({"C1": xd.DataCollection([da], "record")}, "cable")
        out = tmp_path / "results"
        result = pipeline.process(dc, out=str(out), chunks={"time": 30})
        assert [path.name for path in out.iterdir()] == ["C1"]
        assert np.allclose(result["C1"].values, pipeline(da).values)

    def test_a_writer_instance_is_shared_by_every_leaf(self, tmp_path):
        writer = xp.DataFrameWriter(str(tmp_path / "picks.csv"))
        result = picker().process(seed_tree(), out=writer)
        assert len(result) == 4

    def test_an_uninferable_out_raises(self):
        with pytest.raises(TypeError, match="cannot infer a writer"):
            picker().process(seed_tree(), out=42)


class TestCollectionGather:
    """W11: the gather is consulted before anything is chunked."""

    def test_a_channel_level_becomes_a_dimension_before_chunking(self):
        from tests.test_atoms_ml import component_collection, picker_model

        dc = component_collection(["SHZ", "SHN", "SHE"])
        model = picker_model("original")
        expected = xd.pick(dc, model, device="cpu")
        result = xd.pick(..., model, device="cpu").process(
            dc, out=None, chunks={"time": 8}
        )
        assert "channel" not in result.columns  # the level became an axis
        assert result.equals(expected)

    def test_the_gathered_array_stays_virtual(self, monkeypatch, tmp_path):
        from tests.test_atoms_ml import component_trace, picker_model

        keys = ["SHZ", "SHN", "SHE"]
        for index, key in enumerate(keys):
            (tmp_path / key).mkdir()
            trace = component_trace(index, n=64)
            for part, chunk in enumerate(xd.split(trace, 4, "time")):
                chunk.to_netcdf(tmp_path / key / f"{part:03d}.nc")
        dc = xd.DataCollection(
            {
                key: xd.open_mfdataarray(str(tmp_path / key / "*.nc"), vtype="tiles")
                for key in keys
            },
            "channel",
        )
        atom = xd.pick(..., picker_model("original"), device="cpu")
        assert isinstance(atom.gather(dc).data, TileArray)  # nothing was read
        # a limit below the stacked array (1536 B) but above the pick table
        monkeypatch.setitem(Config.config, "memory_limit", 1024)
        with pytest.raises(ValueError, match="process"):
            atom(dc)  # the eager walk refuses to load the stacked array
        assert len(atom.process(dc, chunks={"time": 16})) > 0
