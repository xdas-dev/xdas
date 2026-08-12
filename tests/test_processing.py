import importlib.util
import os
import threading
import time
from pathlib import Path

import hdf5plugin
import numpy as np
import obspy
import pandas as pd
import pytest
import scipy.signal as sp

import xdas as xd
import xdas.processing as xp
from xdas.atoms import Partial, Sequential
from xdas.signal import sosfilt

TIMEOUT = 60.0
"""Seconds any ZMQ wait is given before failing, so that no test can hang."""


class TestDataArrayLoader:
    def test_init(self):
        da = xd.testing.dummy(shape=(1000, 100))
        dl = xp.DataArrayLoader(da, {"time": 100})
        assert dl.da is da
        assert dl.chunk_dim == "time"
        assert dl.chunk_size == 100
        assert dl.max_buffers == 1
        assert dl.max_workers == 1
        assert len(dl) == 10

    @pytest.mark.parametrize(
        "max_buffers,max_workers",
        [
            (1, 1),
            (2, 2),
            (4, 2),
            (8, 4),
        ],
    )
    def test_chunks_integrity(self, max_buffers, max_workers):
        da = xd.testing.dummy(shape=(1000, 100))
        dl = xp.DataArrayLoader(da, {"time": 100}, max_buffers, max_workers)
        chunks = list(dl)
        result = xd.concat(chunks)
        assert result.equals(da)

    def test_error_handling(self):
        da = xd.testing.dummy(shape=(1000, 100))
        with pytest.raises(TypeError):
            xp.DataArrayLoader(None, None)
        with pytest.raises(TypeError):
            xp.DataArrayLoader(da, 100)
        with pytest.raises(ValueError):
            xp.DataArrayLoader(da, {"space": 100})
        with pytest.raises(ValueError):
            xp.DataArrayLoader(da, {"time": 2000})

    def test_unknown_pool_raises(self):
        da = xd.testing.dummy(shape=(1000, 100))
        with pytest.raises(ValueError, match="no worker pool"):
            list(xp.DataArrayLoader(da, {"time": 100}, pool="fork"))


class TestReadOnlyIngress:
    """Chunks may arrive immutable (zero-copy from an object store).

    Atoms honor this by allocating their outputs rather than writing into
    their input; this pins the contract down without needing ray installed.
    """

    def test_pipeline_accepts_readonly_chunks(self):
        da = xd.testing.dummy(shape=(1000, 100))
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])
        expected = xd.concat(
            [sequence(chunk, chunk_dim="time") for chunk in xd.split(da, 10, "time")]
        )
        chunks = [chunk.copy() for chunk in xd.split(da, 10, "time")]
        for chunk in chunks:
            chunk.data.setflags(write=False)
        sequence.reset()
        result = xd.concat([sequence(chunk, chunk_dim="time") for chunk in chunks])
        assert result.equals(expected)
        for chunk, original in zip(chunks, xd.split(da, 10, "time")):
            np.testing.assert_array_equal(chunk.data, original.data)


def test_missing_ray_points_at_the_extra(monkeypatch):
    """Without ray installed, the pool names the optional dependency."""
    import builtins

    from xdas.processing.core import ProcessPool

    real_import = builtins.__import__

    def no_ray(name, *args, **kwargs):
        if name == "ray":
            raise ImportError("no module named ray")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ray)
    with pytest.raises(ImportError, match=r"xdas\[ray\]"):
        ProcessPool(1)


requires_ray = pytest.mark.skipif(
    importlib.util.find_spec("ray") is None, reason="ray is not installed"
)


@requires_ray
@pytest.mark.slow
class TestProcessPool:
    """Worker processes get past the HDF5 lock, shared memory past the pipe."""

    def test_loader_chunks_integrity(self, tmp_path):
        # A virtual array: the manifest of each chunk is what crosses to the
        # worker, which then reads its own files; the loaded chunk comes back
        # through the object store.
        expected = xd.testing.dummy(shape=(1000, 100))
        expected.to_netcdf(tmp_path / "data.nc")
        da = xd.open_dataarray(tmp_path / "data.nc")
        dl = xp.DataArrayLoader(da, {"time": 100}, 4, 2, pool="processes")
        assert xd.concat(list(dl)).equals(expected)

    def test_loader_chunks_are_zero_copy(self):
        # Read-only data is the signature of a store-backed array: the parent
        # mapped shared memory, nothing was pickled back.
        da = xd.testing.dummy(shape=(1000, 100))
        chunks = list(xp.DataArrayLoader(da, {"time": 100}, 2, 2, pool="processes"))
        assert all(not chunk.data.flags.writeable for chunk in chunks)

    def test_loader_equals_threads(self):
        da = xd.testing.dummy(shape=(1000, 100))
        threads = list(xp.DataArrayLoader(da, {"time": 100}, 2, 2))
        processes = list(xp.DataArrayLoader(da, {"time": 100}, 2, 2, "processes"))
        assert xd.concat(processes).equals(xd.concat(threads))

    def test_writer_equals_threads(self, tmp_path):
        da = xd.testing.dummy(shape=(1000, 100))
        chunks = list(xd.split(da, 10, "time"))
        results = []
        for pool in ["threads", "processes"]:
            dirpath = tmp_path / pool
            dirpath.mkdir()
            dw = xp.DataArrayWriter(dirpath, max_buffers=2, max_workers=2, pool=pool)
            for chunk in chunks:
                dw.write(chunk)
            results.append(dw.result())
        assert results[1].load().equals(results[0].load())
        assert results[0].load().equals(da)

    def test_end_to_end(self, tmp_path):
        # which pool the chunks travel through cannot change what comes out.
        xd.testing.dummy(shape=(1000, 100)).to_netcdf(tmp_path / "data.nc")
        da = xd.open_dataarray(tmp_path / "data.nc")
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        results = []
        for pool in ["threads", "processes"]:
            sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])
            loader = xp.DataArrayLoader(da, {"time": 100}, 2, 2, pool=pool)
            dirpath = tmp_path / pool
            dirpath.mkdir()
            writer = xp.DataArrayWriter(dirpath, dim="time")
            results.append(xp.process(sequence, loader, writer))
        assert results[1].load().equals(results[0].load())

    def test_backlog_is_parked_and_ordered(self):
        # More submissions than workers: the excess is parked, launched as
        # capacity frees, and resolved in submission order. The tasks sleep
        # so none can finish during the submit loop and free a slot.
        from xdas.processing.core import ProcessPool

        with ProcessPool(2) as pool:
            futures = [pool.submit(_slow_double, i) for i in range(8)]
            assert sum(future._ref is None for future in futures) == 6
            assert len(pool._running) == 2
            assert [future.result() for future in futures] == [2 * i for i in range(8)]

    def test_errors_propagate(self):
        from xdas.processing.core import ProcessPool

        with ProcessPool(1) as pool, pytest.raises(ValueError, match="broken task"):
            pool.submit(_raise).result()

    def test_shutdown_cancels_backlog(self):
        from concurrent.futures import CancelledError

        from xdas.processing.core import ProcessPool

        pool = ProcessPool(1)
        futures = [pool.submit(_slow_double, i) for i in range(4)]
        pool.shutdown()
        pool.shutdown()  # idempotent: nothing left to wait for
        assert futures[0].result() == 0
        with pytest.raises(CancelledError):
            futures[-1].result()


def _slow_double(x):
    """Worker task used by the pool tests, slow enough to keep a backlog parked."""
    time.sleep(0.2)
    return 2 * x


def _raise():
    """Worker task raising, to check error propagation through the store."""
    raise ValueError("broken task")


class TestDataArrayWriter:
    def test_init(self, tmp_path):
        dw = xp.DataArrayWriter(tmp_path)
        assert dw.dirpath == str(tmp_path)

    @pytest.mark.parametrize(
        "max_buffers,max_workers",
        [
            (1, 1),
            (2, 2),
            (4, 2),
            (8, 4),
        ],
    )
    def test_chunk_integrity(self, max_buffers, max_workers, tmp_path):
        expected = xd.testing.dummy(shape=(1000, 100))
        dw = xp.DataArrayWriter(tmp_path, None, max_buffers, max_workers)
        chunks = xd.split(expected, 10, dim="time")
        for chunk in chunks:
            dw.submit(chunk)
        result = dw.result()
        assert result.equals(expected)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(OSError):
            xp.DataArrayWriter("not_a_directory")
        dirpath = tmp_path / "some_directory"
        xp.DataArrayWriter(dirpath, create_dirs=True)

    def test_passing_wrong_input(self, tmp_path):
        dw = xp.DataArrayWriter(tmp_path, create_dirs=True)
        with pytest.raises(TypeError):
            dw.submit(None)

    def test_empty_chunks_are_dropped(self, tmp_path):
        expected = xd.testing.dummy(shape=(1000, 100))
        dw = xp.DataArrayWriter(tmp_path, dim="time")
        dw.submit(expected.isel(time=slice(0, 0)))
        for chunk in xd.split(expected, 10, dim="time"):
            dw.submit(chunk)
        assert dw.result().equals(expected)

    def test_the_chunked_dimension_need_not_lead(self, tmp_path):
        # joining on the first dimension stacks the chunks along the wrong
        # axis whenever the chunked dimension does not lead the output.
        expected = xd.testing.dummy(shape=(1000, 100)).transpose("distance", "time")
        dw = xp.DataArrayWriter(tmp_path, dim="time")
        for chunk in xd.split(expected, 10, dim="time"):
            dw.submit(chunk)
        assert dw.result().equals(expected)


class TestProcessing:
    def test_stateful(self, tmp_path):
        sample_path = tmp_path / "sample.nc"

        # generate test dataarray
        xd.testing.dummy().to_netcdf(sample_path)
        da = xd.open(sample_path)

        # declare processing sequence
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])

        # monolithic processing
        result1 = sequence(da)

        # chunked processing
        data_loader = xp.DataArrayLoader(da, chunks={"time": 100})
        data_writer = xp.DataArrayWriter(tmp_path)
        result2 = xp.process(
            sequence, data_loader, data_writer
        )  # resets the sequence by default

        # test
        assert result1.equals(result2)

    def test_small_last_chunk(self, tmp_path):
        da = xd.testing.dummy(shape=(1001, 100), datetime=False)

        # declare processing sequence
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])

        # monolithic processing
        sequence(da)

        # chunked processing
        data_loader = xp.DataArrayLoader(da, chunks={"time": 100})
        for da in data_loader:
            pass  # TODO
        # data_writer = xp.DataArrayWriter(tmp_path)
        # result2 = xp.process(
        #     sequence, data_loader, data_writer
        # )  # resets the sequence by default


class TestDataFrameWriter:
    def test_init(self, tmp_path):
        dw = xp.DataFrameWriter(tmp_path / "output.csv")
        assert dw.path == str(tmp_path / "output.csv")
        assert dw.parse_dates is None

    def test_single_dataframe(self, tmp_path):
        dw = xp.DataFrameWriter(tmp_path / "output.csv")
        expected = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        dw.submit(expected)
        result = dw.result()
        assert result.equals(expected)
        assert Path(dw.path).exists()
        result = pd.read_csv(dw.path)
        assert result.equals(expected)

    def test_multiple_dataframes(self, tmp_path):
        dw = xp.DataFrameWriter(tmp_path / "output.csv")
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
        dw.submit(df1)
        dw.submit(df2)
        result = dw.result()
        expected = pd.concat([df1, df2], ignore_index=True)
        assert result.equals(expected)
        assert Path(dw.path).exists()
        result = pd.read_csv(dw.path)
        assert result.equals(expected)

    def test_write_empty_dataframe(self, tmp_path):
        # Empty chunks are accepted and silently dropped (many flushes
        # produce nothing): no file is created for them.
        dw = xp.DataFrameWriter(tmp_path / "output.csv")
        expected = pd.DataFrame()
        dw.submit(expected)
        result = dw.result()
        assert result.equals(expected)
        assert not Path(dw.path).exists()

    def test_with_existing_file(self, tmp_path):
        dw1 = xp.DataFrameWriter(tmp_path / "output.csv")
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        dw1.submit(df1)
        result = dw1.result()

        dw2 = xp.DataFrameWriter(tmp_path / "output.csv")
        df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
        dw2.submit(df2)
        result = dw2.result()

        expected = pd.concat([df1, df2], ignore_index=True)
        assert result.equals(expected)
        result = pd.read_csv(tmp_path / "output.csv")
        assert result.equals(expected)

    def test_missing_directory(self, tmp_path):
        with pytest.raises(OSError):
            xp.DataFrameWriter(tmp_path / "not_a_directory" / "output.csv")
        xp.DataFrameWriter(tmp_path / "some_directory" / "output.csv", create_dirs=True)
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            dw = xp.DataFrameWriter("output.csv", create_dirs=True)
            assert dw.path == "output.csv"
        finally:
            os.chdir(orig)

    def test_passing_wrong_input(self, tmp_path):
        dw = xp.DataFrameWriter(tmp_path / "output.csv")
        with pytest.raises(TypeError):
            dw.submit(None)


class TestZMQ:
    def _publish_and_subscribe(self, packets, address, encoding=None):
        publisher = xp.ZMQPublisher(address, encoding)

        def publish():
            # A recording, wanted whole. Only the publisher can hold a replay
            # back until its audience has landed; a subscriber joining one
            # already under way has no way to recover its first packets.
            publisher.wait_for_subscribers(timeout=TIMEOUT)
            for packet in packets:
                publisher.submit(packet)

        threading.Thread(target=publish).start()
        subscriber = xp.ZMQSubscriber(address, timeout=TIMEOUT)

        result = []
        for n, packet in enumerate(subscriber, start=1):
            result.append(packet)
            if n == len(packets):
                break
        return xd.concat(result)

    def test_publish_and_subscribe(self):
        expected = xd.testing.dummy()
        packets = xd.split(expected, 10)
        address = f"tcp://localhost:{xd.io.get_free_port()}"

        result = self._publish_and_subscribe(packets, address)
        assert result.equals(expected)

    def test_subscriber_joins_a_real_time_flux(self):
        # A real-time publisher streams whether or not anyone is listening, so
        # a subscriber joining it gets the stream from wherever it lands.
        packets = xd.split(xd.testing.dummy(), 10)
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        publisher = xp.ZMQPublisher(address)
        stop = threading.Event()

        def flux():
            while not stop.is_set():
                for packet in packets:
                    if stop.is_set():
                        return
                    publisher.submit(packet)

        thread = threading.Thread(target=flux)
        thread.start()
        try:
            subscriber = xp.ZMQSubscriber(address, timeout=TIMEOUT)
            # The flux greets us as it streams, without ever waiting for us.
            subscriber.wait_until_subscribed()
            received = [next(subscriber) for _ in range(3)]
        finally:
            stop.set()
            thread.join()

        subscriber.wait_until_subscribed()  # greeted already, so this returns
        for packet in received:
            assert any(packet.equals(published) for published in packets)

    def test_subscriber_timeout(self):
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        xp.ZMQPublisher(address)  # binds, but never publishes
        subscriber = xp.ZMQSubscriber(address, timeout=0.1)
        with pytest.raises(TimeoutError, match="no packet received"):
            next(subscriber)

    def test_wait_until_subscribed_needs_a_publisher_that_publishes(self):
        # Only a publisher that publishes can acknowledge anybody: a silent one
        # leaves a subscriber waiting, which is what the timeout is for.
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        xp.ZMQPublisher(address)
        subscriber = xp.ZMQSubscriber(address, timeout=0.1)
        with pytest.raises(TimeoutError, match="no packet received"):
            subscriber.wait_until_subscribed()

    def test_encoding(self):
        expected = xd.testing.dummy()
        packets = xd.split(expected, 10)
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=1e-6)}

        result = self._publish_and_subscribe(packets, address, encoding=encoding)
        assert np.allclose(result.values, expected.values, atol=1e-6)
        result.data = expected.data
        assert result.equals(expected)


class TestStreamWriter:
    def test_without_gap(self, tmp_path):
        data = np.random.randint(low=-1000, high=1000, size=(1000, 10), dtype=np.int32)
        starttime = np.datetime64("2023-01-01T00:00:00")
        endtime = starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1)
        distance = 5.0 * np.arange(data.shape[1])

        da = xd.DataArray(
            data=data,
            coords={
                "time": {
                    "tie_indices": [0, data.shape[0] - 1],
                    "tie_values": [starttime, endtime],
                    "sampling_interval": np.timedelta64(10, "ms"),
                },
                "distance": distance,
            },
        )

        def atom(da, **kwargs):
            return da.to_stream(
                network="NT",
                station="ST{:03}",
                channel="HN1",
                location="00",
                dim={"distance": "time"},
            )

        data_loader = xp.DataArrayLoader(da, chunks={"time": 100})

        kw_merge = {"method": 1}
        kw_write = {"reclen": 4096}
        data_writer = xp.StreamWriter(
            tmp_path, "M", kw_merge, kw_write, output_format="SDS"
        )

        st = xp.process(atom, data_loader, data_writer)

        assert isinstance(st, obspy.Stream)
        assert len(st) == 10
        tr = st[0]
        assert tr.stats.network == "NT"
        assert tr.stats.station == "ST001"
        assert tr.stats.channel == "HN1"
        assert tr.stats.location == "00"
        assert tr.stats.npts == 1000
        assert np.array_equal(tr.data, data[:, 0])
        assert tr.stats.starttime == obspy.UTCDateTime(str(starttime))
        path = (
            tmp_path / "2023" / "NT" / "ST001" / "HN1.D" / "NT.ST001.00.HN1.D.2023.001"
        )
        assert path.exists()
        st = obspy.read(path)
        assert len(st) == 1
        assert len(list(tmp_path.rglob("*.001"))) == 10

    def test_with_gap(self, tmp_path):
        da = xd.DataArray(
            data=np.random.randint(
                low=-1000, high=1000, size=(900, 10), dtype=np.int32
            ),
            coords={
                "time": {
                    "tie_indices": [0, 399, 400, 899],
                    "tie_values": np.array(
                        [
                            "2023-01-01T00:00:00.000",
                            "2023-01-01T00:00:03.990",
                            "2023-01-01T00:00:05.000",
                            "2023-01-01T00:00:09.990",
                        ],
                        dtype="datetime64[ms]",
                    ),
                    "sampling_interval": np.timedelta64(10, "ms"),
                },
                "distance": 5.0 * np.arange(10),
            },
        )

        def atom(da, **kwargs):
            return da.to_stream(
                network="NT",
                station="ST{:03}",
                channel="HN1",
                location="00",
                dim={"distance": "time"},
            )

        data_loader = xp.DataArrayLoader(da, chunks={"time": 100})

        kw_merge = {"method": 1}
        kw_write = {"reclen": 4096}
        data_writer = xp.StreamWriter(
            tmp_path, "M", kw_merge, kw_write, output_format="SDS"
        )

        st = xp.process(atom, data_loader, data_writer)

        assert isinstance(st, obspy.Stream)
        assert len(st) == 10
        tr = st[0]
        assert isinstance(tr.data, np.ma.masked_array)
        assert tr.stats.network == "NT"
        assert tr.stats.station == "ST001"
        assert tr.stats.channel == "HN1"
        assert tr.stats.location == "00"
        tr1, tr2 = tr.split()
        assert tr1.stats.npts == 400
        assert tr2.stats.npts == 500
        assert np.array_equal(tr1.data, da.values[0:400, 0])
        assert np.array_equal(tr2.data, da.values[400:900, 0])
        assert tr1.stats.starttime == obspy.UTCDateTime("2023-01-01T00:00:00.000")
        assert tr2.stats.starttime == obspy.UTCDateTime("2023-01-01T00:00:05.000")
        path = (
            tmp_path / "2023" / "NT" / "ST001" / "HN1.D" / "NT.ST001.00.HN1.D.2023.001"
        )
        assert path.exists()
        st = obspy.read(path)
        assert len(st) == 2
        assert len(list(tmp_path.rglob("*.001"))) == 10

    def test_flat(self, tmp_path):
        data = np.random.randint(low=-1000, high=1000, size=(1000, 10), dtype=np.int32)
        starttime = np.datetime64("2023-01-01T00:00:00")
        endtime = starttime + np.timedelta64(10, "ms") * (data.shape[0] - 1)
        distance = 5.0 * np.arange(data.shape[1])

        da = xd.DataArray(
            data=data,
            coords={
                "time": {
                    "tie_indices": [0, data.shape[0] - 1],
                    "tie_values": [starttime, endtime],
                    "sampling_interval": np.timedelta64(10, "ms"),
                },
                "distance": distance,
            },
        )

        def atom(da, **kwargs):
            return da.to_stream(
                network="NT",
                station="ST{:03}",
                channel="HN1",
                location="00",
                dim={"distance": "time"},
            )

        data_loader = xp.DataArrayLoader(da, chunks={"time": 100})

        path = tmp_path / "flat_output.mseed"
        kw_merge = {"method": 1}
        kw_write = {"reclen": 4096}
        data_writer = xp.StreamWriter(
            path, "M", kw_merge, kw_write, output_format="flat"
        )

        st = xp.process(atom, data_loader, data_writer)

        assert isinstance(st, obspy.Stream)
        assert len(st) == 10
        tr = st[0]
        assert tr.stats.network == "NT"
        assert tr.stats.station == "ST001"
        assert tr.stats.channel == "HN1"
        assert tr.stats.location == "00"
        assert tr.stats.npts == 1000
        assert np.array_equal(tr.data, data[:, 0])
        assert tr.stats.starttime == obspy.UTCDateTime(str(starttime))
        assert path.exists()
        st = obspy.read(path)
        assert len(st) == 10


class TestProcessNoNbytes:
    def test_loader_without_nbytes(self, tmp_path):
        da = xd.testing.dummy(shape=(100, 10))
        chunks = xd.split(da, 10, dim="time")

        class SimpleLoader:
            chunk_dim = "time"

            def __iter__(self):
                return iter(chunks)

        data_writer = xp.DataArrayWriter(tmp_path)

        def atom(x, **kw):
            return x

        result = xp.process(atom, SimpleLoader(), data_writer)
        assert result.equals(da)


class TestDataArrayLoaderMaxBuffers:
    def test_max_buffers_exceeds_chunks(self):
        da = xd.testing.dummy(shape=(10, 5))
        dl = xp.DataArrayLoader(da, {"time": 5}, max_buffers=10)
        chunks = list(dl)
        assert len(chunks) == 2
        result = xd.concat(chunks)
        assert result.equals(da)


class TestDataFrameWriterAliases:
    def test_write_alias(self, tmp_path):
        dw = xp.DataFrameWriter(tmp_path / "output.csv")
        df = pd.DataFrame({"A": [1, 2, 3]})
        dw.write(df)  # use write() alias
        result = dw.result()
        assert result.equals(df)

    def test_create_dirs_no_dirname(self, tmp_path):
        path = tmp_path / "bare.csv"
        dw = xp.DataFrameWriter(path, create_dirs=True)
        assert dw.path == str(path)


class TestStreamWriterEdgeCases:
    def test_flat_missing_directory_raises(self, tmp_path):
        with pytest.raises(OSError):
            xp.StreamWriter(
                tmp_path / "nonexistent_dir" / "out.mseed", "M", output_format="flat"
            )

    def test_invalid_output_format_raises(self, tmp_path):
        with pytest.raises(ValueError, match="output_format"):
            xp.StreamWriter(tmp_path, "M", output_format="invalid")

    def test_submit_wrong_type_raises(self, tmp_path):
        sw = xp.StreamWriter(tmp_path, "M")
        with pytest.raises(TypeError):
            sw.submit("not_a_stream")


class TestZMQPublisherAliases:
    def test_write_alias(self):
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        publisher = xp.ZMQPublisher(address)
        da = xd.testing.dummy()
        publisher.write(da)  # use write() alias

    def test_result_returns_none(self):
        address = f"tcp://localhost:{xd.io.get_free_port()}"
        publisher = xp.ZMQPublisher(address)
        assert publisher.result() is None


class TestHandlerDirect:
    def test_on_closed(self, tmp_path):
        from queue import Queue

        from xdas.processing.core import Handler

        da = xd.testing.dummy(shape=(10, 5), step=(1.0, 10.0), dtype=np.float32)
        path = str(tmp_path / "test.nc")
        da.to_netcdf(path)

        queue = Queue()
        handler = Handler(queue, "xdas")

        class MockEvent:
            src_path = path

        handler.on_closed(MockEvent())
        result = queue.get()
        assert result.equals(da)


class TestRealTimeLoader:
    def test_iter_and_next(self, tmp_path):
        from xdas.processing.core import RealTimeLoader

        loader = RealTimeLoader(str(tmp_path), engine="xdas")
        assert iter(loader) is loader

        # put a DataArray directly into the queue
        da = xd.testing.dummy(shape=(5, 3), step=(1.0, 10.0), dtype=np.float32)
        loader.queue.put(da)
        result = next(loader)
        assert result.equals(da)

        # put None to trigger StopIteration
        loader.queue.put(None)
        with pytest.raises(StopIteration):
            next(loader)
