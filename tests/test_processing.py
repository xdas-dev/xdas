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
from xdas.processing import (
    DataArrayLoader,
    DataArrayWriter,
    DataFrameWriter,
    StreamWriter,
    ZMQPublisher,
    ZMQSubscriber,
    process,
)
from xdas.signal import sosfilt
from xdas.synthetics import wavelet_wavefronts


class TestDataArrayLoader:
    @pytest.mark.timeout(1)
    def test_init_and_stop(self):
        da = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))
        dl = DataArrayLoader(da, {"time": 100})
        assert dl.chunk_dim == "time"
        assert dl.chunk_size == 100
        assert len(dl) == 10
        dl.stop()

    @pytest.mark.timeout(1)
    def test_context_manager(self):
        da = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))
        with DataArrayLoader(da, {"time": 100}) as dl:
            assert dl.da is da

    @pytest.mark.timeout(1)
    def test_chunks_integrity(self):
        da = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))
        dl = DataArrayLoader(da, {"time": 100})
        chunks = [chunk for chunk in dl]
        result = xd.concatenate(chunks)
        assert result.equals(da)

    @pytest.mark.timeout(1)
    def test_error_handling(self):
        da = xd.DataArray(np.random.rand(1000, 100), dims=("time", "distance"))
        with pytest.raises(TypeError):
            DataArrayLoader(None, None)
        with pytest.raises(TypeError):
            DataArrayLoader(da, 100)
        with pytest.raises(ValueError):
            DataArrayLoader(da, {"space": 100})
        with pytest.raises(ValueError):
            DataArrayLoader(da, {"time": 2000})


class TestProcessing:
    def test_stateful(self, tmp_path):
        sample_path = tmp_path / "sample.nc"

        # generate test dataarray
        wavelet_wavefronts().to_netcdf(sample_path)
        da = xd.open(sample_path)

        # declare processing sequence
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])

        # monolithic processing
        result1 = sequence(da)

        # chunked processing
        data_loader = DataArrayLoader(da, chunks={"time": 100})
        data_writer = DataArrayWriter(tmp_path)
        result2 = process(
            sequence, data_loader, data_writer
        )  # resets the sequence by default

        # test
        assert result1.equals(result2)

    def test_small_last_chunk(self, tmp_path):
        da = xd.DataArray(
            data=np.random.randn(1001, 100),
            coords={
                "time": xd.Coordinate["interpolated"].from_block(0, 1001, 0.01),
                "distance": xd.Coordinate["interpolated"].from_block(0, 100, 10.0),
            },
        )

        # declare processing sequence
        sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
        sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])

        # monolithic processing
        result1 = sequence(da)

        # chunked processing
        data_loader = DataArrayLoader(da, chunks={"time": 100})
        for da in data_loader:
            pass
        # data_writer = DataArrayWriter(tmp_path)
        # result2 = process(
        #     sequence, data_loader, data_writer
        # )  # resets the sequence by default

        # # test
        # assert result1.equals(result2)


class TestDataFrameWriter:
    def test_write_and_result(self, tmp_path):
        # Create a DataFrameWriter instance
        writer = DataFrameWriter(tmp_path / "output.csv")

        # Create a DataFrame to write
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Write the DataFrame asynchronously
        writer.write(df)

        # Get the result (wait for the asynchronous task to complete)
        result = writer.result()

        # Check if the result matches the original DataFrame
        assert result.equals(df)

        # Check if the output file exists
        assert Path(writer.path).exists()

        # Check if the output file contains the correct data
        output_df = pd.read_csv(writer.path)
        assert output_df.equals(df)

    def test_write_multiple_dataframes(self, tmp_path):
        # Create a DataFrameWriter instance
        writer = DataFrameWriter(tmp_path / "output.csv")

        # Create multiple DataFrames to write
        df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df2 = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})

        # Write the DataFrames asynchronously
        writer.write(df1)
        writer.write(df2)

        # Get the result (wait for the asynchronous task to complete)
        result = writer.result()

        # Check if the result matches the concatenated DataFrames
        expected_result = pd.concat([df1, df2], ignore_index=True)
        assert result.equals(expected_result)

        # Check if the output file exists
        assert Path(writer.path).exists()

        # Check if the output file contains the correct data
        output_df = pd.read_csv(writer.path)
        assert output_df.equals(expected_result)

    def test_write_empty_dataframe(self, tmp_path):
        # Create a DataFrameWriter instance
        writer = DataFrameWriter(tmp_path / "output.csv")

        # Create an empty DataFrame to write
        df = pd.DataFrame()

        # Write the DataFrame asynchronously
        writer.write(df)

        # Get the result (wait for the asynchronous task to complete)
        result = writer.result()

        # Check if the result matches the original DataFrame
        assert result.equals(df)

        # Check if the output file exists
        assert Path(writer.path).exists()

    def test_write_and_result_with_existing_file(self, tmp_path):
        # Create a DataFrameWriter instance
        output_path = tmp_path / "output.csv"
        writer = DataFrameWriter(output_path)

        # Create a DataFrame to write
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Write the DataFrame asynchronously
        writer.write(df)

        # Get the result (wait for the asynchronous task to complete)
        result = writer.result()

        # Check if the result matches the original DataFrame
        assert result.equals(df)

        # Check if the output file exists
        assert Path(writer.path).exists()

        # Check if the output file contains the correct data
        output_df = pd.read_csv(writer.path)
        assert output_df.equals(df)

        # Create a new DataFrame to write
        new_df = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})

        # Create new Writer instance with the same output file
        writer = DataFrameWriter(output_path)

        # Write the new DataFrame asynchronously
        writer.write(new_df)

        # Get the result (wait for the asynchronous task to complete)
        result = writer.result()

        # Check if the result matches the concatenated DataFrames
        expected_result = pd.concat([df, new_df], ignore_index=True)
        assert result.equals(expected_result)

        # Check if the output file contains the correct data
        output_df = pd.read_csv(writer.path)
        assert output_df.equals(expected_result)


class TestZMQ:
    def _publish_and_subscribe(self, packets, address, encoding=None):
        publisher = ZMQPublisher(address, encoding)

        def publish():
            for packet in packets:
                time.sleep(0.001)
                publisher.submit(packet)

        threading.Thread(target=publish).start()

        subscriber = ZMQSubscriber(address)
        result = []
        for n, packet in enumerate(subscriber, start=1):
            result.append(packet)
            if n == len(packets):
                break
        return xd.concatenate(result)

    def test_publish_and_subscribe(self):
        expected = xd.synthetics.dummy()
        packets = xd.split(expected, 10)
        address = f"tcp://localhost:{xd.io.get_free_port()}"

        result = self._publish_and_subscribe(packets, address)
        assert result.equals(expected)

    def test_encoding(self):
        expected = xd.synthetics.dummy()
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
                },
                "distance": distance,
            },
        )

        atom = lambda da, **kwargs: da.to_stream(
            network="NT",
            station="ST{:03}",
            channel="HN1",
            location="00",
            dim={"distance": "time"},
        )

        data_loader = DataArrayLoader(da, chunks={"time": 100})

        kw_merge = {"method": 1}
        kw_write = {"reclen": 4096}
        data_writer = StreamWriter(
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
                },
                "distance": 5.0 * np.arange(10),
            },
        )
        atom = lambda da, **kwargs: da.to_stream(
            network="NT",
            station="ST{:03}",
            channel="HN1",
            location="00",
            dim={"distance": "time"},
        )

        data_loader = DataArrayLoader(da, chunks={"time": 100})

        kw_merge = {"method": 1}
        kw_write = {"reclen": 4096}
        data_writer = StreamWriter(
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
                },
                "distance": distance,
            },
        )

        atom = lambda da, **kwargs: da.to_stream(
            network="NT",
            station="ST{:03}",
            channel="HN1",
            location="00",
            dim={"distance": "time"},
        )

        data_loader = DataArrayLoader(da, chunks={"time": 100})

        path = tmp_path / "flat_output.mseed"
        kw_merge = {"method": 1}
        kw_write = {"reclen": 4096}
        data_writer = StreamWriter(path, "M", kw_merge, kw_write, output_format="flat")

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
