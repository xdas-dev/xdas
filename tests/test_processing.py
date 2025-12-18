import os
import tempfile
import threading
import time
from glob import glob

import hdf5plugin
import numpy as np
import obspy
import pandas as pd
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


class TestProcessing:
    def test_stateful(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # generate test dataarray
            wavelet_wavefronts().to_netcdf(os.path.join(tempdir, "sample.nc"))
            da = xd.open_dataarray(os.path.join(tempdir, "sample.nc"))

            # declare processing sequence
            sos = sp.iirfilter(4, 0.1, btype="lowpass", output="sos")
            sequence = Sequential([Partial(sosfilt, sos, ..., dim="time", zi=...)])

            # monolithic processing
            result1 = sequence(da)

            # chunked processing
            data_loader = DataArrayLoader(da, chunks={"time": 100})
            data_writer = DataArrayWriter(tempdir)
            result2 = process(
                sequence, data_loader, data_writer
            )  # resets the sequence by default

            # test
            assert result1.equals(result2)


class TestDataFrameWriter:
    def test_write_and_result(self):
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a DataFrameWriter instance
            writer = DataFrameWriter(os.path.join(tmp_dir, "output.csv"))

            # Create a DataFrame to write
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

            # Write the DataFrame asynchronously
            writer.write(df)

            # Get the result (wait for the asynchronous task to complete)
            result = writer.result()

            # Check if the result matches the original DataFrame
            assert result.equals(df)

            # Check if the output file exists
            assert os.path.exists(writer.path)

            # Check if the output file contains the correct data
            output_df = pd.read_csv(writer.path)
            assert output_df.equals(df)

    def test_write_multiple_dataframes(self):
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a DataFrameWriter instance
            writer = DataFrameWriter(os.path.join(tmp_dir, "output.csv"))

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
            assert os.path.exists(writer.path)

            # Check if the output file contains the correct data
            output_df = pd.read_csv(writer.path)
            assert output_df.equals(expected_result)

    def test_write_empty_dataframe(self):
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a DataFrameWriter instance
            writer = DataFrameWriter(os.path.join(tmp_dir, "output.csv"))

            # Create an empty DataFrame to write
            df = pd.DataFrame()

            # Write the DataFrame asynchronously
            writer.write(df)

            # Get the result (wait for the asynchronous task to complete)
            result = writer.result()

            # Check if the result matches the original DataFrame
            assert result.equals(df)

            # Check if the output file exists
            assert os.path.exists(writer.path)

    def test_write_and_result_with_existing_file(self):
        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a DataFrameWriter instance
            writer = DataFrameWriter(os.path.join(tmp_dir, "output.csv"))

            # Create a DataFrame to write
            df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

            # Write the DataFrame asynchronously
            writer.write(df)

            # Get the result (wait for the asynchronous task to complete)
            result = writer.result()

            # Check if the result matches the original DataFrame
            assert result.equals(df)

            # Check if the output file exists
            assert os.path.exists(writer.path)

            # Check if the output file contains the correct data
            output_df = pd.read_csv(writer.path)
            assert output_df.equals(df)

            # Create a new DataFrame to write
            new_df = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})

            # Create new Writer instance with the same output file
            writer = DataFrameWriter(os.path.join(tmp_dir, "output.csv"))

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
    def test_without_gap(self):
        with tempfile.TemporaryDirectory() as tempdir:
            data = np.random.randint(
                low=-1000, high=1000, size=(1000, 10), dtype=np.int32
            )
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
                tempdir, "M", kw_merge, kw_write, output_format="SDS"
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
            path = os.path.join(
                tempdir,
                "2023",
                "NT",
                "ST001",
                "HN1.D",
                "NT.ST001.00.HN1.D.2023.001",
            )
            assert os.path.exists(path)
            st = obspy.read(path)
            assert len(st) == 1
            assert len(glob(os.path.join(tempdir, "**", "*.001"), recursive=True)) == 10

    def test_with_gap(self):
        with tempfile.TemporaryDirectory() as tempdir:
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
                tempdir, "M", kw_merge, kw_write, output_format="SDS"
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
            path = os.path.join(
                tempdir,
                "2023",
                "NT",
                "ST001",
                "HN1.D",
                "NT.ST001.00.HN1.D.2023.001",
            )
            assert os.path.exists(path)
            st = obspy.read(path)
            assert len(st) == 2
            assert len(glob(os.path.join(tempdir, "**", "*.001"), recursive=True)) == 10

    def test_flat(self):
        with tempfile.TemporaryDirectory() as tempdir:
            data = np.random.randint(
                low=-1000, high=1000, size=(1000, 10), dtype=np.int32
            )
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

            path = os.path.join(tempdir, "flat_output.mseed")
            kw_merge = {"method": 1}
            kw_write = {"reclen": 4096}
            data_writer = StreamWriter(
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
            assert os.path.exists(path)
            st = obspy.read(path)
            assert len(st) == 10
