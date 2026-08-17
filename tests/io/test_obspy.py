import numpy as np
import numpy.testing as npt
import obspy
import pytest

import xdas as xd
from xdas.coordinates import Coordinate
from xdas.io.obspy import ObsPyEngine, get_band_code, to_stream
from xdas.virtual import TileArray


def header(station="CH001", channel="HHZ", starttime=0.0, delta=0.01, location="00"):
    return {
        "delta": delta,
        "starttime": obspy.UTCDateTime(starttime),
        "network": "DX",
        "station": station,
        "location": location,
        "channel": channel,
    }


def write(path, traces, **kwargs):
    st = obspy.Stream(
        [obspy.Trace(np.asarray(data, dtype=np.float64), head) for head, data in traces]
    )
    st.write(str(path), format="MSEED", **kwargs)
    return st


class TestScan:
    def test_one_trace_per_obspy_trace(self, tmp_path):
        path = tmp_path / "three.mseed"
        write(
            path,
            [
                (header(channel=f"HH{component}"), np.random.rand(100))
                for component in "ZNE"
            ],
        )
        dc = ObsPyEngine().open_datacollection(path)
        assert dc.fields == ("network", "station", "location", "channel", "record")
        assert list(dc) == ["DX"]
        assert list(dc["DX"]) == ["CH001"]
        assert list(dc["DX"]["CH001"]) == ["00"]
        assert sorted(dc["DX"]["CH001"]["00"]) == ["HHE", "HHN", "HHZ"]
        da = dc["DX"]["CH001"]["00"]["HHZ"][0]
        assert da.dims == ("time",)
        assert da.shape == (100,)
        assert da["network"].values == "DX"
        assert da["station"].values == "CH001"
        assert da["location"].values == "00"
        assert da["channel"].values == "HHZ"
        assert da["time"][0].values == np.datetime64("1970-01-01T00:00:00")
        assert da["time"][-1].values == np.datetime64("1970-01-01T00:00:00.990")

    def test_gaps_become_separate_traces(self, tmp_path):
        path = tmp_path / "gap.mseed"
        write(
            path,
            [
                (header(), np.random.rand(50)),
                (header(starttime=1.0), np.random.rand(40)),
            ],
        )
        dc = ObsPyEngine().open_datacollection(path)
        traces = dc["DX"]["CH001"]["00"]["HHZ"]
        assert len(traces) == 2
        assert [da.sizes["time"] for da in traces] == [50, 40]

    def test_mixed_sampling_rates_are_read(self, tmp_path):
        # the old engine refused these outright
        path = tmp_path / "mixed.mseed"
        write(
            path,
            [
                (header(channel="HHZ", delta=0.01), np.random.rand(100)),
                (header(channel="LHZ", delta=1.0), np.random.rand(10)),
            ],
        )
        dc = ObsPyEngine().open_datacollection(path)
        channels = dc["DX"]["CH001"]["00"]
        assert sorted(channels) == ["HHZ", "LHZ"]
        assert channels["HHZ"][0].sizes["time"] == 100
        assert channels["LHZ"][0].sizes["time"] == 10

    def test_duplicated_id_groups_into_one_sequence(self, tmp_path):
        path = tmp_path / "dup.mseed"
        write(
            path,
            [
                (header(), np.random.rand(50)),
                (header(starttime=10.0), np.random.rand(50)),
                (header(starttime=20.0), np.random.rand(50)),
            ],
        )
        dc = ObsPyEngine().open_datacollection(path)
        assert len(dc["DX"]["CH001"]["00"]["HHZ"]) == 3

    def test_ctype_drives_the_time_coordinate(self, tmp_path):
        path = tmp_path / "one.mseed"
        write(path, [(header(), np.random.rand(100))])
        dc = ObsPyEngine(ctype="dense").open_datacollection(path)
        da = dc["DX"]["CH001"]["00"]["HHZ"][0]
        assert isinstance(da["time"], Coordinate["dense"])

    def test_traces_sharing_everything_are_refused(self, tmp_path):
        path = tmp_path / "twins.mseed"
        write(path, [(header(), np.zeros(50)), (header(), np.ones(50))])
        with pytest.raises(ValueError, match="nothing content-free separates"):
            ObsPyEngine().open_datacollection(path)

    def test_dtype_comes_from_the_encoding(self, tmp_path):
        # headonly leaves `tr.data` an empty float64 array whatever the file
        # holds, so a STEIM-compressed file would scan with the wrong dtype
        path = tmp_path / "steim.mseed"
        st = obspy.Stream([obspy.Trace(np.arange(100, dtype=np.int32), header())])
        st.write(str(path), format="MSEED", encoding="STEIM2")
        dc = ObsPyEngine().open_datacollection(path)
        da = dc["DX"]["CH001"]["00"]["HHZ"][0]
        assert da.dtype == np.int32
        npt.assert_array_equal(da.values, np.arange(100))

    def test_open_dataarray_needs_a_single_trace(self, tmp_path):
        path = tmp_path / "one.mseed"
        write(path, [(header(), np.random.rand(50))])
        da = ObsPyEngine().open_dataarray(path)
        assert da.dims == ("time",)
        assert da.shape == (50,)

        path = tmp_path / "two.mseed"
        write(
            path,
            [
                (header(channel="HHZ"), np.zeros(50)),
                (header(channel="HHN"), np.ones(50)),
            ],
        )
        with pytest.raises(ValueError, match="holds 2 traces"):
            ObsPyEngine().open_dataarray(path)


class TestBlankLocation:
    def test_round_trips_through_netcdf_and_to_stream(self, tmp_path):
        path = tmp_path / "blank.mseed"
        write(path, [(header(location=""), np.random.rand(50))])
        dc = ObsPyEngine().open_datacollection(path)
        # "" cannot be a netCDF group name; "--" is the FDSN convention
        assert list(dc["DX"]["CH001"]) == ["--"]
        da = dc["DX"]["CH001"]["--"]["HHZ"][0]
        assert da["location"].values == "--"
        npt.assert_allclose(da.values, obspy.read(str(path))[0].data)

        dc.to_netcdf(tmp_path / "blank.nc")
        reopened = xd.open_datacollection(tmp_path / "blank.nc")
        assert list(reopened["DX"]["CH001"]) == ["--"]

        stacked = xd.DataArray(
            da.values[None],
            {"space": [0.0], "time": da["time"]},
        )
        dim = {"space": "time"}
        assert to_stream(stacked, location="--", dim=dim)[0].stats.location == ""
        assert to_stream(stacked, location="00", dim=dim)[0].stats.location == "00"


class TestLoadTile:
    def test_pointer_survives_resegmentation(self, tmp_path):
        # the same samples written with different record boundaries: the trace
        # count and the segmentation differ, the pointer must not
        data = np.random.rand(400)
        coarse = tmp_path / "coarse.mseed"
        write(coarse, [(header(), data)], reclen=4096)
        fine = tmp_path / "fine.mseed"
        write(fine, [(header(), data)], reclen=512)
        assert len(obspy.read(str(coarse))) == len(obspy.read(str(fine)))
        for path in (coarse, fine):
            da = ObsPyEngine().open_dataarray(path)
            npt.assert_allclose(da.values, data)
            npt.assert_allclose(da.isel(time=slice(10, 30)).values, data[10:30])

    def test_split_records_still_resolve(self, tmp_path):
        # two files holding the same span, one written as a single trace and
        # one as two abutting traces that `join_contiguous` must fuse back
        data = np.random.rand(200)
        whole = tmp_path / "whole.mseed"
        write(whole, [(header(), data)])
        split = tmp_path / "split.mseed"
        write(
            split,
            [(header(), data[:120]), (header(starttime=1.20), data[120:])],
        )
        # the split file scans as one trace: obspy already rejoins abutting
        # records of the same channel
        da = ObsPyEngine().open_dataarray(split)
        npt.assert_allclose(da.values, data)
        assert ObsPyEngine().open_dataarray(whole).equals(da)

    def test_traces_split_by_data_quality_are_rejoined(self, tmp_path):
        # libmseed hands back one trace per data quality flag; the two are
        # sample-exact contiguous, so each pointer must still resolve against
        # the joined run
        data = np.arange(200.0)
        parts = []
        for index, (values, start, quality) in enumerate(
            [(data[:120], 0.0, "D"), (data[120:], 1.20, "R")]
        ):
            tr = obspy.Trace(values, header(starttime=start))
            tr.stats.mseed = {"dataquality": quality}
            path = tmp_path / f"part_{index}.mseed"
            obspy.Stream([tr]).write(str(path), format="MSEED")
            parts.append(path)
        path = tmp_path / "quality.mseed"
        path.write_bytes(b"".join(part.read_bytes() for part in parts))
        assert len(obspy.read(str(path))) == 2

        dc = ObsPyEngine().open_datacollection(path)
        traces = dc["DX"]["CH001"]["00"]["HHZ"]
        assert [da.sizes["time"] for da in traces] == [120, 80]
        npt.assert_allclose(traces[0].values, data[:120])
        npt.assert_allclose(traces[1].values, data[120:])

    def test_legacy_unsynchronized_manifest_trimmed_and_squeezed(self, tmp_path):
        # the old engine folded a single-channel axis out of the scanned shape
        # and could drop the last sample of each segment
        path = tmp_path / "gap.mseed"
        st = write(
            path,
            [
                (header(), np.arange(50.0)),
                (header(starttime=1.0), np.arange(100.0, 150.0)),
            ],
        )
        data = TileArray.from_tiles(
            str(path),
            (99,),
            np.dtype("float64"),
            {
                "name": "miniseed",
                "method": "unsynchronized",
                "ignore_last_sample": True,
            },
        )
        expected = np.concatenate([tr.data for tr in st])
        npt.assert_allclose(np.asarray(data), np.delete(expected, -1))

    def test_same_start_different_length(self, tmp_path):
        path = tmp_path / "twins.mseed"
        short = np.arange(10.0)
        long = np.arange(100.0, 130.0)
        write(path, [(header(), short), (header(), long)])
        dc = ObsPyEngine().open_datacollection(path)
        traces = dc["DX"]["CH001"]["00"]["HHZ"]
        assert len(traces) == 2
        assert sorted(da.sizes["time"] for da in traces) == [10, 30]
        for da in traces:
            expected = short if da.sizes["time"] == 10 else long
            npt.assert_allclose(da.values, expected)

    def test_missing_run_raises(self, tmp_path):
        path = tmp_path / "one.mseed"
        write(path, [(header(), np.random.rand(50))])
        da = ObsPyEngine().open_dataarray(path)
        # point at a channel the file does not hold
        with pytest.raises(ValueError, match="0 contiguous runs"):
            ObsPyEngine.load_tile(
                str(path),
                (slice(0, 50),),
                network="DX",
                station="CH001",
                location="00",
                channel="HHN",
                starttime=int(da["time"][0].values.astype("datetime64[ns]").view("i8")),
                endtime=int(da["time"][-1].values.astype("datetime64[ns]").view("i8")),
            )


class TestLegacy:
    def test_the_legacy_engine_is_separate(self):
        from xdas.io import Engine
        from xdas.io.miniseed import MiniSEEDEngine

        # `engine="miniseed"` names the engine this one replaced, kept for the
        # views it wrote; see tests/io/test_miniseed.py
        assert Engine["miniseed"] is MiniSEEDEngine
        assert Engine["obspy"] is ObsPyEngine


class TestOpenRouting:
    def make_network(self, dirpath, stations=3, channels="ZNE", chunks=2):
        samples = 50
        for index in range(1, stations + 1):
            for chunk in range(chunks):
                traces = [
                    (
                        header(
                            station=f"CH{index:03d}",
                            channel=f"HH{component}",
                            starttime=chunk * samples * 0.01,
                        ),
                        np.random.rand(samples),
                    )
                    for component in channels
                ]
                write(dirpath / f"CH{index:03d}_{chunk}.mseed", traces)

    def test_one_file_glob_and_list_agree(self, tmp_path):
        self.make_network(tmp_path, stations=1, chunks=1)
        paths = sorted(str(path) for path in tmp_path.glob("*.mseed"))
        one = xd.open(paths[0], engine="obspy")
        glob = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        listed = xd.open(paths, engine="obspy")
        for dc in (one, glob, listed):
            assert dc.fields == (
                "network",
                "station",
                "location",
                "channel",
                "record",
            )
        assert glob.equals(listed)
        assert one.equals(listed)

    def test_contiguous_files_fuse_into_one_array(self, tmp_path):
        self.make_network(tmp_path, stations=1, channels="Z", chunks=3)
        dc = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        sequence = dc["DX"]["CH001"]["00"]["HHZ"]
        assert len(sequence) == 1
        da = sequence[0]
        assert da.sizes["time"] == 150
        assert isinstance(da.data, TileArray)

    def test_gaps_live_in_the_coordinate(self, tmp_path):
        path = tmp_path / "gap.mseed"
        write(
            path,
            [
                (header(), np.random.rand(50)),
                (header(starttime=10.0), np.random.rand(40)),
            ],
        )
        dc = xd.open(path, engine="obspy")
        da = dc["DX"]["CH001"]["00"]["HHZ"][0]
        assert da.sizes["time"] == 90
        parts = xd.split(da, "gaps")
        assert [part.sizes["time"] for part in parts] == [50, 40]

    def test_rate_change_arrives_as_two_elements(self, tmp_path):
        for index, delta in enumerate([0.01, 0.02]):
            write(
                tmp_path / f"chunk_{index}.mseed",
                [(header(starttime=index * 10.0, delta=delta), np.random.rand(50))],
            )
        dc = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        sequence = dc["DX"]["CH001"]["00"]["HHZ"]
        assert len(sequence) == 2
        assert sequence.name == "record"

    def test_select_globs_like_obspy(self, tmp_path):
        self.make_network(tmp_path, stations=3, chunks=1)
        dc = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        result = dc.select(station="CH00[12]", channel="HH?")
        assert sorted(result["DX"]) == ["CH001", "CH002"]
        assert sorted(result["DX"]["CH001"]["00"]) == ["HHE", "HHN", "HHZ"]
        assert dc.query(station="CH001").equals(dc.select(station="CH001"))

    def test_concat_along_channel_folds_the_other_columns(self, tmp_path):
        self.make_network(tmp_path, stations=1, chunks=1)
        dc = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        channels = dc["DX"]["CH001"]["00"]
        da = xd.concat([channels[key][0] for key in sorted(channels)], "channel")
        assert da.dims == ("channel", "time")
        assert da.shape == (3, 50)
        assert da["channel"].values.tolist() == ["HHE", "HHN", "HHZ"]
        assert isinstance(da.data, TileArray)
        # only the varying column unfolds; the other three stay 0-d
        manifest = da.data.to_dataset()
        assert manifest["channel"].ndim == 1
        for column in ("network", "station", "location"):
            assert manifest[column].ndim == 0
        npt.assert_allclose(
            da.values,
            np.stack([channels[key][0].values for key in sorted(channels)]),
        )

    def test_netcdf_round_trip_of_a_per_trace_view(self, tmp_path):
        self.make_network(tmp_path, stations=1, chunks=2)
        dc = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        dc.to_netcdf(tmp_path / "view.nc")
        reopened = xd.open_datacollection(tmp_path / "view.nc")
        da = reopened["DX"]["CH001"]["00"]["HHZ"][0]
        assert isinstance(da.data, TileArray)
        npt.assert_allclose(da.values, dc["DX"]["CH001"]["00"]["HHZ"][0].values)

    def test_auto_detection(self, tmp_path):
        # naming the engine is not required: `AutoEngine` reaches
        # `open_datacollection` too, so the shape does not depend on it
        self.make_network(tmp_path, stations=1, chunks=2)
        named = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        assert xd.open(str(tmp_path / "*.mseed")).equals(named)
        one = xd.open(str(tmp_path / "CH001_0.mseed"))
        assert one.fields == named.fields

    def test_parallel_scan(self, tmp_path):
        self.make_network(tmp_path, stations=2, chunks=2)
        serial = xd.open(str(tmp_path / "*.mseed"), engine="obspy", parallel=1)
        parallel = xd.open(str(tmp_path / "*.mseed"), engine="obspy", parallel=2)
        assert parallel.equals(serial)

    def test_sac_opens_through_the_same_path(self, tmp_path):
        path = tmp_path / "trace.sac"
        data = np.arange(50, dtype=np.float32)
        obspy.Trace(data, header()).write(str(path), format="SAC")
        dc = xd.open(path, engine="obspy")
        da = dc["DX"]["CH001"]["00"]["HHZ"][0]
        assert da.sizes["time"] == 50
        assert da.dtype == np.float32
        npt.assert_allclose(da.values, data)


class TestHelpers:
    def test_get_band_code_out_of_range(self):
        assert get_band_code(0.0) == "X"
        assert get_band_code(6000.0) == "X"

    def test_to_stream_requires_2d(self):
        da = xd.DataArray(np.zeros((2, 3, 4)), dims=("a", "b", "c"))
        with pytest.raises(ValueError, match="2D"):
            to_stream(da)

    def test_stream_round_trip(self, tmp_path):
        TestOpenRouting().make_network(tmp_path, stations=1, chunks=1)
        dc = xd.open(str(tmp_path / "*.mseed"), engine="obspy")
        channels = dc["DX"]["CH001"]["00"]
        da = xd.concat([channels[key][0] for key in sorted(channels)], "channel")
        st = da.to_stream(dim={"channel": "time"})
        assert len(st) == 3
        result = xd.DataArray.from_stream(st)
        npt.assert_allclose(result.values, da.values)
