"""
Run-semantics tests: seams, flush, collections and the commutation invariant.

The testable invariant of the continuous-run model (plan §5.5) is that
splitting anywhere commutes with processing::

    concat(atom(split(da, indices))) == atom(da)

for *arbitrary* split points, including at discontinuities — continuous
elements carry state across, discontinuous ones reset, and tails are flushed.
"""

import warnings

import numpy as np
import pytest

import xdas as xd
from xdas.atoms import (
    STFT,
    Atom,
    Decimate,
    DownSample,
    Filter,
    Integrate,
    Rechunk,
    Sequential,
    State,
)
from xdas.atoms.core import _aschunks
from xdas.testing import dummy


def collect(atom, chunks, dim="time"):
    """Fold *chunks* through *atom*, drain it, and return all output chunks."""
    outs = []
    for chunk in chunks:
        outs += _aschunks(atom(chunk, chunk_dim=dim))
    outs += atom.flush()
    atom.reset()
    return outs


def gappy(da, at=50, gap=10):
    """Split *da* in two runs separated by a gap of *gap* samples."""
    left = da.isel(time=slice(0, at))
    right = da.isel(time=slice(at + gap, None))
    return left, right, xd.concat([left, right], "time")


@pytest.fixture
def da():
    # 101 samples: an awkward length that exercises the flushed tails.
    return dummy(shape=(101, 5))


class TestCommutation:
    # Split points chosen to be awkward: single-sample chunks, unequal sizes.
    splits = [[37, 61], [1, 100], [13, 14, 50, 87], 7]

    factories = [
        lambda: DownSample(3, dim="time"),
        lambda: Rechunk({"time": 7}),
        lambda: Filter((1.0, 10.0)),
        lambda: Filter((None, 10.0), ftype="fir"),
        lambda: Decimate(25.0),
        lambda: STFT(0.16),  # expander: 16-sample windows, 8-sample hops
        lambda: Integrate(),
        lambda: Sequential([Decimate(25.0), Filter((1.0, 10.0)), np.square]),
    ]

    @pytest.mark.parametrize("split", splits)
    @pytest.mark.parametrize("factory", factories)
    def test_chunked_equals_eager(self, da, factory, split):
        expected = factory()(da)
        chunks = xd.split(da, split, "time")
        result = xd.concat(collect(factory(), chunks), "time")
        assert result.coords.equals(expected.coords)
        assert np.allclose(result.values, expected.values, atol=1e-15, rtol=1e-9)

    @pytest.mark.parametrize("factory", factories)
    def test_collection_input_folds(self, da, factory):
        expected = factory()(da)
        collection = factory()(xd.split(da, [13, 50, 87], "time"))
        assert isinstance(collection, xd.DataSequence)
        result = xd.concat(list(collection), "time")
        assert np.allclose(result.values, expected.values, atol=1e-15, rtol=1e-9)


class TestFlush:
    def test_default_is_noop(self):
        assert Atom().flush() == []

    def test_downsample_emits_tail(self, da):
        # 101 % 3 != 0: the strided remainder starts on an output sample.
        atom = DownSample(3, dim="time")
        expected = atom(da)
        assert expected.sizes["time"] == 34
        outs = collect(atom, xd.split(da, 4, "time"))
        assert xd.concat(outs, "time").equals(expected)

    def test_flush_empties_the_buffer(self, da):
        atom = DownSample(3, dim="time")
        atom(da.isel(time=slice(0, 50)), chunk_dim="time")
        assert len(atom.flush()) == 1
        assert atom.flush() == []

    def test_eager_call_is_complete(self, da):
        # Single-call prototyping returns the full output: nothing left over.
        atom = DownSample(3, dim="time")
        atom(da)
        assert atom.flush() == []


class TestSeams:
    def test_gap_resets_state(self, da):
        # Integrate carries a cumulative offset: carrying it across the gap
        # would corrupt the second run.
        left, right, _ = gappy(da)
        atom = Integrate()
        expected = [Integrate()(left), Integrate()(right)]
        result = collect(atom, [left, right])
        assert len(result) == 2
        for out, exp in zip(result, expected):
            assert out.equals(exp)

    def test_gap_flushes_the_tail(self, da):
        # The seam call returns the flushed tail of the old run before the
        # fresh output of the new one.
        left, right, _ = gappy(da, at=50)  # 50 % 3 != 0: pending remainder
        atom = DownSample(3, dim="time")
        outs = _aschunks(atom(left, chunk_dim="time"))
        outs += _aschunks(atom(right, chunk_dim="time"))
        outs += atom.flush()
        expected = xd.concat(
            [DownSample(3, dim="time")(run) for run in (left, right)], "time"
        )
        assert xd.concat(outs, "time").equals(expected)

    def test_rate_change_redesigns(self):
        # Same filter atom, stream whose rate halves mid-way: each run must be
        # filtered with coefficients designed for its own rate.
        left = dummy(shape=(100, 5), step=(0.01, 10.0))
        start = left["time"].end + 5 * left["time"].sampling_interval
        right = dummy(shape=(50, 5), step=(0.02, 10.0))
        right["time"] += start - right["time"].start
        atom = Filter((1.0, 10.0))
        result = collect(atom, [left, right])
        expected = [Filter((1.0, 10.0))(left), Filter((1.0, 10.0))(right)]
        assert len(result) == 2
        for out, exp in zip(result, expected):
            assert np.allclose(out.values, exp.values)

    def test_overlap_raises(self, da):
        atom = Filter((1.0, 10.0))
        atom(da.isel(time=slice(0, 50)), chunk_dim="time")
        with pytest.raises(ValueError, match="overlap"):
            atom(da.isel(time=slice(40, 80)), chunk_dim="time")

    def test_on_discontinuity_raise(self, da):
        left, right, _ = gappy(da)
        atom = Filter((1.0, 10.0))
        atom.on_discontinuity = "raise"
        atom(left, chunk_dim="time")
        with pytest.raises(ValueError, match="discontinuous"):
            atom(right, chunk_dim="time")

    def test_invalid_policy(self, da):
        left, right, _ = gappy(da)
        atom = Filter((1.0, 10.0))
        atom.on_discontinuity = "ignore"
        atom(left, chunk_dim="time")
        with pytest.raises(ValueError, match="on_discontinuity"):
            atom(right, chunk_dim="time")

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_irregular_coordinate_raises(self):
        coords = {
            "time": {"tie_indices": [0, 49], "tie_values": [0.0, 49.0]},
            "distance": {"tie_indices": [0, 4], "tie_values": [0.0, 40.0]},
        }
        da = xd.DataArray(np.random.randn(50, 5), coords)
        atom = Integrate()
        atom(da, chunk_dim="time")
        with pytest.raises(ValueError, match="regular"):
            atom(da, chunk_dim="time")


class TestEagerRuns:
    def test_gappy_input_splits_into_runs(self, da):
        # Filters never cross discontinuities: eager on a gappy record equals
        # per-run processing, re-joined with the gap kept in the coords.
        left, right, joined = gappy(da)
        atom = Filter((1.0, 10.0))
        result = atom(joined)
        assert isinstance(result, xd.DataArray)
        expected = xd.concat(
            [Filter((1.0, 10.0))(run) for run in (left, right)], "time"
        )
        assert np.allclose(result.values, expected.values)
        assert result["time"].get_split_indices().size == 1

    def test_gapless_input_unchanged(self, da):
        assert xd.filter(da, (1.0, 10.0)).sizes["time"] == da.sizes["time"]

    def test_chunked_internal_gap_splits(self, da):
        # The ingress invariant: an internally gappy chunk is split into runs
        # before the seam-aware call, so state never crosses a gap.
        left, right, joined = gappy(da)
        atom = Integrate()
        result = _aschunks(atom(joined, chunk_dim="time"))
        expected = [Integrate()(left), Integrate()(right)]
        assert len(result) == 2
        for out, exp in zip(result, expected):
            assert out.equals(exp)


class TestSplitAnnouncement:
    def test_eager_call_announces_the_split_count(self, da):
        left = da.isel(time=slice(0, 30))
        mid = da.isel(time=slice(40, 60))
        right = da.isel(time=slice(70, None))
        joined = xd.concat([left, mid, right], "time")
        with pytest.warns(UserWarning, match="2 discontinuities along 'time'"):
            xd.filter(joined, (1.0, 10.0))

    def test_singular_wording(self, da):
        _, _, joined = gappy(da)
        with pytest.warns(UserWarning, match="1 discontinuity along 'time'"):
            xd.filter(joined, (1.0, 10.0))

    def test_gapless_input_is_silent(self, da):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            xd.filter(da, (1.0, 10.0))

    def test_every_leaf_of_a_collection_reports(self, da):
        # The message names the source by its start, so two leaves with the
        # same gap count are not deduplicated into one warning.
        _, _, joined = gappy(da)
        other = joined.copy()
        other["time"] = other["time"] + np.timedelta64(1, "h")
        collection = xd.DataCollection({"das1": joined, "das2": other})
        with pytest.warns(UserWarning) as record:
            xd.filter(collection, (1.0, 10.0))
        messages = [str(w.message) for w in record if "discontinuit" in str(w.message)]
        assert len(messages) == 2
        assert messages[0] != messages[1]


class TestCollections:
    def test_mapping_maps_over_leaves(self, da):
        collection = xd.DataCollection({"das1": da, "das2": da})
        result = xd.filter(collection, (1.0, 10.0))
        expected = xd.filter(da, (1.0, 10.0))
        assert result["das1"].equals(expected)
        assert result["das2"].equals(expected)

    def test_chunked_mapping_raises(self, da):
        collection = xd.DataCollection({"das1": da})
        with pytest.raises(NotImplementedError):
            xd.filter(..., (1.0, 10.0))(collection, chunk_dim="time")

    def test_sequence_with_gap(self, da):
        # Continuous elements carry state across, discontinuous ones reset.
        left, right, _ = gappy(da)
        elements = xd.split(left, 2, "time") + xd.split(right, 2, "time")
        atom = Integrate()
        collection = atom(xd.DataCollection(elements))
        result = xd.concat(list(collection), "time")
        expected = xd.concat([Integrate()(left), Integrate()(right)], "time")
        assert np.allclose(result.values, expected.values)


class TestIterChunks:
    def test_matches_eager(self, da):
        pipeline = Sequential([Decimate(25.0), Filter((1.0, 10.0))])
        outs = list(pipeline.iter_chunks(xd.split(da, 5, "time")))
        expected = Sequential([Decimate(25.0), Filter((1.0, 10.0))])(da)
        result = xd.concat(outs, "time")
        assert np.allclose(result.values, expected.values, atol=1e-15, rtol=1e-9)

    def test_explicit_chunk_dim(self, da):
        atom = DownSample(2, dim="distance")
        outs = list(atom.iter_chunks(xd.split(da, 2, "time"), chunk_dim="time"))
        assert xd.concat(outs, "time").equals(DownSample(2, dim="distance")(da))

    def test_resets_at_the_end(self, da):
        atom = Integrate()
        list(atom.iter_chunks(xd.split(da, 3, "time")))
        assert not atom.initialized


class StreamMean(Atom):
    """Test reduction: accumulate per chunk, emit the single result at flush."""

    def __init__(self, dim="time"):
        super().__init__()
        self.dim = dim
        self.numerator = State(...)
        self.denominator = State(...)

    def initialize(self, da, chunk_dim=None, **flags):
        if chunk_dim == self.dim:
            self.numerator = State(0.0 * da.sum(self.dim))
            self.denominator = State(0)
        else:
            self.numerator = State(None)
            self.denominator = State(None)

    def call(self, da, **flags):
        if self.numerator is None:
            return da.mean(self.dim)
        self.numerator = State(self.numerator + da.sum(self.dim))
        self.denominator = State(self.denominator + da.sizes[self.dim])
        return None

    def flush(self):
        if not isinstance(self.numerator, xd.DataArray):
            return []
        out = self.numerator / self.denominator
        self.numerator = State(0.0 * self.numerator)
        self.denominator = State(0)
        return [out]


class TestSTFTRuns:
    def test_frames_never_span_gaps(self, da):
        left, right, both = gappy(da)
        streamed = collect(STFT(0.16), [left, right])
        expected = [STFT(0.16)(left), STFT(0.16)(right)]
        assert len(streamed) == 2
        for out, exp in zip(streamed, expected):
            assert out.coords.equals(exp.coords)
            assert np.allclose(out.values, exp.values)
        # eager on the gappy record splits into the same per-run frames
        eager = STFT(0.16)(both)
        assert np.allclose(xd.concat(streamed, "time").values, eager.values)

    def test_short_run_emits_nothing_when_streaming(self, da):
        # 64-sample windows never fit in a 50-sample run: the buffered tail
        # is dropped at flush, nothing is emitted and nothing raises.
        outs = collect(STFT(0.64), [da.isel(time=slice(0, 50))])
        assert outs == []

    def test_eager_short_record_raises(self, da):
        with pytest.raises(ValueError, match="shorter"):
            STFT(1.28)(da)


class TestReduction:
    def test_streaming_equals_eager(self, da):
        expected = StreamMean()(da)
        outs = collect(StreamMean(), xd.split(da, [37, 61], "time"))
        assert len(outs) == 1
        assert np.allclose(outs[0].values, expected.values)

    def test_call_returns_no_chunk(self, da):
        atom = StreamMean()
        out = atom(da.isel(time=slice(0, 50)), chunk_dim="time")
        assert list(out) == []


class TestRechunk:
    def test_sizes(self, da):
        atom = Rechunk({"time": 30})
        outs = collect(atom, xd.split(da, 7, "time"))
        assert [out.sizes["time"] for out in outs] == [30, 30, 30, 11]

    def test_never_merges_across_gaps(self, da):
        left, right, _ = gappy(da)
        atom = Rechunk({"time": 40})
        outs = collect(atom, [left, right])
        # 50-sample run then 41-sample run: the partial buffer is flushed at
        # the seam instead of being merged with the next run.
        assert [out.sizes["time"] for out in outs] == [40, 10, 40, 1]
        for out in outs:
            assert out["time"].get_split_indices().size == 0

    def test_eager_is_identity(self, da):
        assert xd.rechunk(da, {"time": 30}).equals(da)

    def test_twin_seed(self):
        atom = xd.rechunk(..., {"time": 30})
        assert isinstance(atom, Rechunk)

    def test_invalid_chunks(self):
        with pytest.raises(TypeError):
            Rechunk({"time": 10, "distance": 2})
        with pytest.raises(ValueError):
            Rechunk({"time": 0})


class TestProcess:
    def test_end_of_stream_flush(self, da, tmp_path):
        # The DownSample tail-drop bug: process() must drain the atom.
        import xdas.processing as xp

        atom = DownSample(3, dim="time")
        expected = DownSample(3, dim="time")(da)
        loader = xp.DataArrayLoader(da, {"time": 25})
        writer = xp.DataArrayWriter(tmp_path)
        result = xp.process(atom, loader, writer)
        assert result.equals(expected)
