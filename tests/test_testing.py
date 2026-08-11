import numpy as np
import pandas as pd
import pytest

import xdas as xd
from xdas.atoms import Partial
from xdas.testing import _assert_same


class TestDummy:
    def test_defaults(self):
        da = xd.testing.dummy()
        assert da.shape == (100, 10)
        assert da.dims == ("time", "distance")
        assert da["time"].isregular()
        assert da["distance"].isregular()

    def test_mismatched_shape(self):
        with pytest.raises(ValueError, match="must equal len\\(shape\\)"):
            xd.testing.dummy(dims=("time",), shape=(10, 10))

    def test_mismatched_step(self):
        with pytest.raises(ValueError, match="must equal len\\(dims\\)"):
            xd.testing.dummy(step=(1.0,))

    def test_datetime_step_passthrough(self):
        da = xd.testing.dummy(step=(np.timedelta64(10, "ms"), 10.0))
        assert da["time"].get_sampling_interval() == 0.01


def _shift_after_the_first_chunk(da):
    """Move the time axis by a nanosecond, but not on the first call."""
    seen = getattr(_shift_after_the_first_chunk, "seen", 0) + 1
    _shift_after_the_first_chunk.seen = seen
    if seen == 1:
        return da
    coords = dict(da.coords)
    coords["time"] = da["time"].values + np.timedelta64(1, "ns")
    return xd.DataArray(da.values, coords, da.dims)


class TestAssertChunkInvariant:
    def test_elementwise_pipeline_passes(self):
        da = xd.testing.dummy()
        xd.testing.assert_chunk_invariant(Partial(np.square), da, {"time": 25})

    def test_stateful_pipeline_passes(self):
        da = xd.testing.dummy()
        pipeline = xd.filter(..., (None, 10.0), dim="time")
        xd.testing.assert_chunk_invariant(pipeline, da, {"time": 25})

    def test_chunk_hostile_pipeline_fails(self):
        def normalize(da):
            return da / np.std(da.values)

        da = xd.testing.dummy()
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            xd.testing.assert_chunk_invariant(Partial(normalize), da, {"time": 25})

    def test_reports_a_shape_difference(self):
        def drop_last(da):
            return da.isel(time=slice(None, -1))

        da = xd.testing.dummy()
        with pytest.raises(AssertionError, match="shape differs"):
            xd.testing.assert_chunk_invariant(Partial(drop_last), da, {"time": 25})

    def test_reports_a_dims_difference(self):
        da = xd.testing.dummy()
        atom = Partial(lambda x: x.mean("time"))
        with pytest.raises(AssertionError, match="dims differ"):
            xd.testing.assert_chunk_invariant(atom, da, {"time": 25})

    def test_reports_a_coordinate_difference(self):
        da = xd.testing.dummy()
        _shift_after_the_first_chunk.seen = 0
        atom = Partial(_shift_after_the_first_chunk)
        with pytest.raises(AssertionError, match="coordinate differs"):
            xd.testing.assert_chunk_invariant(atom, da, {"time": 25})

    def test_coord_atol_admits_a_sub_sample_drift(self):
        da = xd.testing.dummy()
        _shift_after_the_first_chunk.seen = 0
        atom = Partial(_shift_after_the_first_chunk)
        xd.testing.assert_chunk_invariant(atom, da, {"time": 25}, coord_atol=2)

    def test_compares_sequences_of_chunks(self):
        left = [xd.testing.dummy(shape=(4, 2)), xd.testing.dummy(shape=(4, 2))]
        _assert_same(left, list(left), 1e-7, 0.0, 0, "result")

    def test_reports_a_chunk_count_difference(self):
        left = [xd.testing.dummy(shape=(4, 2)), xd.testing.dummy(shape=(4, 2))]
        with pytest.raises(AssertionError, match="2 chunks eager vs 1 chunked"):
            _assert_same(left, left[:1], 1e-7, 0.0, 0, "result")

    def test_reports_a_result_that_did_not_join(self):
        eager = xd.testing.dummy(shape=(4, 2))
        with pytest.raises(AssertionError, match="did not join into"):
            _assert_same(eager, [eager], 1e-7, 0.0, 0, "result")

    def test_compares_bare_values(self):
        _assert_same(1.0, 1.0, 1e-7, 0.0, 0, "result")

    def test_reports_a_table_that_is_not_a_table(self):
        frame = pd.DataFrame({"a": [1.0]})
        with pytest.raises(AssertionError, match="chunked gave a DataArray"):
            _assert_same(frame, xd.testing.dummy(shape=(2, 2)), 1e-7, 0.0, 0, "result")

    def test_reports_tables_whose_columns_differ(self):
        frame = pd.DataFrame({"a": [1.0]})
        with pytest.raises(AssertionError, match="columns differ"):
            _assert_same(frame, pd.DataFrame({"b": [1.0]}), 1e-7, 0.0, 0, "result")

    def test_compares_pick_tables_as_sets_of_rows(self):
        # Eager processing walks lane by lane, chunked walks chunk by chunk, so
        # the rows arrive in a different order with the same content.
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        _assert_same(frame, frame.iloc[::-1], 1e-7, 0.0, 0, "result")

    def test_gaps_exercise_the_seams(self):
        da = xd.testing.dummy()
        pipeline = xd.filter(..., (None, 10.0), dim="time")
        xd.testing.assert_chunk_invariant(pipeline, da, {"time": 25}, gaps=2)

    def test_gaps_at_explicit_positions(self):
        da = xd.testing.dummy()
        xd.testing.assert_chunk_invariant(
            Partial(np.square), da, {"time": 25}, gaps=[10, 60]
        )

    def test_gaps_shrink_an_oversized_chunking(self):
        da = xd.testing.dummy()
        # 100 samples minus two gaps leaves 90: the requested size must clamp.
        xd.testing.assert_chunk_invariant(Partial(np.square), da, {"time": 100}, gaps=2)

    def test_cut_invariance_catches_a_cut_sensitive_atom(self):
        def anchor_on_chunk_start(da):
            return da - da.values[0]

        da = xd.testing.dummy()
        atom = Partial(anchor_on_chunk_start)
        with pytest.raises(AssertionError):
            xd.testing.assert_chunk_invariant(atom, da, {"time": 100})

    def test_explicit_cuts(self):
        da = xd.testing.dummy()
        xd.testing.assert_chunk_invariant(
            Partial(np.square), da, {"time": 25}, cuts=[{"time": 13}, {"time": 7}]
        )

    def test_cuts_zero_restores_the_single_split(self):
        da = xd.testing.dummy()
        xd.testing.assert_chunk_invariant(Partial(np.square), da, {"time": 25}, cuts=0)

    def test_inject_gaps_makes_discontinuities(self):
        da = xd.testing.dummy()
        gappy = xd.testing.inject_gaps(da, "time", 2)
        assert gappy.sizes["time"] == 90
        coord = gappy["time"]
        indices = coord.get_split_indices("discontinuities", coord.tolerance)
        assert len(indices) == 2

    def test_inject_gaps_refuses_to_eat_the_record(self):
        da = xd.testing.dummy(shape=(4, 2))
        with pytest.raises(ValueError, match="less than two pieces"):
            xd.testing.inject_gaps(da, "time", [0])

    def test_a_bare_callable_is_not_a_pipeline(self):
        # It cannot be streamed at all — the chunked path hands each chunk a
        # `chunk_dim` keyword, which a plain function does not accept.
        da = xd.testing.dummy()
        with pytest.raises(AttributeError, match="reset"):
            xd.testing.assert_chunk_invariant(np.square, da, {"time": 25})
