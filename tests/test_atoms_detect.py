"""
Tests for the detection atoms of `xdas.atoms.detect`.

`Trigger` joined the public layer with the rest of the task vocabulary, so it
now defaults to `dim="time"` and has a lowercase twin, `xd.trigger`.
"""

import numpy as np
import pandas as pd
import pytest

import xdas as xd
from xdas.atoms import Trigger


def generate():
    return xd.DataArray(
        data=[[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2]],
        coords={
            "space": [0.0],
            "time": {
                "tie_indices": [0, 9],
                "tie_values": [0.0, 9.0],
                "sampling_interval": 1.0,
            },
        },
    )


expected = pd.DataFrame({"space": [0.0, 0.0], "time": [2.0, 7.0], "value": [0.9, 0.7]})


def test_trigger():
    cft = generate()

    # test monolithic processing
    picks = Trigger(thresh=0.5, dim="time")(cft)
    assert picks.equals(expected)

    # test chunked processing
    atom = Trigger(thresh=0.5, dim="time")
    chunks = xd.split(cft, 3, dim="time")
    result = []
    for chunk in chunks:
        picks = atom(chunk, chunk_dim="time")
        result.append(picks)
    result = pd.concat(result, ignore_index=True)
    assert result.equals(expected)


class TestTwin:
    def test_is_a_function_and_the_module_still_imports(self):
        # `xdas.trigger` stays importable as the compat module, but the
        # attribute is the lowercase twin and importing the module does not
        # shadow it.
        assert callable(xd.trigger)
        from xdas.trigger import Trigger as compat

        assert compat is Trigger
        import xdas.trigger  # noqa: F401

        assert callable(xd.trigger)

    def test_eager_call(self):
        assert xd.trigger(generate(), thresh=0.5).equals(expected)

    def test_seed_returns_the_atom(self):
        atom = xd.trigger(..., thresh=0.5)
        assert isinstance(atom, Trigger)
        assert atom(generate()).equals(expected)

    def test_extends_a_pipeline(self):
        pipeline = xd.taper(...) >> xd.trigger(..., thresh=0.5)
        assert isinstance(pipeline(generate()), pd.DataFrame)

    def test_dim_defaults_to_time(self):
        assert Trigger(thresh=0.5).dim == "time"
        assert xd.trigger(generate(), thresh=0.5).equals(expected)


class TestTriggerCoords:
    def generate(self):
        cft = xd.DataArray(
            data=[
                [0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2],
                [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.8, 0.9, 0.1, 0.0],
            ],
            coords={
                "space": [0.0, 100.0],
                "time": {
                    "tie_indices": [0, 9],
                    "tie_values": [0.0, 9.0],
                    "sampling_interval": 1.0,
                },
            },
        )
        return cft.assign_coords(station=("space", ["ST01", "ST02"]))

    def test_annotates_with_a_non_dimensional_coordinate(self):
        cft = self.generate()
        picks = Trigger(thresh=0.5, dim="time", coords=["time", "station"])(cft)
        expected = pd.DataFrame(
            {
                "time": [2.0, 7.0, 7.0],
                "station": ["ST01", "ST01", "ST02"],
                "value": [0.9, 0.7, 0.9],
            }
        )
        assert picks.equals(expected)

    def test_defaults_to_the_dimension_coordinates(self):
        cft = self.generate()
        picks = Trigger(thresh=0.5, dim="time")(cft)
        assert list(picks.columns) == ["space", "time", "value"]

    def test_selects_and_orders_the_requested_columns(self):
        cft = self.generate()
        picks = Trigger(thresh=0.5, dim="time", coords=["station", "space", "time"])(
            cft
        )
        assert list(picks.columns) == ["station", "space", "time", "value"]

    def test_chunked_annotation_matches_monolithic(self):
        cft = self.generate()
        coords = ["time", "station"]
        expected = Trigger(thresh=0.5, dim="time", coords=coords)(cft)
        atom = Trigger(thresh=0.5, dim="time", coords=coords)
        picks = [atom(chunk, chunk_dim="time") for chunk in xd.split(cft, 3, "time")]
        result = pd.concat(picks, ignore_index=True)
        assert result.sort_values(coords, ignore_index=True).equals(
            expected.sort_values(coords, ignore_index=True)
        )

    def test_annotates_with_a_coordinate_along_the_picked_dimension(self):
        # A label riding on the picked dimension is indexed by absolute sample
        # number, so it is read off the labels kept for the whole run rather
        # than off the chunk in hand: a pick found in a chunk already gone
        # still names the right sample.
        cft = self.generate().assign_coords(
            sample=("time", [f"s{n}" for n in range(10)])
        )
        coords = ["time", "sample", "station"]
        expected = Trigger(thresh=0.5, dim="time", coords=coords)(cft)
        assert list(expected["sample"]) == ["s2", "s7", "s7"]
        atom = Trigger(thresh=0.5, dim="time", coords=coords)
        picks = [atom(chunk, chunk_dim="time") for chunk in xd.split(cft, 3, "time")]
        result = pd.concat(picks, ignore_index=True)
        assert result.sort_values(coords, ignore_index=True).equals(
            expected.sort_values(coords, ignore_index=True)
        )

    def test_unknown_coordinate_raises(self):
        cft = self.generate()
        with pytest.raises(KeyError, match="not a coordinate"):
            Trigger(thresh=0.5, dim="time", coords=["time", "elevation"])(cft)


def test_trigger_1d():
    """1D input (no spatial dimension) covers the coords=() branch in _call_numeric."""
    cft = xd.DataArray(
        data=[0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.6, 0.7, 0.3, 0.2],
        coords={
            "time": {
                "tie_indices": [0, 9],
                "tie_values": [0.0, 9.0],
                "sampling_interval": 1.0,
            },
        },
    )
    picks = Trigger(thresh=0.5, dim="time")(cft)
    assert len(picks) == 2
    assert list(picks["time"]) == [2.0, 7.0]


class TestThresholdMapping:
    """W6: thresholds keyed on the `phase` labels rather than on their position."""

    def generate(self, labels=("N", "P", "S")):
        data = {
            "N": [0.0, 0.9, 0.0, 0.0, 0.0],
            "P": [0.0, 0.0, 0.4, 0.0, 0.0],
            "S": [0.0, 0.0, 0.0, 0.8, 0.0],
        }
        return xd.DataArray(
            data=[data[label] for label in labels],
            coords={
                "phase": list(labels),
                "time": {
                    "tie_indices": [0, 4],
                    "tie_values": [0.0, 4.0],
                    "sampling_interval": 1.0,
                },
            },
        )

    def test_one_threshold_per_label(self):
        picks = Trigger(thresh={"P": 0.3, "S": 0.5})(self.generate())
        assert list(picks["phase"]) == ["P", "S"]
        assert list(picks["time"]) == [2.0, 3.0]
        assert list(picks["value"]) == [0.4, 0.8]

    def test_a_lane_below_its_own_threshold_does_not_trigger(self):
        picks = Trigger(thresh={"P": 0.5, "S": 0.5})(self.generate())
        assert list(picks["phase"]) == ["S"]

    def test_unlisted_labels_never_trigger(self):
        # `N` is the loudest lane of the three and carries no entry.
        picks = Trigger(thresh={"P": 0.3, "S": 0.5})(self.generate())
        assert "N" not in set(picks["phase"])

    def test_the_label_order_is_irrelevant(self):
        # The order of a model's labels is a property of its weight set: the
        # same mapping must give the same picks whatever order they come in.
        thresh = {"P": 0.3, "S": 0.5}
        expected = Trigger(thresh=thresh)(self.generate())
        flipped = Trigger(thresh=thresh)(self.generate(("S", "P", "N")))
        columns = ["phase", "time", "value"]
        assert flipped.sort_values(columns, ignore_index=True)[columns].equals(
            expected.sort_values(columns, ignore_index=True)[columns]
        )

    def test_a_scalar_still_applies_to_every_lane(self):
        picks = Trigger(thresh=0.3)(self.generate())
        assert list(picks["phase"]) == ["N", "P", "S"]

    def test_bytes_labels_are_decoded(self):
        cft = self.generate().assign_coords(phase=np.array([b"N", b"P", b"S"]))
        picks = Trigger(thresh={"P": 0.3, "S": 0.5})(cft)
        assert list(picks["time"]) == [2.0, 3.0]

    def test_numeric_labels_are_keyed_by_their_string_form(self):
        # A weight set declaring no `labels` gets positional ones, so the
        # `phase` coordinate is integer-valued and the mapping still keys on it.
        cft = self.generate().assign_coords(phase=[0, 1, 2])
        picks = Trigger(thresh={1: 0.3, 2: 0.5})(cft)
        assert list(picks["phase"]) == [1, 2]

    def test_chunked_matches_monolithic(self):
        cft = self.generate()
        thresh = {"P": 0.3, "S": 0.5}
        expected = Trigger(thresh=thresh)(cft)
        atom = Trigger(thresh=thresh)
        picks = [atom(chunk, chunk_dim="time") for chunk in xd.split(cft, 3, "time")]
        picks += atom.flush()
        result = pd.concat(picks, ignore_index=True)
        assert result.equals(expected)

    def test_without_a_phase_coordinate_raises(self):
        with pytest.raises(ValueError, match="'phase' coordinate, which the data"):
            Trigger(thresh={"P": 0.3})(generate())

    def test_along_the_phase_dimension_raises(self):
        with pytest.raises(ValueError, match="it is the dimension"):
            Trigger(thresh={"P": 0.3}, dim="phase")(self.generate())

    def test_an_unknown_label_raises(self):
        with pytest.raises(KeyError, match=r"\['Pg'\]"):
            Trigger(thresh={"Pg": 0.3})(self.generate())


class TestScalarCoords:
    """W6: 0-d coordinates become constant columns."""

    def generate(self):
        return generate().assign_coords(station="ST01", depth=1000.0)

    def test_named_explicitly(self):
        picks = Trigger(thresh=0.5, coords=["time", "station"])(self.generate())
        assert list(picks.columns) == ["time", "station", "value"]
        assert list(picks["station"]) == ["ST01", "ST01"]

    def test_auto_leads_with_them(self):
        # identity first, measurement last: the scalar coordinates lead, then
        # the other dimension coordinates, then the picked dimension. The tree
        # path of a collection walk takes the same leading position, so a pick
        # table reads the same whichever source its identity came from.
        picks = Trigger(thresh=0.5, coords="auto")(self.generate())
        assert list(picks.columns) == ["station", "depth", "space", "time", "value"]
        assert list(picks["depth"]) == [1000.0, 1000.0]

    def test_auto_does_not_depend_on_the_input_dimension_order(self):
        cft = self.generate()
        expected = Trigger(thresh=0.5)(cft)
        transposed = Trigger(thresh=0.5)(cft.transpose("time", "space"))
        assert list(transposed.columns) == list(expected.columns)

    def test_auto_is_the_default(self):
        expected = Trigger(thresh=0.5, coords="auto")(self.generate())
        assert Trigger(thresh=0.5)(self.generate()).equals(expected)

    def test_none_keeps_the_dimension_coordinates_only(self):
        picks = Trigger(thresh=0.5, coords=None)(self.generate())
        assert list(picks.columns) == ["space", "time", "value"]

    def test_a_constant_column_survives_an_empty_chunk(self):
        cft = self.generate()
        atom = Trigger(thresh=0.5, coords=["station"])
        picks = [atom(chunk, chunk_dim="time") for chunk in xd.split(cft, 5, "time")]
        assert picks[0].empty
        result = pd.concat(picks + atom.flush(), ignore_index=True)
        assert list(result["station"]) == ["ST01", "ST01"]

    def test_an_unknown_coords_string_raises(self):
        with pytest.raises(ValueError, match="must be 'auto', None or a sequence"):
            Trigger(thresh=0.5, coords="all")


class TestFlush:
    """W6: a trigger still open at the end of a run is closed, not lost."""

    def generate(self):
        return xd.DataArray(
            data=[[0.0, 0.1, 0.9, 0.8, 0.7]],
            coords={
                "space": [0.0],
                "time": {
                    "tie_indices": [0, 4],
                    "tie_values": [0.0, 4.0],
                    "sampling_interval": 1.0,
                },
            },
        )

    def test_the_eager_call_closes_the_run(self):
        picks = Trigger(thresh=0.5)(self.generate())
        assert isinstance(picks, pd.DataFrame)
        assert list(picks["time"]) == [2.0]
        assert list(picks["value"]) == [0.9]

    def test_chunk_invariance(self):
        cft = self.generate()
        expected = Trigger(thresh=0.5)(cft)
        atom = Trigger(thresh=0.5)
        picks = [atom(chunk, chunk_dim="time") for chunk in xd.split(cft, 3, "time")]
        picks += atom.flush()
        assert pd.concat(picks, ignore_index=True).equals(expected)

    def test_nothing_open_emits_nothing(self):
        atom = Trigger(thresh=0.5)
        atom(generate(), chunk_dim="time")
        assert atom.flush() == []

    def test_flushing_twice_emits_once(self):
        atom = Trigger(thresh=0.5)
        atom(self.generate(), chunk_dim="time")
        assert len(atom.flush()) == 1
        assert atom.flush() == []

    def test_before_initialization_emits_nothing(self):
        assert Trigger(thresh=0.5).flush() == []

    def test_each_run_of_a_gappy_record_is_closed(self):
        tail = xd.DataArray(
            data=[[0.0, 0.1, 0.7, 0.6, 0.6]],
            coords={
                "space": [0.0],
                "time": {
                    "tie_indices": [0, 4],
                    "tie_values": [10.0, 14.0],
                    "sampling_interval": 1.0,
                },
            },
        )
        picks = Trigger(thresh=0.5)(xd.concat([self.generate(), tail], "time"))
        assert isinstance(picks, pd.DataFrame)
        assert list(picks["time"]) == [2.0, 12.0]

    def test_iter_chunks_flushes(self):
        chunks = list(xd.split(self.generate(), 3, "time"))
        atom = Trigger(thresh=0.5)
        picks = pd.concat(atom.iter_chunks(chunks, "time"), ignore_index=True)
        assert list(picks["time"]) == [2.0]

    def test_one_lane_open_among_several(self):
        cft = xd.DataArray(
            data=[[0.0, 0.9, 0.0], [0.0, 0.9, 0.8]],
            coords={
                "space": [0.0, 100.0],
                "time": {
                    "tie_indices": [0, 2],
                    "tie_values": [0.0, 2.0],
                    "sampling_interval": 1.0,
                },
            },
        )
        picks = Trigger(thresh=0.5)(cft)
        assert list(picks["space"]) == [0.0, 100.0]
        assert list(picks["time"]) == [1.0, 1.0]


class TestChunkSemantics:
    """
    `Trigger` carries its open triggers across chunks along `dim`, elementwise
    across.

    Chunking along a dimension the atom does not pick along must change
    nothing. It used to be false: `annotations` froze the *first* chunk's
    lane coordinates, and `offset`, `coord`, `status`, `index` and `value`
    accumulated on every call, so a run chunked along ``distance`` labelled
    every later chunk's picks with the first chunk's lanes and the wrong
    times.
    """

    def cft(self, nlanes=8, nsamples=40):
        rng = np.random.default_rng(42)
        template = xd.testing.dummy(
            dims=("time", "distance"),
            shape=(nsamples, nlanes),
            datetime=False,
            step=1.0,
        )
        values = rng.random((nsamples, nlanes))
        return xd.DataArray(values, dict(template.coords), ("time", "distance"))

    @pytest.mark.parametrize("size", [7, 13, 40])
    def test_chunking_along_the_picked_dimension_is_invariant(self, size):
        xd.testing.assert_chunk_invariant(
            Trigger(thresh=0.8), self.cft(), {"time": size}
        )

    def test_a_single_lane_is_invariant(self):
        xd.testing.assert_chunk_invariant(
            Trigger(thresh=0.8), self.cft(nlanes=1), {"time": 7}
        )

    @pytest.mark.parametrize("size", [1, 3, 8])
    def test_chunking_along_another_dimension_is_invariant(self, size):
        # Regression: the lanes of a later chunk used to be annotated with the
        # first chunk's `distance` values, on a time axis that kept growing.
        xd.testing.assert_chunk_invariant(
            Trigger(thresh=0.8), self.cft(), {"distance": size}
        )

    def test_each_lane_keeps_its_own_identity_and_time_base(self):
        # Two lanes picking at different samples: the second chunk used to be
        # labelled with the first chunk's `distance` value, and its index to be
        # shifted by the first chunk's length.
        cft = xd.DataArray(
            data=[[0.0, 0.9], [0.9, 0.0], [0.0, 0.0], [0.0, 0.0]],
            coords={
                "time": {
                    "tie_indices": [0, 3],
                    "tie_values": [0.0, 3.0],
                    "sampling_interval": 1.0,
                },
                "distance": [0.0, 100.0],
            },
        )
        eager = Trigger(thresh=0.5)(cft)
        assert sorted(zip(eager["distance"], eager["time"])) == [
            (0.0, 1.0),
            (100.0, 0.0),
        ]
        chunked = Trigger(thresh=0.5).process(cft, chunks={"distance": 1})
        assert sorted(zip(chunked["distance"], chunked["time"])) == sorted(
            zip(eager["distance"], eager["time"])
        )
