"""
Tests for the machine-learning atom and for the fake model the ML suite runs on.

READ THIS BEFORE CHANGING ``xdas/atoms/ml.py``
----------------------------------------------
The two classes at the bottom of this module are a *characterization pin* of
``MLPicker`` as it behaves today, taken deliberately before the ``Annotate`` /
``Picker`` rework (plan §7, W2: "the values are unchanged against the current
``MLPicker`` on the same input").  The rework changes some of that behaviour on
purpose, so the pin is split in two, and the split is the point:

``TestMLPickerValuePin``
    What the rework must **not** change: the numbers.  Every assertion here
    reaches the array through dimension *names* — ``transpose("time", lane,
    "phase")`` before comparing, ``isel`` with a name, never a positional axis
    — so it holds whatever order the output dimensions come back in.  If one of
    these fails after the rework, the characteristic function moved and that is
    a regression.

``TestMLPickerBehaviourTheReworkChanges``
    What the rework is **expected** to change, isolated so it can be updated on
    its own:

    - ``test_output_is_laid_out_sample_last`` — the old ``MLPicker`` forced the
      sample dimension first (``chunk.transpose(self.dim, ...)``) whatever went
      in.  ``Annotate`` keeps the input's order among the other dimensions but
      lays the output out sample-last, so ``("distance", "time")`` in gives
      ``("distance", "phase", "time")`` out.  Do not touch
      ``TestMLPickerValuePin``, which reaches the array by name and holds
      either way.
    - ``test_output_stops_at_the_last_complete_window`` — today the trailing
      samples that no full window covers are dropped.  W1 emits the
      end-aligned final window at ``flush()``, so the output grows to span the
      input.  ``TestMLPickerValuePin`` only ever looks at the leading samples
      the two agree on, so it survives that too.
"""

import numpy as np
import obspy
import pytest
import torch
from seisbench.models import WaveformModel

import xdas as xd
from tests.fakemodel import WEIGHT_SETS, FakeModel, fake_model
from xdas.atoms import Annotate, Filter, MLPicker, Resample, Sequential
from xdas.atoms.ml import _ChannelFilter, _model_filter

#: A short, exactly representable signal: two lanes that are neither equal nor
#: proportional, so a bug that mixes lanes shows up in the golden.
PIN_SIGNAL = np.array(
    [3, -1, 4, -1, 5, -9, 2, -6, 5, 3, -5, 8, 9, -7, 9, 3, -2, 3, -8, 4, 6, 2, -6, 4],
    float,
)

#: Number of samples the current implementation emits for :func:`pin_array`:
#: ``range(0, 24 - 8, 4)`` windows of ``step = in_samples // 2 = 4`` samples.
PIN_SAMPLES = 16


def pin_array(dims, n_lanes=2):
    """Build the pin input, laid out over *dims* (which must contain ``time``)."""
    values = np.stack(
        [np.roll(PIN_SIGNAL, 5 * j) * (j + 1) for j in range(n_lanes)], axis=-1
    )
    lane = next(dim for dim in dims if dim != "time")
    template = xd.testing.dummy(
        dims=("time", lane), shape=(len(PIN_SIGNAL), n_lanes), ctype="interpolated"
    )
    coords = dict(template.coords)
    if lane == "id":
        coords["id"] = np.array([f"A{index:02d}" for index in range(n_lanes)])
    da = xd.DataArray(values, coords, ("time", lane))
    return da if dims == ("time", lane) else da.transpose(*dims)


# ---------------------------------------------------------------------------
# The fake model (plan §3 as an executable contract)
# ---------------------------------------------------------------------------


class TestFakeModelSpansTheWeightSetAxes:
    """Every property §3 lists as per-weights is actually varied by the presets."""

    @staticmethod
    def spread(key, default=None):
        return {
            weights.get(key, default)
            if not isinstance(weights.get(key, default), dict)
            else "dict"
            for weights in WEIGHT_SETS.values()
        }

    def test_component_order(self):
        assert self.spread("component_order") == {"ENZ", "ZNE", "Z12H"}

    def test_in_channels(self):
        assert self.spread("in_channels") == {3, 4}

    def test_label_order_flips(self):
        assert self.spread("labels") == {"NPS", "PSN"}

    def test_sampling_rate(self):
        assert self.spread("sampling_rate") == {50, 100}

    def test_blinding_present_and_absent(self):
        declared = {"blinding" in w["default_args"] for w in WEIGHT_SETS.values()}
        assert declared == {True, False}

    def test_overlap_present_and_absent(self):
        declared = {"overlap" in w["default_args"] for w in WEIGHT_SETS.values()}
        assert declared == {True, False}

    def test_per_phase_thresholds_present_and_absent(self):
        declared = {
            any(key.endswith("_threshold") for key in w["default_args"])
            for w in WEIGHT_SETS.values()
        }
        assert declared == {True, False}

    def test_filters_absent_flat_and_per_channel(self):
        kinds = set()
        for weights in WEIGHT_SETS.values():
            args = weights.get("filter_args")
            kinds.add("absent" if args is None else type(args).__name__)
        assert kinds == {"absent", "tuple", "dict"}
        obs = fake_model("obs")
        assert obs.filter_args == {"??H": ["highpass"]}
        assert obs.filter_kwargs == {"??H": {"freq": 0.5}}


class TestFakeModel:
    """The contract ``MLPicker`` and its successors drive the model through."""

    def test_is_a_waveform_model(self):
        model = fake_model()
        assert isinstance(model, WaveformModel)
        assert isinstance(model, torch.nn.Module)
        assert not model.training  # the factory returns it in eval mode
        assert model.to("cpu") is model

    @pytest.mark.parametrize("name", sorted(WEIGHT_SETS))
    def test_every_preset_exposes_what_the_atom_reads(self, name):
        model = fake_model(name)
        assert isinstance(model.in_samples, int)
        assert model.classes == len(model.labels)
        assert len(model.component_order) == model.in_channels
        assert set(model.default_args) == set(WEIGHT_SETS[name]["default_args"])

    def test_overrides_win_over_the_preset(self):
        model = fake_model("obs", sampling_rate=50, in_samples=16)
        assert model.component_order == "Z12H"
        assert model.sampling_rate == 50
        assert model.in_samples == 16

    def test_forward_is_a_fixed_function_of_the_input(self):
        model = fake_model()  # ZNE, 3 channels, PSN
        batch = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        out = model(batch)
        assert out.shape == (2, model.classes, 4)
        # y[b, k, t] = k + sum_c (c + 1) * x[b, c, t]
        pooled = (batch * torch.tensor([1.0, 2.0, 3.0]).reshape(1, -1, 1)).sum(dim=-2)
        for k in range(model.classes):
            assert torch.equal(out[:, k, :], pooled + k)

    def test_annotate_batch_pre_reads_the_argdict(self):
        model = fake_model()
        batch = torch.tensor([[1.0, -2.0, 4.0]])
        plain = model.annotate_batch_pre(batch, {})
        scaled = model.annotate_batch_pre(batch, {"scale": 10.0})
        np.testing.assert_allclose(plain.numpy(), [[0.25, -0.5, 1.0]], rtol=1e-6)
        np.testing.assert_allclose(scaled.numpy(), [[2.5, -5.0, 10.0]], rtol=1e-6)
        assert model.seen_argdicts == [{}, {"scale": 10.0}]

    def test_annotate_batch_post_transposes_and_blinds(self):
        model = fake_model()
        batch = torch.arange(1 * 3 * 5, dtype=torch.float32).reshape(1, 3, 5)
        out = model.annotate_batch_post(batch.clone(), None, {"blinding": (1, 2)})
        assert out.shape == (1, 5, 3)  # (batch, samples, classes)
        assert torch.isnan(out[0, 0]).all()
        assert torch.isnan(out[0, -2:]).all()
        assert torch.equal(out[0, 1:3], batch[0, :, 1:3].T)

    def test_blinding_falls_back_to_the_class_default_when_undeclared(self):
        # `obs` declares no blinding; the fallback is (0, 0), i.e. blind nothing.
        model = fake_model("obs")
        assert "blinding" not in model.default_args
        batch = torch.ones(1, model.classes, 5)
        out = model.annotate_batch_post(batch, None, {})
        assert not torch.isnan(out).any()

    def test_annotate_args_moves_the_fallback_per_instance(self):
        model = fake_model(annotate_args={"blinding": (2, 0)})
        batch = torch.ones(1, model.classes, 5)
        out = model.annotate_batch_post(batch, None, {})
        assert torch.isnan(out[0, :2]).all()
        assert not torch.isnan(out[0, 2:]).any()
        assert FakeModel._annotate_args["blinding"][1] == (0, 0)  # class untouched

    def test_piggyback_pair_form(self):
        model = fake_model(piggyback=3.0)
        pair = model.annotate_batch_pre(torch.tensor([[1.0, 2.0]]), {})
        assert isinstance(pair, tuple) and len(pair) == 2
        piggyback = pair[1]
        assert piggyback == 3.0
        out = model.annotate_batch_post(
            torch.ones(1, model.classes, 2), piggyback, {"blinding": (0, 0)}
        )
        assert torch.equal(out, torch.full((1, 2, model.classes), 3.0))
        assert model.seen_piggybacks == [3.0]

    @pytest.mark.parametrize("name", ["original", "diting", "geofon"])
    def test_the_atom_can_drive_every_preset_that_declares_blinding(self, name):
        model = fake_model(name)
        picker = MLPicker(model, "time", device="cpu")
        result = picker(pin_array(("time", "distance")))
        assert result.sizes["phase"] == model.classes
        assert list(result.coords["phase"].values) == list(model.labels)

    @pytest.mark.parametrize("name", ["obs", "volpick"])
    def test_undeclared_blinding_falls_back_instead_of_raising(self, name):
        # Neither weight set declares `blinding`. Before W1 the atom read
        # `default_args["blinding"]` directly and raised `KeyError` here (plan
        # §3, consequence 3); now blinding is the model's own business, applied
        # by `annotate_batch_post`, and its fallback is (0, 0) — blind nothing.
        picker = Annotate(fake_model(name), "time", device="cpu")
        result = picker(pin_array(("time", "distance")))
        assert not np.isnan(result.values).any()


# ---------------------------------------------------------------------------
# Characterization pin — see the module docstring
# ---------------------------------------------------------------------------

GOLDEN_CLONE = np.array(
    [[[       np.nan,        np.nan,        np.nan],
      [       np.nan,        np.nan,        np.nan]],
     [[-0.6666667,  0.3333333,  1.3333333],
      [ 6.       ,  7.       ,  8.       ]],
     [[ 2.6666667,  3.6666667,  4.666667 ],
      [ 2.       ,  3.       ,  4.       ]],
     [[-0.6666667,  0.3333333,  1.3333333],
      [-6.       , -5.       , -4.       ]],
     [[ 3.3333335,  4.3333335,  5.3333335],
      [ 4.       ,  5.       ,  6.       ]],
     [[-6.       , -5.       , -4.       ],
      [ 2.5      ,  3.5      ,  4.5      ]],
     [[ 1.3333334,  2.3333335,  3.3333335],
      [-0.8333334,  0.1666667,  1.1666666]],
     [[-4.       , -3.       , -2.       ],
      [ 2.6666667,  3.6666667,  4.666667 ]],
     [[ 3.3333335,  4.3333335,  5.3333335],
      [-0.6666667,  0.3333333,  1.3333333]],
     [[ 2.       ,  3.       ,  4.       ],
      [ 3.3333335,  4.3333335,  5.3333335]],
     [[-3.3333335, -2.3333335, -1.3333335],
      [-6.       , -5.       , -4.       ]],
     [[ 5.3333335,  6.3333335,  7.3333335],
      [ 1.3333334,  2.3333335,  3.3333335]],
     [[ 6.       ,  7.       ,  8.       ],
      [-4.       , -3.       , -2.       ]],
     [[-4.666667 , -3.666667 , -2.666667 ],
      [ 3.3333335,  4.3333335,  5.3333335]],
     [[ 6.       ,  7.       ,  8.       ],
      [ 2.       ,  3.       ,  4.       ]],
     [[ 2.       ,  3.       ,  4.       ],
      [-3.3333335, -2.3333335, -1.3333335]]]
)  # fmt: skip

GOLDEN_Z = np.array(
    [[[       np.nan,        np.nan,        np.nan],
      [       np.nan,        np.nan,        np.nan]],
     [[-0.3333333,  0.6666666,  1.6666666],
      [ 3.       ,  4.       ,  5.       ]],
     [[ 1.3333334,  2.3333335,  3.3333335],
      [ 1.       ,  2.       ,  3.       ]],
     [[-0.3333333,  0.6666666,  1.6666666],
      [-3.       , -2.       , -1.       ]],
     [[ 1.6666667,  2.6666667,  3.6666667],
      [ 2.       ,  3.       ,  4.       ]],
     [[-3.       , -2.       , -1.       ],
      [ 1.25     ,  2.25     ,  3.25     ]],
     [[ 0.6666667,  1.6666667,  2.6666667],
      [-0.4166667,  0.5833333,  1.5833333]],
     [[-2.       , -1.       ,  0.       ],
      [ 1.3333334,  2.3333335,  3.3333335]],
     [[ 1.6666667,  2.6666667,  3.6666667],
      [-0.3333333,  0.6666666,  1.6666666]],
     [[ 1.       ,  2.       ,  3.       ],
      [ 1.6666667,  2.6666667,  3.6666667]],
     [[-1.6666667, -0.6666667,  0.3333333],
      [-3.       , -2.       , -1.       ]],
     [[ 2.6666667,  3.6666667,  4.666667 ],
      [ 0.6666667,  1.6666667,  2.6666667]],
     [[ 3.       ,  4.       ,  5.       ],
      [-2.       , -1.       ,  0.       ]],
     [[-2.3333335, -1.3333335, -0.3333335],
      [ 1.6666667,  2.6666667,  3.6666667]],
     [[ 3.       ,  4.       ,  5.       ],
      [ 1.       ,  2.       ,  3.       ]],
     [[ 1.       ,  2.       ,  3.       ],
      [-1.6666667, -0.6666667,  0.3333333]]]
)  # fmt: skip


def canonical(result, lane):
    """
    Return *result*'s pinned values as ``(time, lane, phase)``, by dimension name.

    Never index a dimension positionally here: the whole point of the pin is
    that it outlives the rework's change of output dimension order.  Only the
    leading :data:`PIN_SAMPLES` samples are taken, so it also outlives W1's
    end-aligned final window extending the output.
    """
    result = result.transpose("time", lane, "phase")
    return result.isel({"time": slice(0, PIN_SAMPLES)}).values


class TestMLPickerValuePin:
    """The numbers today's ``MLPicker`` produces — these must not move."""

    @pytest.mark.parametrize(
        "dims", [("time", "distance"), ("distance", "time"), ("time", "id")]
    )
    def test_das_layouts_agree_with_the_golden(self, dims):
        lane = next(dim for dim in dims if dim != "time")
        picker = MLPicker(fake_model("original"), "time", device="cpu")
        result = picker(pin_array(dims))
        np.testing.assert_allclose(canonical(result, lane), GOLDEN_CLONE, rtol=1e-6)

    def test_named_component_strategy_agrees_with_the_golden(self):
        picker = MLPicker(
            fake_model("original"), "time", device="cpu", component_strategy="Z"
        )
        result = picker(pin_array(("time", "distance")))
        np.testing.assert_allclose(canonical(result, "distance"), GOLDEN_Z, rtol=1e-6)

    @pytest.mark.parametrize("indices", [[7, 13], [4, 8, 12, 16, 20], [1]])
    def test_streamed_chunks_agree_with_the_golden(self, indices):
        picker = MLPicker(fake_model("original"), "time", device="cpu")
        chunks = xd.split(pin_array(("time", "distance")), indices, "time")
        result = xd.concat(list(picker.iter_chunks(chunks)), "time")
        np.testing.assert_allclose(
            canonical(result, "distance"), GOLDEN_CLONE, rtol=1e-6
        )

    def test_the_phase_coordinate_follows_the_model_label_order(self):
        # `original` labels NPS, `geofon` labels PSN — plan §3's order flip.
        for name, labels in (
            ("original", ["N", "P", "S"]),
            ("geofon", ["P", "S", "N"]),
        ):
            picker = MLPicker(fake_model(name), "time", device="cpu")
            result = picker(pin_array(("time", "distance")))
            assert list(result.coords["phase"].values) == labels

    @pytest.mark.parametrize("name", ["original", "geofon"])
    def test_the_phases_can_be_selected_by_label(self, name):
        # the point of labelling the axis: `sel` must work whichever order the
        # weight set declares, since `isel` positions differ between the two.
        picker = MLPicker(fake_model(name), "time", device="cpu")
        cft = picker(pin_array(("time", "distance")))
        result = cft.sel(phase=["P", "S"])
        assert list(result.coords["phase"].values) == ["P", "S"]
        for phase in ("P", "S"):
            np.testing.assert_array_equal(
                result.sel(phase=phase).values, cft.sel(phase=phase).values
            )
        with pytest.raises(KeyError):
            cft.sel(phase="Q")

    @pytest.mark.parametrize(
        "dims", [("time", "distance"), ("distance", "time"), ("time", "id")]
    )
    def test_the_non_sample_coordinates_pass_through_untouched(self, dims):
        lane = next(dim for dim in dims if dim != "time")
        da = pin_array(dims)
        picker = MLPicker(fake_model("original"), "time", device="cpu")
        result = picker(da)
        np.testing.assert_array_equal(
            result.coords[lane].values, da.coords[lane].values
        )

    def test_the_sample_coordinate_starts_at_the_first_input_sample(self):
        da = pin_array(("time", "distance"))
        picker = MLPicker(fake_model("original"), "time", device="cpu")
        result = picker(da)
        np.testing.assert_array_equal(
            result.coords["time"].values[:PIN_SAMPLES],
            da.coords["time"].values[:PIN_SAMPLES],
        )


class TestMLPickerBehaviourTheReworkChanges:
    """
    Two assertions W1+W2 is expected to rewrite. See the module docstring.

    Nothing else in this file encodes either fact, so updating these two is
    enough; if a value assertion in ``TestMLPickerValuePin`` fails as well, the
    characteristic function moved and that is a regression, not the rename.
    """

    @pytest.mark.parametrize(
        "dims", [("time", "distance"), ("distance", "time"), ("time", "id")]
    )
    def test_output_is_laid_out_sample_last(self, dims):
        # W1+W2: the other dimensions keep their order, then `phase`, then the
        # samples — so the characteristic function of one lane is contiguous.
        lane = next(dim for dim in dims if dim != "time")
        picker = MLPicker(fake_model("original"), "time", device="cpu")
        result = picker(pin_array(dims))
        assert result.dims == (lane, "phase", "time")
        assert result.values.flags["C_CONTIGUOUS"]

    def test_output_stops_at_the_last_complete_window(self):
        # W1: `flush()` emits the end-aligned final window, so the output spans
        # the whole input.
        da = pin_array(("time", "distance"))
        picker = MLPicker(fake_model("original"), "time", device="cpu")
        result = picker(da)
        assert da.sizes["time"] == len(PIN_SIGNAL)
        assert result.sizes["time"] == len(PIN_SIGNAL)


# ---------------------------------------------------------------------------
# W1 + W2 — `Annotate`
# ---------------------------------------------------------------------------

#: Spike spacing, equal to the fake model's window, so that every window sees
#: exactly one spike of every component and the peak normalisation is a no-op.
SPIKE_PERIOD = 8


def annotate_model(name="original", **overrides):
    """A preset with a plain 50 % overlap and no blinding, for legible values."""
    overrides.setdefault("default_args", {"overlap": 0.5})
    return fake_model(name, **overrides)


def spikes(ncomp, n=24):
    """Unit spikes: component *c* fires at every sample congruent to *c*."""
    values = np.zeros((n, ncomp))
    for index in range(ncomp):
        values[index::SPIKE_PERIOD, index] = 1.0
    return values


def component_array(labels, dims=("time", "channel"), sample_dim="time", n=24):
    """
    A record whose component *c* is a spike train naming itself.

    Because the fake model weights input slot *k* by ``k + 1``, the value the
    characteristic function takes at sample *c* says which slot component *c*
    was permuted into — see :func:`slot_of`.
    """
    comp_dim = next(dim for dim in dims if dim != sample_dim)
    template = xd.testing.dummy(
        dims=(sample_dim, comp_dim), shape=(n, len(labels)), ctype="interpolated"
    )
    coords = dict(template.coords)
    coords[comp_dim] = np.array(labels)
    da = xd.DataArray(spikes(len(labels), n), coords, (sample_dim, comp_dim))
    return da.transpose(*dims)


def trace_array(n=24, sample_dim="time", index=0):
    """A single trace, spiking at every sample congruent to *index*."""
    template = xd.testing.dummy(
        dims=(sample_dim,), shape=(n,), ctype="interpolated", step=0.01
    )
    values = spikes(index + 1, n)[:, index]
    return xd.DataArray(values, dict(template.coords), (sample_dim,))


def slot_of(result, index, sample_dim="time", **isel):
    """Return the model input slot the component spiking at *index* landed in."""
    value = result.isel({sample_dim: index, "phase": 0, **isel}).values
    return round(float(value)) - 1


class TestAnnotateRenaming:
    """W1: `MLPicker` becomes `Annotate`, the old names warn until 0.4."""

    def test_mlpicker_is_a_deprecated_alias(self):
        with pytest.warns(DeprecationWarning, match="removed in 0.4"):
            picker = MLPicker(annotate_model(), "time", device="cpu")
        assert isinstance(picker, Annotate)

    def test_the_mlpicker_twin_warns_too(self):
        with pytest.warns(DeprecationWarning, match="removed in 0.4"):
            atom = xd.mlpicker(..., annotate_model(), "time", device="cpu")
        assert isinstance(atom, Annotate)

    def test_annotate_has_an_eager_twin(self):
        da = pin_array(("time", "distance"))
        model = annotate_model()
        expected = Annotate(model, "time", device="cpu")(da)
        assert xd.annotate(da, model, "time", device="cpu").equals(expected)
        atom = xd.annotate(..., model, "time", device="cpu")
        assert isinstance(atom, Annotate)


class TestAnnotateReadsTheModelArgdict:
    """W1: the parameters SeisBench reads from the model are no longer invented."""

    def test_argdict_is_the_weight_sets_defaults_plus_the_call_kwargs(self):
        model = fake_model("diting")
        picker = Annotate(model, "time", device="cpu", scale=2.0)
        assert picker.argdict == model.default_args | {"scale": 2.0}
        assert "blinding" in picker.argdict  # what the weight set declares

    def test_every_batch_is_preprocessed_with_that_argdict(self):
        model = fake_model("original")
        Annotate(model, "time", device="cpu", scale=2.0)(pin_array(("time", "id")))
        assert model.seen_argdicts
        assert all(
            argdict == {"overlap": 0.5, "blinding": (1, 1), "scale": 2.0}
            for argdict in model.seen_argdicts
        )

    def test_the_argdict_reaches_the_values(self):
        da = pin_array(("time", "distance"))
        plain = Annotate(annotate_model(), "time", device="cpu")(da)
        scaled = Annotate(annotate_model(), "time", device="cpu", scale=2.0)(da)
        # phase 0 carries no class offset, so the gain shows undiluted
        np.testing.assert_allclose(
            scaled.isel(phase=0).values, 2.0 * plain.isel(phase=0).values, rtol=1e-6
        )

    def test_preprocessing_sees_the_filled_three_dimensional_batch(self):
        # Trap 1: today `annotate_batch_pre` is handed the 2-D staging buffer,
        # before the component slots are filled.
        shapes = []

        class Recording(FakeModel):
            def annotate_batch_pre(self, batch, argdict):
                shapes.append(tuple(batch.shape))
                return super().annotate_batch_pre(batch, argdict)

        model = Recording(
            component_order="ENZ", in_channels=3, default_args={"overlap": 0.5}
        ).eval()
        Annotate(model, "time", device="cpu")(pin_array(("time", "distance")))
        assert shapes and all(shape == (2, 3, 8) for shape in shapes)

    def test_the_piggyback_pair_is_plumbed_through_to_post(self):
        da = pin_array(("time", "distance"))
        model = annotate_model(piggyback=3.0)
        result = Annotate(model, "time", device="cpu")(da)
        plain = Annotate(annotate_model(), "time", device="cpu")(da)
        assert model.seen_piggybacks == [3.0] * len(model.seen_piggybacks)
        assert len(model.seen_piggybacks) > 1
        np.testing.assert_allclose(result.values, 3.0 * plain.values, rtol=1e-6)

    def test_blinding_is_left_to_the_model(self):
        model = annotate_model(default_args={"overlap": 0.5, "blinding": (2, 0)})
        result = Annotate(model, "time", device="cpu")(pin_array(("time", "distance")))
        # one-sided blinding: only the first two samples are covered by nothing
        assert np.isnan(result.isel(time=slice(0, 2)).values).all()
        assert not np.isnan(result.isel(time=slice(2, None)).values).any()

    def test_blinding_of_zero_blinds_nothing(self):
        model = annotate_model(default_args={"overlap": 0.5, "blinding": (0, 0)})
        result = Annotate(model, "time", device="cpu")(pin_array(("time", "distance")))
        assert not np.isnan(result.values).any()


class TestAnnotateWindowing:
    """W1: the overlap, the stacking and the end-aligned final window."""

    @pytest.mark.parametrize(
        "overlap, noverlap", [(0, 0), (0.25, 2), (0.5, 4), (3, 3), (7, 7)]
    )
    def test_overlap_is_read_the_seisbench_way(self, overlap, noverlap):
        model = annotate_model(default_args={"overlap": overlap})
        picker = Annotate(model, "time", device="cpu")
        assert picker.noverlap == noverlap
        assert picker.step == 8 - noverlap

    def test_overlap_falls_back_to_the_models_own_default(self):
        model = annotate_model(default_args={}, annotate_args={"overlap": 4})
        assert Annotate(model, "time", device="cpu").noverlap == 4

    def test_an_overlap_of_a_whole_window_is_refused(self):
        model = annotate_model(default_args={"overlap": 8})
        with pytest.raises(ValueError, match="shorter than one model window"):
            Annotate(model, "time", device="cpu")

    def test_a_zero_overlap_still_windows(self):
        model = annotate_model(default_args={"overlap": 0})
        result = Annotate(model, "time", device="cpu")(pin_array(("time", "distance")))
        assert result.sizes["time"] == len(PIN_SIGNAL)
        assert not np.isnan(result.values).any()

    def test_the_final_window_is_end_aligned(self):
        # 23 samples: the stride leaves a remainder of 3 samples that no
        # grid-aligned window covers, and the output still spans the input.
        da = pin_array(("time", "distance")).isel(time=slice(0, 23))
        result = Annotate(annotate_model(), "time", device="cpu")(da)
        assert result.sizes["time"] == 23
        np.testing.assert_array_equal(
            result.coords["time"].values, da.coords["time"].values
        )

    def test_a_record_of_exactly_one_window_is_annotated(self):
        da = pin_array(("time", "distance")).isel(time=slice(0, 8))
        result = Annotate(annotate_model(), "time", device="cpu")(da)
        assert result.sizes["time"] == 8

    def test_a_record_shorter_than_one_window_raises(self):
        da = pin_array(("time", "distance")).isel(time=slice(0, 7))
        with pytest.raises(ValueError, match="shorter along"):
            Annotate(annotate_model(), "time", device="cpu")(da)

    @pytest.mark.parametrize("indices", [[9, 11], [3, 6, 9], [23]])
    def test_a_chunk_completing_no_window_holds_everything_back(self, indices):
        da = pin_array(("time", "distance"))
        expected = Annotate(annotate_model(), "time", device="cpu")(da)
        picker = Annotate(annotate_model(), "time", device="cpu")
        chunks = list(picker.iter_chunks(xd.split(da, indices, "time"), "time"))
        assert xd.concat(chunks, "time").equals(expected)

    def test_flushing_before_any_window_emits_nothing(self):
        assert Annotate(annotate_model(), "time", device="cpu").flush() == []

    def test_stacking_max_takes_the_running_maximum(self):
        da = pin_array(("time", "distance"))
        average = Annotate(annotate_model(), "time", device="cpu")(da)
        maximum = Annotate(annotate_model(), "time", device="cpu", stacking="max")(da)
        assert maximum.dims == average.dims
        # the leading samples are covered by a single window: nothing to stack
        np.testing.assert_allclose(
            maximum.isel(time=slice(0, 4)).values,
            average.isel(time=slice(0, 4)).values,
            rtol=1e-6,
        )
        assert np.all(maximum.values >= average.values - 1e-6)
        assert not np.allclose(maximum.values, average.values)

    def test_stacking_max_leaves_uncovered_samples_undefined(self):
        model = annotate_model(default_args={"overlap": 0.5, "blinding": (2, 0)})
        result = Annotate(model, "time", device="cpu", stacking="max")(
            pin_array(("time", "distance"))
        )
        assert np.isnan(result.isel(time=slice(0, 2)).values).all()
        assert not np.isnan(result.isel(time=slice(2, None)).values).any()

    def test_an_unknown_stacking_rule_is_refused(self):
        with pytest.raises(ValueError, match="stacking must be"):
            Annotate(annotate_model(), "time", device="cpu", stacking="median")


class TestAnnotateChunkSemantics:
    """
    `Annotate` carries its window across chunks along `dim`, elementwise across.

    Chunking along a dimension the atom does not work along must change
    nothing. It used to be false here: the tail buffer was allocated only
    when the chunking followed `dim`, but refilled on every call, so a run
    chunked along ``distance`` concatenated one chunk's *time* tail onto the
    next chunk's *other lanes* and came out ragged.
    """

    @pytest.mark.parametrize("size", [8, 13, 16, 32])
    def test_chunking_along_the_sample_dimension_is_invariant(self, size):
        da = xd.testing.dummy(dims=("time", "distance"), shape=(64, 3))
        atom = Annotate(annotate_model(), "time", device="cpu")
        xd.testing.assert_chunk_invariant(atom, da, {"time": size})

    @pytest.mark.parametrize("size", [1, 2, 3])
    def test_chunking_along_another_dimension_is_invariant(self, size):
        # Regression: 4 lanes chunked 2 by 2 used to give a `DataSequence` of
        # ragged pieces with overlapping time coordinates and NaN values.
        da = pin_array(("time", "distance"), n_lanes=4)
        atom = Annotate(annotate_model(), "time", device="cpu")
        xd.testing.assert_chunk_invariant(atom, da, {"distance": size})

    def test_chunking_along_another_dimension_leaves_no_tail_behind(self):
        da = pin_array(("time", "distance"), n_lanes=4)
        atom = Annotate(annotate_model(), "time", device="cpu")
        streamed = atom.process(da, chunks={"distance": 2})
        assert isinstance(streamed, xd.DataArray)
        assert streamed.sizes["time"] == da.sizes["time"]
        assert not np.isnan(streamed.values).any()

    def test_a_record_of_exactly_one_window_chunked_across_is_invariant(self):
        # The body completes no window: everything comes out of `flush`.
        da = pin_array(("time", "distance"), n_lanes=4).isel(time=slice(0, 8))
        atom = Annotate(annotate_model(), "time", device="cpu")
        xd.testing.assert_chunk_invariant(atom, da, {"distance": 2})

    def test_a_record_shorter_than_a_window_still_raises_when_chunked_across(self):
        da = pin_array(("time", "distance"), n_lanes=4).isel(time=slice(0, 7))
        atom = Annotate(annotate_model(), "time", device="cpu")
        with pytest.raises(ValueError, match="shorter along"):
            atom.process(da, chunks={"distance": 2})

    def test_the_component_dimension_is_not_a_lane_axis(self):
        # The exemption is about *lanes*: a chunk holding a subset of the
        # components is not a record the model can read.
        da = component_array(["SHE", "SHN", "SHZ"])
        atom = Annotate(annotate_model(), "time", device="cpu")
        with pytest.raises(ValueError, match="component dimension"):
            atom.process(da, chunks={"channel": 2})


class TestAnnotatePostShapeIsNamed:
    """
    W9: a model whose ``annotate_batch_post`` breaks the stacking contract.

    The atom adopts SeisBench's ``(batch, samples, classes)``, which is
    ``PhaseNet``'s convention, not the ``WaveformModel`` default. Surveying
    the shipped SeisBench models, four keep the base default (``CRED``,
    ``GPD``, ``DPPDetector``, ``DPPPicker``) and ``CRED`` is an ``"array"``
    model, so it reaches the accumulation and used to get a bare
    ``RuntimeError`` from the broadcast. None of the 17 cached ``PhaseNet``
    weight sets does this.
    """

    def test_the_base_waveform_model_order_is_named(self):
        model = annotate_model()
        model.annotate_batch_post = lambda batch, piggyback, argdict: batch
        picker = Annotate(model, "time", device="cpu")
        with pytest.raises(ValueError, match=r"not \(8, 3\) = \(in_samples, classes\)"):
            picker(pin_array(("time", "distance")))

    def test_a_window_prediction_of_another_length_is_named(self):
        model = annotate_model()
        model.annotate_batch_post = lambda batch, piggyback, argdict: torch.transpose(
            batch, -1, -2
        )[..., :2, :]
        picker = Annotate(model, "time", device="cpu")
        with pytest.raises(ValueError, match=r"batch ending in \(2, 3\)"):
            picker(pin_array(("time", "distance")))


class TestAnnotateComponents:
    """W2: finding the component dimension and permuting it into the model."""

    def test_components_are_permuted_into_the_model_order(self):
        # ENZ weights fed ZNE data: every component lands in its own slot.
        result = Annotate(annotate_model("original"), "time", device="cpu")(
            component_array(["SHZ", "SHN", "SHE"])
        )
        assert result.dims == ("phase", "time")
        assert [slot_of(result, index) for index in range(3)] == [2, 1, 0]

    def test_the_four_component_obs_layout_needs_no_special_case(self):
        result = Annotate(annotate_model("obs"), "time", device="cpu")(
            component_array(["HHZ", "HH1", "HH2", "HHH"])
        )
        assert [slot_of(result, index) for index in range(4)] == [0, 1, 2, 3]

    def test_horizontal_components_are_matched_flexibly(self):
        # ZNE data on Z12H weights: N feeds the `1` slot and E the `2` slot.
        result = Annotate(annotate_model("obs"), "time", device="cpu")(
            component_array(["HHZ", "HHN", "HHE"])
        )
        assert [slot_of(result, index) for index in range(3)] == [0, 1, 2]

    def test_flexible_horizontal_components_can_be_switched_off(self):
        da = component_array(["HHZ", "HHN", "HHE"])
        picker = Annotate(
            annotate_model("obs"),
            "time",
            components="channel",
            device="cpu",
            flexible_horizontal_components=False,
        )
        with pytest.raises(ValueError, match="not labelled by components"):
            picker(da)

    def test_detection_declines_rather_than_guessing(self):
        # Same data, no explicit `components=`: the dimension simply is not
        # recognised, so it stays a batch axis.
        picker = Annotate(
            annotate_model("obs"),
            "time",
            device="cpu",
            flexible_horizontal_components=False,
        )
        result = picker(component_array(["HHZ", "HHN", "HHE"]))
        assert result.dims == ("channel", "phase", "time")

    def test_duplicated_orientations_raise(self):
        picker = Annotate(annotate_model(), "time", device="cpu")
        with pytest.raises(ValueError, match="repeats component orientations"):
            picker(component_array(["SHZ", "HHZ"]))

    def test_two_candidate_dimensions_raise(self):
        template = xd.testing.dummy(
            dims=("time", "station", "channel"), shape=(24, 2, 3), step=1.0
        )
        coords = dict(template.coords)
        coords["station"] = np.array(["N", "E"])
        coords["channel"] = np.array(["SHZ", "SHN", "SHE"])
        da = xd.DataArray(np.zeros((24, 2, 3)), coords, ("time", "station", "channel"))
        picker = Annotate(annotate_model(), "time", device="cpu")
        with pytest.raises(ValueError, match="several dimensions"):
            picker(da)

    def test_components_false_disables_detection(self):
        picker = Annotate(annotate_model(), "time", components=False, device="cpu")
        result = picker(component_array(["SHZ", "SHN", "SHE"]))
        assert result.dims == ("channel", "phase", "time")
        # each lane is cloned into all three slots: 1 + 2 + 3
        assert round(float(result.isel(time=0, channel=0, phase=0).values)) == 6

    def test_components_names_the_dimension_explicitly(self):
        da = component_array(["SHZ", "SHN", "SHE"])
        named = Annotate(annotate_model(), "time", components="channel", device="cpu")
        detected = Annotate(annotate_model(), "time", device="cpu")
        assert named(da).equals(detected(da))

    def test_components_must_name_a_dimension(self):
        picker = Annotate(annotate_model(), "time", components="sensor", device="cpu")
        with pytest.raises(ValueError, match="is not a dimension of the input"):
            picker(component_array(["SHZ", "SHN", "SHE"]))

    def test_components_must_name_a_labelled_dimension(self):
        picker = Annotate(annotate_model(), "time", components="distance", device="cpu")
        with pytest.raises(ValueError, match="not labelled by components"):
            picker(pin_array(("time", "distance")))

    def test_byte_labels_are_read_like_string_ones(self):
        da = component_array(["SHZ", "SHN", "SHE"])
        expected = Annotate(annotate_model(), "time", device="cpu")(da)
        da.coords["channel"] = np.array([b"SHZ", b"SHN", b"SHE"])
        assert Annotate(annotate_model(), "time", device="cpu")(da).equals(expected)

    def test_a_dimension_without_a_coordinate_is_never_a_candidate(self):
        da = component_array(["SHZ", "SHN", "SHE"])
        bare = xd.DataArray(da.values, {"time": da.coords["time"]}, da.dims)
        result = Annotate(annotate_model(), "time", device="cpu")(bare)
        assert result.dims == ("channel", "phase", "time")


class TestAnnotateLayouts:
    """W2: one trace, one instrument, a grid of instruments."""

    def test_a_single_trace(self):
        result = Annotate(annotate_model(), "time", device="cpu")(trace_array())
        assert result.dims == ("phase", "time")
        assert slot_of(result, 0) == 5  # cloned: 1 + 2 + 3

    def test_one_instrument_either_way_round(self):
        rowwise = component_array(["SHZ", "SHN", "SHE"], dims=("time", "channel"))
        colwise = component_array(["SHZ", "SHN", "SHE"], dims=("channel", "time"))
        picker = Annotate(annotate_model(), "time", device="cpu")
        assert picker(rowwise).equals(picker(colwise))

    def test_a_grid_of_instruments(self):
        template = xd.testing.dummy(
            dims=("station", "channel", "time"),
            shape=(2, 3, 24),
            step=1.0,
            datetime=False,
        )
        coords = dict(template.coords)
        coords["station"] = np.array(["ALPHA", "BRAVO"])
        coords["channel"] = np.array(["SHZ", "SHN", "SHE"])
        values = np.stack([spikes(3).T * (index + 1) for index in range(2)])
        da = xd.DataArray(values, coords, ("station", "channel", "time"))
        result = Annotate(annotate_model("original"), "time", device="cpu")(da)
        assert result.dims == ("station", "phase", "time")
        for station in range(2):
            slots = [slot_of(result, index, station=station) for index in range(3)]
            assert slots == [2, 1, 0]  # the lane gain is normalised away


class TestComponentStrategy:
    """W2: `"auto"` resolving two ways, and each explicit strategy."""

    def test_auto_clones_when_there_is_no_component_dimension(self):
        result = Annotate(annotate_model(), "time", device="cpu")(trace_array())
        assert slot_of(result, 0) == 5  # every slot filled: 1 + 2 + 3

    def test_auto_pads_a_partial_component_set(self):
        result = Annotate(annotate_model("original"), "time", device="cpu")(
            component_array(["SHZ"])
        )
        assert slot_of(result, 0) == 2  # the Z slot of ENZ, the rest zeroed

    @pytest.mark.parametrize(
        "name, first, other", [("original", "E", "Z"), ("diting", "Z", "E")]
    )
    def test_pad_fills_slot_zero_positionally(self, name, first, other):
        # SeisBench's `"pad"` counts to the first slot rather than naming it:
        # that is `E` for ENZ weights and `Z` for ZNE ones.
        da = trace_array()

        def annotate(strategy):
            picker = Annotate(
                annotate_model(name),
                "time",
                component_strategy=strategy,
                device="cpu",
            )
            return picker(da)

        assert annotate("pad").equals(annotate(first))
        assert not annotate("pad").equals(annotate(other))

    def test_clone_needs_a_single_component(self):
        picker = Annotate(
            annotate_model(), "time", component_strategy="clone", device="cpu"
        )
        with pytest.raises(ValueError, match="needs a single component"):
            picker(component_array(["SHZ", "SHN", "SHE"]))

    def test_clone_accepts_a_component_dimension_of_one(self):
        picker = Annotate(
            annotate_model(), "time", component_strategy="clone", device="cpu"
        )
        assert slot_of(picker(component_array(["SHZ"])), 0) == 5

    def test_a_named_slot_overrides_the_label(self):
        picker = Annotate(
            annotate_model("original"), "time", component_strategy="E", device="cpu"
        )
        assert slot_of(picker(component_array(["SHZ"])), 0) == 0

    def test_strict_accepts_a_complete_component_set(self):
        picker = Annotate(
            annotate_model("original"),
            "time",
            component_strategy="strict",
            device="cpu",
        )
        da = component_array(["SHZ", "SHN", "SHE"])
        assert picker(da).equals(Annotate(annotate_model(), "time", device="cpu")(da))

    def test_strict_refuses_a_partial_component_set(self):
        picker = Annotate(
            annotate_model(), "time", component_strategy="strict", device="cpu"
        )
        with pytest.raises(ValueError, match="component_strategy is 'strict'"):
            picker(component_array(["SHZ", "SHN"]))

    def test_strict_refuses_data_without_components(self):
        picker = Annotate(
            annotate_model(), "time", component_strategy="strict", device="cpu"
        )
        with pytest.raises(ValueError, match="needs a component dimension"):
            picker(trace_array())

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match="component_strategy must be one of"):
            Annotate(annotate_model(), "time", component_strategy="nope", device="cpu")


class TestAnnotateAssumesNoName:
    """W2: nothing in the implementation may spell `time` or `channel`."""

    def test_neither_dimension_is_conventionally_named(self):
        da = component_array(
            ["SHZ", "SHN", "SHE"], dims=("samples", "sensor"), sample_dim="samples"
        )
        picker = Annotate(annotate_model("original"), dim="samples", device="cpu")
        result = picker(da)
        assert result.dims == ("phase", "samples")
        slots = [slot_of(result, index, sample_dim="samples") for index in range(3)]
        assert slots == [2, 1, 0]

    def test_the_component_dimension_can_also_be_named_explicitly(self):
        da = component_array(
            ["SHZ", "SHN", "SHE"], dims=("samples", "sensor"), sample_dim="samples"
        )
        named = Annotate(
            annotate_model(), dim="samples", components="sensor", device="cpu"
        )
        assert named(da).dims == ("phase", "samples")

    @pytest.mark.parametrize(
        "dims, alias",
        [(("samples", "sensor"), "first"), (("sensor", "samples"), "last")],
    )
    def test_the_first_and_last_aliases_resolve(self, dims, alias):
        da = component_array(["SHZ", "SHN", "SHE"], dims=dims, sample_dim="samples")
        picker = Annotate(annotate_model(), dim=alias, device="cpu")
        assert picker(da).dims == ("phase", "samples")

    def test_an_unknown_sample_dimension_raises(self):
        picker = Annotate(annotate_model(), dim="nope", device="cpu")
        with pytest.raises(ValueError, match="is not a dimension of the input"):
            picker(pin_array(("time", "distance")))


# ---------------------------------------------------------------------------
# W3 — the filter the weight set ships
# ---------------------------------------------------------------------------

#: Long enough for a 0.5 Hz highpass at 100 Hz to be a filter rather than an
#: edge effect, and random so that no channel is a multiple of another.
FILTER_SIGNAL = np.random.default_rng(0).standard_normal(2000).cumsum()


def waveform(labels=("BHZ", "BH1", "BH2", "BDH"), dim="channel", step=0.01):
    """A record of *labels* channels, each a differently rolled random walk."""
    values = np.stack(
        [np.roll(FILTER_SIGNAL, 137 * index) for index in range(len(labels))], axis=-1
    )
    template = xd.testing.dummy(
        dims=("time", dim),
        shape=values.shape,
        step=(step, 1.0),
        ctype="interpolated",
    )
    coords = dict(template.coords)
    coords[dim] = np.array(labels)
    return xd.DataArray(values, coords, ("time", dim))


def obspy_filtered(da, name, dim="channel", **kwargs):
    """The same record filtered by obspy, the reference W3 has to reproduce."""
    fs = 1.0 / xd.get_sampling_interval(da, "time")
    traces = []
    for index in range(da.sizes[dim]):
        trace = obspy.Trace(np.asarray(da.isel({dim: index}).values).copy())
        trace.stats.sampling_rate = fs
        traces.append(trace)
    obspy.Stream(traces).filter(name, **kwargs)
    return np.stack([trace.data for trace in traces], axis=-1)


class TestChannelFilter:
    """W3: the per-channel form, which SeisBench's own DAS wrapper refuses."""

    def test_only_the_matching_channels_are_filtered(self):
        da = waveform()
        result = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        assert result.dims == da.dims
        untouched = result.isel(channel=slice(0, 3)).values
        assert np.array_equal(untouched, np.asarray(da.values)[:, :3])
        assert not np.allclose(result.isel(channel=3).values, da.values[:, 3])

    def test_the_matching_channel_is_filtered_exactly_as_obspy_does(self):
        da = waveform()
        result = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        expected = obspy_filtered(da, "highpass", freq=0.5)
        assert np.max(np.abs(result.isel(channel=3).values - expected[:, 3])) == 0.0

    def test_a_pattern_matching_nothing_is_a_no_operation(self):
        # SeisBench's `stream.select(channel=...)` on a pattern no trace
        # answers filters an empty stream, which is not an error.
        da = waveform()
        result = _ChannelFilter("??Q", (0.5, None), component_order="Z12H")(da)
        assert result.equals(da)

    def test_data_without_a_channel_dimension_is_left_alone(self):
        da = pin_array(("time", "distance"))
        result = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        assert result.equals(da)

    def test_components_false_disables_the_stage(self):
        da = waveform()
        atom = _ChannelFilter(
            "??H", (0.5, None), component_order="Z12H", components=False
        )
        assert atom(da).equals(da)

    def test_the_channel_dimension_is_found_by_its_labels_not_its_name(self):
        result = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(
            waveform(dim="sensor")
        )
        expected = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(
            waveform()
        )
        assert np.allclose(result.values, expected.values)

    def test_the_channel_dimension_can_be_named_explicitly(self):
        da = waveform(dim="sensor")
        atom = _ChannelFilter(
            "??H", (0.5, None), component_order="Z12H", components="sensor"
        )
        assert not np.allclose(atom(da).values[:, 3], da.values[:, 3])

    def test_horizontal_components_are_matched_flexibly_when_detecting(self):
        # `ZNE` labels against `Z12H` weights: N -> 1 and E -> 2, so the
        # dimension is still recognised and the hydrophone still filtered.
        da = waveform(("BHZ", "BHN", "BHE", "BDH"))
        result = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        assert not np.allclose(result.values[:, 3], da.values[:, 3])

    @pytest.mark.parametrize("indices", [(500,), (137, 900, 1500)])
    def test_chunked_processing_equals_eager_processing(self, indices):
        da = waveform()
        atom = _ChannelFilter("??H", (0.5, None), component_order="Z12H")
        expected = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        chunked = xd.concat(
            [atom(chunk, chunk_dim="time") for chunk in xd.split(da, indices, "time")],
            "time",
        )
        assert np.allclose(chunked.values, expected.values)

    def test_integer_data_is_promoted_rather_than_truncated(self):
        da = waveform()
        da = xd.DataArray(np.asarray(da.values).astype(np.int32), da.coords, da.dims)
        result = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        assert result.dtype == np.float64
        assert np.array_equal(result.values[:, :3], np.asarray(da.values)[:, :3])

    def test_the_parameters_of_filter_are_inherited(self):
        # Being a `Filter`, it takes `zerophase` — which the wrapper this
        # replaced could not express — and the subset still holds.
        da = waveform()
        atom = _ChannelFilter(
            "??H", (0.5, None), component_order="Z12H", zerophase=True
        )
        result = atom(da)
        causal = _ChannelFilter("??H", (0.5, None), component_order="Z12H")(da)
        assert np.array_equal(result.values[:, :3], np.asarray(da.values)[:, :3])
        assert not np.allclose(result.values[:, 3], causal.values[:, 3])

    def test_a_fir_filter_is_refused(self):
        # It shifts the coordinate to compensate its group delay, which would
        # leave the channels the pattern does not match on the wrong samples.
        with pytest.raises(ValueError, match="cannot filter a subset"):
            _ChannelFilter("??H", (0.5, None), ftype="fir")


class TestModelFilter:
    """W3: the stage is built from the weight set, and only when it declares one."""

    @pytest.mark.parametrize("name", ["original", "diting", "geofon"])
    def test_a_weight_set_declaring_none_gets_no_stage(self, name):
        assert _model_filter(fake_model(name)) is None

    def test_the_flat_form_filters_everything(self):
        # `volpick`'s flat filter is invented (no cached weight set ships one),
        # but the form has to work: a 1 Hz highpass on every channel.
        stage = _model_filter(fake_model("volpick"))
        assert isinstance(stage, Filter)
        assert (stage.freq, stage.order, stage.ftype) == ((1.0, None), 4, "iir")
        da = waveform(("BHZ", "BHN", "BHE"))
        expected = obspy_filtered(da, "highpass", freq=1.0)
        assert np.max(np.abs(stage(da).values - expected)) == 0.0

    def test_the_per_channel_form_filters_the_hydrophone_alone(self):
        stage = _model_filter(fake_model("obs"))
        assert isinstance(stage, _ChannelFilter)
        assert stage.pattern == "??H"
        assert stage.freq == (0.5, None)
        assert stage.component_order == "Z12H"
        da = waveform()
        result = stage(da)
        expected = obspy_filtered(da, "highpass", freq=0.5)
        assert np.array_equal(result.values[:, :3], np.asarray(da.values)[:, :3])
        assert np.max(np.abs(result.values[:, 3] - expected[:, 3])) == 0.0

    def test_the_corner_count_defaults_to_the_obspy_one(self):
        # `obs` declares only `freq`, so 4 corners is what obspy would have used
        # — and it is `Filter`'s own default, which is why the match is exact.
        assert _model_filter(fake_model("obs")).order == 4

    def test_a_declared_corner_count_is_honoured(self):
        model = fake_model(
            filter_args=("highpass",), filter_kwargs={"freq": 1.0, "corners": 2}
        )
        assert _model_filter(model).order == 2

    @pytest.mark.parametrize(
        "args, kwargs, freq",
        [
            (("highpass",), {"freq": 2.0}, (2.0, None)),
            (("lowpass",), {"freq": 20.0}, (None, 20.0)),
            (("bandpass",), {"freqmin": 1.0, "freqmax": 20.0}, (1.0, 20.0)),
        ],
    )
    def test_every_obspy_band_that_has_a_filter_equivalent(self, args, kwargs, freq):
        stage = _model_filter(fake_model(filter_args=args, filter_kwargs=kwargs))
        assert stage.freq == freq
        da = waveform(("BHZ", "BHN", "BHE"))
        expected = obspy_filtered(da, args[0], **kwargs)
        assert np.max(np.abs(stage(da).values - expected)) == 0.0

    def test_zerophase_warns_and_doubles_the_order(self):
        # SeisBench's concession, and ours for the same reason: `filtfilt` has
        # no causal streaming form.
        model = fake_model(
            filter_args=("highpass",),
            filter_kwargs={"freq": 1.0, "zerophase": True},
        )
        with pytest.warns(UserWarning, match="no causal streaming form"):
            stage = _model_filter(model)
        assert stage.order == 8

    def test_a_corner_above_half_the_nyquist_is_clamped(self):
        # At 100 Hz the Nyquist is 50 and half of it 25, so a 40 Hz lowpass
        # comes back clamped — the filter stays valid on data at a lower rate.
        model = fake_model(filter_args=("lowpass",), filter_kwargs={"freq": 40.0})
        assert _model_filter(model).freq == (None, pytest.approx(25.0, rel=1e-5))

    def test_the_clamp_follows_the_weight_sets_own_rate(self):
        model = fake_model(
            "diting", filter_args=("lowpass",), filter_kwargs={"freq": 40.0}
        )
        assert _model_filter(model).freq == (None, pytest.approx(12.5, rel=1e-5))

    def test_a_model_without_a_rate_is_not_clamped(self):
        model = fake_model(
            sampling_rate=None, filter_args=("lowpass",), filter_kwargs={"freq": 40.0}
        )
        assert _model_filter(model).freq == (None, 40.0)

    def test_several_patterns_become_a_sequence_applied_in_order(self):
        model = fake_model(
            "obs",
            filter_args={"??Z": ("highpass",), "??H": ("highpass",)},
            filter_kwargs={"??Z": {"freq": 1.0}, "??H": {"freq": 0.5}},
        )
        stage = _model_filter(model)
        assert isinstance(stage, Sequential)
        assert [atom.pattern for atom in stage] == ["??Z", "??H"]
        da = waveform()
        result = stage(da)
        expected = obspy_filtered(da, "highpass", freq=1.0)
        assert np.max(np.abs(result.values[:, 0] - expected[:, 0])) == 0.0
        assert np.array_equal(result.values[:, 1:3], np.asarray(da.values)[:, 1:3])

    def test_overlapping_patterns_filter_a_channel_twice_as_obspy_would(self):
        # Two `stream.select` calls hitting the same trace filter it twice, in
        # declaration order; nothing deduplicates them, here or in SeisBench.
        model = fake_model(
            "obs",
            filter_args={"??H": ("highpass",), "BD?": ("highpass",)},
            filter_kwargs={"??H": {"freq": 0.5}, "BD?": {"freq": 0.5}},
        )
        da = waveform()
        once = _ChannelFilter("??H", (0.5, None), component_order="Z12H")
        twice = _model_filter(model)(da)
        assert not np.allclose(twice.values[:, 3], once(da).values[:, 3])
        assert np.allclose(twice.values[:, 3], once(once(da)).values[:, 3])

    def test_an_empty_declaration_gets_no_stage(self):
        assert _model_filter(fake_model(filter_args={}, filter_kwargs={})) is None

    def test_a_pattern_missing_from_the_kwargs_raises(self):
        model = fake_model(filter_args={"??H": ("highpass",)}, filter_kwargs={})
        with pytest.raises(ValueError, match="in `filter_args` but not in"):
            _model_filter(model)

    def test_a_declaration_naming_no_single_filter_type_raises(self):
        model = fake_model(filter_args=("highpass", "lowpass"), filter_kwargs={})
        with pytest.raises(ValueError, match="exactly one obspy filter type"):
            _model_filter(model)

    def test_a_band_with_no_filter_equivalent_raises(self):
        # `Filter` has no bandstop, and silently skipping the filter would feed
        # the network what it was not trained on — the very bug W3 fixes.
        model = fake_model(
            filter_args=("bandstop",), filter_kwargs={"freqmin": 1.0, "freqmax": 2.0}
        )
        with pytest.raises(ValueError, match="'bandstop' filter, which has no"):
            _model_filter(model)

    def test_the_stage_reads_the_dimension_names_it_is_given(self):
        da = waveform(dim="sensor").rename({"time": "samples"})
        stage = _model_filter(fake_model("obs"), dim="samples", components="sensor")
        assert not np.allclose(stage(da).values[:, 3], da.values[:, 3])


class TestFilterStageOrder:
    """W3: the model's filter runs *before* the resampling, as SeisBench does."""

    def test_filtering_before_resampling_is_not_the_same_operation(self):
        da = waveform(("BHZ", "BHN", "BHE"), step=0.02)  # 50 Hz, model wants 100
        before = (_model_filter(fake_model("volpick")) >> Resample(100.0))(da)
        after = (Resample(100.0) >> _model_filter(fake_model("volpick")))(da)
        assert before.shape == after.shape
        assert not np.allclose(before.values, after.values)

    def test_the_stage_composes_ahead_of_resample_and_annotate(self):
        da = waveform(("BHZ", "BHN", "BHE"), step=0.02)
        model = fake_model("volpick", default_args={"overlap": 0.5})
        pipeline = (
            _model_filter(model)
            >> Resample(model.sampling_rate)
            >> Annotate(model, "time", device="cpu")
        )
        assert [type(stage).__name__ for stage in pipeline] == [
            "Filter",
            "Resample",
            "Annotate",
        ]
        result = pipeline(da)
        assert result.dims == ("phase", "time")
        assert np.isfinite(result.values).any()
