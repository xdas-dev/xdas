"""
Offline stand-ins for SeisBench weight sets, for the machine-learning tests.

Every interesting property of a SeisBench picker belongs to the *weight set*,
not to the architecture: ``component_order``, ``in_channels``, the *order* of
``labels``, ``sampling_rate``, which keys ``default_args`` declares, and whether
the weights ship their own preprocessing filter.  Surveying the 17 cached
``PhaseNet`` weight sets, all of these vary — see :data:`WEIGHT_SETS` for the
five archetypes that between them cover the whole range.

:class:`FakeModel` is a real :class:`seisbench.models.WaveformModel` subclass
whose forward pass is a fixed, trivial function of its input, so tests can
assert exact values without downloading weights or touching the network.  It is
deliberately not a picker that works: it is a picker-shaped contract.

Import it either directly, for :func:`pytest.mark.parametrize` tables::

    from tests.fakemodel import WEIGHT_SETS, fake_model

or through the ``fake_model`` fixture declared in ``tests/conftest.py``.
"""

import numpy as np
import torch
from seisbench.models import WaveformModel

#: Archetypal weight sets, one per combination the real ``PhaseNet`` sets show.
#:
#: The values mirror the cached metadata of the weight set each is named after,
#: except that ``blinding`` and ``overlap`` are scaled to the toy window of
#: :func:`fake_model` (``in_samples=8`` rather than 3001).  Pass a different
#: ``in_samples`` and you must pass a matching ``default_args`` with it.
WEIGHT_SETS = {
    # ENZ, NPS labels, declares both overlap and blinding, no thresholds.
    "original": {
        "component_order": "ENZ",
        "in_channels": 3,
        "labels": "NPS",
        "sampling_rate": 100,
        "default_args": {"overlap": 0.5, "blinding": (1, 1)},
    },
    # ZNE, NPS labels, and the one weight set that does not run at 100 Hz.
    "diting": {
        "component_order": "ZNE",
        "in_channels": 3,
        "labels": "NPS",
        "sampling_rate": 50,
        "default_args": {
            "P_threshold": 0.3,
            "S_threshold": 0.3,
            "blinding": (1, 1),
        },
    },
    # ZNE, PSN labels, thresholds far from the 0.3 fallback and far apart.
    "geofon": {
        "component_order": "ZNE",
        "in_channels": 3,
        "labels": "PSN",
        "sampling_rate": 100,
        "default_args": {
            "P_threshold": 0.5704745853696115,
            "S_threshold": 0.07349645833964447,
            "blinding": (1, 1),
        },
    },
    # Z12H with a fourth (hydrophone) channel and a per-channel filter, and no
    # blinding key at all — the combination that breaks assumptions.
    "obs": {
        "component_order": "Z12H",
        "in_channels": 4,
        "labels": "PSN",
        "sampling_rate": 100,
        "default_args": {"P_threshold": 0.2, "S_threshold": 0.1},
        "filter_args": {"??H": ["highpass"]},
        "filter_kwargs": {"??H": {"freq": 0.5}},
    },
    # No blinding either. The flat filter is *invented*: `obs`'s per-channel
    # one is the only filter any cached PhaseNet weight set declares, and the
    # flat form still has to work, so one preset carries it.
    "volpick": {
        "component_order": "ZNE",
        "in_channels": 3,
        "labels": "PSN",
        "sampling_rate": 100,
        "default_args": {"P_threshold": 0.39, "S_threshold": 0.34},
        "filter_args": ("highpass",),
        "filter_kwargs": {"freq": 1.0},
    },
}


class FakeModel(WaveformModel):
    """
    A ``WaveformModel`` with no weights and a closed-form forward pass.

    The forward pass is

    .. math:: y_{b,k,t} = k + \\sum_c (c + 1) \\, x_{b,c,t}

    so the output reveals both which input slot each component landed in (the
    ``c + 1`` weights) and the order of the classes (the ``k`` offset).
    Preprocessing is a peak normalisation scaled by the ``scale`` annotate
    argument, which makes the result depend on the whole window and therefore
    pins the sliding-window stitching rather than just the arithmetic.

    Parameters
    ----------
    component_order : str, optional
        Component letters in the order the model's input slots expect them,
        e.g. ``"ENZ"``, ``"ZNE"`` or ``"Z12H"``. Defaults to ``"ZNE"``.
    in_channels : int, optional
        Number of input slots. Defaults to 3; the ``obs`` archetype uses 4.
    labels : str or list of str, optional
        Output class labels, *in order*. Defaults to ``"PSN"``.
    classes : int, optional
        Number of output classes. Defaults to ``len(labels)``.
    sampling_rate : float, optional
        Sampling rate the weights were trained at. Defaults to 100.
    in_samples : int, optional
        Model window length in samples. Defaults to 8, small enough that a
        whole test record fits in a golden array.
    default_args : dict, optional
        Exactly what the weight set declares. Left empty by default, so tests
        can add or omit ``blinding``, ``overlap`` and ``*_threshold`` keys one
        at a time.
    annotate_args : dict, optional
        Per-instance overrides of the class-level ``_annotate_args`` defaults,
        given as ``{key: default_value}``. Use this to move the *fallback* a
        weight set falls back to, as opposed to what it declares.
    filter_args, filter_kwargs : optional
        Preprocessing filter the weights ship, flat (a tuple plus a dict) or
        per channel pattern (two dicts keyed identically).
    piggyback : float, optional
        When given, ``annotate_batch_pre`` returns the ``(batch, piggyback)``
        pair rather than a bare tensor, and ``annotate_batch_post`` multiplies
        by it. ``None`` (default) keeps the bare-tensor form.

    Attributes
    ----------
    seen_argdicts : list of dict
        A copy of the argdict every ``annotate_batch_pre`` call received, in
        order. Empty until the model is driven.
    seen_piggybacks : list
        The piggyback every ``annotate_batch_post`` call received, in order.
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window",
        (0, 0),
    )
    _annotate_args["scale"] = ("Gain applied by annotate_batch_pre", 1.0)

    def __init__(
        self,
        component_order="ZNE",
        in_channels=3,
        labels="PSN",
        classes=None,
        sampling_rate=100,
        in_samples=8,
        default_args=None,
        annotate_args=None,
        filter_args=None,
        filter_kwargs=None,
        piggyback=None,
    ):
        super().__init__(
            citation="fake model, for tests only",
            component_order=component_order,
            sampling_rate=sampling_rate,
            output_type="array",
            default_args=dict(default_args or {}),
            in_samples=in_samples,
            pred_sample=(0, in_samples),
            labels=labels,
            filter_args=filter_args,
            filter_kwargs=filter_kwargs,
        )
        self.in_channels = in_channels
        self.classes = len(labels) if classes is None else classes
        self.piggyback = piggyback
        self.seen_argdicts = []
        self.seen_piggybacks = []
        # `_annotate_args` is a class attribute in SeisBench; shadow it per
        # instance so one test can move a fallback without leaking to the next.
        self._annotate_args = dict(type(self)._annotate_args)
        for key, value in (annotate_args or {}).items():
            doc = self._annotate_args.get(key, ("Fake annotate argument", None))[0]
            self._annotate_args[key] = (doc, value)

    def forward(self, x):
        """Return ``(batch, classes, samples)`` as a fixed function of *x*."""
        weights = torch.arange(
            1, self.in_channels + 1, dtype=x.dtype, device=x.device
        ).reshape(1, -1, 1)
        pooled = (x * weights).sum(dim=-2)
        offsets = torch.arange(self.classes, dtype=x.dtype, device=x.device).reshape(
            1, -1, 1
        )
        return pooled.unsqueeze(-2) + offsets

    def annotate_batch_pre(self, batch, argdict):
        """Peak-normalise *batch*, recording the argdict it was given."""
        self.seen_argdicts.append(dict(argdict))
        scale = self._argdict_get_with_default(argdict, "scale")
        peak = batch.abs().amax(dim=-1, keepdim=True)
        normalized = scale * batch / (peak + 1e-10)
        if self.piggyback is None:
            return normalized
        return normalized, self.piggyback

    def annotate_batch_post(self, batch, piggyback, argdict):
        """Transpose to ``(batch, samples, classes)`` and blind the edges."""
        self.seen_piggybacks.append(piggyback)
        batch = torch.transpose(batch, -1, -2)
        if piggyback is not None:
            batch = batch * piggyback
        prenan, postnan = self._argdict_get_with_default(argdict, "blinding")
        if prenan > 0:
            batch[..., :prenan, :] = np.nan
        if postnan > 0:
            batch[..., -postnan:, :] = np.nan
        return batch


def fake_model(name=None, **overrides):
    """
    Build a :class:`FakeModel`, optionally from a :data:`WEIGHT_SETS` preset.

    Parameters
    ----------
    name : str, optional
        Key of :data:`WEIGHT_SETS` to start from. ``None`` (default) starts
        from :class:`FakeModel`'s own defaults.
    **overrides
        Passed to :class:`FakeModel`, overriding the preset key by key.

    Returns
    -------
    FakeModel
        A model in evaluation mode on the CPU.

    Examples
    --------
    >>> from tests.fakemodel import fake_model
    >>> model = fake_model("obs")
    >>> model.component_order, model.in_channels, list(model.labels)
    ('Z12H', 4, ['P', 'S', 'N'])
    >>> "blinding" in model.default_args
    False
    >>> model = fake_model("original", in_samples=16)
    >>> model.component_order, list(model.labels), model.in_samples
    ('ENZ', ['N', 'P', 'S'], 16)
    """
    kwargs = dict(WEIGHT_SETS[name]) if name is not None else {}
    kwargs.update(overrides)
    return FakeModel(**kwargs).eval()
