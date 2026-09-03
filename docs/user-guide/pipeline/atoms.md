---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
import xdas as xd
os.chdir("../../_data")
```

# Composing a processing sequence

*Xdas* ships its processing vocabulary — filtering, resampling, integration,
spectra, machine-learning picking — in two interchangeable faces. The
**function** is the one you write day to day; the **atom** is the same
operation as an object, which composes into a pipeline and streams. Every
function has an atom form and every atom a function form, so a pipeline built
here runs unchanged on an array in memory and, chunk by chunk, on an archive
that does not fit in one (see [](processing.md)).

The parameters are physical throughout — a rate in hertz, a corner frequency,
a window in seconds — rather than the arguments of the SciPy routine
underneath (a decimation factor, a sample count, a normalised frequency). That
is what lets one pipeline be defined once and applied to whatever it is given:
nothing in it silently means something else at another sampling rate. The
SciPy-shaped functions are still there, in {py:mod}`xdas.signal`.

## Applying and composing

Called on data, the function applies:

```{code-cell}
import numpy as np
import xdas as xd

da = xd.synthetics.wavelet_wavefronts()
xd.resample(da, 25.0, dim="time")
```

Called on `...` — the placeholder standing for the data to come — the same
function returns the atom instead, and atoms compose with `>>`:

```{code-cell}
pipeline = (
    xd.taper(..., dim="time")
    >> xd.filter(..., (5.0, None), dim="time")
    >> xd.resample(..., 25.0, dim="time")
)
pipeline
```

Nothing in that pipeline names the data it will be given: `25.0` is the rate to
land on, not a decimation factor, so the same object takes a 50 Hz record and a
1 kHz one to 25 Hz — resampling by a rational ratio where it has to. Calling it
applies it:

```{code-cell}
result = pipeline(da)
result.plot(yincrease=False)
```

The same pipeline can be reused: it is defined once and carries no data.

Ordinary NumPy expressions compose too. Under the `...` seed they are *traced*
— appended to the pipeline rather than computed — so an expression reads as
mathematics:

```{code-cell}
energy = 20 * np.log10(np.abs(xd.resample(..., 25.0, dim="time")))
energy
```

## The vocabulary

| | |
| --- | --- |
| {py:func}`~xdas.filter` | band, low- or high-pass, from a corner pair in Hz |
| {py:func}`~xdas.resample` | to a target `rate`, `interval` or `up`/`down` ratio, by `method="fir"` (default), `"iir"` or `"fft"` |
| {py:func}`~xdas.integrate`, {py:func}`~xdas.differentiate` | in the coordinate's own units |
| {py:func}`~xdas.stft` | window and hop in seconds |
| {py:func}`~xdas.detrend`, {py:func}`~xdas.taper` | whole-record shaping |
| {py:func}`~xdas.hilbert` | analytic signal |
| {py:func}`~xdas.medfilt`, {py:func}`~xdas.sliding_mean_removal` | kernels in seconds or meters |
| {py:func}`~xdas.rechunk` | a streaming-cadence knob, not an operation |
| {py:func}`~xdas.stalta` | short-term over long-term average, windows in seconds |
| {py:func}`~xdas.annotate`, {py:func}`~xdas.trigger`, {py:func}`~xdas.pick` | detection and picking (see [](picking.md)) |

Each has an atom behind it — {py:class}`~xdas.atoms.Filter`,
{py:class}`~xdas.atoms.Resample`, and so on — reached by seeding with `...`.

Below them sits an expert layer, {py:mod}`xdas.atoms.kernel`, holding the exact
primitives these design from the data at the first call:
{py:class}`~xdas.atoms.LFilter`, {py:class}`~xdas.atoms.SOSFilter`,
{py:class}`~xdas.atoms.DownSample` and friends, which take machine parameters —
filter coefficients, integer factors — rather than physical ones. Reach for
them when you need to say exactly what runs; otherwise you should never have to
meet them.

## Wrapping your own functions

Any callable taking a data array as its first argument becomes an atom by
composition — `>>` wraps it:

```{code-cell}
pipeline = xd.taper(..., dim="time") >> np.square
pipeline
```

`Partial` does the same explicitly, and is what to reach for when the extra
arguments have to be given at definition time:

```{code-cell}
from xdas.atoms import Partial

Partial(np.percentile, ..., 90.0, axis=0)
```

## Defining custom atoms

An operation that carries a *state* from one chunk to the next — a recursive
filter, a running mean, a detector — is written as a subclass of `Atom`.
`call` maps one input chunk to zero or more output chunks, and `flush` emits
whatever remains buffered at the end of a run:

```{code-cell}
from xdas.atoms import Atom, State

class MyStatefulRoutine(Atom):

    def __init__(self, a, dim="time"):
        super().__init__()
        # Configuration: kept as-is, shared by clones of this atom
        self.a = a
        self.dim = dim
        # State: reset between runs, carried across the chunks of one run
        self.buffer = State(...)

    def initialize(self, da, **flags):
        # Called on the first chunk of a run, to size the state from the data
        self.buffer = ...

    def call(self, da, **flags):
        # Applied to every chunk; may return nothing, one chunk, or several
        ...

    def flush(self):
        # Called at the end of a run: emit what is still buffered
        return []
```

*Xdas* handles the rest: state is carried across chunk boundaries, flushed and
reset at every gap or sampling-rate change of the input, and `flush` is called
at the end of the stream. {py:func}`xdas.testing.assert_chunk_invariant` checks
that a pipeline gives the very same answer eagerly and chunk by chunk, gaps
included — the thing worth verifying before trusting a custom atom on an
archive.

For executing a pipeline on chunked data, see the next section:
[](processing.md).
