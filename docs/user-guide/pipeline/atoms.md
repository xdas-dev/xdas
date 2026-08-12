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

*Xdas* ships a processing vocabulary — filtering, resampling, integration,
spectra, machine-learning picking — as *atoms*: elementary operations that
compose into a pipeline. A pipeline built this way runs unchanged on an array
in memory and, chunk by chunk, on an archive that does not fit in one (see
[](processing.md)), which is what makes it worth defining one rather than
calling functions in a row.

## Applying and composing

Every atom has a function form at the top level of `xdas`. Called on data, it
applies:

```{code-cell}
import numpy as np
import xdas as xd

da = xd.synthetics.wavelet_wavefronts()
xd.filter(da, (5.0, None), dim="time")
```

Called on `...` — the placeholder standing for the data to come — the same
function returns the atom instead, and atoms compose with `>>`:

```{code-cell}
pipeline = (
    xd.taper(..., dim="time")
    >> xd.filter(..., (5.0, None), dim="time")
    >> xd.decimate(..., 25.0, dim="time")
)
pipeline
```

The parameters are physical: corner frequencies in hertz, target rates in
hertz, window lengths in seconds. They keep their meaning whatever the sampling
rate of the data the pipeline is later given.

Calling the pipeline applies it:

```{code-cell}
result = pipeline(da)
result.plot(yincrease=False)
```

The same pipeline can be reused: it is defined once and carries no data.

Ordinary NumPy expressions compose too. Under the `...` seed they are *traced*
— appended to the pipeline rather than computed — so an expression reads as
mathematics:

```{code-cell}
energy = 20 * np.log10(np.abs(xd.decimate(..., 25.0, dim="time")))
energy
```

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
