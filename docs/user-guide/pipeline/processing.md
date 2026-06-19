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

# Processing larger-than-memory data

## Chunked processing: basic concepts

Given the sheer size of DAS data, it is often impossible to process an entire data set directly in memory. Hence, chunked-based processing is a necessity that requires an additional layer of computational logistics. A naive approach to chunked processing would be to load a chunk of data, apply a `Sequential` pipeline to it (see [*Composing a processing sequence*](atoms.md)), and write the resulting data to disk. Assuming that disk I/O is the limiting factor, this scenario would leave the CPU mostly idle as it has to wait for new data to be read and processed data to be written to disk.

To maximise the pipeline throughput, xdas applies a staggered protocol of reading, processing, and writing data in parallel, as illustrated in the figure below:

![](/_static/processing.svg)

With this approach, execution time is determined by the slowest of the three steps (reading, processing, writing) rather than by the sum of the three, a concept known as *latency hiding*. If, for example, reading and writing a chunk of data takes 2 seconds, and processing takes 1 second, then the total execution time per chunk is 2 seconds instead of 5.

A second feature of xdas, is that it automatically handles state updates and transfer. Many types of filters (e.g. recursive filters and STA/LTA algorithms) rely on some kind of memory of previously seen data, known as the *state* of the filter. The state of each filter needs to be preserved and transferred from one chunk to the next. Moreover, if the computation pipeline gets interrupted and needs to be restarted, the states need to be properly initialised for a seamless continuation. xdas offers optimised filters that handle state updates internally.

## Example

The following example shows how to apply a simple processing pipeline to a large dataset.
First, build and validate the pipeline on a small in-memory subset:

```{code-cell}
:tags: [remove-output]

import numpy as np
import xdas as xd
import xdas.signal as xs
from xdas.atoms import Sequential, Partial, LFilter
from xdas.processing import process, DataArrayLoader, DataArrayWriter
from scipy.signal import iirfilter

da = xd.synthetics.wavelet_wavefronts()

b, a = iirfilter(4, 0.1, btype="high")

atom = Sequential(
    [
        Partial(xs.decimate, 2, ftype="fir", dim="distance"),
        LFilter(b, a, dim="time"),
        Partial(np.square),
    ]
)

monolithic = atom(da)
```

Then apply the same pipeline chunk-by-chunk using {py:func}`~xdas.processing.process`.
The {py:class}`~xdas.processing.DataArrayLoader` splits the input into fixed-size chunks
along a given dimension, while {py:class}`~xdas.processing.DataArrayWriter` collects and
writes each processed chunk to a directory on disk:

```{code-cell}
:tags: [remove-output]

import os
os.makedirs("output", exist_ok=True)

dl = DataArrayLoader(da, chunks={"time": 100})
dw = DataArrayWriter("output")
chunked = process(atom, dl, dw)

assert chunked.equals(monolithic)
```

```{code-cell}
:tags: [remove-cell]

import shutil
shutil.rmtree("output")
```

The result is identical to the monolithic run but can scale to datasets that do not fit in
memory. The loader and writer can be swapped for other variants — for example,
{py:class}`~xdas.processing.ZMQPublisher` to stream results over a network (see
[](streaming.md)).
