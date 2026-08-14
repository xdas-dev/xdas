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

Given the sheer size of DAS data, it is often impossible to process an entire
data set directly in memory. Hence, chunk-based processing is a necessity that
requires an additional layer of computational logistics. A naive approach would
be to load a chunk of data, apply a pipeline to it (see
[*Composing a processing sequence*](atoms.md)), and write the result to disk.
Assuming that disk I/O is the limiting factor, this scenario would leave the
CPU mostly idle as it has to wait for new data to be read and processed data to
be written to disk.

To maximise the pipeline throughput, xdas applies a staggered protocol of
reading, processing, and writing data in parallel, as illustrated in the figure
below:

![](/_static/processing.svg)

With this approach, execution time is determined by the slowest of the three
steps (reading, processing, writing) rather than by the sum of the three, a
concept known as *latency hiding*. If, for example, reading and writing a chunk
of data takes 2 seconds, and processing takes 1 second, then the total
execution time per chunk is 2 seconds instead of 5.

A second feature of xdas is that it handles the *state* of the pipeline. Many
operations (recursive filters, decimation, STA/LTA detectors) carry a memory of
the data already seen, which must be transferred from one chunk to the next.
Xdas does this for you, and it does it *knowing where the runs are*: state is
carried across chunks that follow one another, and flushed and reset wherever
the input has a gap or changes sampling rate — so a chunked run answers exactly
what the same pipeline answers in one piece.

## Example

Build and validate the pipeline on a small in-memory subset:

```{code-cell}
import numpy as np
import xdas as xd

da = xd.synthetics.wavelet_wavefronts()

pipeline = (
    xd.decimate(..., 0.02, dim="distance")
    >> xd.filter(..., (5.0, None), dim="time")
    >> np.square
)

monolithic = pipeline(da)
```

Then run the very same pipeline chunk by chunk with
{py:meth}`~xdas.atoms.Atom.process`, which infers what to read from the source
it is given and what to write from the `out` destination:

```{code-cell}
:tags: [remove-output]

chunked = pipeline.process(da, chunks={"time": 100}, out="output")
```

```{code-cell}
chunked.equals(monolithic)
```

```{code-cell}
:tags: [remove-cell]

import shutil
shutil.rmtree("output")
```

The result is identical to the monolithic run but scales to datasets that do
not fit in memory.

## Sources and destinations

`process` dispatches on what it is given, so the same pipeline serves every
scale:

| `source` | what happens |
| --- | --- |
| a `DataArray` | applied in one piece, or in chunks with `chunks=` |
| a virtual array | streamed, `chunks="auto"` following the storage layout |
| a path, a directory or a glob | opened, then streamed |
| a `DataCollection` | walked leaf by leaf, each leaf streamed |
| `xdas.watch(dir)` | a directory followed as files arrive (see [](streaming.md)) |
| `"tcp://..."` | subscribed to over ZeroMQ |

and on the destination it is given:

| `out` | what happens |
| --- | --- |
| `None` | the output chunks are accumulated and returned |
| a directory | written there, joined along the chunked dimension |
| a `*.csv` file | appended to, for pipelines that emit tables |
| `"tcp://..."` | published over ZeroMQ |
| a writer instance | used as configured |

`out=None` is the convenient form and the dangerous one: the result must fit in
memory. Beyond the `"memory_limit"` configuration entry it raises rather than
filling the machine. The default is a quarter of the memory the process can
use — of the machine's, or of what a container or a batch scheduler allows it,
whichever is smaller — so it scales with where the code runs:

```{code-cell}
xd.config.get("memory_limit") / 2**30  # GiB
```

Raise it, or lower it, with `xd.config.set("memory_limit", 64 * 2**30)`.

The explicit form remains available and is what to reach for to configure the
ends themselves — a process pool, a compression, a writer of another kind:

```{code-cell}
:tags: [remove-output]

from xdas.processing import DataArrayLoader, DataArrayWriter, process

os.makedirs("output", exist_ok=True)
dl = DataArrayLoader(da, chunks={"time": 100})
dw = DataArrayWriter("output")
chunked = process(pipeline, dl, dw)
```

```{code-cell}
:tags: [remove-cell]

shutil.rmtree("output")
```
