---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
import xdas as xd
os.chdir("../_data")
```

# Process big dataarrays

```{warning}
The API of this part of xdas is still experimental.
```

## Chunked processing: basic concepts

Given the sheer size of DAS data, it is often impossible to process an entire data set directly in memory. Hence, chunked-based processing is a necessity that requires an additional layer of computational logistics. A naive approach to chunked processing would be to load a chunk of data, apply a `Sequential` pipeline to it (see [*Composing a processing sequence*](atoms.md)), and write the resulting data to disk. Assuming that disk I/O is the limiting factor, this scenario would leave the CPU mostly idle as it has to wait for new data to be read and processed data to be written to disk.

To maximise the pipeline throughput, xdas applies a staggered protocol of reading, processing, and writing data in parallel, as illustrated in the figure below:

![](/_static/processing.svg)

With this approach, execution time is determined by the slowest of the three steps (reading, processing, writing) rather than by the sum of the three, a concept known as *latency hiding*. If, for example, reading and writing a chunk of data takes 2 seconds, and processing takes 1 second, then the total execution time per chunk is 2 seconds instead of 5.

A second feature of xdas, is that it automatically handles state updates and transfer. Many types of filters (e.g. recursive filters and STA/LTA algorithms) rely on some kind of memory of previously seen data, known as the *state* of the filter. The state of each filter needs to be preserved and transferred from one chunk to the next. Moreover, if the computation pipeline gets interrupted and needs to be restarted, the states need to be properly initialised for a seamless continuation. xdas offers optimised filters that handle state updates internally.

## Example

**TODO**