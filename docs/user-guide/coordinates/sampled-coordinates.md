---
file_format: mystnb
kernelspec:
  name: python3
---

# Sampled Coordinates

## Overview

A {py:class}`~xdas.coordinates.SampledCoordinate` describes a coordinate whose values are
spaced by a **fixed sampling interval**. Unlike {py:class}`~xdas.coordinates.InterpCoordinate`,
which uses piecewise linear interpolation between arbitrary tie points,
`SampledCoordinate` exploits the regularity of the grid and stores only:

- `tie_values` — the start value of each contiguous segment.
- `tie_lengths` — the number of samples in each segment.
- `sampling_interval` — the fixed step shared by all segments.

This makes it more compact and numerically stable than the interpolated variant, and it
maps directly to the block-based time representation used in the miniSEED and SEED formats.

```{code-cell}
import numpy as np
import xdas as xd
from xdas.coordinates import SampledCoordinate

coord = SampledCoordinate(
    {
        "tie_values": [0.0, 200.0],
        "tie_lengths": [10, 8],
        "sampling_interval": 10.0,
    }
)
coord
```

The two segments start at 0 and 200, each stepping by 10. The gap between them (from 100
to 200) is explicit — there is simply no segment covering that range.

## Materialising values

Calling `.values` returns the full coordinate vector as a dense NumPy array:

```{code-cell}
coord.values
```

Individual values are obtained from indices with `.get_value` and the reverse mapping
(index from value) with `.to_index`:

```{code-cell}
coord.get_value(3)
```

```{code-cell}
coord.to_index(30.0)
```

## Datetime coordinates

The most common use of `SampledCoordinate` is for the time axis of DAS data. The
`sampling_interval` must be a `numpy.timedelta64`:

```{code-cell}
t0 = np.datetime64("2024-01-01T00:00:00.000", "ms")
dt = np.timedelta64(4, "ms")  # 250 Hz

coord = SampledCoordinate(
    {
        "tie_values": [t0, t0 + np.timedelta64(1, "s")],
        "tie_lengths": [250, 250],
        "sampling_interval": dt,
    }
)
coord
```

## Gaps and multi-segment coordinates

Multiple segments represent an acquisition with gaps. Each element of `tie_values` marks
the start of one contiguous block, and `tie_lengths` gives its duration in samples:

```{code-cell}
t0 = np.datetime64("2024-01-01T00:00:00.000", "ms")
dt = np.timedelta64(4, "ms")

segments = [
    (t0,                             500),   # 2 s block
    (t0 + np.timedelta64(3, "s"),   500),   # another 2 s block after a 1 s gap
]
coord = SampledCoordinate(
    {
        "tie_values": [s[0] for s in segments],
        "tie_lengths": [s[1] for s in segments],
        "sampling_interval": dt,
    }
)
coord
```

## Simplifying near-regular coordinates

When opening files whose timestamps are not perfectly aligned (e.g. NTP-synchronized
acquisitions), small drifts create many short segments. The `simplify` method merges
segments whose start time is within `tolerance` of the expected position:

```{code-cell}
t0 = np.datetime64("2024-01-01T00:00:00.000", "ms")
dt = np.timedelta64(4, "ms")

# Simulate a one-sample drift at the boundary
t1_drifted = t0 + np.timedelta64(2000, "ms") + np.timedelta64(1, "ms")

coord = SampledCoordinate(
    {
        "tie_values": [t0, t1_drifted],
        "tie_lengths": [500, 500],
        "sampling_interval": dt,
    }
)
print("Before:", len(coord.tie_values), "segments")

tol = np.timedelta64(10, "ms")
coord = coord.simplify(tol)
print("After: ", len(coord.tie_values), "segments")
coord
```

## When to use SampledCoordinate vs InterpCoordinate

| | `SampledCoordinate` | `InterpCoordinate` |
|---|---|---|
| Sampling | Strictly uniform (one `sampling_interval`) | Variable (piecewise linear) |
| Memory | Very compact | Compact |
| Use case | DAS / seismic time axes, uniform grids | Non-uniform grids, GPS-corrected time |
| miniSEED/SEED compatible | Yes | No |
