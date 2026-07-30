---
file_format: mystnb
kernelspec:
  name: python3
---

# Interpolated Coordinates

## Overview

Because DAS data are generally sampled with a constant sampling rate,
keeping the corresponding value for each index as a dense array is
inefficient.  *xdas* stores such coordinates using the
[CF convention][CF] through the
{py:class}`~xdas.coordinates.InterpCoordinate` class.  Only a few tie
points are kept; intermediate values are recovered by linear interpolation.
Discontinuities are marked by two consecutive tie points at adjacent
indices, as illustrated below:

![](/_static/coordinate.svg)

The resulting representation is sparse but contains all the information
needed to exactly recover the original dense coordinate vector.

## Creating an InterpCoordinate

The {py:class}`~xdas.coordinates.InterpCoordinate` constructor takes
`tie_indices` and `tie_values` as keys in a dict.  The code below
corresponds with the example illustrated in the figure above:

```{code-cell}
import xdas as xd

coord = xd.Coordinate(
    {
        "tie_indices": [0, 9, 19, 20, 29],
        "tie_values": [0.0, 90.0, 190.0, 400.0, 490.0],
    }
)
coord
```

`xd.Coordinate(...)` acts as a factory and returns an
{py:class}`~xdas.coordinates.InterpCoordinate` when the dict contains
`tie_indices` and `tie_values`.

The coordinate behaves like a numpy array — indexing and slicing work
out of the box.  Note that when specifying a step greater than 1, tie
points may shift slightly to remain on the sampled grid.

```{code-cell}
coord = coord[1:-3:2]
coord
```

## Label-based selection

A major advantage of {py:class}`~xdas.coordinates.InterpCoordinate` is
that it enables label-based selection.  To retrieve the integer index
corresponding to a given value, use the {py:meth}`~xdas.coordinates.AxisCoordinate.to_index`
method:

```{code-cell}
coord.to_index(430.0)
```

```{warning}
To enable label-based selection, `tie_values` must be strictly increasing
(no overlaps).  To deal with small overlaps, use
{py:meth}`~xdas.coordinates.InterpCoordinate.simplify` with a tolerance
large enough to absorb them.
```

## Gaps and overlaps

Gaps and overlaps can be identified from the tie-point positions and
extracted with:

```{code-cell}
coord.get_discontinuities()
```

Gaps represent missing data and are generally not problematic; overlaps
usually arise from labelling errors and should be resolved.

Using the {py:meth}`~xdas.coordinates.InterpCoordinate.simplify` method,
the coordinate can be compressed with controlled accuracy using the
[Ramer–Douglas–Peucker algorithm][RDP].  In the example below, the
second tie point carries no additional information and is safely discarded:

```{code-cell}
coord = coord.simplify(tolerance=0.0)
coord
```

## Regular coordinates

An interpolated coordinate can optionally carry a nominal
`sampling_interval` (and a `tolerance` bounding the allowed jitter around
it), making it *regular*.  Signal-processing routines (filtering, FFT,
resampling) require a regular coordinate to obtain a clean sample rate;
{py:meth}`~xdas.coordinates.Coordinate.isregular` tells whether a
coordinate carries one.  Coordinates built by the file engines or by
{py:meth}`~xdas.coordinates.InterpCoordinate.from_block` are regular out
of the box:

```{code-cell}
coord = xd.Coordinate(
    {
        "tie_indices": [0, 9],
        "tie_values": [0.0, 90.0],
        "sampling_interval": 10.0,
    }
)
coord.isregular()
```

An irregular coordinate whose values are in fact evenly spaced can be
promoted explicitly with
{py:meth}`~xdas.coordinates.AxisCoordinate.to_regular`, which infers the
spacing when it is not given and raises on genuinely irregular axes.
Data saved by earlier *xdas* versions carries no declared spacing; for
now, signal-processing routines fall back to inferring one and emit a
{py:exc}`FutureWarning` telling you the tolerance required — promote the
coordinate as shown below to silence it:

```{code-cell}
coord = xd.Coordinate({"tie_indices": [0, 9], "tie_values": [0.0, 90.0]})
coord.to_regular().get_sampling_interval()
```

For jittery axes, pass a `tolerance`: the declared spacing is accepted as
long as every continuous segment stays within it.  The stored tolerance
is also the default accuracy budget of
{py:meth}`~xdas.coordinates.InterpCoordinate.simplify`, so chunk seams
introduced by piecewise processing fuse back automatically on
concatenation.  `simplify(regularize=True)` combines both steps: it drops
redundant tie points and promotes the result to regular when the
surviving segments admit a single spacing within the budget.

## Temporal coordinates

The most common use of interpolated coordinates in *xdas* is handling
long time series.  By default *xdas* uses `"datetime64[us]"` dtype.
Microseconds are used because interpolation internally converts
`datetime64` to POSIX floats, which cannot safely represent finer
resolution.

```{code-cell}
import numpy as np

coord = xd.Coordinate(
    {
        "tie_indices": [0, 3600 * 100],
        "tie_values": [
            np.datetime64("2023-01-01T00:00:00"),
            np.datetime64("2023-01-01T01:00:00"),
        ],
    }
)
coord.to_index(slice("2023-01-01T00:10:00", "2023-01-01T00:20:00"))
```

[CF]: <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#compression-by-coordinate-subsampling>
[RDP]: <https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm>
