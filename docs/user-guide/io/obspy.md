---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
os.chdir("../../_data")

import warnings
warnings.filterwarnings("ignore")

import obspy
import numpy as np

np.random.seed(0)

network = "NX"
stations = ["SX001", "SX002", "SX003", "SX004", "SX005", "SX006", "SX007"]
location = "00"
channels = ["HHZ", "HHN", "HHE"]

nchunk = 5
chunk_duration = 60
starttimes = [
    obspy.UTCDateTime("2024-01-01T00:00:00") + idx * chunk_duration
    for idx in range(nchunk)
]
delta = 0.01
failure = 0.1

for station in stations:
    for starttime in starttimes:
        if np.random.rand() < failure:
            continue
        for channel in channels:
            data = np.random.randn(round(chunk_duration / delta))
            header = {
                "delta": delta,
                "starttime": starttime,
                "network": network,
                "station": station,
                "location": location,
                "channel": channel,
            }
            tr = obspy.Trace(data, header)
            endtime = starttime + chunk_duration
            dirpath = f"{network}/{station}"
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            fname = f"{network}.{station}.{location}.{channel}__{starttime}_{endtime}.mseed"
            path = os.path.join(dirpath, fname)
            tr.write(path)

```

# Working with miniSEED and other ObsPy formats

*Xdas* reads seismological data through ObsPy, with the engine named `"obspy"`
after the library rather than after any one format: decoding is
{py:func}`obspy.read`, so miniSEED, SAC, GSE2, SEG-2 and everything else ObsPy
supports goes through the same path.

The engine mirrors {py:func}`obspy.read` exactly: **one contiguous ObsPy
`Trace` becomes one lazy `DataArray`**, and the collection mirrors the
`Stream`, nested on the four levels of the SEED hierarchy. Nothing is decoded
at this point — the scan only records where each trace lives.

```{note}
The legacy `"miniseed"` engine is a different reader, kept for the code written
against it: it returns one stacked-channel array per file rather than a
collection. See [](data-formats.md).
```

## Reading

Our synthetic dataset holds 7 stations of 3 channels each, cut into
one-minute files, with some files missing. Point {py:func}`xdas.open` at them —
the directory layout does not have to be described, since the SEED identifiers
inside the files already say where each trace belongs.

```{code-cell}
import numpy as np
import xdas as xd

dc = xd.open("NX/*/*.mseed", engine="obspy")
dc
```

Each level is named, and the leaves are `acquisition` sequences: `xdas.open`
combines what it scanned, so contiguous traces have been fused into a single
lazy array, gaps have moved *into* the time coordinate, and a new element
appears only where something genuinely changed — a different sampling rate, a
different data type. That is why the level is no longer called `trace`.

## Selecting

Because the levels are named, {py:meth}`~xdas.DataCollection.select` gives the
semantics of `obspy.Stream.select`, with shell-style globbing on the keys:

```{code-cell}
dc.select(station="SX00[123]", channel="HH?")
```

`select` chooses *which* leaves are kept; {py:meth}`~xdas.DataMapping.sel`
trims *inside* each leaf by coordinate label. Indexing works too, and reads
like the seed id it is:

```{code-cell}
dc["NX"]["SX001"]["00"]["HHZ"][0]
```

```{note}
A blank location code, which ObsPy spells `""`, becomes `"--"` — the FDSN
convention, and the only spelling that can be a group name when the collection
is written to netCDF.
```

## Availability

```{code-cell}
:tags: [remove-output]

xd.plot_availability(dc.select(channel="HHZ"), dim="time")
```
```{code-cell}
:tags: [remove-input]
from IPython.display import HTML
fig = xd.plot_availability(dc.select(channel="HHZ"), dim="time")
HTML(fig.to_html())
```

Some data is missing. The gaps are not holes in the collection — they live in
each channel's time coordinate, and {py:func}`xdas.split` recovers the original
contiguous segments exactly:

```{code-cell}
da = dc["NX"]["SX003"]["00"]["HHZ"][0]
[part.sizes["time"] for part in xd.split(da, "gaps")]
```

## Overlapping data

Files often share a sample at their seam, or an acquisition restarts slightly
before it stopped. Those overlaps are visible as backward steps of the
coordinate, and {py:func}`xdas.trim_overlaps` resolves them by dropping the
duplicated samples — never resampling, never filling, always on a sample
boundary:

```python
dc = xd.trim_overlaps(dc)  # the later data wins, as obspy's merge(method=1)
dc = xd.trim_overlaps(dc, keep="first")
```

It recurses over a collection, preserving the tree. If instead you want to keep
every copy and look at them, `xd.split(da, "overlaps")` cuts them apart.

## Stacking channels and stations

As often, the different channels of a station are synchronized. They can be
collapsed into a dimension of a two-dimensional array with
{py:func}`xdas.stack`, which stays lazy: the level's keys become the new
coordinate, and the identifiers that do not vary stay scalar.

```{code-cell}
da = xd.stack(dc["NX"]["SX001"]["00"], "channel")[0]
da
```

All stations here are synchronized to GPS time, so once a time range without
missing data is selected they can be stacked in turn into an N-dimensional
array, ready for array analysis:

```{code-cell}
sub = dc.sel(time=slice("2024-01-01T00:01:00", "2024-01-01T00:02:59.99"))
da = xd.stack(xd.stack(sub["NX"], "channel"), "station")["00"][0]
da
```

In this example, we simply stack the energy.

```{code-cell}
trace = np.square(da).mean("channel").mean("station")
trace.plot(ylim=(0, 3))
```

All the processing capabilities of Xdas can be applied to the dataset. We encourage readers to explore the various possibilities.
