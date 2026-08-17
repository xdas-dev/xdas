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
```

# Picking seismic phases

*Xdas* runs SeisBench models as atoms, so picking a whole network is
{py:func}`xdas.open` followed by {py:func}`xdas.pick`:

```python
import seisbench.models as sbm
import xdas as xd

model = sbm.PhaseNet.from_pretrained("original")

dc = xd.open("CX_HH/*.mseed")
picks = xd.pick(dc, model)
```

`picks` is one flat `pandas.DataFrame` for the whole network — a single array
works just as well. Everything the model needs — the filter its weights ship,
the sampling rate it was trained at, how its components are ordered, which
labels it emits and at what threshold each is picked — is read off the weight
set, so nothing above has to be repeated.

```{note}
SeisBench and PyTorch are not documentation dependencies, so the cells that run
a model are shown with the output they produced rather than re-executed here.
Everything else on this page — opening the archive, cutting it at its gaps,
resampling it — runs when this page is built, on waveforms fetched at build
time.
```

## The data

One hour of the CX network (IPOC, northern Chile) on 1 May 2014, a month into
the aftershock sequence of the M8.2 Iquique earthquake — the archive SeisBench
builds its own catalog example from. Eighteen stations, three `HH?` components
each, all recorded at 100 Hz, which happens to be PhaseNet's own rate.
{py:func}`xdas.open` gives the SEED tree, without decoding anything (see
[](../io/obspy.md)):

```{code-cell}
import xdas as xd

dc = xd.open("CX_HH/*.mseed")
dc.select(station="PB01")
```

One station, `PB04`, was down until 11:43 and lost two seconds of data twelve
minutes later. That gap is not a hole in the collection — it lives in the time
coordinate, and the pipeline treats what it separates as two continuous runs,
each windowed, filtered and triggered on its own:

```{code-cell}
da = dc["CX"]["PB04"]["--"]["HHZ"][0]
[part.sizes["time"] for part in xd.split(da, "gaps")]
```

## What `xd.pick` runs

{py:func}`xdas.pick` is the eager face of the {py:class}`~xdas.atoms.Picker`
atom, which assembles the pipeline `model.classify(stream)` runs in SeisBench —
the filter the weight set declares, resampling to its rate,
{py:class}`~xdas.atoms.Annotate`, then {py:class}`~xdas.atoms.Trigger` — and
nothing else. The stages come from the weights, so two pickers over the same
model class differ:

```python
>>> from xdas.atoms import Picker
>>> picker = Picker(model)
>>> [type(stage).__name__ for stage in picker]
['Resample', 'Annotate', 'Trigger']
>>> picker[-1].thresh
{'P': 0.3, 'S': 0.3}
```

```python
>>> obs = Picker(sbm.PhaseNet.from_pretrained("obs"))
>>> [type(stage).__name__ for stage in obs]
['_ChannelFilter', 'Resample', 'Annotate', 'Trigger']
>>> obs[-1].thresh
{'P': 0.2, 'S': 0.1}
```

Three stages for one weight set and four for the other, from one architecture:
`obs` declares a 0.5 Hz highpass on its hydrophone channel (`??H`) and
`original` declares no filter at all. The target rate is the model's own —
`diting` runs at 50 Hz, not 100 — and the thresholds are read per phase off the
weight set, the noise class deliberately getting no entry so that it is never
picked. Each stage is droppable: `resample=False`, `filter=False`, an explicit
`thresh=` overriding the weight set.

The `Resample` stage is still assembled here even though this data is already
at 100 Hz; it simply has nothing to do, and passes its input through unchanged.
The last section of this page is about what happens when it does.

Nothing had to say that `HHE`, `HHN` and `HHZ` are the three components of one
instrument either. Walking the collection, the picker is offered each level
before the walk descends into it and claims the `channel` one, stacking it into
the component dimension the model expects (this is {py:func}`xdas.stack`, which
you can also call yourself); the station level, whose keys are not channel
codes, is walked past untouched. `components=False` turns the recognition off
and picks leaf by leaf.

## The pick table

```python
>>> picks.head()
  network station location  record phase                    time     value
0      CX   MNMCX       --       0     S 2014-05-01 11:00:07.530  0.746849
1      CX   MNMCX       --       0     P 2014-05-01 11:01:57.080  0.905384
2      CX   MNMCX       --       0     S 2014-05-01 11:02:28.100  0.467770
3      CX   MNMCX       --       0     P 2014-05-01 11:03:08.490  0.397209
4      CX   MNMCX       --       0     P 2014-05-01 11:04:39.600  0.425320
>>> len(picks)
369
>>> picks["phase"].value_counts()
phase
P    219
S    150
Name: count, dtype: int64
```

Each named level of the tree contributes a column, filled with the key the leaf
was reached by, and the columns are ordered identity first and measurement
last: the tree path (or the scalar coordinates of a single array), then the
other dimension coordinates, then the dimension the picks were found along,
then the value. A pick table therefore reads the same however the collection
was nested and whatever order the input's dimensions came in. The `channel`
level contributes nothing, because it became a dimension before the walk
reached a leaf — a pick belongs to an instrument, not to one of its components.

Seventeen stations appear, not eighteen: `PB04` came back too late, and too
quiet, to trigger anything in this hour.

The rows are not sorted. They come out leaf by leaf — every `MNMCX` pick before
every `PATCX` pick — and inside a leaf each phase is in time order, but the two
phases are interleaved rather than merged, so the `time` column itself is not
monotonic. Sort the frame if the order matters.

## Taking the pipeline apart

The stages are ordinary atoms, so the same thing can be run one piece at a
time. Two minutes of one station, its three components stacked into an array:

```{code-cell}
da = xd.stack(dc["CX"]["PB07"]["--"], "channel")[0]
sub = da.sel(time=slice("2014-05-01T11:20:00", "2014-05-01T11:22:00")).load()
sub
```

{py:class}`~xdas.atoms.Annotate` consumes the component dimension and appends
its classes as a `phase` dimension, laying the samples out last so that the
characteristic function of one phase is contiguous:

```python
>>> cft = xd.annotate(sub, model)
>>> cft
<xdas.DataArray (phase: 3, time: 12000)>
[[nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]]
Coordinates:
    network: 'CX'
    station: 'PB07'
    location: '--'
  * phase (phase): ['N' ... 'S']
  * time (time): 2014-05-01T11:20:00.008 to 2014-05-01T11:21:59.998
```

The `nan` values at both ends of each row are the ends of the record, blinded
by the model's own `annotate_batch_post` as SeisBench blinds them; the values
in between are the characteristic function.

The `phase` coordinate carries the model's own labels, so a class is selected
by name — `cft.sel(phase="P")` — and never by position. That matters more than
it looks: the label order is a property of the weight set. It is `NPS` here,
but `PSN` on fourteen of the seventeen cached PhaseNet weight sets — including
`iquique`, trained on this very sequence — so anything positional silently
addresses the wrong phase on the next set of weights. {py:class}`~xdas.atoms.Trigger` keys its thresholds
the same way:

```python
>>> xd.trigger(cft, thresh={"P": 0.3, "S": 0.3})
  network station location phase                       time     value
0      CX    PB07       --     P 2014-05-01 11:20:08.798393  0.947520
1      CX    PB07       --     P 2014-05-01 11:20:50.268393  0.971352
2      CX    PB07       --     S 2014-05-01 11:20:15.898393  0.841123
3      CX    PB07       --     S 2014-05-01 11:20:57.418393  0.937744
```

A label the mapping does not name is never triggered, which is how the noise
class keeps being computed and carried without ever producing a pick. The four
picks are the same four the whole-hour run found on this station, but not to
the sample: the strongest keeps its timing exactly, while the others move by
ten or twenty milliseconds, and the first changes confidence markedly
(0.67 to 0.95). The model sees two minutes of record here instead of an hour,
so its windows fall in different places.

## The same walk at two scales

Nothing above is specific to seismometers: a DAS acquisition is a lane per
channel, and a DAS archive is a collection like any other. Take a synthetic
cable pair, each recorded as two consecutive records:

```python
>>> from xdas.synthetics import randn_wavefronts
>>> da = randn_wavefronts().isel(distance=slice(None, None, 200))
>>> das = xd.DataCollection(
...     {
...         cable: xd.DataCollection(
...             xd.split(da.sel(distance=slice(*bounds)), 2, dim="time"), "record"
...         )
...         for cable, bounds in {"east": (0, 40000), "west": (60000, 100000)}.items()
...     },
...     "cable",
... )
>>> das
<xdas.DataCollection: 4 leaves, 937.5 KB>
cable  record
east        0  (time: 10000, distance: 3)  234.4 KB
            1  (time: 10000, distance: 3)  234.4 KB
west        0  (time: 10000, distance: 3)  234.4 KB
            1  (time: 10000, distance: 3)  234.4 KB
```

Calling the picker walks that tree in memory:

```python
>>> picker = xd.pick(..., model)
>>> picker(das)
  cable  record  distance phase                    time     value
0  east       0   20000.0     P 2024-01-01 00:00:39.170  0.390122
1  east       0   40000.0     P 2024-01-01 00:00:35.640  0.402109
2  east       0   40000.0     S 2024-01-01 00:00:39.930  0.371556
3  east       0       0.0     S 2024-01-01 00:00:53.560  0.346529
4  west       0   60000.0     P 2024-01-01 00:00:35.560  0.305741
5  west       0   60000.0     S 2024-01-01 00:00:39.880  0.330890
6  west       0   80000.0     P 2024-01-01 00:00:38.960  0.452030
```

`process()` walks the very same tree, but streams each leaf in chunks instead
of loading it — which is the only form left once a leaf is an archive rather
than an array. With `out=None` the results are accumulated and merged, and the
two answers are the same table:

```python
>>> streamed = picker.process(das, chunks={"time": 5000}, out=None)
>>> streamed.equals(picker(das))
True
```

The state carries across the chunks and across the records of a sequence,
and the tables are labelled as each leaf is produced, so a sink can be given
the rows directly. A `*.csv` destination is *shared* — every leaf appends to
one table, the `cable` and `record` columns keeping the rows apart:

```python
>>> picker.process(das, chunks={"time": 5000}, out="picks.csv")
```

A directory destination fans out instead, one subdirectory per leaf mirroring
the tree path. See [](processing.md) for the rest of the source and sink
vocabulary, and [](streaming.md) for picking a stream as it arrives.

## How close is this to SeisBench?

Close enough to be worth stating precisely — and the answer splits in two,
depending on whether the resampler has anything to do.

### At the model's own rate

This hour is recorded at 100 Hz, which is what `original` was trained at, so
the `Resample` stage passes its input through untouched and every other stage
is measured on exactly the same samples. Over the eighteen stations, twenty
minutes of each, against `model.classify(stream)`:

- annotation is **bit-identical** — a maximum absolute difference of exactly 0
  over 6.3 million samples — once both sides are given the same batch size. The
  ~1e-7 that appears otherwise is float32 convolution non-associativity between
  a one-window and a 256-window batch, not a difference of method;
- triggering differs in exactly one place, a sample whose value equals the
  threshold exactly: ObsPy's `trigger_onset` turns on at `>= thresh`, *xdas* at
  `> thresh`. It did not occur anywhere in this dataset;
- end to end, `xd.pick` and `model.classify` each produced **56 picks, every
  one of them on the same sample**.

That is not a property of one weight set. Repeated over all seventeen cached
PhaseNet weight sets — four stations, ten minutes each — annotation is
bit-identical on every one of them, exactly 0 over 11.8 million samples, and so
is the preprocessing filter wherever a weight set declares one (only `obs`
does: exactly 0 over 960 thousand samples). Fed the same resampled data, the
two sides produced **650 picks each, every one on the same sample**. Letting
each side resample for itself, they differ by a single pick out of 651 — and
that one difference, together with timing shifts of up to 0.92 s, comes
entirely from `diting`, the only weight set in the sweep that does not run at
100 Hz.

### What the resampler costs

Which leaves the resampler as the one real deviation, and it is a deviation of
passband rather than of care. SeisBench resamples with ObsPy's
`Trace.resample`, which defaults to `window="hann"` and applies that window *in
the frequency domain*: unity at DC, zero at Nyquist, and half the amplitude at
half the input Nyquist. *Xdas* resamples with a polyphase FIR — the only form
that can run chunk by chunk — which is flat across that band.

The same eighteen stations also record at 20 Hz, on their `BH?` channels, and
those are enough to show the window directly. Dividing one spectrum by the
other leaves the taper and nothing else:

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import obspy

bh = xd.open("CX_BH/*.mseed")

ratios = []
for station in ["PB01", "PB05", "PB09", "PB11", "PB16"]:
    ours = xd.resample(bh["CX"][station]["--"]["BHZ"][0].load(), 100.0)
    theirs = obspy.read(f"CX_BH/CX.{station}.mseed").select(channel="BHZ")[0]
    theirs.resample(100.0)
    n = min(ours.sizes["time"], theirs.stats.npts)
    a = np.abs(np.fft.rfft(np.asarray(ours.values[:n], "float64")))
    b = np.abs(np.fft.rfft(theirs.data[:n].astype("float64")))
    ratios.append(b / np.where(a == 0, np.nan, a))

freq = np.fft.rfftfreq(n, 0.01)
ratio = np.convolve(np.nanmedian(ratios, axis=0), np.ones(201) / 201, mode="same")

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(freq, ratio)
ax.set(
    xlim=(0, 10),
    ylim=(0, 1.15),
    xlabel="frequency [Hz]",
    ylabel="obspy / xdas amplitude",
)
fig.tight_layout()
```

Unity at the left, a half at 5 Hz — half of the 10 Hz input Nyquist — and
nothing left by 10 Hz. On 20 Hz records that taper sits squarely on the band
where P and S energy lives, and the consequence is easy to understate if you
only look at the waveforms. Over the eighteen stations, twenty minutes each:

- the resampled **waveforms** barely differ: a median relative RMS of 1.3 %
  (0.1 % to 6.0 % across the stations);
- the **characteristic function** the model computes from them differs
  enormously: a median maximum difference of 0.49, on a scale that runs from
  0 to 1;
- and so the **picks** differ: 44 from *xdas* against 21 from SeisBench, with
  only 15 in common;
- fed the *same* resampled data, both sides produce 21 picks, every one of them
  on the same sample.

A one-percent difference in the waveform becomes a doubling of the detections,
because the percent that goes missing is the part the network was looking at.
Reproducing SeisBench here would mean reproducing a worse resampler, so *xdas*
does not. `resample=False` drops the stage — pass it when the data is already
at the model's rate, or to compare the two implementations on the same
waveforms.
