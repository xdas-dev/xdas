# Picking seismic phases

*Xdas* runs SeisBench models as atoms, so picking a whole network is
{py:func}`xdas.open` followed by {py:func}`xdas.pick`:

```python
import seisbench.models as sbm
import xdas as xd

model = sbm.PhaseNet.from_pretrained("original")

dc = xd.open("20May2026_LabuanBajo/*.mseed")
picks = xd.pick(dc, model)
```

`picks` is one flat `pandas.DataFrame` for the whole network — a single array
works just as well. Everything the model needs — the filter its weights ship,
the sampling rate it was trained at, how its components are ordered, which
labels it emits and at what threshold each is picked — is read off the weight
set, so nothing above has to be repeated.

```{note}
Unlike the rest of this guide, the code on this page is not executed when the
documentation is built: it needs SeisBench and its downloaded weights. Every
output shown was produced by running the code as written — on the day of data
described below, or, for the DAS example, on the synthetic collection built
there.
```

## The data

One day of a temporary network in eastern Indonesia: 8 short-period stations,
three `SH?` components each, six of them recorded at 40 Hz and two (`LBFI`,
`LEMFI`) at 50 Hz — PhaseNet wants 100. {py:func}`xdas.open` gives the SEED
tree, without decoding anything (see [](../io/obspy.md)):

```python
>>> dc.select(station="DBNFM")
Network:
  IA:
    Station:
      DBNFM:
        Location:
          --:
            Channel:
              SHE:
                Acquisition:
                  0: <xdas.DataArray (time: 3456001)>
              SHN:
                Acquisition:
                  0: <xdas.DataArray (time: 3456001)>
              SHZ:
                Acquisition:
                  0: <xdas.DataArray (time: 3456001)>
```

One station, `OMBFM`, started at 00:43 and lost a second of data twelve minutes
later. That gap is not a hole in the collection — it lives in the time
coordinate, and the pipeline treats what it separates as two continuous runs,
each windowed, filtered and triggered on its own:

```python
>>> da = dc["IA"]["OMBFM"]["--"]["SHZ"][0]
>>> [part.sizes["time"] for part in xd.split(da, "gaps")]
[28160, 3130760]
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

Nothing had to say that `SHE`, `SHN` and `SHZ` are the three components of one
instrument either. Walking the collection, the picker is offered each level
before the walk descends into it and claims the `channel` one, stacking it into
the component dimension the model expects (this is {py:func}`xdas.stack`, which
you can also call yourself); the station level, whose keys are not channel
codes, is walked past untouched. `components=False` turns the recognition off
and picks leaf by leaf.

## The pick table

```python
>>> picks.head()
  network station location  acquisition phase                        time     value
0      IA   DBNFM       --            0     P  2026-05-20 00:22:43.446860  0.410896
1      IA   DBNFM       --            0     P  2026-05-20 00:58:37.956860  0.544187
2      IA   DBNFM       --            0     P  2026-05-20 02:09:35.936860  0.683518
3      IA   DBNFM       --            0     P  2026-05-20 02:16:24.486860  0.733029
4      IA   DBNFM       --            0     P  2026-05-20 02:50:49.036860  0.342554
>>> len(picks)
1918
>>> picks["phase"].value_counts()
phase
P    985
S    933
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

The rows are not sorted: they come out lane by lane and leaf by leaf, so a
station's `P` picks precede its `S` picks. Sort the frame if the order matters.

## Taking the pipeline apart

The stages are ordinary atoms, so the same thing can be run one piece at a
time. Two minutes of one station, its three components stacked into an array:

```python
>>> da = xd.stack(dc["IA"]["DBNFM"]["--"], "channel")[0]
>>> sub = da.sel(time=slice("2026-05-20T00:22:00", "2026-05-20T00:24:00")).load()
>>> sub
<xdas.DataArray (channel: 3, time: 4800)>
[[-1827 -1832 -1783 ... -1788 -2109 -2192]
 [  688   725   623 ...   656   742   828]
 [ 2990  2976  2972 ...  2867  2906  2755]]
Coordinates:
    network: 'IA'
    station: 'DBNFM'
    location: '--'
  * time (time): 2026-05-20T00:22:00.021 to 2026-05-20T00:23:59.996
  * channel (channel): ['SHE' ... 'SHZ']
```

{py:class}`~xdas.atoms.Annotate` consumes the component dimension and appends
its classes as a `phase` dimension, laying the samples out last so that the
characteristic function of one phase is contiguous:

```python
>>> cft = xd.annotate(xd.resample(sub, 100.0), model)
>>> cft
<xdas.DataArray (phase: 3, time: 12000)>
[[nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]
 [nan nan nan ... nan nan nan]]
Coordinates:
    network: 'IA'
    station: 'DBNFM'
    location: '--'
  * phase (phase): ['N' ... 'S']
  * time (time): 2026-05-20T00:21:59.771 to 2026-05-20T00:23:59.761
```

The `nan` values at both ends of each row are the ends of the record, blinded
by the model's own `annotate_batch_post` as SeisBench blinds them; the values
in between are the characteristic function.

The `phase` coordinate carries the model's own labels, so a class is selected
by name — `cft.sel(phase="P")` — and never by position. That matters more than
it looks: the label order is a property of the weight set, `NPS` here and `PSN`
for most other PhaseNet weights, so anything positional silently addresses the
wrong phase on the next set of weights. {py:class}`~xdas.atoms.Trigger` keys its
thresholds the same way:

```python
>>> xd.trigger(cft, thresh={"P": 0.3, "S": 0.3})
  network station location phase                       time     value
0      IA   DBNFM       --     P 2026-05-20 00:22:43.461860  0.711655
1      IA   DBNFM       --     P 2026-05-20 00:23:53.941860  0.578763
```

A label the mapping does not name is never triggered, which is how the noise
class keeps being computed and carried without ever producing a pick. Both the
timing and the list differ a little from the whole-day run above — the first
pick moves by 15 ms and a second, weaker one appears — because the model, and
before it the resampler, see two minutes of record here instead of a day.

## The same walk at two scales

Nothing above is specific to seismometers: a DAS acquisition is a lane per
channel, and a DAS archive is a collection like any other. Take a synthetic
cable pair, each recorded as two consecutive acquisitions:

```python
>>> from xdas.synthetics import randn_wavefronts
>>> da = randn_wavefronts().isel(distance=slice(None, None, 200))
>>> das = xd.DataCollection(
...     {
...         cable: xd.DataCollection(
...             xd.split(da.sel(distance=slice(*bounds)), 2, dim="time"), "acquisition"
...         )
...         for cable, bounds in {"east": (0, 40000), "west": (60000, 100000)}.items()
...     },
...     "cable",
... )
>>> das
Cable:
  east:
    Acquisition:
      0: <xdas.DataArray (time: 10000, distance: 3)>
      1: <xdas.DataArray (time: 10000, distance: 3)>
  west:
    Acquisition:
      0: <xdas.DataArray (time: 10000, distance: 3)>
      1: <xdas.DataArray (time: 10000, distance: 3)>
```

Calling the picker walks that tree in memory:

```python
>>> picker = xd.pick(..., model)
>>> picker(das)
  cable  acquisition  distance phase                    time     value
0  east            0       0.0     S 2024-01-01 00:00:53.560  0.346529
1  east            0   20000.0     P 2024-01-01 00:00:39.170  0.390122
2  east            0   40000.0     P 2024-01-01 00:00:35.640  0.402109
3  east            0   40000.0     S 2024-01-01 00:00:39.930  0.371556
4  west            0   60000.0     P 2024-01-01 00:00:35.560  0.305741
5  west            0   60000.0     S 2024-01-01 00:00:39.880  0.330890
6  west            0   80000.0     P 2024-01-01 00:00:38.960  0.452030
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

The state carries across the chunks and across the acquisitions of a sequence,
and the tables are labelled as each leaf is produced, so a sink can be given
the rows directly. A `*.csv` destination is *shared* — every leaf appends to
one table, the `cable` and `acquisition` columns keeping the rows apart:

```python
>>> picker.process(das, chunks={"time": 5000}, out="picks.csv")
```

A directory destination fans out instead, one subdirectory per leaf mirroring
the tree path. See [](processing.md) for the rest of the source and sink
vocabulary, and [](streaming.md) for picking a stream as it arrives.

## How close is this to SeisBench?

Close enough to be worth stating precisely. Stage by stage, against
`model.classify(stream)` over the 17 cached PhaseNet weight sets and the 8
stations above:

- the preprocessing filter is **bit-identical** — a maximum absolute difference
  of exactly 0 over 9.3 million samples;
- annotation is bit-identical too, once both sides use the same batch size:
  exactly 0 on all 17 weight sets, the ~1e-6 seen otherwise being float32
  convolution non-associativity between one-window and 256-window batches;
- triggering differs in exactly one place, a sample whose value equals the
  threshold exactly — ObsPy's `trigger_onset` turns on at `>= thresh`, xdas at
  `> thresh`. It never happened on this dataset;
- fed the *same* resampled data, `xd.pick` and `model.classify` produced 1979
  picks each over the reference day, **every one on the same sample**.

Which leaves the resampler as the one real deviation, and it is a deviation of
passband rather than of care. SeisBench resamples with ObsPy's
`Trace.resample`, which defaults to `window="hann"` and applies that window *in
the frequency domain*: unity at DC, zero at Nyquist, and **half the amplitude
at half the input Nyquist**. *Xdas* resamples with a polyphase FIR — the only
form that can run chunk by chunk — which is flat across that band. On this
40 Hz data the two differ by 6.3 % relative RMS (median over the sweep), enough
to move most picks: 1918 picks instead of 1979 over the reference day.
Reproducing SeisBench here would mean reproducing a worse resampler, so *xdas*
does not. `resample=False` drops the stage — pass it when the data is already
at the model's rate, or to compare the two implementations on the same
waveforms.
