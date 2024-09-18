---
file_format: mystnb
kernelspec:
  name: python3
---

```{code-cell}
:tags: [remove-cell]

import os
os.chdir("../_data")

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

# Working with Large-N Seismic Arrays

The virtualization capabilities of Xdas make it a good candidate for working with the large datasets produced by large-N seismic arrays.

In this section, we will present several examples of how to handle long and numerous time series.

```{note}
This part encourages experimenting with seismic data. Depending on the most common use cases users find, this could lead to changes in development direction.
```

## Exploring a dataset

We will start by exploring a synthetic dataset composed of 7 stations, each with 3 channels (HHZ, HHN, HHE). Each trace for each station and channel is stored in multiple one-minute-long files. Some files are missing, resulting in data gaps. The sampling rate is 100 Hz. The data is organized in a directory structure that groups files by station.

To open the dataset, we will use the `open_mfdatatree` function. This function takes a pattern that describes the directory structure and file names. The pattern is a string containing placeholders for the network, station, location, channel, start time, and end time. Placeholders for named fields are enclosed in curly braces, while simple brackets are used for varying parts of the file name that will be concatenated into different acquisitions (meant for changes in acquisition parameters).

Next, we will plot the availability of the dataset.

```{code-cell}
:tags: [remove-output]

import xdas as xd

pattern = "NX/{station}/NX.{station}.00.{channel}__[acquisition].mseed"
dc = xd.open_mfdatatree(pattern, engine="miniseed")
xd.plot_availability(dc, dim="time")
```
```{code-cell}
:tags: [remove-input]
from IPython.display import HTML
fig = xd.plot_availability(dc, dim="time")
HTML(fig.to_html())
```

We can see that indeed some data is missing. Yet, as often, the different channels are synchronized. We can therefore reorganize the data by concatenating the channels of each station.

```{code-cell}
:tags: [remove-output]

dc = xd.DataCollection(
    {
        station: xd.concatenate(objs, dim="channel")
        for station in dc
        for objs in zip(*[dc[station][channel] for channel in dc[station]]) 
    },
    name="station",
)
xd.plot_availability(dc, dim="time")
```
```{code-cell}
:tags: [remove-input]
from IPython.display import HTML
fig = xd.plot_availability(dc, dim="time")
HTML(fig.to_html())
```

In our case, all stations are synchronized to GPS time. By selecting a time range where no data is missing, we can concatenate the stations to obtain an N-dimensional array representation of the dataset.

```{code-cell}
dc = dc.sel(time=slice("2024-01-01T00:01:00", "2024-01-01T00:02:59.99"))
da = xd.concatenate((dc[station] for station in dc), dim="station")
da
```

This is useful for performing array analysis. In this example, we simply stack the energy.

```{code-cell}  
trace = np.square(da).mean("channel").mean("station")
trace.plot(ylim=(0, 3))
```

All the processing capabilities of Xdas can be applied to the dataset. We encourage readers to explore the various possibilities.
