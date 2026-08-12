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

# Streaming data 

Xdas allows to stream data over any network using [ZeroMQ](https://zeromq.org). Xdas use the Publisher and Subscriber patterns meaning that on one node the data is published and that any number of subscribers can receive the data stream. 

Streaming data with Xdas is done by simply dumping each chunk to NetCDF binaries and to send those as packets. This ensure that each packet is self described and that feature such as compression are available (which can be very helpful to minimize the used bandwidth). 

Xdas implements the {py:class}`~xdas.processing.ZMQPublisher` and {py:class}`~xdas.processing.ZMQSubscriber`.Those object can respectively be used as a Writer and a Loader as described in the [](processing) section. Both are initialized by giving an network address. The publisher use the `submit` method to send packets while the subscriber is an infinite iterator that yields packets.

In this section, we will mimic the use of several machine by using multithreading, where each thread is supposed to be a different machine. In real-life application, the publisher and subscriber are generally called in different machine or software.

## Simple use case

```{code-cell}
import threading
import time

import xdas as xd
from xdas.processing import ZMQPublisher, ZMQSubscriber
```

First we generate some data and split it into packets

```{code-cell}
da = xd.testing.dummy()
packets = xd.split(da, 5)
```

We then publish the packets on machine 1.

```{code-cell}
address = f"tcp://localhost:{xd.io.get_free_port()}"
publisher = ZMQPublisher(address)

def publish():
    for packet in packets:
        publisher.submit(packet)
        # give a chance to the subscriber to connect in time and to get the last packet
        time.sleep(0.1)  

machine1 = threading.Thread(target=publish)
machine1.start()
```

Let's receive the packets on machine 2.

```{code-cell}
subscriber = ZMQSubscriber(address)

packets = []

def subscribe():
    for packet in subscriber:
        packets.append(packet)

machine2 = threading.Thread(target=subscribe)
machine2.start()
```

Now we wait for machine 1 to finish sending its packet and see if everything went well.

```{code-cell}
machine1.join()
print(f"We received {len(packets)} packets!")
assert xd.concatenate(packets).equals(da)
```

## Using encoding

To reduce the volume of the transmitted data, compression is often useful. Xdas enable the use of the ZFP algorithm when storing data but also when streaming it. Encoding is declared the same way.

```{code-cell}
:tags: [remove-output]

import hdf5plugin

address = f"tcp://localhost:{xd.io.get_free_port()}"
encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=1e-6)}
publisher = ZMQPublisher(address, encoding)  # Add encoding here, the rest is the same
```

```{note}
Xdas also implements the ZeroMQ protocol used by the OptoDAS interrogators by ASN. Equivalent {py:class}`~xdas.io.asn.ZMQPublisher` and {py:class}`~xdas.io.asn.ZMQSubscriber` can be found in {py:mod}`xdas.io.asn`. This can be useful to get data in real-time from one instrument of that kind. Note that compression is not available with that protocol yet.
```

## Processing a stream

A pipeline consumes and produces a stream by naming the address on either end,
so nothing about the pipeline itself changes between replaying an archive and
following an instrument:

```python
pipeline.process("tcp://localhost:5556", out="tcp://*:5557")
```

A directory that is still being filled is the other unbounded source:
{py:func}`xdas.watch` follows it as files arrive, where a bare directory path
means "process what is there and stop".

```python
pipeline.process(xd.watch("/incoming", engine="febus"), out="results/")
```

Unbounded sources are processed until they are stopped. Interrupting with
`Ctrl-C` flushes the pipeline, closes the destination cleanly and returns what
was written; `until=` stops on its own at a coordinate value:

```python
pipeline.process(xd.watch("/incoming"), out="results/", until="2026-05-20T12:00:00")
```

Gaps are announced as they arrive rather than upfront — a stream cannot be
inspected ahead of time — and each one flushes and restarts the state of every
stage, exactly as it does on an archive.
