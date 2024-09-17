---
file_format: mystnb
kernelspec:
  name: python3
---

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
da = xd.synthetics.dummy()
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

encoding = {"chunks": (10, 10), **hdf5plugin.Zfp(accuracy=1e-6)}
publisher = ZMQPublisher(address, encoding)  # Add encoding here, the rest is the same
```


