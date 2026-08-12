```{eval-rst}
.. currentmodule:: xdas.processing
```

# xdas.processing

## Functions

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   process
   watch
   get_source
   get_writer
```

## Loaders

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataArrayLoader
   RealTimeLoader
```

### DataArrayLoader

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataArrayLoader.nbytes
```

## Writers

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataArrayWriter
   DataFrameWriter
   StreamWriter
   ZMQPublisher
   ZMQSubscriber
   ResultWriter
```

### DataArrayWriter

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DataArrayWriter.submit
   DataArrayWriter.write
   DataArrayWriter.shutdown
   DataArrayWriter.result
```

### ZMQPublisher

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   ZMQPublisher.submit
   ZMQPublisher.write
   ZMQPublisher.result
   ZMQPublisher.wait_for_subscribers
```

### ZMQSubscriber

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   ZMQSubscriber.wait_until_subscribed
```