```{eval-rst}
.. currentmodule:: xdas.atoms
```

# xdas.atoms

## Base Atom class

Constructor

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Atom
```

Attributes

```{eval-rst}
.. autosummary::

   Atom.state
   Atom.initialized
```

Methods

```{eval-rst}
.. autosummary::

   Atom.initialize
   Atom.initialize_from_state
   Atom.call
   Atom.flush
   Atom.reset
   Atom.process
   Atom.iter_chunks
   Atom.save_state
   Atom.set_state
   Atom.load_state
```

## Core atoms

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Sequential
   State
```

### Partial

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Partial
```

Attributes

```{eval-rst}
.. autosummary::

   Partial.stateful
```

Methods

```{eval-rst}
.. autosummary::

   Partial.call
   Partial.from_state
   Partial.get_state
```

## Decorators and composition

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   as_function
   atomized
   compose
```

## Task atoms

Public processing vocabulary with physical parameters only. Each task atom has
a function form exported at the top level of `xdas` (e.g. `xdas.filter`,
`xdas.decimate`).

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Decimate
   Differentiate
   Filter
   Integrate
   Resample
   STFT
```

```{eval-rst}
.. currentmodule:: xdas
```

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   decimate
   detrend
   differentiate
   filter
   hilbert
   integrate
   medfilt
   rechunk
   resample
   sliding_mean_removal
   stft
   taper
```

```{eval-rst}
.. currentmodule:: xdas.atoms
```

## Signal processing

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   FIRFilter
   IIRFilter
   MLPicker
   ResamplePoly
   Trigger
```

## Kernel atoms

Expert layer (`xdas.atoms.kernel`): exact stateful primitives with machine
parameters, designed by the task atoms from the data at the first call.

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DownSample
   LFilter
   Polyphase
   Rechunk
   SOSFilter
   UpSample
```