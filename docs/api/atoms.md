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
   Atom.gather
   Atom.merge
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

Public processing vocabulary with physical parameters only.

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Differentiate
   Filter
   Integrate
   Resample
   STFT
```

## Function forms

Every atom has a function form exported at the top level of `xdas`: called on
data it applies eagerly, called on `...` it returns the atom.

```{eval-rst}
.. currentmodule:: xdas
```

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   annotate
   detrend
   differentiate
   filter
   hilbert
   integrate
   medfilt
   pick
   rechunk
   resample
   sliding_mean_removal
   stft
   taper
   trigger
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
   ResamplePoly
```

## Detection and picking

`Annotate` runs a SeisBench model window by window, `Trigger` turns the
characteristic function it produces into a pick table, and `Picker` is the
whole pipeline a weight set describes — its own filter, its own sampling rate,
its own per-phase thresholds. Each has a lowercase functional twin at the top
level of `xdas` (`xdas.annotate`, `xdas.trigger`, `xdas.pick`).

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   Annotate
   Trigger
   Picker
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
