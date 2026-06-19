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
   Atom.reset
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

## Decorators

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   atomized
```

## Signal processing

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   DownSample
   FIRFilter
   IIRFilter
   LFilter
   MLPicker
   ResamplePoly
   SOSFilter
   Trigger
   UpSample
```