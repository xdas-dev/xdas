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

## Decorators and composition

```{eval-rst}
.. autosummary::
   :toctree: ../_autosummary

   as_function
   atomized
   compose
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
   SOSFilter
   UpSample
```