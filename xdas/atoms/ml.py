"""
Machine-learning atom: :class:`MLPicker` wraps SeisBench models as pipeline atoms.

Torch and SeisBench are loaded lazily so they remain optional dependencies.
"""

import importlib

import numpy as np

from ..core.dataarray import DataArray
from ..core.routines import concat
from .core import Atom, State


class LazyModule:
    """Defer importing *name* until the first attribute access."""

    def __init__(self, name):
        self._name = name
        self._module = None

    def __getattr__(self, name):
        if self._module is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError:
                raise ImportError(
                    f"{self._name} is not installed by default, "
                    f"please install is manually"
                )
        return getattr(self._module, name)


torch = LazyModule("torch")


class MLPicker(Atom):
    """
    Wraps a SeisBench phase-picking model as a streaming :class:`Atom`.

    Uses an overlapping sliding-window strategy to apply the model to
    arbitrarily long data and to stitch the per-segment probability outputs
    back into a continuous DataArray.

    Parameters
    ----------
    model : seisbench.models.WaveformModel
        A SeisBench model in evaluation mode (will be moved to *device*).
    dim : str
        Dimension name along which the model slides (usually ``"time"``).
    device : str or torch.device, optional
        Torch device.  Defaults to CUDA if available, else CPU.
    component_strategy : str, optional
        How to fill the channel dimension: ``"clone"`` replicates the
        single-component signal, or pass a component letter to select it.
    """

    def __init__(self, model, dim, device=None, component_strategy="clone"):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        valid = {"clone", *model.component_order}
        if component_strategy not in valid:
            raise ValueError(f"component_strategy must be one of {valid}")
        self.device = device
        self.model = model.eval().to(self.device)
        self.dim = dim
        self.component_strategy = component_strategy
        self.buffer = State(...)
        self.batch_size = State(...)
        self.circular_input = State(...)
        self.model_input = State(...)  # TODO: should not be a state
        self.circular_output = State(...)
        self.circular_counts = State(...)

    @property
    def nperseg(self):
        """Number of samples per segment (= model input length)."""
        return self.model.in_samples

    @property
    def noverlap(self):
        """Number of overlapping samples between consecutive segments."""
        return self.nperseg // 2

    @property
    def step(self):
        """Stride between the start of consecutive segments."""
        return self.nperseg - self.noverlap

    @property
    def phases(self):
        """List of phase label strings produced by the model."""
        return list(self.model.labels)

    @property
    def in_channels(self):
        """Number of input channels the model expects."""
        return self.model.in_channels

    @property
    def classes(self):
        """Number of output classes (phases) the model produces."""
        return self.model.classes

    @property
    def blinding(self):
        """``(left, right)`` blinding samples from the model's default args."""
        return self.model.default_args["blinding"]

    def initialize(self, da, chunk_dim=None, **flags):
        """Allocate circular buffers sized to *da*'s batch and segment dimensions."""
        self.batch_size = State(
            np.prod([size for dim, size in da.sizes.items() if not dim == self.dim])
        )
        self.circular_input = State(
            torch.zeros(
                self.batch_size, self.nperseg, dtype=torch.float32, device=self.device
            )
        )
        self.model_input = State(
            torch.zeros(
                self.batch_size,
                self.in_channels,
                self.nperseg,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.circular_output = State(
            torch.zeros(
                self.batch_size,
                self.classes,
                self.nperseg,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.circular_counts = State(
            torch.zeros(
                self.batch_size, self.nperseg, dtype=torch.int32, device=self.device
            )
        )
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        """Run the model over *da*, managing a carry-over buffer for chunked input."""
        if self.buffer is None:
            out = self._process(da)
        else:
            da = concat([self.buffer, da], self.dim)
            out = self._process(da)
            divpoint = out.sizes[self.dim]
            self.buffer = State(da.isel({self.dim: slice(divpoint, None)}))
        return out

    def _process(self, da):
        self._initialize(da)
        chunks = []
        for idx in range(0, da.sizes[self.dim] - self.nperseg, self.step):
            data = self._push_chunk(da, idx)
            self._roll(self.circular_input, data)
            self._roll(self.circular_counts, 0)
            self._roll(self.circular_output, 0.0)
            normalized = self.model.annotate_batch_pre(self.circular_input, {})
            if self.component_strategy == "clone":
                self.model_input[:] = normalized.unsqueeze(1)
            else:
                ch = list(self.model.component_order).index(self.component_strategy)
                self.model_input[:, ch, :] = normalized
            self._run_model()
            data = self._pull_completed()
            chunk = self._attach_metadata(data, da, idx)
            chunk = chunk.transpose(self.dim, ...)  # TODO: does it make sense?
            chunks.append(chunk)
        return concat(chunks, self.dim)

    def _initialize(self, da):
        chunk = da.isel({self.dim: slice(0, self.noverlap)})
        chunk = chunk.transpose(..., self.dim)
        chunk = torch.tensor(chunk.values, dtype=torch.float32, device=self.device)
        self.circular_input[:, self.step :] = chunk

    def _push_chunk(self, da, idx):
        chunk = da.isel({self.dim: slice(idx + self.noverlap, idx + self.nperseg)})
        chunk = chunk.transpose(..., self.dim)
        return torch.tensor(chunk.values, dtype=torch.float32, device=self.device)

    def _roll(self, buffer, values):
        buffer[..., : self.noverlap] = buffer[..., self.step :]
        buffer[..., self.noverlap :] = values

    def _run_model(self):
        with torch.no_grad():
            out = self.model(self.model_input)
        slc = slice(self.blinding[0], -self.blinding[1])
        self.circular_counts[:, slc] += 1
        self.circular_output[:, :, slc] += out[:, :, slc]

    def _pull_completed(self):
        data = (
            self.circular_output[:, :, : self.step]
            / self.circular_counts[:, None, : self.step]
        )
        return data.cpu().numpy()

    def _attach_metadata(self, data, da, idx):
        coords = da.coords.copy()
        coords[self.dim] = coords[self.dim][idx : idx + self.step]
        coords["phase"] = self.phases
        dims = tuple(dim for dim in da.dims if not dim == self.dim) + (
            "phase",
            self.dim,
        )
        return DataArray(data, coords, dims, da.name, da.attrs)
