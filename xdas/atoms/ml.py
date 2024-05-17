import importlib

import numpy as np

from ..atoms import Atom, State
from ..core.dataarray import DataArray
from ..core.routines import concatenate


class LazyModule:
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
    def __init__(self, model, dim, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device
        self.model = model.eval().to(self.device)
        self.dim = dim
        self.buffer = State(...)
        self.batch_size = State(...)
        self.circular_input = State(...)
        self.model_input = State(...)  # TODO: should not be a state
        self.circular_output = State(...)
        self.circular_counts = State(...)

    @property
    def nperseg(self):
        return self.model.in_samples

    @property
    def noverlap(self):
        return self.nperseg // 2

    @property
    def step(self):
        return self.nperseg - self.noverlap

    @property
    def phases(self):
        return list(self.model.labels)

    @property
    def in_channels(self):
        return self.model.in_channels

    @property
    def classes(self):
        return self.model.classes

    @property
    def blinding(self):
        return self.model.default_args["blinding"]

    def initialize(self, da, chunk_dim=None, **flags):
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
        if self.buffer is None:
            out = self._process(da)
        else:
            da = concatenate([self.buffer, da], self.dim)
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
            self._normalize(self.circular_input, out=self.model_input[:, 1, :])
            self._run_model()
            data = self._pull_completed()
            chunk = self._attach_metadata(data, da, idx)
            chunk = chunk.transpose(self.dim, ...)  # TODO: does it make sense?
            chunks.append(chunk)
        return concatenate(chunks, self.dim)

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

    @staticmethod
    def _normalize(x, out=None):
        if out is None:
            out = torch.empty_like(x)
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        torch.sub(x, mean, out=out)
        torch.div(out, std, out=out)
        return out

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
