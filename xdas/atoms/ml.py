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
import torch  # TODO: remove


class MLPicker(Atom):
    def __init__(self, model, dim, compile=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)
        self.dim = dim
        self.compile = compile
        self.buffer = State(...)

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
    def blinding(self):
        return self.model.default_args["blinding"]

    def initialize(self, da, chunk_dim=None, **flags):
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
        batch_size = np.prod(
            [size for dim, size in da.sizes.items() if not dim == self.dim]
        )
        circular_input = torch.zeros(
            batch_size, self.nperseg, dtype=torch.float32, device=self.device
        )
        model_input = torch.zeros(
            batch_size, 3, self.nperseg, dtype=torch.float32, device=self.device
        )
        circular_output = torch.zeros(
            batch_size, 3, self.nperseg, dtype=torch.float32, device=self.device
        )
        circular_counts = torch.zeros(
            batch_size, self.nperseg, dtype=torch.int32, device=self.device
        )
        self._initialize(da, circular_input)
        chunks = []
        for idx in range(0, da.sizes[self.dim] - self.nperseg, self.step):
            data = self._push_chunk(da, idx)
            self._roll(circular_input, data)
            self._roll(circular_counts, 0)
            self._roll(circular_output, 0.0)
            self._normalize(circular_input, out=model_input[:, 1, :])
            self._run_model(model_input, circular_counts, circular_output)
            data = self._pull_completed(circular_counts, circular_output)
            chunk = self._attach_metadata(data, da, idx)
            chunk = chunk.transpose(self.dim, ...)  # TODO: reorder better
            chunks.append(chunk)
        return concatenate(chunks, self.dim)

    def _initialize(self, da, circular_input):
        chunk = da.isel({self.dim: slice(0, self.noverlap)})
        chunk = chunk.transpose(..., self.dim)
        chunk = torch.tensor(chunk.values, dtype=torch.float32, device=self.device)
        circular_input[:, self.step :] = chunk

    def _push_chunk(self, da, idx):
        chunk = da.isel({self.dim: slice(idx + self.noverlap, idx + self.nperseg)})
        chunk = chunk.transpose(..., self.dim)
        return torch.tensor(chunk.values, dtype=torch.float32, device=self.device)

    def _roll(self, buffer, values):
        buffer[..., : self.noverlap] = buffer[..., self.step :]
        buffer[..., self.noverlap :] = values

    def _normalize(self, x, out=None):
        if out is None:
            out = torch.empty_like(x)
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        torch.sub(x, mean, out=out)
        torch.div(out, std, out=out)
        return out

    def _run_model(self, model_input, circular_counts, circular_output):
        with torch.no_grad():
            out = self.model(model_input)

            # assign output
        slc = slice(self.blinding[0], -self.blinding[1])
        circular_counts[:, slc] += 1
        circular_output[:, :, slc] += out[:, :, slc]

    def _pull_completed(self, circular_counts, circular_output):
        data = (
            circular_output[:, :, : self.step] / circular_counts[:, None, : self.step]
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
