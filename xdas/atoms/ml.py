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

    def initialize(self, da, chunk_dim=None, **flags):
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        if self.buffer is None:
            out = self._doit(da)
        else:
            da = concatenate([self.buffer, da], self.dim)
            out = self._doit(da)
            divpoint = out.sizes[self.dim]
            self.buffer = State(da.isel({self.dim: slice(divpoint, None)}))
        return out

    def _batch_process(self, da):
        chunks = []
        for idx in range(0, da.sizes[self.dim] - self.nperseg, self.step):
            chunk = da.isel({self.dim: slice(idx, idx + self.nperseg)})
            chunk = chunk.transpose(..., self.dim)
            # data
            data = chunk.data
            # TODO: data = data.reshape(-1, data.shape[-1])
            with torch.no_grad():  # TODO: avoiding sending data twice to GPU
                t = torch.tensor(chunk.data, dtype=torch.float32, device=self.device)
                if self.compile:
                    t = self._process_compiled(t)
                else:
                    t = self._process(t)
                data = t.cpu().numpy()
            # TODO: data = data.reshape(...)
            # coords
            coords = chunk.coords.copy()
            coords[self.dim] = coords[self.dim][
                self.noverlap // 2 : -self.noverlap // 2
            ]
            coords["phase"] = self.phases
            # dims
            dims = chunk.dims[:-1] + ("phase",) + chunk.dims[-1:]
            # pack
            chunk = DataArray(data, coords, dims, da.name, da.attrs)
            chunk = chunk.transpose(self.dim, ...)  # TODO: reorder better
            chunks.append(chunk)
        return concatenate(chunks, self.dim)

    def _doit(self, da):
        batch_size = np.prod(
            [size for dim, size in da.sizes.items() if not dim == self.dim]
        )

        # allocate memory
        rolling_input = torch.zeros(
            batch_size, self.nperseg, dtype=torch.float32, device=self.device
        )
        model_input = torch.zeros(
            batch_size, 3, self.nperseg, dtype=torch.float32, device=self.device
        )
        rolling_output = torch.zeros(
            batch_size, 3, self.nperseg, dtype=torch.float32, device=self.device
        )
        rolling_counts = torch.zeros(
            batch_size, self.nperseg, dtype=torch.int32, device=self.device
        )

        # initialize rolling_buffer
        chunk = da.isel({self.dim: slice(0, self.noverlap)})
        chunk = chunk.transpose(..., self.dim)
        chunk = torch.tensor(chunk.values, dtype=torch.float32, device=self.device)
        rolling_input[:, self.step :] = chunk

        chunks = []
        for idx in range(0, da.sizes[self.dim] - self.nperseg, self.step):

            # roll buffer
            rolling_input[:, : self.noverlap] = rolling_input[:, self.step :]

            # send data
            chunk = da.isel({self.dim: slice(idx + self.noverlap, idx + self.nperseg)})
            chunk = chunk.transpose(..., self.dim)
            chunk = torch.tensor(chunk.values, dtype=torch.float32, device=self.device)
            rolling_input[:, self.noverlap :] = chunk

            # TODO: data = data.reshape(-1, data.shape[-1])

            # normalize into input_buffer
            self._normalize(rolling_input, out=model_input[:, 1, :])

            # run model
            with torch.no_grad():
                out = self.model(model_input)

            # roll buffers
            rolling_counts[:, : self.noverlap] = rolling_counts[:, self.step :]
            rolling_counts[:, self.noverlap :] = 0
            rolling_output[:, :, : self.noverlap] = rolling_output[:, :, self.step :]
            rolling_output[:, :, self.noverlap :] = 0.0

            rolling_counts[:, 250:-250] += 1
            rolling_output[:, :, 250:-250] += out[:, :, 250:-250]

            chunk = (
                rolling_output[:, :, : self.step] / rolling_counts[:, None, : self.step]
            )

            # retrieve results
            chunk = chunk.cpu().numpy()

            # TODO: data = data.reshape(...)

            chunks.append(chunk)
        return np.concatenate(chunks, axis=-1)

    def _process(self, x):
        self._normalize(x)
        y = torch.zeros(
            (*x.shape[:-1], 3, x.shape[-1]), dtype=x.dtype, device=self.device
        )
        y[:, 1, :] = x
        y = self.model(y)
        y = y[:, :, self.noverlap // 2 : -self.noverlap // 2]
        return y

    def _normalize(self, x, out=None):
        if out is None:
            out = torch.empty_like(x)
        std, mean = torch.std_mean(x, dim=-1, keepdim=True)
        torch.sub(x, mean, out=out)
        torch.div(out, std, out=out)
        return out

    _process_compiled = torch.compile(_process)
