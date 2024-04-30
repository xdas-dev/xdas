import importlib

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
    def __init__(self, model, dim, compile=True):
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
    def _torch(self):
        try:
            return importlib.import_module("torch")
        except ImportError:
            raise ImportError(
                "pytorch is not installed, use `pip install torch` to install it"
            )

    def initialize(self, da, chunk_dim=None, **flags):
        if chunk_dim == self.dim:
            self.buffer = State(da.isel({self.dim: slice(0, 0)}))
        else:
            self.buffer = State(None)

    def call(self, da, **flags):
        if self.buffer is None:
            out = self.process(da)
        else:
            da = concatenate([self.buffer, da], self.dim)
            out = self.process(da)
            divpoint = out.sizes[self.dim]
            self.buffer = State(da.isel({self.dim: slice(divpoint, None)}))
        return out

    def process(self, da):
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

    def _process(self, t):
        t = t - t.mean(-1, keepdim=True)
        t = t / t.std(-1, keepdim=True)
        t = t[:, None, :].repeat(1, self.model.in_channels, 1)
        t = self.model(t)
        t = t[:, :, self.noverlap // 2 : -self.noverlap // 2]
        return t

    _process_compiled = torch.compile(_process)
