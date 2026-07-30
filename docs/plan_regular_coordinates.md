---
orphan: true
---

# Design: regular coordinates — settling the open questions

Status: implemented on this branch (2026-07-29).
Branch: `feature/fixed-interp-coords`, targeting a PR to `dev` (0.2.8, untagged).

This document settles the four design questions left open by the regular-
coordinate work, so the remaining implementation (engine emission, docs pass,
failing tests) has a fixed contract to build against. It supersedes the
never-written `docs/plan_propagate_simplify_kwargs.md` referenced by
`concat`'s docstring.

## Background: the model as implemented

An `InterpCoordinate` may carry two optional metadata entries:

- `sampling_interval` — the nominal sample spacing. Its presence is what makes
  the coordinate *regular* (`isregular()`).
- `tolerance` — the allowed jitter around that spacing. The validity invariant,
  checked at construction (`_is_valid_sampling_interval`), is per continuous
  segment: `|num - si * den| <= 2 * tolerance`, evaluated at the dtype
  resolution (integer division for datetime64, so sub-resolution drift is
  always absorbed).

`from_block` produces regular coordinates; `_to_regular` enforces or infers a
spacing (raising when it cannot); `simplify(tolerance, reduce, regularize)`
spends an accuracy budget on tie-point reduction and optional promotion to
regular; `_concat` is strict (keeps the spacing only when both sides agree
exactly, takes `max` of tolerances, otherwise drops to irregular).

## D1. Public API surface: `to_regular` public, `infer_regular` private

**Decision.** Promote `_to_regular` to public `to_regular`, defined on
`AxisCoordinate` (not just `InterpCoordinate`), honouring the rule that a
public coordinate method exists on the whole axis hierarchy or not at all:

- `InterpCoordinate.to_regular(sampling_interval=None, tolerance=None)` —
  current `_to_regular` behaviour: enforce the given spacing, inferring it when
  omitted, raising `ValueError` when the tie points cannot be described by a
  single spacing within `tolerance`.
- `SampledCoordinate.to_regular(...)` — regular by construction: with no
  arguments return a copy; with explicit arguments validate them against the
  stored interval and raise on mismatch.
- `DenseCoordinate.to_regular(...)` — *conversion*: return a regular
  `InterpCoordinate` built from the dense values (reduce within `tolerance`,
  then enforce the spacing), raising when the values are genuinely irregular.
  Returning a different subclass is acceptable: the `to_` prefix already
  signals a conversion, and this is the natural "make this axis usable by
  signal processing" entry point.

`_infer_regular` stays private. It is an implementation detail of
`to_regular`/`simplify` (the Chebyshev-center fit); exposing it publicly on
only one subclass would recreate the partial-interface problem, and its
diagnostic value is available through `to_regular`'s behaviour and error
message. `docs/api/coordinates.md` must drop the `infer_regular` entry and the
release notes keep advertising `to_regular` (now truthfully).

Consequence: `get_sampling_interval` (module level, `core.py:1244`) loses its
`hasattr(coord, "_to_regular")` duck-typing — see D3.

## D2. What "regular" means per subclass (the Dense question)

**Decision.** *Regular* means "carries an explicit nominal sampling interval",
uniformly:

- `InterpCoordinate`: regular iff `sampling_interval` metadata is present.
- `SampledCoordinate`: always regular (the interval is part of its data).
- `DenseCoordinate`: **never regular**. `get_sampling_interval` returns `None`
  unconditionally, dropping the current end-to-end average. The average makes
  `isregular()` vacuously true for any dense axis and silently hands a
  meaningless rate to signal routines on jittery data — the exact failure mode
  this branch exists to eliminate. A dense axis that really is evenly sampled
  becomes regular explicitly, via `to_regular` (D1) or
  `simplify(regularize=True)`.
- `ScalarCoordinate`: `isregular()` moves to the `Coordinate` base and returns
  `False` there; `AxisCoordinate` overrides it with the current
  `get_sampling_interval() is not None`. This makes the release-notes claim
  ("on the base ABC") true and removes the `AttributeError` on scalar coords.

## D3. The `get_sampling_interval` contract: strict, one choke point

Three layers, each with a single behaviour:

1. **Primitive** — `coord.get_sampling_interval(cast=True)`: return the
   nominal interval, or `None` when the coordinate is not regular. Never
   raises, never infers, O(1).
2. **Conversion** — `coord.to_regular(...)`: the only place inference and
   enforcement happen. Raises with an actionable message on genuinely
   irregular axes.
3. **Convenience** — `xdas.get_sampling_interval(da, dim)`: return the nominal
   interval when the coordinate is regular, otherwise **raise** `ValueError`
   telling the user how to fix it (open the files with a `tolerance`, or
   `da[dim] = da[dim].to_regular(tolerance=...)`). The current silent
   `_to_regular()` fallback is removed: it hides an O(n log n) inference in
   every FFT/filter call and only ever succeeds on exactly-uniform axes anyway
   (the implicit epsilon tolerance rejects any real jitter), so its benefit is
   marginal and its implicitness is not.

   *Amendment (2026-07-30):* data saved by earlier versions carries no
   `sampling_interval` metadata, so raising immediately would break every
   signal-processing call on existing archives. For one deprecation cycle the
   helper therefore falls back to inference on irregular coordinates: it infers
   the spacing (and, for `InterpCoordinate`, the minimal tolerance that
   validates it via the Chebyshev fit), emits a `FutureWarning` stating both
   values and the migration path, and returns the inferred spacing. Dense
   coordinates go through the strict `to_regular()` (uniform axes work, jittery
   ones still raise — the old end-to-end average was a silent wrong answer not
   worth preserving). Raising remains only where no spacing can be inferred at
   all. The strict behaviour described above becomes the default when the
   deprecation completes.

**Migration.** All signal-consuming code goes through layer 3 — including
`xdas/signal.py`, which currently open-codes the strict check six times
(`d = coords[dim].get_sampling_interval(); if d is None: raise ...`). Revert
those to the module-level helper so the error message and the policy live in
one place, and keep `fft.py`, `spectral.py`, `atoms/`, `picking.py`,
`miniseed.py` on the helper. Net user-visible behaviour: every signal routine
raises the *same* error on irregular axes, and none of them raise on data
opened through the engines once D5 lands.

Also fix `DataArrayList`-style compatibility checking
(`routines.py:919-922`): `get_sampling_interval` returning `None` for the
incoming chunk must produce a `CompatibilityError`, not a `TypeError` inside
`np.isclose`.

## D4. Tolerance semantics and propagation

**Meaning.** `tolerance` is a *declared jitter bound carried by the
coordinate*: the promise that every continuous segment satisfies
`|num - si * den| <= 2 * tolerance` at the dtype resolution. It is data, not a
processing parameter — processing functions take a *budget* argument that may
default to it.

**Propagation rules** (R1–R2 already implemented, kept as-is):

- **R1 — slicing/striding** (`_slice`): spacing scales by the step, tolerance
  is preserved.
- **R2 — raw concatenation** (`_concat`): strict; equal spacings are kept with
  `max` of tolerances, anything else drops to irregular. Reconciliation is the
  job of user-facing routines via `simplify`.
- **R3 — derived rates must carry their quantization error.** Any operation
  that synthesizes a new nominal spacing that is not exactly representable in
  the coordinate dtype must record the representation error in `tolerance`
  instead of claiming `0`. Concretely for `Upsample(factor)` on datetime axes:
  `new_delta = delta // factor` truncates, so the coordinate must carry
  `tolerance >= (delta - factor * new_delta)` (2 ns in the failing test) on
  top of the inherited tolerance. This is what makes chunk seams land within
  tolerance of the nominal grid.
- **R4 — `simplify(tolerance=None)` defaults to the coordinate's own stored
  tolerance** (falling back to the current zero-like default when the
  coordinate has none). Rationale: the coordinate has already declared "my
  values are only meaningful to within `tolerance`"; a canonicalisation pass
  that refuses to spend that declared slack is pointless strictness. This
  applies to `concat(tolerance=None)` too, per-coordinate. `tolerance=False`
  keeps its "no simplification" meaning; an explicit scalar overrides.
- **R5 — no unconditional widening.** `InterpCoordinate.simplify` on a regular
  coordinate currently stores `self.tolerance + tolerance` whenever `reduce`
  runs. Replace with: after reduction, keep the original tolerance if it still
  validates, and only widen (to the smallest valid value, bounded by
  `self.tolerance + budget`) when it does not. Without this, chunked and
  unchunked pipelines can never produce `equals()` coordinates because the
  chunked path concatenates and re-simplifies.

**Why this fixes `test_upsample`.** Each upsampled chunk carries
`sampling_interval = 6_666_666 ns, tolerance = 2 ns` (R3). `_concat` keeps the
spacing (R2). `concat`'s simplify defaults its budget to the stored 2 ns (R4),
Douglas-Peucker drops the seam tie points (they deviate ≤ 2 ns from the global
line), and R5 keeps `tolerance = 2 ns` — identical to the unchunked result.

**Defaults alignment.** `concat` and `concat_coords` currently disagree
(`regularize=False, tolerance=None` vs `regularize=True, tolerance=False`).
Align `concat_coords` to `concat`: `reduce=True, regularize=False,
tolerance=None` (with R4's meaning). `regularize` stays opt-in for this PR —
with engines emitting regular coordinates (D5) and R2 preserving them,
multi-file opens stay regular without promotion, so the conservative default
costs nothing; flipping it can be revisited once propagation has soaked.

## D5. IO emission (scope confirmed, design only sketched here)

Engines construct per-file time/space coordinates with
`InterpCoordinate.from_block(start, size, step)` (the existing `# TODO: use
from_block` sites in `prodml`, `terra15`, `asn`, plus `miniseed.read_stream`
and ObsPy `from_stream`, which must also build at ns resolution to round-trip
`to_stream`). Per-file tolerance is `0`: within one file the grid is exact by
construction. Cross-file jitter is reconciled where it appears — at
`concat`/`open_mfdataarray` time via the user-supplied `tolerance` (R4/R2).
`from_stream` uses `stats.delta`; engines use the file's metadata rate.

## Acceptance criteria

- `tests/test_atoms.py::TestFilters::test_upsample` and
  `tests/test_dataarray.py::TestIO::test_stream` pass without weakening the
  assertions.
- `xd.signal.*`, `xd.fft.*`, `xd.spectral.*`, and the atoms raise one uniform,
  actionable error on irregular axes, and raise nothing on engine-opened data.
- Release notes, `docs/api/coordinates.md`, and the user guide describe only
  APIs that exist (`to_regular` public, `infer_regular` gone from docs).
- `concat`'s docstring no longer references this document's missing
  predecessor.
