import os
import shutil
import sys
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor

import cloudpickle
import numpy as np
import pytest
from loky import process_executor

import xdas as xd
from xdas.processing.core import AUTO_CHUNK_NBYTES, get_pool
from xdas.processing.pools import (
    _ARENAS,
    PREFIX,
    Arena,
    ProcessFuture,
    ProcessPool,
    ShmRef,
    _directory,
    _init_worker,
    _offloadable,
    _park,
    _resolve,
    _run,
    _unlink,
    attach,
    sweep,
    view,
)

# The tasks below are sent to worker processes, which would have to import this
# module to unpickle them by reference -- and cannot: pytest imports the test
# suite without putting its directory on the path, so `tests` is a package only
# in the parent. Sending them by value keeps the worker from needing it.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


def _double(x):
    """A task returning something that is not a chunk."""
    return 2 * x


def _raise():
    """A task that fails, to check errors propagate and slots come back."""
    raise ValueError("broken task")


def _identity(chunk):
    """A task taking a chunk in, so the argument path gets exercised."""
    return chunk


def _make(length):
    """A task producing a chunk without being sent one, so nothing is staged."""
    return xd.testing.dummy(shape=(length, 10))


def _sum(chunk):
    """A task reading a staged argument without sending an array back."""
    return float(chunk.values.sum())


class TestArena:
    def test_slots_are_distinct_and_reused(self):
        arena = Arena(3, 1024)
        try:
            slots = [arena.reserve() for _ in range(3)]
            offsets = [offset for _, offset, _ in slots]
            assert sorted(offsets) == [0, 1024, 2048]
            assert arena.reserve() is None  # exhausted, callers fall back
            arena.release(offsets[0])
            assert arena.reserve()[1] == offsets[0]
        finally:
            arena.close()

    def test_close_unlinks_but_keeps_mappings_valid(self):
        arena = Arena(2, 1024)
        _, offset, _ = arena.reserve()
        ref = ShmRef(arena.path, offset, (4,), "<f8")
        data = view(ref)
        data[:] = [1.0, 2.0, 3.0, 4.0]
        arena.close()
        assert not os.path.exists(arena.path)
        # The pages outlive the name: whoever still holds a chunk can read it.
        np.testing.assert_array_equal(data, [1.0, 2.0, 3.0, 4.0])

    def test_create_refuses_when_it_cannot_fit(self):
        # An arena that would claim more than its share of the filesystem is
        # not worth a bus error later on.
        free = shutil.disk_usage(_directory()).free
        assert Arena.create(4, free) is None

    def test_sweep_removes_arenas_of_dead_owners(self):
        stale = os.path.join(_directory(), f"{PREFIX}999999999-deadbeef")
        with open(stale, "wb") as file:
            file.write(b"\0")
        keep = os.path.join(_directory(), f"{PREFIX}{os.getpid()}-cafecafe")
        with open(keep, "wb") as file:
            file.write(b"\0")
        try:
            sweep()
            assert not os.path.exists(stale)
            assert os.path.exists(keep)  # ours, still running
        finally:
            for path in (stale, keep):
                if os.path.exists(path):
                    os.unlink(path)

    def test_worker_init_discounts_the_arena_from_the_leak_check(self):
        # Shared pages count as resident memory, so without this a worker
        # writing a chunk looks to loky like it is leaking and gets recycled.
        arena = Arena(2, 1024)
        before = process_executor._MAX_MEMORY_LEAK_SIZE
        try:
            _init_worker(arena.path, arena.size)
            assert process_executor._MAX_MEMORY_LEAK_SIZE == before + arena.size
            assert arena.path in _ARENAS
        finally:
            process_executor._MAX_MEMORY_LEAK_SIZE = before
            arena.close()

    def test_attach_is_idempotent(self):
        arena = Arena(2, 1024)
        try:
            mapping = _ARENAS[arena.path]
            attach(arena.path, arena.size)
            assert _ARENAS[arena.path] is mapping
        finally:
            arena.close()


class TestShmRef:
    def test_presents_the_array_interface(self):
        ref = ShmRef("/nowhere", 0, (2, 3), "<f4")
        assert ref.shape == (2, 3)
        assert ref.ndim == 2
        assert ref.nbytes == 24
        assert "shape=(2, 3)" in repr(ref)

    def test_resolving_an_unmapped_arena_says_so(self):
        ref = ShmRef("/nowhere", 0, (2,), "<f4")
        with pytest.raises(RuntimeError, match="not mapped"):
            np.asarray(ref)

    def test_resolves_through_the_array_protocol(self):
        arena = Arena(2, 1024)
        try:
            _, offset, _ = arena.reserve()
            ref = ShmRef(arena.path, offset, (3,), "<f8")
            view(ref)[:] = [1.0, 2.0, 3.0]
            np.testing.assert_array_equal(np.asarray(ref), [1.0, 2.0, 3.0])
            np.testing.assert_array_equal(np.asarray(ref, dtype="int64"), [1, 2, 3])
        finally:
            arena.close()


class TestOffloadable:
    def test_only_plain_numeric_chunks_qualify(self):
        da = xd.testing.dummy(shape=(10, 10))
        assert _offloadable(da, da.nbytes)
        assert not _offloadable(da, da.nbytes - 1)  # bigger than the slot
        assert not _offloadable("not a chunk", 2**20)
        assert not _offloadable(da.isel(time=slice(0, 0)), 2**20)  # empty

    def test_lazy_chunks_stay_on_the_pickle_path(self, tmp_path):
        # A virtual array is kilobytes of manifest; there is nothing to park.
        xd.testing.dummy(shape=(100, 10)).to_netcdf(tmp_path / "data.nc")
        da = xd.open_dataarray(tmp_path / "data.nc")
        assert not _offloadable(da, 2**20)


@pytest.mark.slow
class TestProcessPool:
    def test_chunk_comes_back_through_the_arena(self):
        expected = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1, 1, expected.nbytes) as pool:
            result = pool.submit(_identity, expected).result()
        assert result.equals(expected)
        assert not result.data.flags.writeable

    def test_result_is_cached_across_calls(self):
        da = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1, 1, da.nbytes) as pool:
            future = pool.submit(_identity, da)
            assert future.result() is future.result()
            assert future.done()

    def test_non_chunk_results_pass_through(self):
        with ProcessPool(1) as pool:
            assert pool.submit(_double, 21).result() == 42

    def test_errors_propagate_and_free_the_slot(self):
        with ProcessPool(1) as pool:
            future = pool.submit(_raise)
            with pytest.raises(ValueError, match="broken task"):
                future.result()
            assert len(pool._arena._free) == pool._arena.nslots

    def test_oversized_chunks_take_the_pickle_path(self):
        da = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1, 1, 1024) as pool:  # slots far too small for the chunk
            result = pool.submit(_identity, da).result()
        assert result.equals(da)
        assert result.data.flags.writeable  # it was pickled, not mapped

    def test_staged_arguments_are_read_by_the_worker(self):
        da = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1, 1, da.nbytes) as pool:
            assert pool.submit(_sum, da).result() == pytest.approx(da.values.sum())

    def test_staged_argument_slots_come_back(self):
        da = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1, 2, da.nbytes) as pool:
            for _ in range(6):  # more rounds than there are slots
                assert pool.submit(_sum, da).result() == pytest.approx(da.values.sum())

    def test_slots_are_recycled_when_chunks_are_dropped(self):
        nbytes = xd.testing.dummy(shape=(100, 10)).nbytes
        with ProcessPool(1, 1, nbytes) as pool:
            for _ in range(2 * pool._arena.nslots):
                pool.submit(_make, 100).result()  # dropped straight away
            assert len(pool._arena._free) == pool._arena.nslots

    def test_held_chunks_hold_their_slots(self):
        nbytes = xd.testing.dummy(shape=(100, 10)).nbytes
        with ProcessPool(1, 1, nbytes) as pool:
            nslots = pool._arena.nslots
            held = [pool.submit(_make, 100).result() for _ in range(2)]
            assert len(pool._arena._free) == nslots - 2
            del held

    def test_exhausted_arena_falls_back_rather_than_blocking(self):
        da = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1, 1, da.nbytes) as pool:
            taken = []
            while (slot := pool._arena.reserve()) is not None:
                taken.append(slot)
            extra = pool.submit(_identity, da).result()
        assert extra.equals(da)
        assert extra.data.flags.writeable  # no slot left, so it was pickled

    def test_cancel_returns_the_slot(self):
        with ProcessPool(1) as pool:
            arena = pool._arena
            nslots = arena.nslots
            future = ProcessFuture(pool, Future(), arena.reserve())
            assert len(arena._free) == nslots - 1
            assert future.cancel()
            assert len(arena._free) == nslots

    def test_a_cancelled_task_still_raises(self):
        with ProcessPool(1) as pool:
            future = ProcessFuture(pool, Future(), pool._arena.reserve())
            future.cancel()
            with pytest.raises(CancelledError):
                future.result()

    def test_a_timeout_keeps_the_slot(self):
        # The task may still be writing into it, so the slot is not reusable.
        with ProcessPool(1) as pool:
            future = ProcessFuture(pool, Future(), pool._arena.reserve())
            free = len(pool._arena._free)
            with pytest.raises(TimeoutError):
                future.result(timeout=0.01)
            assert len(pool._arena._free) == free

    def test_without_shared_memory_it_is_a_plain_process_pool(self, monkeypatch):
        monkeypatch.setattr(Arena, "create", staticmethod(lambda *args: None))
        da = xd.testing.dummy(shape=(100, 10))
        with ProcessPool(1) as pool:
            assert pool._arena is None
            result = pool.submit(_identity, da).result()
        assert result.equals(da)

    def test_shutdown_unlinks_the_arena(self):
        pool = ProcessPool(1)
        path = pool._arena.path
        assert os.path.exists(path)
        pool.shutdown()
        assert not os.path.exists(path)

    def test_cancelling_a_running_task_changes_nothing(self):
        with ProcessPool(1) as pool:
            future = pool.submit(_double, 1)
            future.result()  # over, so it cannot be cancelled any more
            free = len(pool._arena._free)
            assert not future.cancel()
            assert len(pool._arena._free) == free


class TestWorkerSide:
    """What `_run` does inside a worker, exercised here in the parent.

    These run in worker processes in real use, where coverage cannot see
    them; calling them directly is what pins their behaviour down.
    """

    def test_a_result_is_parked_in_the_outbox(self):
        arena = Arena(2, 2**16)
        try:
            da = xd.testing.dummy(shape=(10, 10))
            path, offset, capacity = arena.reserve()
            result = _run(_identity, (path, offset, capacity), (da,), {})
            assert isinstance(result.data, ShmRef)
            np.testing.assert_array_equal(view(result.data), da.values)
        finally:
            arena.close()

    def test_a_result_too_big_for_the_outbox_is_left_alone(self):
        arena = Arena(2, 1024)
        try:
            da = xd.testing.dummy(shape=(100, 10))
            result = _run(_identity, (*arena.reserve()[:2], 1024), (da,), {})
            assert isinstance(result.data, np.ndarray)
        finally:
            arena.close()

    def test_without_an_outbox_the_result_passes_through(self):
        da = xd.testing.dummy(shape=(10, 10))
        assert _run(_identity, None, (da,), {}).data is da.data

    def test_a_parked_argument_is_resolved_to_a_view(self):
        arena = Arena(2, 2**16)
        try:
            da = xd.testing.dummy(shape=(10, 10))
            path, offset, _ = arena.reserve()
            staged = _park(da, path, offset)
            resolved = _resolve(staged)
            assert isinstance(resolved.data, np.ndarray)
            np.testing.assert_array_equal(resolved.values, da.values)
        finally:
            arena.close()

    def test_an_ordinary_argument_is_left_alone(self):
        assert _resolve(42) == 42


class TestGetPool:
    def test_unknown_name_lists_what_there_is(self):
        with pytest.raises(ValueError, match="no worker pool named 'fork'"):
            get_pool("fork", 1)

    def test_threads_ignores_the_arena_arguments(self):
        with get_pool("threads", 2) as pool:
            assert isinstance(pool, ThreadPoolExecutor)
            assert pool.submit(abs, -1).result() == 1

    @pytest.mark.slow
    def test_processes_sizes_its_slots_from_the_chunk(self):
        with get_pool("processes", 1, 1, 4096) as pool:
            assert isinstance(pool, ProcessPool)
            assert pool._arena.slot_nbytes == 4096

    @pytest.mark.slow
    def test_slots_default_to_the_auto_chunk_budget(self):
        with get_pool("processes", 1) as pool:
            assert pool._arena.slot_nbytes == AUTO_CHUNK_NBYTES


def test_unlinking_a_missing_arena_is_not_an_error():
    _unlink(os.path.join(_directory(), "xdas-shm-does-not-exist"))
