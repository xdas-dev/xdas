import numpy as np
import pytest
import scipy.signal as sp

import xdas
import xdas.parallel as xp
from xdas.parallel import (
    BYTES_PER_WORKER,
    concatenate,
    get_workers_count,
    parallelize,
)


class TestParallelize:
    def test_ufunc(self):
        func = parallelize()(np.square)
        x = np.random.rand(2, 3, 5)
        expected = np.square(x)
        result = func(x)
        assert np.array_equal(result, expected)

    def test_two_in_one_out(self):
        func = parallelize(split_axis=(0, 0))(np.multiply)
        x = np.random.rand(2, 3, 5)
        y = np.random.rand(2, 3, 5)
        expected = np.multiply(x, y)
        result = func(x, y)
        assert np.array_equal(result, expected)

    def test_lfilter(self):
        # axis=-1
        func = parallelize(
            split_axis=(None, None, 0, None, 0),
            concat_axis=(0, 0),
        )(sp.lfilter)
        b = [1, 1]
        a = [1, 1]
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((20, 3, 1))
        y_exp, zf_exp = sp.lfilter(b, a, x, -1, zi)
        y_res, zf_res = func(b, a, x, -1, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)
        # axis=0
        func = parallelize(
            split_axis=(None, None, 1, None, 1),
            concat_axis=(1, 1),
        )(sp.lfilter)
        b = [1, 1]
        a = [1, 1]
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((1, 3, 50))
        y_exp, zf_exp = sp.lfilter(b, a, x, 0, zi)
        y_res, zf_res = func(b, a, x, 0, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)

    def test_sosfilter(self):
        # axis=-1
        func = parallelize(
            split_axis=(None, 0, None, 1),
            concat_axis=(0, 1),
        )(sp.sosfilt)
        sos = np.ones((5, 6))
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((5, 20, 3, 2))
        y_exp, zf_exp = sp.sosfilt(sos, x, -1, zi)
        y_res, zf_res = func(sos, x, -1, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)
        # axis=0
        func = parallelize(
            split_axis=(None, 1, None, 2),
            concat_axis=(1, 2),
        )(sp.sosfilt)
        sos = np.ones((5, 6))
        x = np.random.rand(20, 3, 50)
        zi = np.zeros((5, 2, 3, 50))
        y_exp, zf_exp = sp.sosfilt(sos, x, 0, zi)
        y_res, zf_res = func(sos, x, 0, zi)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(zf_res, zf_exp)

    def test_ignore_one_output(self):
        func = parallelize()(sp.resample)
        x = np.random.rand(20, 3, 50)
        t = np.arange(50)
        y_exp, t_exp = sp.resample(x, 30, t, axis=-1)
        y_res, t_res = func(x, 30, t, axis=-1)
        assert np.array_equal(y_res, y_exp)
        assert np.array_equal(t_res, t_exp)

    def test_parallel_multiple_outputs(self):
        # Force 2 workers to hit the parallel output path (line 104)
        func = parallelize(split_axis=(0, 0), concat_axis=(0, 0), parallel=2)(
            lambda x, y: (x + y, x - y)
        )
        x = np.random.rand(10, 5)
        y = np.random.rand(10, 5)
        result = func(x, y)
        assert len(result) == 2

    def test_parallel_size_mismatch(self):
        # Two inputs with different sizes along split axis → raises ValueError
        func = parallelize(split_axis=(0, 0), parallel=2)(np.add)
        x = np.random.rand(10, 5)
        y = np.random.rand(8, 5)  # different size on axis 0
        with pytest.raises(ValueError, match="mismatch in size"):
            func(x, y)

    def test_parallel_single_output(self):
        # parallel=2 + single-output function → covers fn tuplize path and output[0] return
        func = parallelize(parallel=2)(np.square)
        x = np.random.rand(10, 5)
        result = func(x)
        np.testing.assert_array_equal(result, np.square(x))

    def test_input_ndim_less_than_split_axis(self):
        # array ndim <= split_axis → early return from fn
        func = parallelize(split_axis=2)(np.square)
        x = np.random.rand(5)  # ndim=1, split_axis=2: 1 <= 2 → early exit
        result = func(x)
        np.testing.assert_array_equal(result, np.square(x))


class TestConcatenate:
    def test_concatenate(self):
        arrays = [np.random.rand(100, 20) for _ in range(100)]
        expected = np.concatenate(arrays)
        result = concatenate(arrays)
        assert np.array_equal(expected, result)
        expected = np.concatenate(arrays, axis=1)
        result = concatenate(arrays, axis=1)
        assert np.array_equal(expected, result)

    def test_different_ndims(self):
        with pytest.raises(ValueError, match="same number of dimensions"):
            concatenate([np.ones((3, 4)), np.ones((3,))])

    def test_different_dtypes(self):
        with pytest.raises(ValueError, match="same dtype"):
            concatenate(
                [np.ones((3,), dtype=np.float32), np.ones((3,), dtype=np.float64)]
            )

    def test_different_shape_other_axis(self):
        with pytest.raises(ValueError, match="same shape"):
            concatenate([np.ones((3, 4)), np.ones((3, 5))])

    def test_out_parameter(self):
        arrays = [np.ones((5, 3)), np.ones((5, 3))]
        out = np.empty((10, 3))
        result = concatenate(arrays, out=out)
        assert np.array_equal(result, np.ones((10, 3)))
        assert result is out

    def test_out_wrong_shape(self):
        arrays = [np.ones((5, 3)), np.ones((5, 3))]
        out = np.empty((9, 3))  # wrong shape
        with pytest.raises(ValueError, match="does not match"):
            concatenate(arrays, out=out)


class TestGetWorkersCount:
    def test_none_uses_config(self):
        # conftest sets n_workers=1
        assert get_workers_count(None) == 1

    def test_bool_true(self):
        import os

        assert get_workers_count(True) == os.cpu_count()

    def test_bool_false(self):
        assert get_workers_count(False) == 1

    def test_int(self):
        assert get_workers_count(4) == 4

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be either None, bool or int"):
            get_workers_count("invalid")

    def test_the_count_follows_the_work(self):
        previous = xdas.config.get("n_workers")
        xdas.config.set("n_workers", 16)
        try:
            # Splitting costs more than it saves until there is real work per
            # thread, so a small array is not split at all.
            assert get_workers_count(None, nbytes=0) == 1
            assert get_workers_count(None, nbytes=BYTES_PER_WORKER - 1) == 1
            assert get_workers_count(None, nbytes=4 * BYTES_PER_WORKER) == 4
            # The configured value stays the ceiling however large the array.
            assert get_workers_count(None, nbytes=1000 * BYTES_PER_WORKER) == 16
        finally:
            xdas.config.set("n_workers", previous)

    def test_an_explicit_request_ignores_the_work(self):
        assert get_workers_count(4, nbytes=0) == 4
        assert get_workers_count(False, nbytes=10**12) == 1


class FakeExecutor:
    """A pool that records how it was built instead of spawning anything."""

    built = []

    def __init__(self, max_workers, timeout=None, initializer=None):
        self.max_workers = max_workers
        self.timeout = timeout
        self.initializer = initializer
        self.alive = True
        FakeExecutor.built.append(self)

    def shutdown(self, kill_workers=False):
        self.alive = False
        self.killed = kill_workers


@pytest.fixture
def fake(monkeypatch):
    """Stub out the loky executor, and leave no pool standing either way."""
    xp.shutdown_scan_pool()
    FakeExecutor.built = []
    monkeypatch.setattr(xp, "ProcessPoolExecutor", FakeExecutor)
    yield FakeExecutor
    xp.shutdown_scan_pool()


class TestGetScanWorkers:
    def test_a_small_scan_stays_in_the_calling_process(self):
        # Starting the pool costs more than scanning this few files ever does.
        assert xp.get_scan_workers(None, 1) == 1
        assert xp.get_scan_workers(None, xp.SCAN_THRESHOLD - 1) == 1

    def test_a_large_scan_takes_the_configured_pool(self):
        assert xp.get_scan_workers(None, xp.SCAN_THRESHOLD) == xdas.config.get(
            "scan_workers"
        )

    def test_the_pool_size_follows_the_configured_worker_count(self):
        previous = xdas.config.get("scan_workers")
        xdas.config.set("scan_workers", 3)
        try:
            assert xp.get_scan_workers(None, 1000) == 3
        finally:
            xdas.config.set("scan_workers", previous)

    def test_an_explicit_request_is_honoured_however_few_the_files(self):
        # The threshold is a default, not a veto on what the caller asked for.
        assert xp.get_scan_workers(4, 1) == 4
        assert xp.get_scan_workers(False, 10_000) == 1

    def test_an_explicit_request_refuses_nonsense(self):
        with pytest.raises(TypeError):
            xp.get_scan_workers("invalid", 1000)


class TestGetScanPool:
    def test_it_builds_a_pool_that_warms_and_expires(self, fake):
        pool = xp.get_scan_pool(4)
        assert pool.max_workers == 4
        # A finite timeout is the only thing that reaps workers orphaned by a
        # parent killed with SIGKILL, which atexit never sees.
        assert pool.timeout == xp.SCAN_TIMEOUT
        assert pool.timeout is not None
        # Workers import xdas at spawn, so they warm concurrently.
        assert pool.initializer is xp._warm

    def test_the_same_pool_is_shared_across_scans(self, fake):
        assert xp.get_scan_pool(4) is xp.get_scan_pool(4)
        assert len(fake.built) == 1

    def test_a_different_worker_count_replaces_the_pool(self, fake):
        first = xp.get_scan_pool(4)
        second = xp.get_scan_pool(8)
        assert second is not first
        assert not first.alive
        assert second.max_workers == 8

    def test_warming_a_worker_imports_xdas(self):
        # Runs in a worker in real use, so it is exercised here on its own.
        xp._warm()


class TestShutdownScanPool:
    def test_it_kills_the_workers_and_forgets_the_pool(self, fake):
        pool = xp.get_scan_pool(4)
        xp.shutdown_scan_pool()
        assert not pool.alive
        assert pool.killed
        assert xp._pool is None
        assert xp._pool_workers is None

    def test_it_is_harmless_when_no_pool_is_up(self, fake):
        xp.shutdown_scan_pool()
        xp.shutdown_scan_pool()
        assert fake.built == []
