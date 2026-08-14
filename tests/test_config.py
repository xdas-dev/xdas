import os

import xdas.config as xc


class TestDefaults:
    def test_the_two_parallel_paths_have_their_own_knob(self):
        # One count cannot serve both: threads are sized against one array,
        # scan workers against the cost of starting a process. Tuning either
        # must leave the other alone.
        previous = xc.get("n_workers")
        xc.set("n_workers", 2)
        try:
            assert xc.get("scan_workers") == xc.Config.config["scan_workers"]
        finally:
            xc.set("n_workers", previous)

    def test_neither_default_follows_the_core_count_upwards(self):
        # A 48-core machine must not split a small array 48 ways, nor boot 48
        # interpreters to read a few headers.
        assert xc.Config.config["n_workers"] <= xc.THREAD_CAP
        assert xc.Config.config["scan_workers"] <= xc.SCAN_WORKERS


class TestTotalMemory:
    def test_reads_the_machine(self):
        # Whatever the machine, the answer must be a plausible byte count.
        total = xc.total_memory()
        assert isinstance(total, int)
        assert total > 2**28  # no machine xdas runs on has under 256 MiB

    def test_a_cgroup_limit_wins_over_the_machine(self, monkeypatch, tmp_path):
        limit = tmp_path / "memory.max"
        limit.write_text("1073741824\n")  # 1 GiB, far under any real machine
        monkeypatch.setattr(xc, "CGROUP_LIMITS", (str(limit),))
        assert xc.total_memory() == 2**30

    def test_an_unlimited_cgroup_loses_to_the_machine(self, monkeypatch, tmp_path):
        # cgroup v2 spells an absent limit "max"; v1 writes a sentinel larger
        # than any machine. Neither may become the answer.
        for content in ("max\n", "9223372036854771712\n"):
            limit = tmp_path / "memory.max"
            limit.write_text(content)
            monkeypatch.setattr(xc, "CGROUP_LIMITS", (str(limit),))
            physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            assert xc.total_memory() == physical

    def test_missing_files_are_skipped(self, monkeypatch, tmp_path):
        monkeypatch.setattr(xc, "CGROUP_LIMITS", (str(tmp_path / "absent"),))
        physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        assert xc.total_memory() == physical

    def test_falls_back_when_nothing_can_be_read(self, monkeypatch):
        def unavailable(name):
            raise ValueError(name)

        monkeypatch.setattr(xc, "CGROUP_LIMITS", ())
        monkeypatch.setattr(xc.os, "sysconf", unavailable)
        assert xc.total_memory() == xc.FALLBACK_MEMORY


class TestMemoryLimit:
    def test_default_is_a_share_of_the_machine(self):
        assert xc.get("memory_limit") == int(xc.MEMORY_FRACTION * xc.total_memory())

    def test_set_overrides_it(self):
        previous = xc.get("memory_limit")
        try:
            xc.set("memory_limit", 123)
            assert xc.get("memory_limit") == 123
        finally:
            xc.set("memory_limit", previous)
