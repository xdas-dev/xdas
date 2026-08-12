"""
Global runtime configuration for xdas (e.g. worker-thread count).

Use :func:`get` and :func:`set` to read and write configuration values.
"""

import os
from typing import ClassVar

MEMORY_FRACTION = 0.25
"""Share of the machine's memory the default ``"memory_limit"`` allows."""

FALLBACK_MEMORY = 32 * 2**30
"""Memory assumed when the machine's cannot be determined."""

CGROUP_LIMITS = (
    "/sys/fs/cgroup/memory.max",  # cgroup v2
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
)
"""Where a container or a batch scheduler declares the memory of a process."""


def total_memory():
    """
    Return the memory this process can use, in bytes.

    The smallest of what the machine has and what a cgroup allows it: a
    container or a scheduler allocation is what the process actually gets,
    whatever the machine holds. The unlimited sentinel a cgroup writes when
    there is no limit is larger than the physical memory, so it loses on its
    own. Falls back to `FALLBACK_MEMORY` where neither can be read (Windows,
    a sandboxed filesystem).

    Returns
    -------
    int
        Usable memory in bytes.
    """
    limits = []
    for path in CGROUP_LIMITS:
        try:
            with open(path) as file:
                value = file.read().strip()
        except OSError:
            continue
        if value.isdigit():  # "max" spells out an absent limit
            limits.append(int(value))
    try:
        limits.append(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        pass
    return min(limits) if limits else FALLBACK_MEMORY


class Config:
    """Global configuration store backed by a plain dict."""

    config: ClassVar[dict] = {
        "n_workers": os.cpu_count(),
        # A guard against footguns, not a budget: it must sit far enough above
        # what one legitimately loads at once that it only ever fires on a
        # mistake, which a fixed number cannot do across a laptop and a node
        # with a terabyte.
        "memory_limit": int(MEMORY_FRACTION * total_memory()),
    }


def get(key):
    """
    Return the current value of configuration key *key*.

    Parameters
    ----------
    key : str
        Configuration key (e.g. ``"n_workers"``).

    Returns
    -------
    object
        The stored configuration value.
    """
    return Config.config[key]


def set(key, value):
    """
    Set configuration key *key* to *value*.

    Parameters
    ----------
    key : str
        Configuration key (e.g. ``"n_workers"``).
    value : object
        New value to store.
    """
    Config.config[key] = value
