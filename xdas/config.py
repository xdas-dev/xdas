"""
Global runtime configuration for xdas (e.g. worker-thread count).

Use :func:`get` and :func:`set` to read and write configuration values.
"""

import os


class Config:
    """Global configuration store backed by a plain dict."""

    config = {"n_workers": os.cpu_count()}


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
