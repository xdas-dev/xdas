import pytest

import xdas


def pytest_configure(config):
    xdas.config.set("n_workers", 1)


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="slow test, use --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
